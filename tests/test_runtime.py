"""Runtime bridge tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from src.config import Settings
from src.inference.scorer import RuntimeScorer, score_window
from src.preprocessing.normalization import DeterministicRecordingNormalizer
from src.runtime.adaptation import AdaptationLayer, default_profile
from src.runtime.baseline import BaselineInference
from src.runtime.contracts import EEGChunk
from src.runtime.decision_engine import DecisionEngine
from src.runtime.eeg_frame import EEGFrame
from src.runtime.engine import StreamingEngine
from src.runtime.router import SourceRouter
from src.runtime.run_engine import build_router, build_source, stream_outputs
from src.runtime.sources.playback_source import PlaybackSource
from src.runtime.sources.synthetic_source import SyntheticSource
from src.runtime.stream_transport import (
    decode_chunk,
    encode_chunk,
    pack_chunk_datagram,
    required_stream_channel_names,
    unpack_chunk_datagram,
)
from src.runtime.window_buffer import WindowBuffer
from src.training.common import PreprocessorBundle


def _canonicalize_fixture_window(window: np.ndarray) -> tuple[np.ndarray, list[str]]:
    settings = Settings()
    channel_names = list(settings.cyton_channel_names)
    resampled = resample_poly(window, up=250, down=128, axis=-1)
    canonical = np.zeros((len(channel_names), resampled.shape[1]), dtype=float)
    mapping = {
        "C3": resampled[0],
        "C4": resampled[1],
        "P3": resampled[2],
        "P4": resampled[3],
    }
    canonical[channel_names.index("Fp1")] = 0.5 * mapping["C3"]
    canonical[channel_names.index("Fp2")] = 0.5 * mapping["C4"]
    canonical[channel_names.index("C3")] = mapping["C3"]
    canonical[channel_names.index("C4")] = mapping["C4"]
    canonical[channel_names.index("P3")] = mapping["P3"]
    canonical[channel_names.index("P4")] = mapping["P4"]
    canonical[channel_names.index("O1")] = 0.5 * mapping["P3"]
    canonical[channel_names.index("O2")] = 0.5 * mapping["P4"]
    return canonical, channel_names


def test_runtime_scorer_scores_canonical_window(trained_artifacts: dict[str, Path]) -> None:
    payload = np.load(trained_artifacts["eegmat_root"] / "Subject02_2.npz", allow_pickle=True)
    window, channel_names = _canonicalize_fixture_window(payload["signal"])
    scorer = RuntimeScorer(artifacts_root=trained_artifacts["artifacts_root"])
    result = scorer.score_window(window, sampling_rate=250.0, channel_names=channel_names)
    assert 0.0 <= result["concentration_probability"] <= 1.0
    assert 0.0 <= result["stress_score"] <= 100.0


def test_runtime_scorer_matches_one_shot_scorer(trained_artifacts: dict[str, Path]) -> None:
    payload = np.load(trained_artifacts["eegmat_root"] / "Subject03_2.npz", allow_pickle=True)
    window, channel_names = _canonicalize_fixture_window(payload["signal"])
    scorer = RuntimeScorer(artifacts_root=trained_artifacts["artifacts_root"])
    persistent = scorer.score_window(window, sampling_rate=250.0, channel_names=channel_names)
    one_shot = score_window(
        window,
        sampling_rate=250.0,
        channel_names=channel_names,
        artifacts_root=trained_artifacts["artifacts_root"],
    )
    assert persistent["concentration_probability"] == one_shot["concentration_probability"]
    assert persistent["stress_score"] == one_shot["stress_score"]
    assert persistent["quality_label"] == one_shot["quality_label"]


def test_runtime_scorer_fails_on_missing_required_channels(trained_artifacts: dict[str, Path]) -> None:
    payload = np.load(trained_artifacts["eegmat_root"] / "Subject02_2.npz", allow_pickle=True)
    window, channel_names = _canonicalize_fixture_window(payload["signal"])
    keep = [name for name in channel_names if name not in {"P3", "P4"}]
    reduced = window[[channel_names.index(name) for name in keep], :]
    scorer = RuntimeScorer(artifacts_root=trained_artifacts["artifacts_root"])
    try:
        scorer.score_window(reduced, sampling_rate=250.0, channel_names=keep)
    except ValueError as exc:
        assert "Missing required channels" in str(exc)
    else:
        raise AssertionError("Expected scoring to fail when baseline-required channels are missing.")


def test_preprocessor_can_skip_training_trim_for_runtime_windows() -> None:
    bundle = PreprocessorBundle(
        task_name="stress",
        profile_name="test",
        channel_names=["C3"],
        raw_normalizer=DeterministicRecordingNormalizer(
            channel_names=["C3"],
            apply_recording_robust_scaling=False,
        ),
        trim_seconds_start=1.0,
        trim_seconds_end=1.0,
        bandpass_low=None,
        bandpass_high=None,
        notch_freq=None,
    )
    signal = np.arange(30, dtype=float).reshape(1, 30)
    trimmed, _ = bundle.transform_raw_with_sampling_rate(signal, sampling_rate=10.0, channel_names=["C3"])
    untrimmed, _ = bundle.transform_raw_with_sampling_rate(
        signal,
        sampling_rate=10.0,
        channel_names=["C3"],
        apply_trim=False,
    )
    assert trimmed.shape == (1, 10)
    assert untrimmed.shape == (1, 30)


def test_window_buffer_emits_expected_windows() -> None:
    channel_names = list(Settings().cyton_channel_names)
    buffer = WindowBuffer(
        channel_names=channel_names,
        sample_rate=250.0,
        window_seconds=2.0,
        stride_seconds=0.5,
        buffer_seconds=10.0,
    )
    for index in range(3):
        frame = EEGFrame(
            timestamp=index * 0.5,
            sample_rate=250.0,
            channel_names=channel_names,
            data=np.full((8, 125), fill_value=index, dtype=float),
            source="synthetic",
        )
        buffer.append(frame)
        assert buffer.pop_ready_windows() == []
    buffer.append(
        EEGFrame(
            timestamp=1.5,
            sample_rate=250.0,
            channel_names=channel_names,
            data=np.full((8, 125), fill_value=3, dtype=float),
            source="synthetic",
        )
    )
    first = buffer.pop_ready_windows()
    assert len(first) == 1
    assert first[0].data.shape == (8, 500)
    buffer.append(
        EEGFrame(
            timestamp=2.0,
            sample_rate=250.0,
            channel_names=channel_names,
            data=np.full((8, 125), fill_value=4, dtype=float),
            source="synthetic",
        )
    )
    second = buffer.pop_ready_windows()
    assert len(second) == 1
    assert second[0].timestamp > first[0].timestamp


def test_window_buffer_uses_exact_average_quarter_second_stride() -> None:
    channel_names = list(Settings().cyton_channel_names)
    buffer = WindowBuffer(
        channel_names=channel_names,
        sample_rate=250.0,
        window_seconds=2.0,
        stride_seconds=0.25,
        buffer_seconds=10.0,
    )
    windows = []
    for index in range(8):
        buffer.append(
            EEGFrame(
                timestamp=index * 0.5,
                sample_rate=250.0,
                channel_names=channel_names,
                data=np.full((8, 125), fill_value=index, dtype=float),
                source="synthetic",
            )
        )
        windows.extend(buffer.pop_ready_windows())
    starts = [int(round(window.timestamp_start * 250.0)) for window in windows]
    assert starts == [0, 62, 125, 187, 250, 312, 375, 437, 500]


def test_synthetic_source_emits_canonical_frames() -> None:
    source = SyntheticSource()
    source.start()
    first = source.read_frame()
    second = source.read_frame()
    source.stop()
    assert first.data.shape == (8, 125)
    assert second.timestamp > first.timestamp
    assert first.source == "synthetic"


def test_synthetic_source_accepts_custom_channel_layout() -> None:
    channel_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "Cz"]
    source = SyntheticSource(channel_names=channel_names)
    source.start()
    frame = source.read_frame()
    source.stop()
    assert frame.channel_names == channel_names
    assert frame.data.shape == (len(channel_names), 125)


def test_playback_source_reads_saved_recording(tmp_path: Path) -> None:
    channel_names = list(Settings().cyton_channel_names)
    path = tmp_path / "recording.npz"
    signal = np.arange(8 * 500, dtype=float).reshape(8, 500)
    np.savez(path, signal=signal, sampling_rate=250.0, channel_names=np.asarray(channel_names))
    source = PlaybackSource(path)
    source.start()
    first = source.read_frame()
    second = source.read_frame()
    source.stop()
    assert first.data.shape == (8, 125)
    assert second.timestamp > first.timestamp
    assert first.channel_names == channel_names


def test_source_router_switches_sources_and_tags_metadata(tmp_path: Path) -> None:
    channel_names = list(Settings().cyton_channel_names)
    path = tmp_path / "recording.npz"
    signal = np.arange(8 * 250, dtype=float).reshape(8, 250)
    np.savez(path, signal=signal, sampling_rate=250.0, channel_names=np.asarray(channel_names))
    router = SourceRouter(
        {
            "synthetic": SyntheticSource(),
            "playback": PlaybackSource(path),
        }
    )
    router.set_active("synthetic")
    synthetic_chunk = router.read_chunk()
    assert synthetic_chunk.metadata["source_name"] == "synthetic"
    router.set_active("playback")
    playback_chunk = router.read_chunk()
    assert playback_chunk.metadata["source_name"] == "playback"
    router.stop()


def test_run_engine_helpers_stream_synthetic_outputs(trained_artifacts: dict[str, Path]) -> None:
    class Args:
        source = "synthetic"
        chunk_seconds = 0.5
        concentration = 0.7
        stress = 0.25
        seed = 42
        input = ""
        serial_port = ""

    source = build_source(Args())
    router = build_router("synthetic", source)
    engine = DecisionEngine(
        router=router,
        baseline=BaselineInference(artifacts_root=trained_artifacts["artifacts_root"]),
    )
    outputs = list(stream_outputs(router=router, engine=engine, max_chunks=5))
    router.stop()
    assert len(outputs) == 3
    assert outputs[0]["source"] == "synthetic"
    assert outputs[0]["state"] in {"neutral", "focused", "stressed", "mixed"}


def test_adaptation_default_profile_is_near_identity() -> None:
    channel_names = list(Settings().cyton_channel_names)
    chunk = EEGChunk(
        timestamp_start=0.0,
        timestamp_end=0.5,
        sample_rate=250.0,
        channel_names=channel_names,
        data=np.arange(8 * 125, dtype=float).reshape(8, 125),
    )
    adaptation = AdaptationLayer(profile=default_profile())
    result = adaptation.transform(chunk)
    assert np.allclose(result.chunk.data, chunk.data)
    assert result.quality["profile_enabled"] is False


def test_stream_chunk_round_trip() -> None:
    chunk = EEGChunk(
        timestamp_start=1.25,
        timestamp_end=1.75,
        sample_rate=250.0,
        channel_names=["Fp1", "Fp2"],
        data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float),
        metadata={"source_name": "synthetic", "seed": 42},
    )
    decoded = decode_chunk(encode_chunk(chunk))
    assert decoded.timestamp_start == pytest.approx(chunk.timestamp_start)
    assert decoded.timestamp_end == pytest.approx(chunk.timestamp_end)
    assert decoded.channel_names == chunk.channel_names
    assert np.allclose(decoded.data, chunk.data)
    assert decoded.metadata == chunk.metadata


def test_stream_chunk_udp_datagram_round_trip() -> None:
    chunk = EEGChunk(
        timestamp_start=2.0,
        timestamp_end=2.5,
        sample_rate=250.0,
        channel_names=["Fp1", "Fp2", "C3"],
        data=np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float),
        metadata={"source_name": "synthetic", "seed": 7},
    )
    decoded, sequence = unpack_chunk_datagram(pack_chunk_datagram(chunk, sequence=11))
    assert sequence == 11
    assert decoded.timestamp_start == pytest.approx(chunk.timestamp_start)
    assert decoded.timestamp_end == pytest.approx(chunk.timestamp_end)
    assert decoded.channel_names == chunk.channel_names
    assert np.allclose(decoded.data, chunk.data)
    assert decoded.metadata == chunk.metadata


def test_baseline_inference_matches_runtime_scorer(trained_artifacts: dict[str, Path]) -> None:
    payload = np.load(trained_artifacts["eegmat_root"] / "Subject03_2.npz", allow_pickle=True)
    window, channel_names = _canonicalize_fixture_window(payload["signal"])
    scorer = RuntimeScorer(artifacts_root=trained_artifacts["artifacts_root"])
    baseline = BaselineInference(scorer=scorer)
    result = scorer.score_window(window, sampling_rate=250.0, channel_names=channel_names)
    prediction = baseline.predict_with_details(window, sampling_rate=250.0, channel_names=channel_names)
    assert prediction.scores.concentration == pytest.approx(result["concentration_probability"])
    assert prediction.scores.stress == pytest.approx(result["stress_score"] / 100.0)
    assert prediction.scores.quality == pytest.approx(result["quality_score"] / 100.0)


def test_required_stream_channel_names_cover_saved_models(trained_artifacts: dict[str, Path]) -> None:
    channel_names = required_stream_channel_names(trained_artifacts["artifacts_root"])
    scorer = RuntimeScorer(artifacts_root=trained_artifacts["artifacts_root"])
    expected = []
    for task_name in ("concentration", "stress", "artifact"):
        if task_name not in scorer.models:
            continue
        for channel_name in scorer.models[task_name].preprocessor.channel_names:
            if channel_name not in expected:
                expected.append(channel_name)
    assert channel_names == expected


def test_decision_engine_ignores_source_identity(trained_artifacts: dict[str, Path]) -> None:
    payload = np.load(trained_artifacts["eegmat_root"] / "Subject02_2.npz", allow_pickle=True)
    window, channel_names = _canonicalize_fixture_window(payload["signal"])
    chunks = [
        EEGChunk(
            timestamp_start=index * 0.5,
            timestamp_end=(index + 1) * 0.5,
            sample_rate=250.0,
            channel_names=channel_names,
            data=window[:, index * 125 : (index + 1) * 125],
            metadata={"source_name": "synthetic"},
        )
        for index in range(4)
    ]
    mirror_chunks = [
        EEGChunk(
            timestamp_start=chunk.timestamp_start,
            timestamp_end=chunk.timestamp_end,
            sample_rate=chunk.sample_rate,
            channel_names=list(chunk.channel_names),
            data=chunk.data.copy(),
            metadata={"source_name": "cyton"},
        )
        for chunk in chunks
    ]
    first_engine = DecisionEngine(baseline=BaselineInference(artifacts_root=trained_artifacts["artifacts_root"]))
    second_engine = DecisionEngine(baseline=BaselineInference(artifacts_root=trained_artifacts["artifacts_root"]))
    first_outputs = []
    second_outputs = []
    for first_chunk, second_chunk in zip(chunks, mirror_chunks, strict=True):
        first_outputs.extend(first_engine.process_chunk(first_chunk))
        second_outputs.extend(second_engine.process_chunk(second_chunk))
    assert len(first_outputs) == len(second_outputs) == 1
    assert first_outputs[0].state == second_outputs[0].state
    assert first_outputs[0].concentration_raw == pytest.approx(second_outputs[0].concentration_raw)
    assert first_outputs[0].stress_raw == pytest.approx(second_outputs[0].stress_raw)
    assert first_outputs[0].quality == pytest.approx(second_outputs[0].quality)


def test_streaming_engine_scores_after_five_chunks_with_quarter_second_stride(trained_artifacts: dict[str, Path]) -> None:
    source = SyntheticSource()
    source.start()
    engine = StreamingEngine(artifacts_root=trained_artifacts["artifacts_root"])
    outputs = []
    for _ in range(3):
        outputs.extend(engine.process_frame(source.read_frame()))
    assert outputs == []
    outputs.extend(engine.process_frame(source.read_frame()))
    outputs.extend(engine.process_frame(source.read_frame()))
    source.stop()
    assert len(outputs) == 3
    assert outputs[0].metadata["decision_state"] in {"neutral", "focused", "stressed", "mixed"}
    assert outputs[2].timestamp > outputs[1].timestamp > outputs[0].timestamp


def test_streaming_engine_scores_generated_union_channel_stream(trained_artifacts: dict[str, Path]) -> None:
    channel_names = required_stream_channel_names(trained_artifacts["artifacts_root"])
    source = SyntheticSource(channel_names=channel_names)
    source.start()
    engine = StreamingEngine(artifacts_root=trained_artifacts["artifacts_root"])
    outputs = []
    for _ in range(5):
        outputs.extend(engine.process_frame(source.read_frame()))
    source.stop()
    assert len(outputs) == 3
    assert outputs[0].source == "synthetic"
    assert 0.0 <= outputs[0].concentration_probability <= 1.0
    assert 0.0 <= outputs[0].stress_score <= 100.0
