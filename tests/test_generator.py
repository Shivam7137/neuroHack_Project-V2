"""Generator dataset and optional model tests."""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets.base import RawDatasetBundle, RawRecording
from src.generator.data.window_dataset import build_canonical_window_dataset
from src.generator.inference.sampler import GeneratorCondition, SyntheticSampler


def _bandpower(signal: np.ndarray, sample_rate: float, low: float, high: float) -> float:
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    mask = (freqs >= low) & (freqs < high)
    return float(np.mean(spectrum[mask])) if np.any(mask) else 0.0


def test_canonical_window_dataset_preserves_subject_metadata() -> None:
    bundle = RawDatasetBundle.from_recordings(
        [
            RawRecording(
                signal=np.ones((4, 768), dtype=float),
                sampling_rate=128.0,
                channel_names=["C3", "C4", "P3", "P4"],
                subject_id="subject_a",
                session_id="session_a",
                source_dataset="fixture",
                raw_label="high",
                mapped_label=1,
            )
        ]
    )
    dataset = build_canonical_window_dataset(
        bundle,
        target_channel_names=["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"],
        target_sampling_rate=250.0,
        window_seconds=2.0,
        stride_seconds=0.5,
    )
    assert dataset.windows.shape[1:] == (8, 500)
    assert dataset.metadata["subject_id"].nunique() == 1
    assert dataset.metadata.iloc[0]["subject_id"] == "subject_a"


def test_synthetic_sampler_supports_continuity_state() -> None:
    sampler = SyntheticSampler(
        channel_names=["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"],
        sample_rate=250.0,
        random_seed=3,
    )
    condition = GeneratorCondition(concentration_level=0.6, stress_level=0.2)
    first = sampler.sample(condition, duration_sec=0.5)
    second = sampler.sample(condition, duration_sec=0.5, carry_state=first.carry_state)
    assert first.data.shape == (8, 125)
    assert second.data.shape == (8, 125)
    assert np.isfinite(first.data).all()
    assert np.isfinite(second.data).all()


def test_synthetic_sampler_biases_posterior_alpha_and_frontal_theta() -> None:
    channel_names = ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]
    sampler = SyntheticSampler(
        channel_names=channel_names,
        sample_rate=250.0,
        random_seed=17,
    )
    focused = sampler.sample(GeneratorCondition(concentration_level=0.95, stress_level=0.05), duration_sec=4.0)
    stressed = sampler.sample(GeneratorCondition(concentration_level=0.05, stress_level=0.9), duration_sec=4.0)

    fp_alpha = _bandpower(focused.data[0], focused.sample_rate, 8.0, 13.0)
    o_alpha = _bandpower(focused.data[-1], focused.sample_rate, 8.0, 13.0)
    fp_theta = _bandpower(stressed.data[0], stressed.sample_rate, 4.0, 8.0)
    o_theta = _bandpower(stressed.data[-1], stressed.sample_rate, 4.0, 8.0)

    assert o_alpha > fp_alpha
    assert fp_theta > o_theta


@pytest.mark.skipif(pytest.importorskip("importlib.util").find_spec("torch") is None, reason="torch not installed")
def test_torch_generator_models_and_losses() -> None:
    import torch

    from src.generator.losses.diversity import diversity_loss
    from src.generator.losses.spectral import bandpower_loss, covariance_loss, multiresolution_stft_loss, reconstruction_loss
    from src.generator.losses.teacher import teacher_feature_loss, teacher_output_loss
    from src.generator.models.autoencoder import EEGAutoencoder
    from src.generator.models.cvae import EEGConditionalVAE

    batch = torch.randn(2, 8, 500)
    condition = torch.rand(2, 2)
    autoencoder = EEGAutoencoder()
    ae_output = autoencoder(batch)
    cvae = EEGConditionalVAE()
    cvae_output = cvae(batch, condition)
    loss = reconstruction_loss(ae_output["reconstruction"], batch)
    loss = loss + multiresolution_stft_loss(ae_output["reconstruction"], batch)
    loss = loss + bandpower_loss(ae_output["reconstruction"], batch)
    loss = loss + covariance_loss(ae_output["reconstruction"], batch)
    loss = loss + diversity_loss(cvae_output["reconstruction"])
    loss = loss + teacher_output_loss(condition[:, :1], condition[:, :1], condition[:, 1:], condition[:, 1:])
    loss = loss + teacher_feature_loss(torch.ones(2, 4), torch.ones(2, 4))
    assert ae_output["reconstruction"].shape == batch.shape
    assert cvae_output["reconstruction"].shape == batch.shape
    assert torch.isfinite(loss)
