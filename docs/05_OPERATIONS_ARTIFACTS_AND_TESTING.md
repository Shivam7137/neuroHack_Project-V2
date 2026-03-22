# Operations, Artifacts, And Testing

This document is the practical operator guide for setting up the project, running the main commands, and understanding what gets written to disk.

## Installation

Base install:

```bash
python -m pip install -r requirements.txt
```

Optional extras from `pyproject.toml`:

```bash
python -m pip install ".[edf]"
python -m pip install ".[artifact]"
python -m pip install ".[runtime]"
python -m pip install ".[generator]"
python -m pip install ".[dev]"
```

Extra meanings:

- `edf`: EDF and MNE support
- `artifact`: optional artifact-cleaning stack
- `runtime`: BrainFlow-backed Cyton support
- `generator`: Torch-backed synthetic generator stack
- `dev`: pytest

## Environment Variables

Configuration is resolved by `src/config.py`.

Most important variables:

- `DATA_ROOT`
- `EEGMAT_ROOT`
- `STRESS_DATA_ROOT`
- `TUAR_ROOT`
- `EEGDENOISENET_ROOT`
- `ARTIFACTS_ROOT`
- `WINDOW_SECONDS`
- `STRIDE_SECONDS`
- `BANDPASS_LOW`
- `BANDPASS_HIGH`
- `NOTCH_FREQ`
- `RANDOM_SEED`
- `AUTO_DOWNLOAD_EEGMAT`
- `ENABLE_PYPREP`
- `ENABLE_AUTOREJECT`
- `CLEANUP_LEVEL`
- `CONCENTRATION_CLEANUP_LEVEL`
- `STRESS_CLEANUP_LEVEL`
- `ARTIFACT_CLEANUP_LEVEL`
- `CYTON_SERIAL_PORT`
- `CYTON_CHANNEL_NAMES`
- `RUNTIME_CHUNK_SECONDS`
- `RUNTIME_WINDOW_SECONDS`
- `RUNTIME_STRIDE_SECONDS`
- `RUNTIME_BUFFER_SECONDS`

Bootstrap a local config with:

```bash
copy .env.example .env
```

## Common Commands

### Training

```bash
python -m src.training.train_concentration
python -m src.training.train_stress
python -m src.training.train_artifact
python -m src.training.benchmark_cleanup --profiles none,light,medium,heavy
python -m src.training.prepare_stew
python -m src.training.train_runtime_bundle --channel-names F7,F3,F4,F8,P7,P8,O1,O2 --artifacts-root artifacts/runtime_v1
```

### Evaluation and inference

```bash
python -m src.evaluation.evaluate_all
python -m src.inference.scorer --input path/to/window.npy --sampling-rate 128
```

### Runtime

```bash
python -m src.runtime.setup_interface --source generator
python -m src.runtime.calibrate_user --source generator --user-id demo_user
python -m src.runtime.run_engine --source synthetic --chunks 10
python -m src.runtime.run_engine --source cyton --serial-port COM5
python -m src.runtime.engine_interface
python -m src.runtime.demo_live --serial-port COM5 --frames 100
python -m src.runtime.demo_playback --output artifacts/playback/demo_playback.npz --duration-sec 20
python -m src.runtime.run_playback --input artifacts/playback/demo_playback.npz
python -m src.runtime.export_playback_formats --input artifacts/playback/demo_playback.npz --output-dir artifacts/playback/export
python -m src.runtime.stream_generate --transport udp
python -m src.runtime.stream_classify --transport udp
```

### Generator

```bash
python -m src.generator.training.train_engine_from_teacher --task concentration --duration-sec 2.0 --sample-rate 128 --maxiter 20
```

## Artifact Directory Map

Current top-level artifact directories in the repo:

- `artifacts/concentration`
- `artifacts/stress`
- `artifacts/artifact`
- `artifacts/cleanup_benchmark`
- `artifacts/generator`
- `artifacts/playback`
- `artifacts/runtime_v1`
- `artifacts/stew`

Typical saved files in task directories:

- `model.pkl`
- `preprocessor.pkl`
- `summary.json`
- `comparison.json`
- metrics JSON files
- prediction CSV files
- plots

Runtime-specific additions:

- `runtime_bundle_summary.json`
- `users/<user_id>/profile.json`
- `users/<user_id>/profile.pkl`
- `users/<user_id>/sessions/<session_id>/phase_*.npz`
- `users/<user_id>/sessions/<session_id>/summary.json`

STEW-specific bundle:

- prepared train, validation, and test arrays
- binary and ordinal labels
- metadata CSVs
- preprocessing pickle
- prep summary JSON

## Tests

The repository uses `pytest`.

Run everything with:

```bash
pytest
```

Test files and coverage areas:

- `tests/test_features.py`: feature stability and channel-order behavior
- `tests/test_splitters.py`: subject-aware split behavior
- `tests/test_cleanup.py`: cleanup-level transformations
- `tests/test_stress_loader.py`: local stress dataset label and file discovery
- `tests/test_stew_loader.py`: STEW parsing and normalization
- `tests/test_artifact_loaders.py`: TUAR and EEGdenoiseNet loading
- `tests/test_artifact_support.py`: calibration round trips and optional cleaner support
- `tests/test_inference.py`: inference channel handling and failures
- `tests/test_runtime.py`: runtime routing, source helpers, transport, and engine behavior
- `tests/test_setup_calibration.py`: calibration flow, animation behavior, and runtime setup outputs
- `tests/test_playback_tools.py`: playback generation and export
- `tests/test_pipeline_smoke.py`: end-to-end smoke coverage
- `tests/test_prepare_stew.py`: STEW preparation CLI outputs
- `tests/test_generator.py`: generator datasets, models, and sampling
- `tests/test_engine_training.py`: teacher-guided engine training
- `tests/test_teacher_api.py`: baseline teacher API stability

## References And Background Material

The `references/` directory is not imported as application code. It is background material for development and experimentation.

Current checked-in reference set includes:

- `references/ao-cognitive-load-estimation/README.md`
- `references/ao-cognitive-load-estimation/Paper.docx`
- `references/ao-cognitive-load-estimation/screenshots.pdf`

## Practical Guidance

- Treat serialized preprocessors as part of the model contract, not as optional metadata.
- Retrain a runtime bundle when you change the runtime channel list.
- Use subject-aware splits and existing training helpers instead of ad hoc notebook pipelines if you want comparable results.
- Keep `artifacts/runtime_v1/` separate from generic task artifacts when working on personalized runtime flows.
