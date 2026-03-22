# Project Reference

## What This Repository Does

This repository implements a baseline EEG decoding pipeline with three parallel outputs:

- concentration scoring from the EEGMAT dataset
- stress scoring from a local manifest-based dataset or STEW-style text recordings
- artifact-quality classification from TUAR-style and EEGdenoiseNet-style data

It also includes:

- offline training and evaluation CLIs
- one-shot inference for saved EEG windows
- a runtime bridge for live Cyton streams, playback files, and synthetic sources
- an experimental synthetic EEG generator stack trained against the baseline models

## High-Level Flow

1. Load raw recordings from one of the dataset loaders in `src/datasets`.
2. Apply deterministic preprocessing from `src/preprocessing`.
3. Split recordings by subject to avoid subject leakage.
4. Window each recording into fixed-length EEG segments.
5. Convert windows into engineered features with `src/features`.
6. Train and select a baseline model from `src/models`.
7. Save artifacts, reports, and metrics under `artifacts/`.
8. Reuse the saved model bundle from `src/inference` or `src/runtime`.

## Repository Layout

```text
.
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- .env.example
|-- data/
|-- artifacts/
|-- references/
|-- src/
|   |-- baseline/
|   |-- datasets/
|   |-- evaluation/
|   |-- features/
|   |-- generator/
|   |-- inference/
|   |-- models/
|   |-- preprocessing/
|   |-- runtime/
|   |-- training/
|   |-- utils/
|   `-- config.py
`-- tests/
```

## Configuration

Configuration is centralized in [`src/config.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\config.py). The `Settings` dataclass resolves all runtime paths and training defaults.

Important environment variables:

- `DATA_ROOT`: base data directory
- `EEGMAT_ROOT`: concentration dataset root
- `STRESS_DATA_ROOT`: stress dataset root
- `TUAR_ROOT`: TUAR-style artifact root
- `EEGDENOISENET_ROOT`: EEGdenoiseNet root
- `ARTIFACTS_ROOT`: output directory for trained models and reports
- `WINDOW_SECONDS`, `STRIDE_SECONDS`: window generation parameters
- `BANDPASS_LOW`, `BANDPASS_HIGH`, `NOTCH_FREQ`: signal filtering configuration
- `RANDOM_SEED`: reproducibility seed
- `ENABLE_PYPREP`, `ENABLE_AUTOREJECT`: optional cleaning integrations
- `CLEANUP_LEVEL`, `CONCENTRATION_CLEANUP_LEVEL`, `STRESS_CLEANUP_LEVEL`, `ARTIFACT_CLEANUP_LEVEL`: treatment profiles
- `CYTON_SERIAL_PORT`, `CYTON_CHANNEL_NAMES`: live runtime stream configuration
- `RUNTIME_CHUNK_SECONDS`, `RUNTIME_WINDOW_SECONDS`, `RUNTIME_STRIDE_SECONDS`, `RUNTIME_BUFFER_SECONDS`: runtime scoring parameters

Default cleanup policy currently used by the training CLIs:

- concentration: `light`
- stress: `none`
- artifact: `none`

## Package Guide

### `src/baseline`

- [`teacher_api.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\baseline\teacher_api.py): wraps the trained baseline models behind a teacher-style API for the synthetic generator stack

### `src/datasets`

- [`base.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\datasets\base.py): shared dataset dataclasses and loader contract
- [`eegmat_loader.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\datasets\eegmat_loader.py): concentration dataset loader for PhysioNet EEGMAT files
- [`stress_local_loader.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\datasets\stress_local_loader.py): local manifest-driven stress dataset loader
- [`stew_loader.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\datasets\stew_loader.py): dedicated STEW loader for workload text files and ratings
- [`artifact_tuar_loader.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\datasets\artifact_tuar_loader.py): TUAR-style artifact loader
- [`eegdenoisenet_loader.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\datasets\eegdenoisenet_loader.py): EEGdenoiseNet epoch loader and synthetic artifact augmentation helpers
- [`artifact_common.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\datasets\artifact_common.py): canonical artifact label normalization utilities

### `src/preprocessing`

- [`filters.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\preprocessing\filters.py): detrending, bandpass, notch filtering, and combined preprocessing
- [`normalization.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\preprocessing\normalization.py): channel alignment, rereferencing, robust scaling, and deterministic normalizers
- [`windowing.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\preprocessing\windowing.py): fixed-length sliding-window generation
- [`splitters.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\preprocessing\splitters.py): subject-aware train/val/test and k-fold splitting
- [`quality.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\preprocessing\quality.py): per-window quality checks
- [`cleanup.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\preprocessing\cleanup.py): cleanup treatment levels applied before feature extraction
- [`cleaners.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\preprocessing\cleaners.py): optional `pyprep` and `autoreject` integrations
- [`calibration.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\preprocessing\calibration.py): calibration profile creation and persistence for inference/runtime

### `src/features`

- [`feature_builder.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\features\feature_builder.py): canonical feature assembly entry point
- [`bandpower.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\features\bandpower.py): frequency-domain EEG features
- [`hjorth.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\features\hjorth.py): Hjorth activity, mobility, and complexity
- [`temporal_stats.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\features\temporal_stats.py): time-domain summary statistics

### `src/models`

- [`model_factory.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\models\model_factory.py): model registry and spec lookup
- [`concentration_model.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\models\concentration_model.py): concentration model defaults
- [`stress_model.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\models\stress_model.py): stress label maps and model defaults

### `src/training`

- [`common.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\training\common.py): shared preparation, training, selection, evaluation, and artifact saving logic
- [`train_concentration.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\training\train_concentration.py): concentration training CLI
- [`train_stress.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\training\train_stress.py): stress training CLI
- [`artifact_pipeline.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\training\artifact_pipeline.py): artifact-sidecar preparation and model training
- [`train_artifact.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\training\train_artifact.py): artifact-quality training CLI
- [`benchmark_cleanup.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\training\benchmark_cleanup.py): benchmark cleanup profiles across tasks
- [`prepare_stew.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\training\prepare_stew.py): STEW preprocessing and artifact generation
- [`build_calibration_profile.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\training\build_calibration_profile.py): calibration profile builder CLI

### `src/evaluation`

- [`metrics.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\evaluation\metrics.py): concentration and stress metrics
- [`artifact_metrics.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\evaluation\artifact_metrics.py): artifact-quality metrics
- [`reports.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\evaluation\reports.py): prediction tables and JSON report payloads
- [`artifact_reports.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\evaluation\artifact_reports.py): artifact prediction tables and summaries
- [`plots.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\evaluation\plots.py): confusion matrices, ROC curves, distributions, and scatter plots
- [`evaluate_all.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\evaluation\evaluate_all.py): aggregate evaluation entry point

### `src/inference`

- [`scorer.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\inference\scorer.py): one-shot inference API and CLI for scoring a single EEG window

Core classes in the inference path:

- `LoadedTaskModel`: persisted model bundle wrapper
- `PreparedTaskInput`: normalized input window state
- `RuntimeScorer`: cached scorer reused by runtime streaming

### `src/runtime`

This package exposes the live and playback bridge around the same scoring pipeline.

- [`eeg_frame.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\eeg_frame.py): canonical runtime frame container
- [`window_buffer.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\window_buffer.py): sliding buffer that emits scoring windows
- [`adapters.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\adapters.py): conversion between runtime frames and model inputs
- [`engine.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\engine.py): shared runtime scoring engine
- [`treatment.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\treatment.py): hook point for runtime-only signal treatments
- [`playback_tools.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\playback_tools.py): generate and export canonical playback recordings
- [`export_playback_formats.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\export_playback_formats.py): export CLI for BrainFlow/OpenBCI outputs
- [`run_playback.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\run_playback.py): replay a saved recording through the runtime engine
- [`demo_live.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\demo_live.py): short Cyton demo CLI
- [`demo_playback.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\demo_playback.py): generate sample playback data
- [`playback_interface.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\playback_interface.py): simple desktop playback/export UI

Runtime sources live under `src/runtime/sources`:

- [`base_source.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\sources\base_source.py): source interface
- [`cyton_source.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\sources\cyton_source.py): BrainFlow-backed OpenBCI Cyton stream
- [`playback_source.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\sources\playback_source.py): playback-from-disk source
- [`synthetic_source.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\sources\synthetic_source.py): synthetic signal source

### `src/generator`

This package is an experimental synthetic EEG stack. It is optional and requires the `generator` extra.

- `data/window_dataset.py`: canonical dataset builder for generator training
- `models/autoencoder.py`: EEG autoencoder
- `models/cvae.py`: conditional VAE
- `losses/diversity.py`: diversity regularization
- `losses/spectral.py`: spectral and covariance losses
- `losses/teacher.py`: teacher-guided losses
- `inference/sampler.py`: procedural and learned synthetic sampler
- `training/train_engine_from_teacher.py`: fit a procedural engine against the trained teacher models

### `src/utils`

- [`io.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\utils\io.py): JSON, pickle, dataframe, and directory persistence helpers
- [`logging_utils.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\utils\logging_utils.py): shared logger setup
- [`seed.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\utils\seed.py): global seeding helpers

## Main CLI Entry Points

### Training

```bash
python -m src.training.train_concentration [--cleanup-level none|light|medium|heavy]
python -m src.training.train_stress [--mode classifier|regressor] [--cleanup-level none|light|medium|heavy]
python -m src.training.train_artifact
python -m src.training.benchmark_cleanup --profiles none,light,medium,heavy
python -m src.training.prepare_stew [--data-root PATH] [--artifacts-root PATH]
python -m src.training.build_calibration_profile --input PATH --output PATH --sampling-rate 128 --channel-names C3,C4,P3,P4
```

### Evaluation and Inference

```bash
python -m src.evaluation.evaluate_all
python -m src.inference.scorer --input path/to/window.npy --sampling-rate 128
python -m src.inference.scorer --input path/to/window.npy --sampling-rate 128 --channel-names Fp1,Fp2,C3,C4,P3,P4,O1,O2
python -m src.inference.scorer --input path/to/window.npy --sampling-rate 128 --calibration-profile artifacts/calibration.pkl
```

### Runtime and Playback

```bash
python -m src.runtime.demo_live --serial-port COM3 --frames 100
python -m src.runtime.demo_playback --output artifacts/runtime/demo_playback.npz --duration-sec 20
python -m src.runtime.run_playback --input artifacts/runtime/demo_playback.npz
python -m src.runtime.export_playback_formats --input artifacts/runtime/demo_playback.npz --output-dir artifacts/runtime/export
python -m src.runtime.playback_interface
```

### Synthetic Generator

```bash
python -m src.generator.training.train_engine_from_teacher --task concentration --duration-sec 2.0 --sample-rate 128 --maxiter 20
```

## Data Contracts

### Concentration Data

Loaded through `EEGMATLoader`. The project expects the PhysioNet EEGMAT dataset, with a local fallback for `.npz` files during development.

Accepted local `.npz` fields:

- `signal`
- `sampling_rate`
- optional `channel_names`

### Stress Data

Loaded through `StressLocalLoader`. Preferred layout:

- `STRESS_DATA_ROOT/manifest.csv`
- required columns: `file_path`, `subject_id`, `label`

Optional columns:

- `session_id`
- `sampling_rate`
- `channel_names`
- `file_format`

Fallback layout:

- label folders such as `natural`, `low`, `mid`, `high`
- files under those folders

Supported file types:

- `.npy`
- `.npz`
- `.csv`
- `.edf` when EDF extras are installed
- `.txt` for STEW-style recordings

### STEW Data

Loaded through `STEWLoader` or preprocessed via `prepare_stew.py`.

Expected layout:

- `ratings.txt` optional
- `sub##_lo.txt`
- `sub##_hi.txt`

Labeling rules:

- `_lo` maps to low workload
- `_hi` maps to high workload
- `ratings.txt` converts `1-3` to low, `4-6` to medium, `7-9` to high

### Artifact Data

Artifact training combines:

- TUAR-style recordings for labeled artifact classes
- EEGdenoiseNet epochs for clean and synthetic-noise augmentation

Preferred TUAR manifest columns:

- `file_path`
- `subject_id`
- `label`

Preferred EEGdenoiseNet manifest columns:

- `file_path`
- `sample_type`

Fallback label folders:

- TUAR: `clean`, `eyem`, `chew`, `shiv`, `elpp`, `musc`
- EEGdenoiseNet: `clean`, `eog`, `emg`

## Artifacts Written by the Project

The exact file set depends on the CLI, but the repository consistently writes outputs under `ARTIFACTS_ROOT`.

Common output categories:

- trained estimator bundles
- preprocessing bundles and scalers
- feature name lists
- CSV prediction tables
- JSON summary reports
- evaluation plots
- calibration profiles
- runtime playback exports

STEW preparation writes a dedicated `artifacts/stew/` bundle containing:

- recording and window metadata CSVs
- subject split CSV
- train, validation, and test arrays
- binary and ordinal labels
- feature names JSON
- preprocessing bundle pickle
- summary JSON

## Tests

The repository uses `pytest` and keeps coverage focused on the pipeline contracts rather than only pure units.

Test coverage by area:

- [`tests/test_features.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_features.py): feature stability and channel-order behavior
- [`tests/test_splitters.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_splitters.py): subject isolation in dataset splits
- [`tests/test_cleanup.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_cleanup.py): cleanup treatment behavior
- [`tests/test_stress_loader.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_stress_loader.py): stress dataset label extraction
- [`tests/test_stew_loader.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_stew_loader.py): STEW parsing and training-only normalization
- [`tests/test_artifact_loaders.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_artifact_loaders.py): TUAR and EEGdenoiseNet loading behavior
- [`tests/test_artifact_support.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_artifact_support.py): optional cleaner fallbacks and calibration round trips
- [`tests/test_inference.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_inference.py): channel reordering and missing-channel failures
- [`tests/test_runtime.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_runtime.py): runtime scorer, sources, buffering, and engine behavior
- [`tests/test_playback_tools.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_playback_tools.py): playback generation and export
- [`tests/test_pipeline_smoke.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_pipeline_smoke.py): end-to-end training and inference smoke tests
- [`tests/test_generator.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_generator.py): synthetic generator datasets, sampler continuity, and torch-backed models
- [`tests/test_engine_training.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_engine_training.py): procedural engine fitting against the teacher API
- [`tests/test_teacher_api.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_teacher_api.py): baseline teacher API output stability
- [`tests/test_prepare_stew.py`](C:\Users\shiva\School Stuff\NeuroHack V2\tests\test_prepare_stew.py): STEW preprocessing CLI outputs

Run the full suite with:

```bash
pytest
```

## References

The `references/` directory contains external research and helper scripts used as background material rather than live package code.

Current checked-in reference set:

- `references/ao-cognitive-load-estimation/README.md`
- `references/ao-cognitive-load-estimation/Paper.docx`
- `references/ao-cognitive-load-estimation/screenshots.pdf`
- helper scripts for feature extraction, dataset generation, and SVM training

## Recommended Reading Order

If you are new to the repository, read in this order:

1. [`README.md`](C:\Users\shiva\School Stuff\NeuroHack V2\README.md)
2. [`src/config.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\config.py)
3. [`src/training/common.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\training\common.py)
4. The dataset loader for the task you care about
5. [`src/inference/scorer.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\inference\scorer.py)
6. [`src/runtime/engine.py`](C:\Users\shiva\School Stuff\NeuroHack V2\src\runtime\engine.py) if you need live scoring

