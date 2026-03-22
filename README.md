# EEG Decoder Baseline Pipeline

Baseline EEG decoder training pipeline for two independent scores:

- concentration score from EEGMAT
- stress score from a local manifest-driven dataset
- artifact-quality sidecar score from TUAR-style artifact data

For a full repository walkthrough, module map, CLI inventory, data contracts, and test coverage, see [`docs/PROJECT_REFERENCE.md`](C:\Users\shiva\School Stuff\NeuroHack V2\docs\PROJECT_REFERENCE.md).

## Setup

```bash
python -m pip install -r requirements.txt
```

Optional EDF support:

```bash
python -m pip install ".[edf]"
```

Optional runtime and generator extras:

```bash
python -m pip install ".[runtime]"
python -m pip install ".[generator]"
```

For local development and tests, the EEGMAT loader also accepts `SubjectXX_1.npz` and `SubjectXX_2.npz` files with `signal`, `sampling_rate`, and optional `channel_names`.

## Configuration

Copy the example config and adjust dataset roots:

```bash
copy .env.example .env
```

Key environment variables:

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
- `ENABLE_PYPREP`
- `ENABLE_AUTOREJECT`
- `CYTON_SERIAL_PORT`
- `CYTON_CHANNEL_NAMES`
- `RUNTIME_CHUNK_SECONDS`
- `RUNTIME_WINDOW_SECONDS`
- `RUNTIME_STRIDE_SECONDS`
- `RUNTIME_BUFFER_SECONDS`

## Run Commands

```bash
python -m src.training.train_concentration
python -m src.training.train_stress
python -m src.training.train_artifact
python -m src.training.train_concentration --cleanup-level medium
python -m src.training.train_stress --cleanup-level medium
python -m src.training.benchmark_cleanup --profiles none,light,medium,heavy
python -m src.training.build_calibration_profile --input path/to/calibration.npy --output artifacts/calibration.pkl --sampling-rate 128 --channel-names C3,C4,P3,P4
python -m src.training.prepare_stew
python -m src.evaluation.evaluate_all
python -m src.inference.scorer --input path/to/window.npy --sampling-rate 128 --calibration-profile artifacts/calibration.pkl
```

Official cleanup defaults from the benchmark:

- concentration: `light`
- stress: `none`
- artifact: `none`

Override them with:

- `CONCENTRATION_CLEANUP_LEVEL`
- `STRESS_CLEANUP_LEVEL`
- `ARTIFACT_CLEANUP_LEVEL`
- or per-run `--cleanup-level ...`

## Runtime Bridge

The runtime bridge adds a canonical Cyton-shaped stream contract:

- `8 channels`
- `250 Hz`
- fixed channel order from `CYTON_CHANNEL_NAMES`

Core modules:

- `src.runtime.eeg_frame.EEGFrame`
- `src.runtime.sources.SyntheticSource`
- `src.runtime.sources.PlaybackSource`
- `src.runtime.sources.CytonSource`
- `src.runtime.engine.StreamingEngine`
- `src.baseline.teacher_api.TeacherAPI`

The current batch scorer remains available, and `RuntimeScorer` now caches loaded model artifacts for repeated streaming use.

## STEW Preparation

Expected local STEW layout:

- `STRESS_DATA_ROOT/ratings.txt` (optional)
- `STRESS_DATA_ROOT/sub01_lo.txt`
- `STRESS_DATA_ROOT/sub01_hi.txt`
- ...

Filename parsing rules:

- `sub##_lo.txt` -> subject `sub##`, low/rest workload, binary label `0`
- `sub##_hi.txt` -> subject `sub##`, high/multitask workload, binary label `1`

`ratings.txt` is optional. When present it is parsed as:

- `subject_id, low_rating, high_rating`
- ratings `1-3 -> low`, `4-6 -> medium`, `7-9 -> high`
- ordinal targets become `0.0`, `0.5`, `1.0`

Prepare STEW artifacts with:

```bash
python -m src.training.prepare_stew
```

Artifacts are written to:

- `artifacts/stew/metadata_recordings.csv`
- `artifacts/stew/metadata_windows.csv`
- `artifacts/stew/subject_split.csv`
- `artifacts/stew/X_train.npy`
- `artifacts/stew/X_val.npy`
- `artifacts/stew/X_test.npy`
- `artifacts/stew/y_train_binary.npy`
- `artifacts/stew/y_val_binary.npy`
- `artifacts/stew/y_test_binary.npy`
- `artifacts/stew/y_train_ordinal.npy`
- `artifacts/stew/y_val_ordinal.npy`
- `artifacts/stew/y_test_ordinal.npy`
- `artifacts/stew/feature_names.json`
- `artifacts/stew/preprocessing.pkl`
- `artifacts/stew/prep_summary.json`

## Stress Dataset Assumptions

Preferred layout:

- `STRESS_DATA_ROOT/manifest.csv`
- each row contains `file_path`, `subject_id`, `label`

Optional manifest columns:

- `session_id`
- `sampling_rate`
- `channel_names`
- `file_format`

Fallback layout without a manifest:

- label folders named `natural`, `low`, `mid`, `high`
- files placed underneath those label folders
- subject ID inferred from the nearest stable parent folder or filename stem

Supported local file formats:

- `.npy`
- `.npz`
- `.csv`
- `.edf` when EDF support is installed

## Artifact Dataset Assumptions

Preferred TUAR-style layout:

- `TUAR_ROOT/manifest.csv`
- each row contains `file_path`, `subject_id`, `label`

Optional TUAR manifest columns:

- `session_id`
- `sampling_rate`
- `channel_names`

Fallback TUAR layout without a manifest:

- label folders named `clean`, `eyem`, `chew`, `shiv`, `elpp`, `musc`

Preferred EEGdenoiseNet layout:

- `EEGDENOISENET_ROOT/manifest.csv`
- each row contains `file_path`, `sample_type`

Optional EEGdenoiseNet manifest columns:

- `subject_id`
- `sampling_rate`
- `channel_names`

Fallback EEGdenoiseNet layout without a manifest:

- `EEGDENOISENET_ROOT/clean`
- `EEGDENOISENET_ROOT/eog`
- `EEGDENOISENET_ROOT/emg`
