# Datasets And Preprocessing

This document covers how raw EEG data enters the project and how it is normalized into model-ready windows.

## Supported Dataset Families

### Concentration data

Loader: `src/datasets/eegmat_loader.py`

Expected source:

- PhysioNet EEGMAT dataset
- local development fallback using `.npz` files

Accepted local `.npz` fields:

- `signal`
- `sampling_rate`
- optional `channel_names`

Default root resolution:

- `EEGMAT_ROOT` if set
- `./data/eeg-during-mental-arithmetic-tasks-1.0.0` if present
- otherwise `./data/eegmat`

### Stress data

Loaders:

- `src/datasets/stress_local_loader.py`
- `src/datasets/stew_loader.py`

Preferred local stress layout:

- `STRESS_DATA_ROOT/manifest.csv`
- required columns: `file_path`, `subject_id`, `label`

Optional stress manifest columns:

- `session_id`
- `sampling_rate`
- `channel_names`
- `file_format`

Fallback local stress layout:

- label folders such as `natural`, `low`, `mid`, `high`
- files nested below those folders

Supported local stress formats:

- `.npy`
- `.npz`
- `.csv`
- `.edf` when EDF extras are installed
- `.txt` for STEW-style recordings

### STEW workload data

Expected layout:

- `ratings.txt` optional
- `sub##_lo.txt`
- `sub##_hi.txt`

Parsing rules:

- `_lo` means low workload and binary label `0`
- `_hi` means high workload and binary label `1`
- `ratings.txt` maps `1-3` to low, `4-6` to medium, `7-9` to high

Default root resolution:

- `STRESS_DATA_ROOT` if set
- `./data/STEW Dataset` if present
- otherwise `./data/stress`

### Artifact data

Loaders:

- `src/datasets/artifact_tuar_loader.py`
- `src/datasets/eegdenoisenet_loader.py`

Preferred TUAR layout:

- `TUAR_ROOT/manifest.csv`
- required columns: `file_path`, `subject_id`, `label`

Fallback TUAR label folders:

- `clean`
- `eyem`
- `chew`
- `shiv`
- `elpp`
- `musc`

Preferred EEGdenoiseNet layout:

- `EEGDENOISENET_ROOT/manifest.csv`
- required columns: `file_path`, `sample_type`

Fallback EEGdenoiseNet label folders:

- `clean`
- `eog`
- `emg`

## Shared Dataset Contracts

The project uses shared dataset dataclasses from `src/datasets/base.py`:

- raw recording bundles before windowing
- metadata with subject IDs and optional session IDs
- windowed datasets used downstream by training helpers

The training stack is designed around subject-level splits so train, validation, and test sets do not mix windows from the same person unless a workflow explicitly chooses otherwise.

## Channel Policy

### Concentration task

The concentration profile is derived from EEGMAT. The training code removes channels such as `A2-A1` and `ECG` and keeps the remaining EEG channels in canonical order.

### Stress task

Stress training keeps the canonical channel order present in the incoming data. When the source is STEW, the preprocessing profile also enables average rereferencing and trims both the beginning and end of the recordings.

### Runtime bundle

Runtime-compatible training can override the channel list so concentration and stress share the same eight-channel layout defined by `src/runtime/constants.py`.

## Preprocessing Stages

Core implementation lives in:

- `src/preprocessing/filters.py`
- `src/preprocessing/normalization.py`
- `src/preprocessing/windowing.py`
- `src/preprocessing/cleanup.py`
- `src/preprocessing/quality.py`
- `src/preprocessing/calibration.py`

The typical order is:

1. Align or validate channel names.
2. Apply filtering.
3. Apply cleanup treatment if enabled.
4. Resample if the profile requires a target sampling rate.
5. Trim task-specific leading or trailing time when required.
6. Normalize recordings deterministically.
7. Window the signal.
8. Reject windows that fail quality gates.
9. Build features from accepted windows.

## Filtering Defaults

Defaults come from `src/config.py`:

- bandpass low cutoff: `1.0`
- bandpass high cutoff: `40.0`
- notch frequency: `50.0`

These can be changed with:

- `BANDPASS_LOW`
- `BANDPASS_HIGH`
- `NOTCH_FREQ`

## Cleanup Levels

Cleanup is controlled by:

- `CLEANUP_LEVEL`
- `CONCENTRATION_CLEANUP_LEVEL`
- `STRESS_CLEANUP_LEVEL`
- `ARTIFACT_CLEANUP_LEVEL`

Recognized levels in the CLI surface:

- `none`
- `light`
- `medium`
- `heavy`

Optional advanced cleaners live in `src/preprocessing/cleaners.py` and are activated only when the corresponding extras and environment flags are available.

## Window Quality Control

The current deterministic quality checks in `src/preprocessing/quality.py` reject windows for:

- flat channel variance
- extreme amplitude
- extreme variance
- line-noise excess when notch checking is enabled
- invalid PSD or invalid signal values

The thresholds come from `Settings`:

- `flat_variance_threshold`
- `max_abs_amplitude_threshold`
- `max_variance_threshold`
- `max_line_noise_ratio`
- `line_noise_frequency`

## Splitting

Split logic is handled by `src/preprocessing/splitters.py`.

Default ratios:

- train: `0.70`
- validation: `0.15`
- test: `0.15`

The project emphasizes subject-aware splits so evaluation reflects generalization across people instead of memorization across windows.

## Feature Extraction

Feature building is centered on `src/features/feature_builder.py`.

Feature families:

- bandpower features from `src/features/bandpower.py`
- Hjorth activity, mobility, and complexity from `src/features/hjorth.py`
- temporal statistics from `src/features/temporal_stats.py`

Serialized preprocessors store:

- the exact feature name list
- channel order
- scaling state
- filtering parameters
- cleanup profile
- quality configuration
- task-specific metadata such as class names or target sampling rate

## Calibration Profiles

`src/preprocessing/calibration.py` handles saved calibration profiles that can later be applied at inference time or by runtime artifact scoring.

The standalone builder CLI is:

```bash
python -m src.training.build_calibration_profile --input PATH --output PATH --sampling-rate 128 --channel-names C3,C4,P3,P4
```

## STEW Preparation Outputs

`python -m src.training.prepare_stew` writes a dedicated bundle under `artifacts/stew/` including:

- recording metadata
- window metadata
- subject split CSV
- train, validation, and test arrays
- binary and ordinal labels
- feature names
- preprocessing bundle
- summary JSON
