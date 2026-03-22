# Project Overview

This documentation set covers the Python EEG project in this repository and intentionally ignores `NeuroBend/`.

## What The Project Does

The repository implements a baseline EEG decoding stack with three main outputs:

- concentration scoring from EEGMAT-style data
- stress scoring from a local manifest-driven dataset or STEW-style text recordings
- artifact-quality scoring from TUAR-style and EEGdenoiseNet-style artifact data

The same repository also contains:

- offline training and evaluation pipelines
- one-shot inference for saved EEG windows
- runtime scoring for synthetic, playback, and live Cyton streams
- user calibration and profile-based personalization for runtime use
- an experimental generator stack trained against the baseline models

## End-To-End Flow

1. Load raw recordings from dataset loaders in `src/datasets`.
2. Apply deterministic preprocessing from `src/preprocessing`.
3. Split by subject to reduce leakage.
4. Window recordings into fixed-length EEG segments.
5. Build engineered features with `src/features`.
6. Train and select models from `src/models`.
7. Save models, preprocessors, reports, plots, and summaries under `artifacts/`.
8. Reuse saved artifacts for inference and runtime scoring.

## Repository Shape

Top-level directories that matter for the Python project:

- `src/`: implementation code
- `tests/`: pytest coverage for loaders, preprocessing, training, runtime, and generator logic
- `artifacts/`: trained outputs, evaluation reports, runtime bundles, playback exports, and saved user profiles
- `data/`: expected local dataset roots
- `docs/`: project documentation
- `references/`: external papers and helper material

## Package Map

- `src/config.py`: central `Settings` dataclass for paths, preprocessing defaults, model defaults, runtime defaults, and cleanup policy
- `src/datasets`: EEGMAT, local stress, STEW, TUAR, and EEGdenoiseNet loaders
- `src/preprocessing`: filtering, normalization, cleanup, windowing, calibration, quality checks, and split logic
- `src/features`: bandpower, Hjorth, and temporal-stat feature extraction
- `src/models`: model registry plus concentration and stress model defaults
- `src/training`: task CLIs, common training pipeline, cleanup benchmarking, STEW preparation, and runtime bundle training
- `src/evaluation`: metrics, reports, and plots
- `src/inference`: saved-model loading and single-window scoring
- `src/runtime`: live and playback sources, routing, setup flow, decision engine, desktop interfaces, transport helpers, and user profiles
- `src/baseline`: teacher API used by the generator stack
- `src/generator`: experimental synthetic EEG training and sampling code
- `src/utils`: I/O, logging, and seed helpers

## Main Project Tracks

### Baseline modeling

This is the main supervised pipeline. It produces concentration, stress, and artifact models plus serialized preprocessors that define the expected channel layout, feature list, filtering, scaling, and cleanup policy.

### Runtime scoring

The runtime stack turns Cyton-like EEG chunks into smoothed concentration and stress decisions. It supports synthetic generation, playback files, or live BrainFlow-backed Cyton input.

### Personalization

The setup flow walks a user through guided states, records accepted windows, builds a user profile, and later uses that profile to remap raw concentration and stress outputs into personalized ranges.

### Synthetic generator

The generator modules are optional. They use baseline models as a teacher signal for an experimental learned EEG generation stack built around datasets, losses, samplers, and Torch-backed models.

## Current Defaults Worth Knowing

- Python requirement: `>=3.11,<3.15`
- main runtime stream: `8` channels
- default training window length: `2.0` seconds
- default training stride: `0.5` seconds
- default runtime chunk length: `0.5` seconds
- default runtime scoring window: `2.0` seconds
- default runtime stride: `0.25` seconds
- default cleanup policy:
  - concentration: `light`
  - stress: `none`
  - artifact: `none`

## Documentation Map

This five-document set is split as follows:

1. `docs/01_PROJECT_OVERVIEW.md`: project scope, architecture, and package map
2. `docs/02_DATASETS_AND_PREPROCESSING.md`: data contracts, channel policy, preprocessing, and feature extraction
3. `docs/03_TRAINING_EVALUATION_AND_INFERENCE.md`: training workflows, evaluation outputs, inference, and runtime bundle training
4. `docs/04_RUNTIME_AND_PERSONALIZATION.md`: runtime architecture, sources, streaming, setup, and user profiles
5. `docs/05_OPERATIONS_ARTIFACTS_AND_TESTING.md`: installation, environment variables, commands, artifact directories, and tests

## Important Notes

- The canonical repository reference is still `docs/PROJECT_REFERENCE.md`.
- The docs in this set are organized by workflow instead of by file listing only.
- The Python project can be understood independently of `NeuroBend/`.
