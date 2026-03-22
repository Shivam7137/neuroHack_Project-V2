# Training, Evaluation, And Inference

This document covers the supervised modeling workflows and the saved artifacts they produce.

## Core Training Pipeline

The shared implementation lives in `src/training/common.py`.

High-level responsibilities:

- load raw dataset bundles
- determine the preprocessing profile for the task
- prepare windowed train, validation, and test sets
- train candidate models
- select the best model
- evaluate on validation and test splits
- save the model, preprocessor, metrics, prediction tables, plots, and summaries

Important shared types:

- `PreprocessorBundle`
- `PreparedData`
- `TrainingResult`
- `WindowQualitySummary`

## Task-Specific Training Entry Points

### Concentration

```bash
python -m src.training.train_concentration [--cleanup-level none|light|medium|heavy]
```

Default model candidates from `Settings`:

- `logistic_regression`
- `linear_svm`
- `rbf_svm`
- `random_forest`

### Stress

```bash
python -m src.training.train_stress [--mode classifier|regressor] [--cleanup-level none|light|medium|heavy]
```

Default regressor candidates:

- `svr_rbf`
- `stress_regressor_random_forest`
- `stress_regressor_hist_gradient_boosting`

Classifier benchmark candidates:

- `multinomial_logistic_regression`
- `linear_svm`
- `rbf_svm`
- `random_forest_classifier`

### Artifact sidecar

```bash
python -m src.training.train_artifact
```

Default artifact candidates:

- `multinomial_logistic_regression`
- `linear_svm`
- `random_forest_classifier`

## Cleanup Benchmarking

The repository includes a benchmark helper for cleanup policy tradeoffs:

```bash
python -m src.training.benchmark_cleanup --profiles none,light,medium,heavy
```

This compares cleanup profiles across the supported tasks and writes outputs under `artifacts/cleanup_benchmark/`.

## Runtime Bundle Training

The runtime-specific training entry point is:

```bash
python -m src.training.train_runtime_bundle --channel-names F7,F3,F4,F8,P7,P8,O1,O2 --artifacts-root artifacts/runtime_v1
```

Current CLI flags:

- `--channel-names`
- `--artifacts-root`
- `--concentration-cleanup`
- `--stress-cleanup`

What it does:

- retrains concentration and stress against one shared channel layout
- saves the bundle under `artifacts/runtime_v1/`
- validates that the saved bundle is compatible with the runtime stream
- writes `runtime_bundle_summary.json`

## Evaluation Outputs

The evaluation layer lives in `src/evaluation/`.

Important modules:

- `metrics.py`
- `artifact_metrics.py`
- `reports.py`
- `artifact_reports.py`
- `plots.py`
- `evaluate_all.py`

Typical saved outputs for concentration and stress:

- `model.pkl`
- `preprocessor.pkl`
- `summary.json`
- `comparison.json`
- `metrics_val.json`
- `metrics_test.json`
- `metrics_pairwise_val.json`
- `metrics_pairwise_test.json`
- `metrics_recording_val.json`
- `metrics_recording_test.json`
- `predictions_val.csv`
- `predictions_test.csv`
- `predictions_pairwise_val.csv`
- `predictions_pairwise_test.csv`
- `predictions_recording_val.csv`
- `predictions_recording_test.csv`
- task-specific plots such as confusion matrices, ROC curves, score distributions, and scatter plots

## Artifact Layout By Task

Common top-level artifact directories:

- `artifacts/concentration`
- `artifacts/stress`
- `artifacts/artifact`
- `artifacts/runtime_v1`
- `artifacts/stew`
- `artifacts/cleanup_benchmark`

The stress and concentration directories already contain the standard serialized model and evaluation report outputs expected by inference and runtime.

## One-Shot Inference

The inference surface is `src/inference/scorer.py`.

CLI:

```bash
python -m src.inference.scorer --input path/to/window.npy --sampling-rate 128
python -m src.inference.scorer --input path/to/window.npy --sampling-rate 128 --channel-names Fp1,Fp2,C3,C4,P3,P4,O1,O2
python -m src.inference.scorer --input path/to/window.npy --sampling-rate 128 --calibration-profile artifacts/calibration.pkl
```

Accepted input formats:

- `.npy`
- `.npz`
- `.csv`

Inference responsibilities:

- load trained models and preprocessors
- select the correct artifact directory based on cleanup settings
- transform one EEG window with the serialized preprocessing policy
- build features with the stored feature list
- run concentration and stress scoring
- optionally run artifact-quality scoring when artifact artifacts exist
- optionally apply a saved calibration profile

Important inference classes:

- `LoadedTaskModel`
- `PreparedTaskInput`
- `RuntimeScorer`

## How Saved Models Stay Compatible

Compatibility depends on the serialized `PreprocessorBundle`. That bundle records:

- required channel names
- filtering settings
- cleanup level
- target sampling rate
- trimming policy
- feature names
- scaling state
- task-specific prediction metadata

This means the runtime and inference layers both depend on the training output being structurally compatible with the incoming stream.

## Generator Training

The generator stack is optional and requires the `generator` extra.

Main entry point:

```bash
python -m src.generator.training.train_engine_from_teacher --task concentration --duration-sec 2.0 --sample-rate 128 --maxiter 20
```

The generator uses the baseline teacher API to optimize a synthetic EEG engine against baseline task behavior rather than only raw waveform reconstruction.
