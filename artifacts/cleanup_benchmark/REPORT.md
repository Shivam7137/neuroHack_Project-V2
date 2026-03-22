# Cleanup Benchmark Report

Date: 2026-03-22

## Objective

Evaluate whether pre-training EEG cleanup treatment improves downstream model quality for:

- concentration classification
- stress prediction

Cleanup profiles benchmarked:

- `none`
- `light`
- `medium`
- `heavy`

Command used:

```bash
python -m src.training.benchmark_cleanup --profiles none,light,medium,heavy
```

Primary summary source:

- [`benchmark_summary.json`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/benchmark_summary.json)

## Primary Results

### Concentration Test Metrics

| Cleanup | ROC AUC | Balanced Accuracy | F1 | Recording ROC AUC | Pairwise Accuracy |
|---|---:|---:|---:|---:|---:|
| none | 0.7352 | 0.6871 | 0.5193 | 0.8889 | 0.8333 |
| light | 0.8009 | 0.7303 | 0.5643 | 0.8889 | 0.8333 |
| medium | 0.7913 | 0.7237 | 0.5581 | 0.8889 | 1.0000 |
| heavy | 0.7775 | 0.7047 | 0.5378 | 0.9167 | 1.0000 |

Interpretation:

- Cleanup improves concentration relative to `none`.
- `light` is the best overall concentration profile on window-level test metrics.
- `heavy` improves recording-level ROC AUC and pairwise accuracy, but loses to `light` on the main window-level metrics.

### Stress Test Metrics

| Cleanup | Macro F1 | MAE | Spearman | Recording Spearman | Pairwise Accuracy |
|---|---:|---:|---:|---:|---:|
| none | 0.2368 | 0.2640 | 0.4066 | 0.5269 | 1.0000 |
| light | 0.2186 | 0.2764 | 0.3626 | 0.3505 | 0.8571 |
| medium | 0.2720 | 0.2773 | 0.3666 | 0.3661 | 0.8571 |
| heavy | 0.2866 | 0.2717 | 0.4044 | 0.4934 | 0.8571 |

Interpretation:

- Stress does not show a clean win from cleanup.
- `heavy` gives the best stress `macro_f1`.
- `none` remains best on test `Spearman`, test `MAE`, recording `Spearman`, and pairwise accuracy.
- `light` is clearly worse than baseline for stress.

## Delta Vs Baseline

Baseline is `none`.

### Concentration Test Delta vs `none`

| Cleanup | Delta ROC AUC | Delta Balanced Accuracy | Delta F1 |
|---|---:|---:|---:|
| light | +0.0657 | +0.0432 | +0.0450 |
| medium | +0.0561 | +0.0366 | +0.0388 |
| heavy | +0.0423 | +0.0176 | +0.0185 |

### Stress Test Delta vs `none`

| Cleanup | Delta Macro F1 | Delta MAE | Delta Spearman |
|---|---:|---:|---:|
| light | -0.0182 | +0.0124 | -0.0440 |
| medium | +0.0353 | +0.0133 | -0.0400 |
| heavy | +0.0498 | +0.0077 | -0.0023 |

Interpretation:

- Concentration benefits across all non-baseline cleanup levels.
- Stress trades classification improvement for worse error/ranking behavior.

## Recommendation

Use:

- `--cleanup-level light` for concentration
- `--cleanup-level none` for stress if ranking/error metrics matter most
- `--cleanup-level heavy` for stress only if macro F1 is the priority metric

## Plot Index

### Concentration Plots

#### none

- [`confusion_matrix.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/none/concentration/plots/confusion_matrix.png)
- [`probability_distribution.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/none/concentration/plots/probability_distribution.png)
- [`roc_curve.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/none/concentration/plots/roc_curve.png)

#### light

- [`confusion_matrix.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/light/concentration/plots/confusion_matrix.png)
- [`probability_distribution.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/light/concentration/plots/probability_distribution.png)
- [`roc_curve.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/light/concentration/plots/roc_curve.png)

#### medium

- [`confusion_matrix.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/medium/concentration/plots/confusion_matrix.png)
- [`probability_distribution.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/medium/concentration/plots/probability_distribution.png)
- [`roc_curve.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/medium/concentration/plots/roc_curve.png)

#### heavy

- [`confusion_matrix.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/heavy/concentration/plots/confusion_matrix.png)
- [`probability_distribution.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/heavy/concentration/plots/probability_distribution.png)
- [`roc_curve.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/heavy/concentration/plots/roc_curve.png)

### Stress Plots

#### none

- [`confusion_matrix.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/none/stress/plots/confusion_matrix.png)
- [`score_distribution.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/none/stress/plots/score_distribution.png)
- [`score_scatter.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/none/stress/plots/score_scatter.png)

#### light

- [`confusion_matrix.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/light/stress/plots/confusion_matrix.png)
- [`score_distribution.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/light/stress/plots/score_distribution.png)
- [`score_scatter.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/light/stress/plots/score_scatter.png)

#### medium

- [`confusion_matrix.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/medium/stress/plots/confusion_matrix.png)
- [`score_distribution.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/medium/stress/plots/score_distribution.png)
- [`score_scatter.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/medium/stress/plots/score_scatter.png)

#### heavy

- [`confusion_matrix.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/heavy/stress/plots/confusion_matrix.png)
- [`score_distribution.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/heavy/stress/plots/score_distribution.png)
- [`score_scatter.png`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/heavy/stress/plots/score_scatter.png)

## Supporting Artifacts

Each profile contains the full run outputs for both tasks:

- model pickle
- preprocessor pickle
- validation/test metrics
- recording-level metrics
- pairwise metrics
- prediction CSVs
- plots
- summary JSON

Folders:

- [`none`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/none)
- [`light`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/light)
- [`medium`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/medium)
- [`heavy`](/Users/shiva/School%20Stuff/NeuroHack%20V2/artifacts/cleanup_benchmark/heavy)
