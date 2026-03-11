# Hyperparameter Stability

Retraining HurdleTemporalTrainer across 6 parameter configurations to verify
that the chosen hyperparameters are not a fragile optimum.

## Results

| Configuration | max_depth | learning_rate | max_iter | AUC-ROC | Time (s) |
|---|---|---|---|---|---|
| depth=4 lr=0.10 | 4 | 0.1 | 300 | 0.8216 | 5.8 |
| depth=6 lr=0.10 (baseline) **(baseline)** | 6 | 0.1 | 300 | 0.8163 | 8.3 |
| depth=8 lr=0.10 | 8 | 0.1 | 300 | 0.8568 | 10.5 |
| depth=6 lr=0.05 | 6 | 0.05 | 300 | 0.8871 | 9.5 |
| depth=6 lr=0.20 | 6 | 0.2 | 300 | 0.762 | 8.7 |
| depth=6 lr=0.10 iter=150 | 6 | 0.1 | 150 | 0.8169 | 5.7 |

## Summary

| Metric | Value |
|---|---|
| AUC-ROC range | 0.7620 – 0.8871 |
| AUC-ROC std | 0.0387 |
| Stable (std < 0.02) | **NO ⚠️** |

The model shows **sensitivity** to hyperparameter choice — consider further tuning.

![Hyperparameter sensitivity plot](hyperparam_sensitivity.png)