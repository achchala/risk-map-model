# Statistical Significance: Diebold-Mariano Test

## Setup
- **Null hypothesis H₀:** The hurdle model and historical-rate baseline have equal predictive accuracy
- **Loss function:** Squared error (MSE-based)
- **Standard error:** Newey-West HAC with lag h=1

## Result
| Metric | Value |
|---|---|
| n (test windows) | 311,072 |
| Mean loss differential d̄ | -0.000243 |
| DM statistic | -3.6369 |
| p-value (two-sided) | 0.0003 |
| Significant (p < 0.05) | **YES** |

## Interpretation
**Hurdle model outperforms** the historical-rate baseline.
The difference is statistically significant (p < 0.05). We reject H₀.

> A negative DM statistic means the model's squared errors are **smaller** than the baseline's,
> confirming genuine predictive improvement beyond what historical segment rates alone provide.