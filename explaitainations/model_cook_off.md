# Model Selection: Why the Two-Stage Hurdle Model

---

## Poisson GLM

**What it is:**
A generalized linear model that treats crash counts as Poisson-distributed. It fits a linear combination of features through a log link: `log(λ) = β₀ + β₁x₁ + ... + βₙxₙ`, then predicts `λ = e^(linear combination)`. The Poisson distribution has one parameter λ which serves as both the mean and the variance.

**Why it failed:**
- The Poisson assumption is `mean = variance`. In this dataset the variance of crash counts is 287× the mean — the data is massively overdispersed. The GLM's loss function is mis-specified from the start; it's penalizing predictions under a distribution that doesn't describe the data.
- With 98.47% zeros, the log-likelihood is dominated by zero windows. The model's optimal strategy under Poisson loss becomes "predict a very small λ everywhere." This minimizes total error but produces a flat, near-zero risk surface that is useless for ranking — every segment looks equally risky.
- It is linear in the features after the log transform. It cannot learn interactions like "freezing AND arterial AND school zone AND rush hour" without those being manually specified.

---

## Negative Binomial GLM

**What it is:**
An extension of the Poisson GLM that adds a dispersion parameter `α`, allowing variance to exceed the mean: `variance = λ + α·λ²`. This relaxes the mean=variance constraint.

**Why it failed:**
- It handles overdispersion correctly but still models all zeros with a single mechanism. In this data there are two types of zero: *structural zeros* (segments where crashes almost never happen regardless of conditions) and *sampling zeros* (segments where crashes do happen but didn't during this specific window). The NB GLM can't distinguish these — it treats every zero as the same kind of absence.
- Because it doesn't separate occurrence from intensity, it still over-weights the sea of zeros during fitting. The result is better calibrated on the full count distribution but still poor at ranking — it can't reliably identify the top 5% of high-risk windows.
- Still linear in features after the log transform. Same interaction limitation as Poisson GLM.

---

## Zero-Inflated Negative Binomial (ZINB)

**What it is:**
A two-component mixture model that explicitly separates zeros into two sources. Component 1 is a binary model that generates "always-zero" outcomes (structural zeros). Component 2 is a Negative Binomial that generates counts including zero (sampling zeros). The two components are fit jointly via EM or maximum likelihood.

**Why it failed:**
- Conceptually the right idea — it acknowledges the two-process structure. But the parameters of both components are estimated simultaneously under a shared likelihood. This joint optimization is sensitive to initialization and tends to have flat or multimodal likelihood surfaces, making convergence unreliable.
- No native gradient boosting implementation exists. ZINB is a parametric GLM — it fits linear predictors, so it shares the same feature interaction limitation as the Poisson and NB GLMs.
- Slower to train and harder to tune than tree-based models. With 66 features and hundreds of thousands of rows, the EM fitting becomes computationally expensive and unstable.
- The end result is similar to hurdle in structure but worse in practice because it lacks the predictive power of gradient boosting and requires careful statistical tuning for each dataset.

---

## XGBoost (count regression)

**What it is:**
A gradient boosted tree ensemble that optimizes a differentiable loss over sequential shallow trees. Far more flexible than any GLM — it can learn arbitrary non-linear interactions between features without manual specification. Can be configured with a Poisson or squared error loss to predict counts.

**Why it failed:**
- The core problem: it uses a single model to predict the count directly, including all zeros. At 98.47% zeros, gradient updates are dominated by the majority class. The trees learn mostly "when is this a zero?" and get very little gradient signal from the rare high-count events that actually matter for routing.
- It doesn't natively separate occurrence from intensity. To predict a window with 5 crashes, the same trees must also correctly predict the 98.47% of windows with 0 crashes. These two tasks have partially conflicting gradient signals — features that push predictions toward 0 for non-crash windows also suppress predictions for crash windows.
- Unlike the hurdle, there is no Stage 2 that trains exclusively on crash windows. The rare 5-crash events contribute ~1.5% of the gradient. They are statistically invisible to the optimization.

---

## Why the Hurdle Wins

Each rejected model fails for one of two reasons: either it **misspecifies the distribution** (Poisson, NB) or it **conflates two separate problems** (ZINB without boosting, XGBoost single-stage). The hurdle solves both:

- Stage 1 uses gradient boosting purely for the binary question, with the full 98.47% zero signal directed at one task — no interference from count estimation
- Stage 2 uses gradient boosting purely for the count question, trained only on crash windows — the distribution it sees is no longer zero-inflated
- The tree ensemble in both stages captures non-linear feature interactions that no GLM can
- Isotonic calibration is applied post-hoc on a held-out set — something GLMs don't need but that tree classifiers specifically require because their outputs are scores, not probabilities
