# Confidence Score Analysis Report

## Problem Summary

You noticed that many confidence scores are rated at 100%. After analysis, here's what we found:

## Key Findings

### Overall Statistics
- **Total segments analyzed**: 65,133
- **Mean confidence**: 64.89%
- **Segments with exactly 100% confidence**: 1,340 (2.06% of all segments)

### The Real Issue
**60.83% of HIGH RISK predictions have exactly 100% confidence!**

- Out of 2,203 high-risk segments, 1,340 have exactly 1.0 (100%) confidence
- **0% of LOW RISK predictions** have 100% confidence
- **0% of MEDIUM RISK predictions** have 100% confidence

This means the model is being overconfident specifically for high-risk predictions.

## How Confidence Scores Are Calculated

The confidence score is calculated as:
```python
confidence = np.max(probabilities, axis=1)
```

Where `probabilities` comes from `model.predict_proba(X_scaled)`, which returns a probability distribution over the three classes (low, medium, high).

**Confidence = the maximum probability among all classes**

For example:
- If probabilities are `[0.2, 0.1, 0.7]` → confidence = 0.7 (70%)
- If probabilities are `[0.0, 0.0, 1.0]` → confidence = 1.0 (100%)

## Why This Is Happening

### Random Forest Behavior
Random Forest's `predict_proba()` calculates probabilities as the fraction of trees that vote for each class. When **all trees vote for the same class**, the probability becomes exactly 1.0.

This is happening for high-risk predictions because:
1. **Strong feature discrimination**: The model has found features that strongly indicate high risk
2. **All trees agree**: Every tree in the forest is voting for "high risk" for these segments
3. **Possible overfitting**: The model may be too certain about high-risk patterns

### Is This Wrong?

Technically, **no** - this is expected Random Forest behavior. However, it may not be ideal because:
- It doesn't reflect model uncertainty well
- Real-world predictions should rarely be 100% certain
- It could indicate the model is overfitting to training patterns

## Potential Solutions

### Option 1: Probability Calibration (Recommended)
Use `CalibratedClassifierCV` to calibrate probabilities, making them more realistic:

```python
from sklearn.calibration import CalibratedClassifierCV

# Wrap the Random Forest with calibration
calibrated_model = CalibratedClassifierCV(
    base_estimator=rf_model,
    method='isotonic',  # or 'sigmoid'
    cv=5
)
```

This will:
- Make probabilities more calibrated to actual outcomes
- Reduce overconfidence
- Provide better uncertainty estimates

### Option 2: Adjust Random Forest Hyperparameters
Increase uncertainty by:
- Increasing `min_samples_leaf` (forces more samples per leaf)
- Decreasing `max_depth` (prevents overfitting)
- Increasing `min_samples_split` (requires more data to split)

### Option 3: Use a Different Confidence Metric
Instead of max probability, use:
- **Entropy-based confidence**: `1 - entropy(probabilities)` (higher entropy = lower confidence)
- **Margin-based confidence**: `max_prob - second_max_prob` (difference between top 2)
- **Calibrated probabilities**: Use calibrated probabilities instead of raw RF probabilities

### Option 4: Cap Maximum Confidence
Artificially cap confidence at a maximum (e.g., 0.95) to avoid 100% scores:

```python
confidence = min(np.max(probabilities, axis=1), 0.95)
```

## Recommended Action

1. **Immediate**: Implement probability calibration to reduce overconfidence
2. **Short-term**: Review features for potential data leakage (features that directly indicate high risk)
3. **Long-term**: Consider ensemble methods that provide better uncertainty quantification

## Code Locations

Confidence is calculated in:
- `src/visualization/risk_mapper.py` (line 681)
- `backend-api/app.py` (line 319)
- `src/models/model_trainer.py` (line 239)

## Next Steps

1. Run the diagnostic script: `python analyze_confidence_scores.py`
2. Review the feature importance to check for data leakage
3. Implement probability calibration
4. Re-train the model and compare confidence distributions
