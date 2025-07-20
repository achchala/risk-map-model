import matplotlib.pyplot as plt
import numpy as np
from model_random_forest import roads, y, model, X_test, y_test

# Predict on test set
y_pred = model.predict(X_test)

# Scatter plot: Predicted vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Crash Count")
plt.ylabel("Predicted Crash Count")
plt.title("Predicted vs Actual Crash Counts")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.show()

# Residuals
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30)
plt.xlabel("Residual (Actual - Predicted)")
plt.title("Residuals Distribution")
plt.show()
