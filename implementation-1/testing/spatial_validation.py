import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from model_random_forest import roads, y, model, X_test, y_test

# Predict on test set
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Add residuals to the corresponding road segments (assuming index alignment)
roads_test = roads.loc[X_test.index].copy()
roads_test["residual"] = residuals

# Plot residuals on the map
roads_test.plot(column="residual", cmap="bwr", legend=True, linewidth=0.5)
plt.title("Residuals (Actual - Predicted) per Road Segment")
plt.show()

# --- Spatial Cross-Validation Example ---
# Split by region (e.g., 'DISTRICT' or 'NEIGHBOURHOOD_158')
region_col = "DISTRICT"  # Change as appropriate for your data
if region_col in roads.columns:
    unique_regions = roads[region_col].dropna().unique()
    for region in unique_regions:
        region_mask = roads[region_col] == region
        X_region = roads.loc[
            region_mask, ["SPEED_LIMIT", "NUMBER_OF_LANES", "high_speed"]
        ].fillna(0)
        y_region = roads.loc[region_mask, "crash_count"]
        if len(X_region) < 10:
            continue  # Skip very small regions
        X_train = roads.loc[
            ~region_mask, ["SPEED_LIMIT", "NUMBER_OF_LANES", "high_speed"]
        ].fillna(0)
        y_train = roads.loc[~region_mask, "crash_count"]
        model.fit(X_train, y_train)
        y_pred_region = model.predict(X_region)
        r2 = np.corrcoef(y_region, y_pred_region)[0, 1] if len(y_region) > 1 else np.nan
        print(
            f"Region: {region}, R^2: {r2:.3f}, RMSE: {np.sqrt(np.mean((y_region - y_pred_region)**2)):.3f}"
        )
else:
    print(
        f"Column '{region_col}' not found in roads. Please update region_col to a valid column."
    )
