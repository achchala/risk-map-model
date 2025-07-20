import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load data (assume spatial join and feature engineering already done)
from spatial_join_crashes import spatial_join_crashes
from data_loading import load_data
from feature_engineering import extract_features

roads, speed, lanes, road_class, ksi = load_data()
roads = extract_features(roads, speed, lanes, road_class)
roads = spatial_join_crashes(roads, ksi)

# Prepare features and target
feature_cols = ["SPEED_LIMIT", "NUMBER_OF_LANES", "high_speed"]
X = roads[feature_cols].fillna(0)
y = roads["crash_count"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Random Forest R^2: {r2_score(y_test, y_pred):.3f}")
print(f"Random Forest RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")
