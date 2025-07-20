from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def train_model(roads, target_column="crash_count"):
    # Prepare features and target
    feature_cols = [
        "SPEED_LIMIT",
        "NUM_LANES",
        "ROAD_CLASS",
        "high_speed",
    ]  # adjust as needed
    X = roads[feature_cols].fillna(0)
    y = roads[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"Test R^2 score: {score:.3f}")
    return model


# Example usage
if __name__ == "__main__":
    from data_loading import load_data
    from feature_engineering import extract_features

    roads, speed, lanes, road_class, ksi = load_data()
    features = extract_features(roads, speed, lanes, road_class, ksi)
    model = train_model(features)
