import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def generate_synthetic_data(n_rows=50000, random_state=42):
    rng = np.random.default_rng(random_state)
    device_os_choices = ["iOS", "Android", "Other"]
    ad_position_choices = ["banner", "native", "interstitial"]
    app_category_choices = ["Gaming", "Social", "News"]

    device_os = rng.choice(device_os_choices, size=n_rows, p=[0.35, 0.55, 0.10])
    ad_position = rng.choice(ad_position_choices, size=n_rows, p=[0.6, 0.25, 0.15])
    app_category = rng.choice(app_category_choices, size=n_rows, p=[0.4, 0.4, 0.2])
    hour_of_day = rng.integers(0, 24, size=n_rows)

    df = pd.DataFrame({
        "device_os": device_os,
        "ad_position": ad_position,
        "app_category": app_category,
        "hour_of_day": hour_of_day,
    })

    base_prob = np.full(n_rows, 0.01)
    base_prob += np.where(df["device_os"] == "iOS", 0.03, 0.0)
    base_prob += np.where(df["app_category"] == "Social", 0.02, 0.0)
    base_prob += np.where(df["ad_position"] == "interstitial", 0.015, 0.0)
    base_prob += np.where((df["hour_of_day"] >= 18) & (df["hour_of_day"] <= 22), 0.01, 0.0)
    base_prob += np.where((df["hour_of_day"] >= 7) & (df["hour_of_day"] <= 9), 0.005, 0.0)
    base_prob += np.where((df["device_os"] == "iOS") & (df["app_category"] == "Social"), 0.02, 0.0)

    probs = np.clip(base_prob, 0.0, 0.99)
    clicks = rng.binomial(1, probs)
    df["click"] = clicks
    return df


def build_pipeline(categorical_features, numerical_features, random_state=42):
    # Handle OneHotEncoder constructor differences across scikit-learn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical_features),
            ("num", StandardScaler(), numerical_features),
        ]
    )

    clf = LogisticRegression(solver="liblinear", C=1.0, max_iter=200, random_state=random_state)

    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", clf)])
    return pipeline


def train_and_save(df, output_path="ctr_model.pkl", random_state=42):
    X = df.drop(columns=["click"])
    y = df["click"]

    cat_feats = ["device_os", "ad_position", "app_category"]
    num_feats = ["hour_of_day"]

    print("Building pipeline...")
    pipeline = build_pipeline(cat_feats, num_feats, random_state=random_state)

    print("Splitting data and training model...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=random_state)
    pipeline.fit(X_train, y_train)

    train_score = pipeline.score(X_train, y_train)
    val_score = pipeline.score(X_val, y_val)
    print(f"Training score: {train_score:.4f}")
    print(f"Validation score: {val_score:.4f}")

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving trained pipeline to {output_path}...")
    joblib.dump(pipeline, output_path)
    print("Save complete.")


def main():
    parser = argparse.ArgumentParser(description="Train a CTR logistic regression model on synthetic data")
    parser.add_argument("--rows", type=int, default=50000, help="Number of synthetic rows to generate")
    parser.add_argument("--out", type=str, default="ctr_model.pkl", help="Output path for the trained model")
    args = parser.parse_args()

    print(f"Generating synthetic dataset ({args.rows} rows)...")
    df = generate_synthetic_data(n_rows=args.rows)
    print("Data generation complete. Sample:")
    print(df.head())

    train_and_save(df, output_path=args.out)


if __name__ == "__main__":
    main()
