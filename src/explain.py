from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/loan_applications.csv")
MODEL_PATH = Path("models/fraud_detector.pkl")
OUTPUT_DIR = Path("notebooks")
RANDOM_STATE = 42


TEXT_COLUMNS_TO_DROP = [
    "applicant_id",
    "full_name",
    "ssn",
    "address",
    "employer_name",
    "email",
    "phone_number",
]


PLAIN_ENGLISH_FEATURES = {
    "annual_income": "stated annual income",
    "credit_score": "credit score",
    "loan_amount": "requested loan amount",
    "time_at_address_months": "time at current address",
    "time_at_employer_months": "time at current employer",
    "num_recent_inquiries": "recent credit inquiries",
    "applicant_age": "applicant age",
}


def load_model_bundle(model_path: Path) -> tuple[object, list[str]]:
    # Load the trained model artifact created by train_model.py.
    # The artifact includes both the XGBoost model and the exact feature order
    # used during training, which prevents accidental schema mismatches.
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        return bundle["model"], bundle["feature_columns"]

    raise ValueError(
        "Expected the model file to contain a dictionary with "
        "'model' and 'feature_columns'. Re-run src/train_model.py if needed."
    )


def prepare_features(data_path: Path, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    # Load the same raw application file used during model training.
    df = pd.read_csv(data_path)

    # Drop text and identifier columns because the model was trained only on
    # numeric, model-ready features.
    df = df.drop(columns=TEXT_COLUMNS_TO_DROP)

    # Convert date_of_birth into applicant_age, matching train_model.py.
    dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
    reference_date = pd.Timestamp("today").normalize()
    df["applicant_age"] = ((reference_date - dob).dt.days / 365.25).round(1)
    df = df.drop(columns=["date_of_birth"])

    # Separate the target from the explanatory fields before encoding.
    y = df["is_fraud"]
    X = df.drop(columns=["is_fraud"])

    # One-hot encode categorical fields such as loan_purpose, matching the
    # approach used during training.
    X = pd.get_dummies(X, drop_first=False)
    X = X.fillna(X.median(numeric_only=True))

    # Align the current dataset to the feature names and order saved with the
    # trained model. Missing columns are filled with 0 for safe scoring.
    X = X.reindex(columns=feature_columns, fill_value=0)

    return X, y


def create_test_split(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    # Use the same 80/20 split settings as train_model.py so explanations are
    # based on the held-out test population rather than the full dataset.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_test, y_test


def generate_shap_values(model: object, X_test: pd.DataFrame) -> shap.Explanation:
    # TreeExplainer is optimized for tree-based models like XGBoost and tells
    # us how each feature pushed predictions higher or lower.
    explainer = shap.TreeExplainer(model)
    return explainer(X_test)


def save_summary_bar_plot(shap_values: shap.Explanation, output_path: Path) -> None:
    # Chart 1: The summary bar plot ranks the strongest overall fraud drivers.
    plt.figure()
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title("Top 10 SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_beeswarm_plot(shap_values: shap.Explanation, output_path: Path) -> None:
    # Chart 2: The beeswarm plot shows both importance and whether each feature
    # generally pushes fraud risk higher or lower.
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.title("SHAP Feature Impact Direction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_single_case_waterfall(
    model: object,
    shap_values: shap.Explanation,
    X_test: pd.DataFrame,
    output_path: Path,
) -> None:
    # Chart 3: Pick the highest-risk test application and explain exactly which
    # features drove that application's fraud score.
    fraud_probabilities = model.predict_proba(X_test)[:, 1]
    highest_risk_position = int(fraud_probabilities.argmax())

    plt.figure()
    shap.plots.waterfall(shap_values[highest_risk_position], max_display=10, show=False)
    plt.title("Why the Highest-Risk Application Was Flagged")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def readable_feature_name(feature_name: str) -> str:
    if feature_name.startswith("loan_purpose_"):
        purpose = feature_name.replace("loan_purpose_", "").replace("_", " ")
        return f"loan purpose: {purpose}"
    return PLAIN_ENGLISH_FEATURES.get(feature_name, feature_name.replace("_", " "))


def print_business_summary(shap_values: shap.Explanation) -> None:
    # Average absolute SHAP values give a plain ranking of which fields most
    # influenced the model's fraud predictions across the test set.
    mean_abs_shap = pd.Series(
        abs(shap_values.values).mean(axis=0),
        index=shap_values.feature_names,
    ).sort_values(ascending=False)

    print("\nTop 5 fraud indicators in plain English:")
    for rank, (feature, value) in enumerate(mean_abs_shap.head(5).items(), start=1):
        feature_label = readable_feature_name(feature)
        print(f"{rank}. {feature_label} increases fraud risk by {value:.4f}")


def main() -> None:
    # Step 1: Load the trained model and prepare the application data using the
    # same transformations as the training pipeline.
    model, feature_columns = load_model_bundle(MODEL_PATH)
    X, y = prepare_features(DATA_PATH, feature_columns)
    X_test, _ = create_test_split(X, y)

    # Step 2: Generate SHAP values for the held-out test set.
    shap_values = generate_shap_values(model, X_test)

    # Step 3: Save all requested explainability charts as PNG files.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_summary_bar_plot(shap_values, OUTPUT_DIR / "shap_summary.png")
    save_beeswarm_plot(shap_values, OUTPUT_DIR / "shap_beeswarm.png")
    save_single_case_waterfall(model, shap_values, X_test, OUTPUT_DIR / "shap_single_case.png")

    # Step 4: Print a short business summary suitable for non-technical review.
    print_business_summary(shap_values)

    print("\nSaved SHAP visualizations:")
    print(f"- {OUTPUT_DIR / 'shap_summary.png'}")
    print(f"- {OUTPUT_DIR / 'shap_beeswarm.png'}")
    print(f"- {OUTPUT_DIR / 'shap_single_case.png'}")


if __name__ == "__main__":
    main()
