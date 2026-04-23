from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


DATA_PATH = Path("data/raw/loan_applications.csv")
MODEL_PATH = Path("models/fraud_detector.pkl")
CONFUSION_MATRIX_PATH = Path("notebooks/confusion_matrix.png")
RANDOM_STATE = 42


def load_and_prepare_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    # Step 1: Load the raw dataset so we can build a supervised fraud model.
    df = pd.read_csv(data_path)

    # These fields are identifiers or free-text values that are not directly
    # usable by the model in raw form, so we remove them from the feature set.
    columns_to_drop = [
        "applicant_id",
        "full_name",
        "ssn",
        "address",
        "employer_name",
        "email",
        "phone_number",
    ]
    df = df.drop(columns=columns_to_drop)

    # Convert date of birth into an age feature so the model receives a numeric
    # representation instead of a raw date string.
    dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
    reference_date = pd.Timestamp("today").normalize()
    df["applicant_age"] = ((reference_date - dob).dt.days / 365.25).round(1)
    df = df.drop(columns=["date_of_birth"])

    # One-hot encode the remaining categorical column(s), such as loan purpose,
    # so every model input is numeric and suitable for XGBoost.
    feature_df = pd.get_dummies(df.drop(columns=["is_fraud"]), drop_first=False)

    # Fill any remaining numeric gaps defensively in case upstream data changes.
    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))

    # Separate predictors from the target label used during supervised learning.
    X = feature_df
    y = df["is_fraud"]

    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    # Step 3: Compute scale_pos_weight so the model pays appropriate attention
    # to the minority class instead of over-optimizing for the majority class.
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / max(positive_count, 1)

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )

    # Fit the model on the training portion of the dataset.
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    # Step 4: Use both hard predictions and probabilities so we can review
    # threshold-based accuracy metrics as well as ranking quality via ROC-AUC.
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(cm)

    # Save a labeled heatmap so model performance can be reviewed visually
    # alongside the printed classification metrics.
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix — Synthetic Identity Fraud Detection")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")


def save_model(model: XGBClassifier, feature_columns: list[str], output_path: Path) -> None:
    # Step 5: Save both the trained model and the feature column order so the
    # same schema can be reused later for scoring new loan applications.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
        },
        output_path,
    )
    print(f"\nModel saved to {output_path}")


def main() -> None:
    # Step 1: Load the dataset and transform it into a numeric modeling table.
    X, y = load_and_prepare_data(DATA_PATH)

    # Step 2: Split the data into training and test sets so evaluation happens
    # on records the model has not seen during fitting.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Test set size: {X_test.shape[0]} rows")
    print(f"Number of features: {X.shape[1]}")

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, X.columns.tolist(), MODEL_PATH)


if __name__ == "__main__":
    main()
