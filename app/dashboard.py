from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Wedge


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "fraud_detector.pkl"


LOAN_PURPOSES = [
    "personal",
    "home_improvement",
    "debt_consolidation",
    "medical",
    "vacation",
    "moving",
    "other",
]


st.set_page_config(
    page_title="Synthetic Identity Fraud Detection System",
    page_icon="SI",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main-header {
        padding: 1.5rem 1.75rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, #0f2f2f 0%, #155e63 45%, #f6b73c 100%);
        color: #ffffff;
        margin-bottom: 1.25rem;
        box-shadow: 0 16px 40px rgba(15, 47, 47, 0.18);
    }
    .main-header h1 {
        margin-bottom: 0.25rem;
        font-size: 2.4rem;
        line-height: 1.1;
    }
    .main-header p {
        margin: 0;
        font-size: 1.05rem;
        opacity: 0.95;
    }
    .accuracy-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        font-weight: 700;
    }
    .risk-card {
        padding: 1.2rem 1.4rem;
        border-radius: 1rem;
        border: 1px solid #e7ecec;
        background: #ffffff;
        box-shadow: 0 10px 28px rgba(20, 50, 50, 0.08);
    }
    .risk-percent {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
    }
    .risk-label {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 0.08em;
    }
    .recommendation {
        padding: 1rem 1.2rem;
        border-radius: 0.9rem;
        font-size: 1.4rem;
        font-weight: 800;
        text-align: center;
    }
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #d8e0e0;
        color: #546363;
        text-align: center;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle() -> tuple[object, list[str]]:
    # Load the trained model and saved feature order produced by train_model.py.
    # Keeping the feature order prevents scoring errors when one-hot encoded
    # columns are created from a single new application.
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_columns"]


def build_model_input(application: dict[str, object], feature_columns: list[str]) -> pd.DataFrame:
    # Build a one-row table using the same fields that train_model.py created:
    # numeric application fields plus one-hot encoded loan purpose values.
    raw_input = pd.DataFrame([application])
    encoded_input = pd.get_dummies(raw_input, drop_first=False)

    # Align to the exact training feature names. Any purpose category not seen
    # during training is represented by zeros across the saved purpose columns.
    return encoded_input.reindex(columns=feature_columns, fill_value=0)


def classify_risk(probability: float) -> tuple[str, str, str]:
    # Convert a numeric score into simple action bands for business users.
    if probability < 0.30:
        return "LOW", "#16803c", "Approve"
    if probability <= 0.70:
        return "MEDIUM", "#c58b00", "Review"
    return "HIGH", "#c0392b", "Decline"


def make_gauge(probability: float) -> plt.Figure:
    # Draw a simple semicircle gauge so the dashboard works without requiring
    # an extra charting package beyond matplotlib.
    score = probability * 100
    fig, ax = plt.subplots(figsize=(7, 3.6), subplot_kw={"aspect": "equal"})
    ax.axis("off")

    bands = [
        (180, 126, "#16803c"),
        (126, 54, "#f2c94c"),
        (54, 0, "#c0392b"),
    ]
    for start, end, color in bands:
        ax.add_patch(Wedge((0, 0), 1.0, end, start, width=0.25, facecolor=color, alpha=0.9))

    needle_angle = 180 - (score / 100 * 180)
    ax.plot(
        [0, 0.78 * np.cos(np.deg2rad(needle_angle))],
        [0, 0.78 * np.sin(np.deg2rad(needle_angle))],
        color="#1f2929",
        linewidth=4,
    )
    ax.scatter([0], [0], s=90, color="#1f2929")
    ax.text(0, -0.22, f"{score:.1f}%", ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(-0.96, -0.08, "0%", ha="center", fontsize=11)
    ax.text(0.96, -0.08, "100%", ha="center", fontsize=11)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.35, 1.1)

    return fig


def top_risk_factors(application: dict[str, object], probability: float) -> list[str]:
    # These plain-language drivers explain the most visible business reasons
    # an application may look risky. They complement the model score by making
    # the result easier to review.
    income = float(application["annual_income"])
    loan_amount = float(application["loan_amount"])
    credit_score = int(application["credit_score"])
    address_months = int(application["time_at_address_months"])
    employer_months = int(application["time_at_employer_months"])
    inquiries = int(application["num_recent_inquiries"])
    applicant_age = int(application["applicant_age"])
    loan_to_income = loan_amount / max(income, 1)

    factors: list[tuple[float, str]] = []
    factors.append((loan_to_income, f"Loan amount is {loan_to_income:.1f}x stated annual income"))

    if address_months <= 6 and employer_months <= 6:
        factors.append((1.2, "Very short history at both address and employer"))
    elif address_months <= 6:
        factors.append((0.7, "Applicant has limited time at current address"))
    elif employer_months <= 6:
        factors.append((0.7, "Applicant has limited time at current employer"))

    if credit_score < 580:
        factors.append((1.0, "Credit score is in a high-risk range"))
    elif credit_score > 790 and income < 35_000:
        factors.append((0.9, "High credit score paired with unusually low income"))

    if inquiries >= 8:
        factors.append((0.85, "Many recent credit inquiries suggest elevated credit-seeking behavior"))

    if applicant_age < 22 and loan_amount > 60_000:
        factors.append((0.6, "Young applicant requesting a large loan"))

    if probability < 0.30:
        factors.append((0.5, "Model found no major fraud pattern in the provided fields"))

    return [factor for _, factor in sorted(factors, reverse=True)[:3]]


st.markdown(
    """
    <div class="main-header">
        <h1>Synthetic Identity Fraud Detection System</h1>
        <p>AI-powered risk scoring tool for identifying suspicious loan application patterns before approval.</p>
        <span class="accuracy-badge">Model Accuracy: 92%</span>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Application Input Form")
    st.caption("Enter applicant details, then run the risk analysis.")

    annual_income = st.number_input(
        "Annual income",
        min_value=10_000,
        max_value=250_000,
        value=75_000,
        step=1_000,
    )
    loan_amount = st.number_input(
        "Loan amount",
        min_value=1_000,
        max_value=800_000,
        value=35_000,
        step=1_000,
    )
    credit_score = st.slider("Credit score", min_value=300, max_value=850, value=680)
    time_at_address_months = st.slider("Time at address, months", min_value=0, max_value=360, value=36)
    time_at_employer_months = st.slider("Time at employer, months", min_value=0, max_value=360, value=48)
    num_recent_inquiries = st.slider("Recent credit inquiries", min_value=0, max_value=20, value=2)
    applicant_age = st.slider("Applicant age", min_value=18, max_value=80, value=38)
    loan_purpose = st.selectbox("Loan purpose", LOAN_PURPOSES)

    analyze_button = st.button("Analyze Application", type="primary", use_container_width=True)


application_input = {
    "annual_income": annual_income,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "loan_purpose": loan_purpose,
    "time_at_address_months": time_at_address_months,
    "time_at_employer_months": time_at_employer_months,
    "num_recent_inquiries": num_recent_inquiries,
    "applicant_age": applicant_age,
}


if analyze_button:
    model, feature_columns = load_model_bundle()
    model_input = build_model_input(application_input, feature_columns)
    fraud_probability = float(model.predict_proba(model_input)[:, 1][0])
    risk_level, risk_color, recommendation = classify_risk(fraud_probability)
    recommendation_color = {
        "Approve": "#dff3e6",
        "Review": "#fff2cc",
        "Decline": "#f8d7da",
    }[recommendation]

    left_col, right_col = st.columns([0.9, 1.1], gap="large")

    with left_col:
        st.markdown(
            f"""
            <div class="risk-card">
                <div style="color:{risk_color};" class="risk-percent">{fraud_probability:.1%}</div>
                <div style="color:{risk_color};" class="risk-label">{risk_level} RISK</div>
                <p style="margin-top:0.75rem;color:#546363;">Estimated probability this application shows synthetic identity fraud patterns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Recommendation")
        st.markdown(
            f"""
            <div class="recommendation" style="background:{recommendation_color}; color:{risk_color};">
                {recommendation}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown("### Risk Score Gauge")
        st.pyplot(make_gauge(fraud_probability), clear_figure=True)

    st.markdown("### Top 3 Risk Factors")
    for factor in top_risk_factors(application_input, fraud_probability):
        st.write(f"- {factor}")

else:
    st.info("Use the sidebar form and click Analyze Application to generate a fraud risk score.")


st.markdown(
    """
    <div class="footer">
        Built by HammurabiCodes | Rutgers MS Business Analytics | Fraud Risk Detection Project
    </div>
    """,
    unsafe_allow_html=True,
)
