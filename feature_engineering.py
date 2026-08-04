"""
Shared feature-engineering logic for the Diabetes Prediction mini-project (Milestone 4).

This module is imported both by the modeling notebook (where it is wrapped in a
sklearn FunctionTransformer and baked into the saved model pipeline) and by the
Streamlit UI (UI.py), so that a single definition guarantees the UI builds exactly
the same features the model was trained on.

Engineered features (per the Milestone 2 "Proposed Enhancements" section):
  1. hba1c_glucose_interaction — HbA1c_level * blood_glucose_level
  2. bmi_category              — WHO-standard BMI bucket
  3. age_risk_bucket           — age-based diabetes-risk bucket (ADA screening age 45 cutoff)
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with engineered columns added.

    Expects columns: age, bmi, HbA1c_level, blood_glucose_level
    (the raw columns from diabetes_prediction_dataset.csv).
    Safe to call on a single-row DataFrame (e.g. from the Streamlit UI).
    """
    out = df.copy()

    # 1. HbA1c x blood glucose interaction term.
    # Motivated by the pairwise-plot insight in the Milestone 2 report: patients with
    # glucose 100-150 but HbA1c > ~7 are still frequently diabetic, i.e. the two
    # features separate classes better jointly than either alone.
    out["hba1c_glucose_interaction"] = out["HbA1c_level"] * out["blood_glucose_level"]

    # 2. WHO-standard BMI category.
    # underweight <18.5, normal 18.5-24.9, overweight 25-29.9, obese >=30.
    bmi_bins = [0, 18.5, 25, 30, 100]
    bmi_labels = ["underweight", "normal", "overweight", "obese"]
    out["bmi_category"] = pd.cut(
        out["bmi"], bins=bmi_bins, labels=bmi_labels, right=False
    ).astype(str)

    # 3. Age-based risk bucket.
    # The American Diabetes Association recommends screening begin at age 45 for
    # average-risk adults, with risk increasing further past 65.
    age_bins = [0, 45, 65, 150]
    age_labels = ["under_45", "45_to_64", "65_plus"]
    out["age_risk_bucket"] = pd.cut(
        out["age"], bins=age_bins, labels=age_labels, right=False
    ).astype(str)

    return out


# Column groups used to build the ColumnTransformer in the modeling notebook.
CATEGORICAL_FEATURES = ["gender", "smoking_history", "bmi_category", "age_risk_bucket"]
NUMERICAL_FEATURES = ["age", "bmi", "HbA1c_level", "blood_glucose_level", "hba1c_glucose_interaction"]
BINARY_PASSTHROUGH_FEATURES = ["hypertension", "heart_disease"]
RAW_FEATURE_ORDER = [
    "gender", "age", "hypertension", "heart_disease", "smoking_history",
    "bmi", "HbA1c_level", "blood_glucose_level",
]
