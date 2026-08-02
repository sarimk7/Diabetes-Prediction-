import streamlit as st
from joblib import load
import pandas as pd

# Load trained model pipeline
model = load("/Users/sarimkazmi/Downloads/LogisticRegression_diabetes_model.joblib")

st.title("Diabetes Prediction App")
st.write("Enter patient information below:")

# Input fields
gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

age = st.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=30
)

hypertension = st.selectbox(
    "Hypertension",
    ["No", "Yes"]
)

heart_disease = st.selectbox(
    "Heart Disease",
    ["No", "Yes"]
)

smoking_history = st.selectbox(
    "Smoking History",
    [
        "never",
        "former",
        "current",
        "ever",
        "not current"
    ]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)

hba1c_level = st.number_input(
    "HbA1c Level",
    min_value=0.0,
    max_value=20.0,
    value=5.7,
    step=0.1
)

blood_glucose_level = st.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=400,
    value=100
)


# Prediction
if st.button("Predict Diabetes Status"):

    # Create dataframe with original training features
    input_features = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [1 if hypertension == "Yes" else 0],
        "heart_disease": [1 if heart_disease == "Yes" else 0],
        "smoking_history": [smoking_history],
        "bmi": [bmi],
        "HbA1c_level": [hba1c_level],
        "blood_glucose_level": [blood_glucose_level]
    })

    # Optional debug
    st.write("Input Data:")
    st.dataframe(input_features)

    # Predict
    prediction = model.predict(input_features)[0]
    probability = model.predict_proba(input_features)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Patient is predicted to have diabetes")
    else:
        st.success("✅ Patient is predicted not to have diabetes")

    st.write(
        f"Diabetes Probability: {probability:.2%}"
    )



