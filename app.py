import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)

st.title("Heart Disease Prediction System")
st.write(
    "Enter the patient's information below to generate a prediction "
    "using the trained SVC model."
)

# Load the trained model
model_path = Path(__file__).parent / "svc_trained_model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)


# User inputs
age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50,
    step=1
)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=[1, 2, 3, 4]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=50,
    max_value=250,
    value=150,
    step=1
)


# Prediction
if st.button("Predict Heart Disease"):

    patient_data = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"]
    )

    prediction = model.predict(patient_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Heart Disease")
    else:
        st.success("Prediction: No Heart Disease")

    st.write("Input values used by the model:")
    st.dataframe(patient_data)