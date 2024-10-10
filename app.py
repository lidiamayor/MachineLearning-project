import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

def main():
    # Load the trained model (make sure to save the model previously as a .pkl file)
    # For example, if your model is called "stroke_model.pkl"
    with open("stroke_model.pkl", "rb") as file:
        model = pickle.load(file)

    st.title('Stroke Predictor')

    # User input for data
    age = st.slider("Age", 1, 100, 30)
    glucose_level = st.slider("Glucose Level", 50.0, 300.0, 100.0)
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 60.0, 25.0)

    # Job type selection
    if age >= 16:
        job_type = st.selectbox("Job Type", 
                                ("Never worked", "Private", "Government Job", "Self-employed"))
        # Convert selected option to numeric values
        if job_type == "Never worked":
            job_type_val = 1
        elif job_type == "Private":
            job_type_val = 2
        elif job_type == "Government Job":
            job_type_val = 3
        else:
            job_type_val = 4
    else:
        job_type_val = 0

    # Ask about medical conditions
    heart_disease = st.selectbox("Do you have any heart disease?", ("No", "Yes"))
    hypertension = st.selectbox("Do you have hypertension?", ("No", "Yes"))

    # Convert "Yes" or "No" options to binary values
    heart_disease_val = 1 if heart_disease == "Yes" else 0
    hypertension_val = 1 if hypertension == "Yes" else 0

    input_data = pd.DataFrame({
        'age': [age],
        'hypertension': [hypertension_val],
        'heart_disease': [heart_disease_val],
        'work_type': [job_type_val],
        'avg_glucose_level': [glucose_level],
        'bmi': [bmi]
    })
    # Button to predict
    if st.button("Predict Stroke Probability"):
        # Perform prediction
        stroke_probability = model.predict_proba(input_data)[0][1]  # Probability of class '1' (stroke)

        # Show the result
        st.subheader(f"The probability of having a stroke is: {stroke_probability * 100}%")


if __name__ == '__main__':
    main()