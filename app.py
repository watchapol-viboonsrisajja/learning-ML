import streamlit as st
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Load model
model = joblib.load("best_model.pkl")

# Streamlit UI
st.title("My first ML App (Study on Imbalanced Data Classification by 67xxxx ")

# Input fields
features = []
for i in range(11):  # Adjust based on dataset
    value = st.number_input(f"Feature_{i}", value=0.0)
    features.append(value)

# Prediction
if st.button("Predict"):
    prediction = model.predict([np.array(features)])
    st.write(f"Predicted Class: {prediction[0]}")
