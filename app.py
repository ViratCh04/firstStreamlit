import pandas as pd
import streamlit as st
import joblib

# Title
st.header("Streamlit Basic Deployment")

# Input Height
height = st.number_input("Enter Height")

# Input Weight
weight = st.number_input("Enter Weight")

# Select eye colour dropdown
eyes = st.selectbox("Select Eye Colour", ("Blue", "Brown"))

if st.button("Submit"):
    # unpickle classifier
    clf = joblib.load("clf.pkl")
    
    # Storing inputs into DataFrame
    X = pd.DataFrame([[height, weight, eyes]],
                    columns = ["Height", "Weight", "Eye"])
    X = X.replace(["Brown", "Blue"], [1, 0])

    prediction = clf.predict(X)[0]
    
    # Output Prediction
    st.text(f"This input is for a {prediction}")