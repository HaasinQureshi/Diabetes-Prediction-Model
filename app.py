import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the models (or load them dynamically based on user choice later)
# Define the prediction function
def predict_diabetes(features, model_choice):
    # Load the appropriate model based on user choice
    if model_choice == 'KNN':
        with open('knn_model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_choice == 'Decision Tree':
        with open('tree_model.pkl', 'rb') as f:
            model = pickle.load(f)
    elif model_choice == 'MLP':
        with open('mlp_model.pkl', 'rb') as f:
            model = pickle.load(f)
    
    # Convert the input features into a numpy array (if necessary)
    features = np.array(features).reshape(1, -1)
    
    # Make the prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)

    return prediction, prediction_proba

# Streamlit UI
st.title("Diabetes Prediction App")

# Define user inputs for model prediction (based on features in your dataset)
def user_input_features():
    pregnancies = st.number_input('Pregnancies', min_value=0)
    glucose = st.number_input('Glucose', min_value=0)
    blood_pressure = st.number_input('Blood Pressure', min_value=0)
    skin_thickness = st.number_input('Skin Thickness', min_value=0)
    insulin = st.number_input('Insulin', min_value=0)
    bmi = st.number_input('BMI', min_value=0.0)
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0)
    age = st.number_input('Age', min_value=0)

    # Return the input features as a list or dataframe
    return [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

# Get user input
input_features = user_input_features()

# Select model
model_choice = st.selectbox("Select Model", ("KNN", "Decision Tree", "MLP"))

# Predict using the selected model
if st.button('Predict'):
    prediction, prediction_proba = predict_diabetes(input_features, model_choice)
    
    # Display results
    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write("The model predicts that the individual has diabetes.")
    else:
        st.write("The model predicts that the individual does not have diabetes.")

    st.subheader('Prediction Probability')
    st.write(f"Probability of no diabetes: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of diabetes: {prediction_proba[0][1]:.2f}")


