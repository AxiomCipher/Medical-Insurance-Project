import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load Model
model = joblib.load('models/model.pkl')

st.title('Medical Insurance Prediction')
st.markdown("""
    In order to predict your medical insurance cost, enter your information and click on "Predict Insurance Cost"
""")

st.sidebar.title('Medical Prediction')
st.sidebar.write('Enter your details')

age = st.number_input(
    'Age',
    min_value=18,
    max_value=90,
    value=20
)

sex = st.selectbox(
    'Gender',
    ['male', 'female']
)

bmi = st.number_input(
    'BMI',
    min_value=8,
    max_value=90,
    value=10
)

children = st.number_input(
    'Number of Children',
    min_value=0,
    max_value=10,
    value=2
)

smoker = st.selectbox(
    'Smoking Status',
    ['yes', 'no']
)

region = st.selectbox(
    'Region',
    ['southwest', 'southeast', 'northwest', 'northeast']
)

sample_data = pd.DataFrame({
    'age':[age],
    'sex':[sex],
    'bmi':[bmi],
    'children':[children],
    'smoker':[smoker],
    'region':[region]
})

if st.button('Predict Insurance Cost'):
    predictions = np.round(model.predict(sample_data), 2)
    st.success(predictions[0])
