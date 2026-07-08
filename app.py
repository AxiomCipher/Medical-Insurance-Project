import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load Model
model = joblib.load('./models/model.pkl')

st.title('Medical Insurance Prediction Using Random Forest')
st.markdown("""
    In order to predict your medical insurance cost, enter your information and click on "Predict"
""")

st.sidebar.title('Medeical Prediction')
st.sidebar.write('Enter your details')

age = st.sidebar.number_input(
    'Age',
    min_value=18,
    max_value=90,
    value=20
)

sex = st.sidebar.selectbox(
    'Gender',
    ['male', 'female']
)

bmi = st.sidebar.number_input(
    'BMI',
    min_value=8,
    max_value=90,
    value=10
)

children = st.sidebar.number_input(
    'Number of Children',
    min_value=0,
    max_value=10,
    value=2
)

smoker = st.sidebar.selectbox(
    'Smoking Status',
    ['yes', 'no']
)

region = st.sidebar.selectbox(
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