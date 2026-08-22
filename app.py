import streamlit as st
import pandas as pd
import joblib

# 1. Load the saved XGBoost model
# Ensure 'xgb_best_model.pkl' is in the same directory as this file
model = joblib.load('xgb_best_model.pkl')

# 2. Page Configuration
st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="centered")
st.title("🏦 Bank Customer Churn Prediction")
st.write("Enter the customer's details below to predict whether they will stay or exit the bank.")

st.divider()

# 3. User Inputs
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4])

with col2:
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=1000.0)
    gender = st.selectbox("Gender", ["Male", "Female"])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    has_crcard = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active = st.selectbox("Is Active Member?", ["Yes", "No"])

st.divider()

# 4. Prediction Logic
if st.button("Predict Churn 🚀", use_container_width=True):
    
    # Encode categorical variables exactly as done during training
    gender_val = 1 if gender == "Male" else 0
    has_crcard_val = 1 if has_crcard == "Yes" else 0
    is_active_val = 1 if is_active == "Yes" else 0
    
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    # Create a DataFrame with the exact same columns as X_train
    input_data = pd.DataFrame([[
        credit_score, gender_val, age, tenure, balance, num_products, 
        has_crcard_val, is_active_val, estimated_salary, geo_germany, geo_spain
    ]], columns=[
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain'
    ])

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    # Display results
    if prediction == 1:
        st.error(f"⚠️ Warning: This customer is likely to CHURN! (Churn Probability: {prediction_proba:.1%})")
    else:
        st.success(f"✅ Safe: This customer is likely to STAY. (Churn Probability: {prediction_proba:.1%})")