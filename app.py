
import streamlit as st
import joblib
import pandas as pd

# Load the trained model pipeline
# Make sure the 'LogisticRegression_churn_model.pkl' file is available in the environment where you run the streamlit app
try:
    loaded_pipeline = joblib.load('LogisticRegression_churn_model.pkl')
except FileNotFoundError:
    st.error("Model file 'LogisticRegression_churn_model.pkl' not found. Please ensure it's in the same directory.")
    st.stop() # Stop the app execution if model is not found

st.title("Customer Churn Prediction App")

st.write("""
Predict if a customer is likely to churn based on their characteristics.
Fill in the customer details below and click 'Predict Churn'.
""")

# Create input fields for each feature used in the model
# Refer to your original feature lists: numerical_features, categorical_features

# Numerical Features
st.subheader("Numerical Features")
tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=1, step=1)
monthly_charges = st.slider("Monthly Charges", min_value=0.0, max_value=120.0, value=20.0, step=0.1)

# Categorical Features
st.subheader("Categorical Features")
partner = st.selectbox("Partner", options=['Yes', 'No'])
dependents = st.selectbox("Dependents", options=['Yes', 'No'])
phone_service = st.selectbox("Phone Service", ['Yes','No'])
multiple_lines = st.selectbox("Multiple Lines", options=['Yes', 'No'])
internet_service = st.selectbox("Internet Service",
                                ['DSL','Fiber optic','No'])
online_security = st.selectbox("Online Security", options=['Yes', 'No'])
online_backup = st.selectbox("Online Backup", options=['Yes', 'No'])
device_protection = st.selectbox("Device Protection", options=['Yes', 'No'])
tech_support = st.selectbox("Tech Support", options=['Yes', 'No'])
streaming_tv = st.selectbox("Streaming TV", options=['Yes', 'No'])
streaming_movies = st.selectbox("Streaming Movies", options=['Yes', 'No'])
contract = st.selectbox("Contract", options=['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox("Paperless Billing", options=['Yes', 'No'])
payment_method = st.selectbox("Payment Method", options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

# Create a button to trigger prediction
if st.button("Predict Churn"):
    # Create a DataFrame from the input values
    input_data = pd.DataFrame({
        'tenure':[tenure],
        'MonthlyCharges':[monthly_charges],
        'Partner':[partner],
        'Dependents':[dependents],
        'PhoneService':[phone_service],
        'MultipleLines':[multiple_lines],
        'InternetService':[internet_service],
        'OnlineSecurity':[online_security],
        'OnlineBackup':[online_backup],
        'DeviceProtection':[device_protection],
        'TechSupport':[tech_support],
        'StreamingTV':[streaming_tv],
        'StreamingMovies':[streaming_movies],
        'Contract':[contract],
        'PaperlessBilling':[paperless_billing],
        'PaymentMethod':[payment_method]
    })

    # Ensure correct column order
    input_data = input_data[[
        'tenure','MonthlyCharges','Partner','Dependents','PhoneService',
        'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
        'Contract','PaperlessBilling','PaymentMethod'
    ]]

    # Make prediction
    prediction_encoded = loaded_pipeline.predict(input_data)
    prediction_proba = loaded_pipeline.predict_proba(input_data)[:,1]
    threshold = 0.67
    churn_status = 1 if prediction_proba[0] >= threshold else 0
    churn_status = "Churn" if prediction_encoded[0] == 1 else "No Churn"

    st.subheader("Prediction Result")

    if churn_status == "Churn":
        st.error(f"Based on the provided information, this customer is predicted to {churn_status}.")
    else:
        st.success(f"Based on the provided information, this customer is predicted to {churn_status}")

    st.write(f"Probability of Churn: **{prediction_proba[0]:.2f}**")
