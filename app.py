
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
multiple_lines = st.selectbox("Multiple Lines", options=['Yes', 'No'])
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
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'Partner': [partner],
        'Dependents': [dependents],
        'MultipleLines': [multiple_lines],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method]
    })

    # Make prediction using the loaded pipeline
    prediction_encoded = loaded_pipeline.predict(input_data)
    prediction_proba = loaded_pipeline.predict_proba(input_data)[:, 1] # Probability of churn (class 1)

    # Decode the prediction (assuming 0 is No Churn, 1 is Churn)
    # You'll need the LabelEncoder used during training to decode correctly
    # Since you fit the LabelEncoder on 'y', we can assume y_encoded is 0 for No Churn, 1 for Churn
    # A more robust way would be to save and load the LabelEncoder as well.
    # For simplicity here, we'll just use the integer output and probability.
    threshold = 0.67
    churn_status = 1 if prediction_proba[0] >= threshold else 0
    #churn_status = "Churn" if prediction_encoded[0] == 1 else "No Churn"


    # Display the prediction
    st.subheader("Prediction Result")
    if churn_status == "Churn":
        st.error(f"Based on the provided information, this customer is predicted to **{churn_status}**.")
    else:
        st.success(f"Based on the provided information, this customer is predicted to **{churn_status}**.")

    st.write(f"Probability of Churn: **{prediction_proba[0]:.2f}**")

# Instructions on how to run this script in Colab (optional, for user reference)
