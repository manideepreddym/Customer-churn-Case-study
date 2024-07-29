import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the pre-trained model
model_path = 'E:\\projects\\End-to-end-project---Customer-churn-main\\Model.pkl'
try:
    model = load(model_path)
    st.success(f'Model loaded successfully from {model_path}')
except Exception as e:
    st.error(f'Error loading model: {e}')
    model = None

# Initialize encoders and scalers based on training data
encoder = OneHotEncoder(drop='first', sparse_output=False)
scaler = StandardScaler()

# Preprocessing functions
def preprocess_data(df):
    # Define fixed categories for encoding
    categories = [
        ['Yes', 'No'],  # Dependents
        ['Yes', 'No'],  # OnlineSecurity
        ['Yes', 'No'],  # OnlineBackup
        ['Yes', 'No'],  # DeviceProtection
        ['Yes', 'No'],  # TechSupport
        ['Month-to-month', 'One year', 'Two year'],  # Contract
        ['Yes', 'No']  # PaperlessBilling
    ]
    
    # Fit the encoder and scaler based on the data
    if encoder.categories_ != categories:
        encoder = OneHotEncoder(drop='first', sparse_output=False, categories=categories)
        encoder.fit(categories)
    
    # Encode categorical features
    categorical_features = df[['Dependents', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                               'TechSupport', 'Contract', 'PaperlessBilling']]
    encoded_categorical = encoder.transform(categorical_features)
    
    # Fit and transform the scaler based on numerical features
    numerical_features = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    if not hasattr(scaler, 'mean_'):
        scaler.fit(numerical_features)
    scaled_numerical = scaler.transform(numerical_features)
    
    # Combine processed features
    processed_data = np.hstack([encoded_categorical, scaled_numerical])
    
    return processed_data

# Streamlit app
st.title('Customer Churn Prediction')

# Input fields
st.sidebar.header('Input Features')
dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
tenure = st.sidebar.slider('Tenure', min_value=0, max_value=72, value=1)
online_security = st.sidebar.selectbox('OnlineSecurity', ['Yes', 'No'])
online_backup = st.sidebar.selectbox('OnlineBackup', ['Yes', 'No'])
device_protection = st.sidebar.selectbox('DeviceProtection', ['Yes', 'No'])
tech_support = st.sidebar.selectbox('TechSupport', ['Yes', 'No'])
contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.sidebar.selectbox('PaperlessBilling', ['Yes', 'No'])
monthly_charges = st.sidebar.number_input('MonthlyCharges', min_value=0.0, max_value=200.0, value=29.85)
total_charges = st.sidebar.number_input('TotalCharges', min_value=0.0, max_value=10000.0, value=556.85)

# Create DataFrame
input_data = pd.DataFrame({
    'Dependents': [dependents],
    'tenure': [tenure],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
})

# Predict if the model is loaded
if model:
    if st.button('Predict'):
        try:
            # Preprocess the input data
            input_data_transformed = preprocess_data(input_data)
            
            # Predict
            prediction = model.predict(input_data_transformed)
            probability = model.predict_proba(input_data_transformed)[:, 1]

            if prediction == 1:
                st.write("This Customer is likely to be Churned!")
                st.write(f"Confidence level is {np.round(probability[0] * 100, 2)}%")
            else:
                st.write("This Customer is likely to Continue!")
                st.write(f"Confidence level is {np.round(probability[0] * 100, 2)}%")
        except Exception as e:
            st.error(f'Error during prediction: {e}')
else:
    st.warning("Model is not loaded. Please check the model path and file.")
