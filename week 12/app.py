import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(page_title="Healthcare Persistency Predictor", layout="wide")

st.title("Healthcare Persistency Classification Dashboard")
st.markdown("**Team:** The Closer | **Week:** 12")

# Check for models
REQUIRED_FILES = ['lr_model.pkl', 'rf_model.pkl', 'xgb_model.pkl', 'scaler.pkl', 'feature_names.pkl', 'target_encoder.pkl']
missing_files = [f for f in REQUIRED_FILES if not os.path.exists(f)]

if missing_files:
    st.error(f"Missing model files: {', '.join(missing_files)}. Please run `model_training.ipynb` first to generate them.")
    st.stop()

# Load Assets
@st.cache_resource
def load_assets():
    lr = joblib.load('lr_model.pkl')
    rf = joblib.load('rf_model.pkl')
    xgb = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('feature_names.pkl')
    # target_enc = joblib.load('target_encoder.pkl') # Not strictly needed if we know classes
    return lr, rf, xgb, scaler, features

lr_model, rf_model, xgb_model, scaler, feature_names = load_assets()

# Sidebar - Model Selection
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "XGBoost"])

# Load Data for Input Schema (only headers)
@st.cache_data
def load_data_schema():
    # Read only a small chunk to get columns and types
    if os.path.exists('Healthcare_dataset.xlsx'):
        df = pd.read_excel('Healthcare_dataset.xlsx', sheet_name='Dataset', nrows=10)
        if 'Ptid' in df.columns:
            df = df.drop(columns=['Ptid'])
        return df
    return None

df_schema = load_data_schema()

if df_schema is not None:
    st.sidebar.subheader("Prediction Input")
    
    input_data = {}
    
    # Dynamically create inputs
    with st.form("prediction_form"):
        st.write("### Patient Details")
        cols = st.columns(2)
        idx = 0
        
        for col in df_schema.columns:
            if col == 'Persistency_Flag':
                continue
                
            with cols[idx % 2]:
                if pd.api.types.is_numeric_dtype(df_schema[col]):
                    input_data[col] = st.number_input(f"{col}", value=float(df_schema[col].mean()))
                else:
                    input_data[col] = st.selectbox(f"{col}", options=df_schema[col].dropna().unique())
            idx += 1
            
        submitted = st.form_submit_button("Predict Persistency")

    if submitted:
        # Preprocess Input
        input_df = pd.DataFrame([input_data])
        
        # One-Hot Encoding (Alignment)
        # 1. Get dummies
        input_encoded = pd.get_dummies(input_df)
        
        # 2. Align with training features
        # Create a df with all 0s for training features
        input_aligned = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Update with actual values where columns match
        common_cols = list(set(input_encoded.columns) & set(feature_names))
        input_aligned[common_cols] = input_encoded[common_cols]
        
        # Scale
        input_scaled = scaler.transform(input_aligned)
        
        # Predict
        if model_choice == "Logistic Regression":
            model = lr_model
        elif model_choice == "Random Forest":
            model = rf_model
        else:
            model = xgb_model
            
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        
        st.subheader("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Class", "Persistent" if prediction == 1 else "Non-Persistent")
            
        with col2:
            st.metric("Probability (Persistent)", f"{proba[1]:.2%}")
            
        if model_choice == "Random Forest":
             st.info("Random Forest is an ensemble model offering good balance and feature importance visibility.")
        elif model_choice == "XGBoost":
            st.info("XGBoost is a boosting model often providing high accuracy but can be less interpretable.")
        else:
             st.info("Logistic Regression is a linear model, highly interpretable and simple.")

else:
    st.warning("Dataset not found! Cannot build input form.")

st.markdown("---")
st.markdown("### Model Performance Overview")
st.write("Run the training notebook to populate performance charts here.")
# Placeholder for static performance if wanted
