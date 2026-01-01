# Week 12: Healthcare Persistency Prediction

This project aims to predict the `Persistency_Flag` of patients using various machine learning models (Linear, Ensemble, Boosting) and visualize the results in an interactive dashboard.

## Structure

- `model_training.ipynb`: Jupyter notebook for data preprocessing, model training, and evaluation.
- `app.py`: Streamlit application for the dashboard and user interface.
- `requirements.txt`: List of Python dependencies.
- `Healthcare_dataset.xlsx`: Dataset used for analysis (ensure this file is present).

## Setup & Usage

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train Models**:
    Open and run `model_training.ipynb` to train the models. This will generate the necessary `.pkl` files (`lr_model.pkl`, `rf_model.pkl`, `xgb_model.pkl`, etc.) required for the dashboard.

3.  **Run Dashboard**:
    After training, run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Models Explored

- **Logistic Regression**: Baseline linear model for high explainability.
- **Random Forest**: Ensemble model for better performance and feature importance.
- **XGBoost**: Gradient boosting model for high accuracy.

## Team
- The Closer
