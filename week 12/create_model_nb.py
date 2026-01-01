import json

NOTEBOOK_NAME = "model_training.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Data Glacier Week 12: Model Selection and Building\n",
            "**Team:** The Closer\n",
            "\n",
            "## Objective\n",
            "To build and compare classification models for predicting `Persistency_Flag`. We will explore Linear, Ensemble, and Boosting families."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import seaborn as sns\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "import xgboost as xgb\n",
            "from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score\n",
            "import joblib\n",
            "\n",
            "# Load Data\n",
            "df = pd.read_excel('Healthcare_dataset.xlsx', sheet_name='Dataset')\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Data Preprocessing"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Data Cleaning\n",
            "# Identify missing values\n",
            "print(df.isnull().sum()[df.isnull().sum() > 0])\n",
            "\n",
            "# For simplicity in this demo, we will drop columns with too many missing values or impute simple ones.\n",
            "# Assuming standard cleaning is needed. \n",
            "# NOTE: Adjust based on specific dataset inspection.\n",
            "\n",
            "# Encoding Target Variable\n",
            "le = LabelEncoder()\n",
            "df['Persistency_Flag'] = le.fit_transform(df['Persistency_Flag'])\n",
            "joblib.dump(le, 'target_encoder.pkl')\n",
            "print(\"Target Class Mapping:\", dict(zip(le.classes_, le.transform(le.classes_))))\n",
            "\n",
            "# Feature Selection (Drop ID columns if any)\n",
            "if 'Ptid' in df.columns:\n",
            "    df = df.drop(columns=['Ptid'])\n",
            "\n",
            "# Categorical Encoding\n",
            "# We will use OneHotEncoding for categorical features\n",
            "categorical_cols = df.select_dtypes(include=['object']).columns\n",
            "df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)\n",
            "\n",
            "# Split Data\n",
            "X = df_encoded.drop(columns=['Persistency_Flag'])\n",
            "y = df_encoded['Persistency_Flag']\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            "\n",
            "# Scaling\n",
            "scaler = StandardScaler()\n",
            "X_train_scaled = scaler.fit_transform(X_train)\n",
            "X_test_scaled = scaler.transform(X_test)\n",
            "\n",
            "joblib.dump(scaler, 'scaler.pkl')\n",
            "joblib.dump(X.columns, 'feature_names.pkl')  # Save feature names for inference\n",
            "\n",
            "print(\"Training Shape:\", X_train_scaled.shape)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Model Training"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. Logistic Regression (Linear Family)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "lr_model = LogisticRegression(random_state=42, max_iter=1000)\n",
            "lr_model.fit(X_train_scaled, y_train)\n",
            "y_pred_lr = lr_model.predict(X_test_scaled)\n",
            "\n",
            "print(\"Logistic Regression Report:\")\n",
            "print(classification_report(y_test, y_pred_lr))\n",
            "print(\"ROC-AUC:\", roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:, 1]))\n",
            "\n",
            "joblib.dump(lr_model, 'lr_model.pkl')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. Random Forest (Ensemble Family)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "rf_model = RandomForestClassifier(random_state=42, n_estimators=100)\n",
            "rf_model.fit(X_train_scaled, y_train)\n",
            "y_pred_rf = rf_model.predict(X_test_scaled)\n",
            "\n",
            "print(\"Random Forest Report:\")\n",
            "print(classification_report(y_test, y_pred_rf))\n",
            "print(\"ROC-AUC:\", roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1]))\n",
            "\n",
            "joblib.dump(rf_model, 'rf_model.pkl')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. XGBoost (Boosting Family)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)\n",
            "xgb_model.fit(X_train_scaled, y_train)\n",
            "y_pred_xgb = xgb_model.predict(X_test_scaled)\n",
            "\n",
            "print(\"XGBoost Report:\")\n",
            "print(classification_report(y_test, y_pred_xgb))\n",
            "print(\"ROC-AUC:\", roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1]))\n",
            "\n",
            "joblib.dump(xgb_model, 'xgb_model.pkl')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Model Evaluation & Comparison"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "results = {\n",
            "    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],\n",
            "    'Accuracy': [\n",
            "        accuracy_score(y_test, y_pred_lr),\n",
            "        accuracy_score(y_test, y_pred_rf),\n",
            "        accuracy_score(y_test, y_pred_xgb)\n",
            "    ],\n",
            "    'ROC-AUC': [\n",
            "        roc_auc_score(y_test, lr_model.predict_proba(X_test_scaled)[:, 1]),\n",
            "        roc_auc_score(y_test, rf_model.predict_proba(X_test_scaled)[:, 1]),\n",
            "        roc_auc_score(y_test, xgb_model.predict_proba(X_test_scaled)[:, 1])\n",
            "    ]\n",
            "}\n",
            "\n",
            "results_df = pd.DataFrame(results)\n",
            "print(results_df)\n",
            "\n",
            "# Plot Feature Importance (for RF)\n",
            "importances = rf_model.feature_importances_\n",
            "indices = np.argsort(importances)[::-1]\n",
            "feature_names = X.columns\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "plt.title(\"Feature Importances (Random Forest)\")\n",
            "plt.bar(range(X.shape[1]), importances[indices], align=\"center\")\n",
            "plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    }
]

notebook_content = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(NOTEBOOK_NAME, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=4)

print(f"Generated {NOTEBOOK_NAME}")
