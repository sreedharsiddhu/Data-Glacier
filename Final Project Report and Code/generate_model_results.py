import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, roc_curve

# Try importing xgboost, else fallback
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Skipping XGBoost.")

# Ensure plot directory
if not os.path.exists('plots'):
    os.makedirs('plots')

def generate_model_results():
    print("Loading data...")
    df = pd.read_excel('Healthcare_dataset.xlsx', sheet_name='Dataset')
    
    # --- Preprocessing ---
    # Drop ID
    if 'Ptid' in df.columns:
        df = df.drop(columns=['Ptid'])
        
    # Drop rows with simplistic missing value handling for this demo
    # In a real scenario, we'd do imputation
    columns_to_drop = df.columns[df.isnull().sum() > 3000] # Drop cols with excessive missing
    df = df.drop(columns=columns_to_drop)
    df = df.dropna()

    # Target Encoding
    le = LabelEncoder()
    # Ensure it's string type for consistency before encoding
    df['Persistency_Flag'] = le.fit_transform(df['Persistency_Flag'].astype(str))

    # Feature Encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=['Persistency_Flag'])
    y = df['Persistency_Flag']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Modeling ---
    models = {}
    
    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf

    # 3. XGBoost
    if HAS_XGB:
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        xgb_model.fit(X_train_scaled, y_train)
        models['XGBoost'] = xgb_model

    # --- Plotting Results ---
    
    # ROC Curve Comparison
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend()
    plt.savefig('plots/model_roc_curve.png')
    plt.close()
    
    # Confusion Matrices
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        filename = f"plots/cm_{name.replace(' ', '_').lower()}.png"
        plt.savefig(filename)
        plt.close()

    # Feature Importance (RF)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10] # Top 10
    
    plt.figure(figsize=(10, 6))
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.barh(range(10), importances[indices], align='center')
    plt.yticks(range(10), [X.columns[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    print("Model results generated in 'plots/'.")

if __name__ == "__main__":
    generate_model_results()
