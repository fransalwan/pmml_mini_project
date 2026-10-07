import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

def load_data(filepath="data/breast_cancer.csv"):
    """
    Load dataset from CSV file.
    Target: 1 = Malignant (cancerous), 0 = Benign (non-cancerous).
    """
    if not os.path.exists(filepath):
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame.copy()
        df['diagnosis'] = df['target'].map({0: 'Malignant', 1: 'Benign'})
        df['target'] = (df['target'] == 0).astype(int)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
    else:
        df = pd.read_csv(filepath)
    
    feature_cols = [c for c in df.columns if c not in ['target', 'diagnosis']]
    X = df[feature_cols].values
    y = df['target'].values
    return df, X, y, feature_cols

def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into 80% Train-Validation and 20% Holdout Test with stratification.
    StandardScaler is fit strictly on train_val to prevent data leakage.
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_val_scaled = scaler.fit_transform(X_train_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_val_scaled, X_test_scaled, y_train_val, y_test, scaler

def get_kfold_splits(n_splits=5, random_state=42):
    """
    Return StratifiedKFold cross-validator.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

