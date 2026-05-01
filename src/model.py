import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

def train_and_evaluate_classifiers(df: pd.DataFrame, target_col: str = 'Machine failure', save_path: str = 'models/classifier.pkl'):
    """
    Train and evaluate multiple classifiers, returning the best one.
    """
    # Features (exclude target and RUL which is the regression target)
    drop_cols = [target_col, 'RUL']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target_col]
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    results = {}
    best_f1 = -1
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'Confusion Matrix': cm,
            'Model': model
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            
    # Save best model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, save_path)
    
    # Optional: save columns for inference
    joblib.dump(list(X.columns), str(Path(save_path).with_name('feature_columns.pkl')))
    
    return results, best_model_name
