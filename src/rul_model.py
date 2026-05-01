import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_and_evaluate_rul_model(df: pd.DataFrame, target_col: str = 'RUL', save_path: str = 'models/rul_model.pkl'):
    """
    Train and evaluate a regression model for Remaining Useful Life (RUL).
    """
    # Exclude Machine failure and RUL from features
    drop_cols = ['Machine failure', target_col]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df[target_col]
    
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    
    return {
        'Model': model,
        'MAE': mae,
        'RMSE': rmse,
        'y_test': y_test,
        'y_pred': y_pred
    }
