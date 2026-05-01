import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Tuple

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features based on industrial principles.
    """
    df = df.copy()
    
    # 1. Encode 'Type' (L=0, M=1, H=2)
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    if 'Type' in df.columns:
        df['Type'] = df['Type'].map(type_mapping)
        
    # Remove brackets from column names for XGBoost compatibility
    df.columns = [col.replace('[', '').replace(']', '').replace('<', '').strip() for col in df.columns]
    
    # 2. Engineered Features
    df['temp_diff'] = df['Process temperature K'] - df['Air temperature K']
    df['power'] = df['Torque Nm'] * df['Rotational speed rpm']
    df['wear_rate'] = df['Tool wear min'] / df['Rotational speed rpm']
    
    # RUL calculation: Assuming max tool wear in the dataset represents failure point
    max_wear = df['Tool wear min'].max()
    df['RUL'] = max_wear - df['Tool wear min']
    
    return df

def preprocess_and_scale(df: pd.DataFrame, is_training: bool = True, scaler_path: str = 'models/scaler.pkl') -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Engineer features and scale them.
    """
    df_engineered = engineer_features(df)
    
    # Define features to scale
    features_to_scale = [
        'Air temperature K', 'Process temperature K', 
        'Rotational speed rpm', 'Torque Nm', 'Tool wear min',
        'temp_diff', 'power', 'wear_rate'
    ]
    
    # Initialize or load scaler
    if is_training:
        scaler = StandardScaler()
        df_engineered[features_to_scale] = scaler.fit_transform(df_engineered[features_to_scale])
        # Ensure models directory exists
        Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        df_engineered[features_to_scale] = scaler.transform(df_engineered[features_to_scale])
        
    return df_engineered, scaler

def save_processed_data(df: pd.DataFrame, output_path: str = 'data/processed.csv'):
    """Save the processed dataset."""
    df.to_csv(output_path, index=False)
