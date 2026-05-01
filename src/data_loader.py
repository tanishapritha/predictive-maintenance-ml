import pandas as pd
from pathlib import Path
from typing import Tuple

def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load the predictive maintenance dataset and drop unused columns.
    
    Args:
        filepath: Path to the raw CSV file.
        
    Returns:
        DataFrame containing only the relevant features and target.
    """
    df = pd.read_csv(filepath)
    
    # Columns to drop: UDI, Product ID, and specific failure modes
    # as we only predict the general 'Machine failure'
    cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return df

if __name__ == "__main__":
    df = load_data('../data/ai4i2020.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
