import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, Any

def plot_confusion_matrix(cm, labels=['Normal', 'Failure'], title='Confusion Matrix'):
    """Plot a heatmap for the confusion matrix."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()

def plot_feature_importance(model, feature_names, title='Feature Importance'):
    """Plot feature importances from a tree-based model."""
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute.")
        
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    return plt.gcf()

def format_metrics_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Format classification results into a pandas DataFrame for easy comparison."""
    data = []
    for model_name, metrics in results_dict.items():
        data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['Accuracy']:.4f}",
            'Precision': f"{metrics['Precision']:.4f}",
            'Recall': f"{metrics['Recall']:.4f}",
            'F1 Score': f"{metrics['F1']:.4f}"
        })
    return pd.DataFrame(data).set_index('Model')

def plot_rul_predictions(y_actual, y_pred, title='Predicted vs Actual RUL'):
    """Plot scatter of actual vs predicted RUL with identity line."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_actual, y_pred, alpha=0.5, color='teal')
    
    # Identity line
    max_val = max(max(y_actual), max(y_pred))
    min_val = min(min(y_actual), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.title(title)
    plt.xlabel('Actual RUL')
    plt.ylabel('Predicted RUL')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()
