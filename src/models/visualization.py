"""
Visualization module for machine learning analysis.

This module contains functions for creating various plots used in ML analysis,
including data exploration, model evaluation, and feature analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.metrics import confusion_matrix

def plot_feature_distributions(X: pd.DataFrame, numeric_features: List[str], 
                             figsize: tuple = (20, 10)) -> None:
    """
    Plot distributions of numeric features.
    
    Args:
        X (pd.DataFrame): Feature matrix
        numeric_features (List[str]): List of numeric feature names
        figsize (tuple): Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    X[numeric_features].hist(bins=30)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple = (10, 8)) -> None:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df (pd.DataFrame): DataFrame to plot correlation matrix for
        figsize (tuple): Figure size (width, height)
    """
    corr = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         title: str = 'Confusion Matrix',
                         figsize: tuple = (6, 5)) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        title (str): Plot title
        figsize (tuple): Figure size (width, height)
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                   title: str = 'ROC Curve',
                   figsize: tuple = (8, 6)) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred_proba (np.ndarray): Predicted probabilities
        title (str): Plot title
        figsize (tuple): Figure size (width, height)
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model: any, feature_names: List[str], 
                          title: str = 'Feature Importance',
                          figsize: tuple = (10, 6)) -> None:
    """
    Plot feature importance for models that support it.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (List[str]): List of feature names
        title (str): Plot title
        figsize (tuple): Figure size (width, height)
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model does not support feature importance plotting")
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=figsize)
    plt.bar(range(len(importances)), sorted_importances)
    plt.xticks(range(len(importances)), sorted_features, rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.show() 