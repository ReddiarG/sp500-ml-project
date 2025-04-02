"""
Model evaluation module.

This module contains functions for evaluating machine learning models,
including calculation of various metrics and visualization of results.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate various classification metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_proba (Optional[np.ndarray]): Predicted probabilities
    
    Returns:
        Dict[str, float]: Dictionary of metric names and values
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

def evaluate_model(model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a trained model on validation and test sets.
    
    Args:
        model (Pipeline): Trained pipeline
        X_val (pd.DataFrame): Validation feature matrix
        y_val (pd.Series): Validation target variable
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target variable
    
    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing metrics for both sets
    """
    # Validation predictions
    y_pred_val = model.predict(X_val)
    y_pred_proba_val = model.predict_proba(X_val)[:, 1]
    
    # Test predictions
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    val_metrics = calculate_metrics(y_val, y_pred_val, y_pred_proba_val)
    test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test)
    
    # Log results
    logger.info("\nValidation Set Metrics:")
    for metric, value in val_metrics.items():
        logger.info(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    logger.info("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    return {
        'validation': val_metrics,
        'test': test_metrics
    }

def analyze_feature_importance(model: Pipeline, feature_names: list) -> pd.DataFrame:
    """
    Analyze feature importance for models that support it.
    
    Args:
        model (Pipeline): Trained pipeline
        feature_names (list): List of feature names
    
    Returns:
        pd.DataFrame: DataFrame with feature importance scores
    """
    # Get the classifier from the pipeline
    classifier = model.named_steps['classifier']
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = np.abs(classifier.coef_[0])
    else:
        raise ValueError("Model does not support feature importance analysis")
    
    # Create DataFrame with feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df

def compare_models(models: Dict[str, Pipeline], X_val: pd.DataFrame, y_val: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Compare multiple models and their performance metrics.
    
    Args:
        models (Dict[str, Pipeline]): Dictionary of model names and trained pipelines
        X_val (pd.DataFrame): Validation feature matrix
        y_val (pd.Series): Validation target variable
        X_test (pd.DataFrame): Test feature matrix
        y_test (pd.Series): Test target variable
    
    Returns:
        pd.DataFrame: DataFrame comparing model performances
    """
    results = []
    
    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val, X_test, y_test)
        
        # Combine validation and test metrics
        model_results = {
            'Model': name,
            'Validation Accuracy': metrics['validation']['accuracy'],
            'Validation F1': metrics['validation']['f1_score'],
            'Test Accuracy': metrics['test']['accuracy'],
            'Test F1': metrics['test']['f1_score']
        }
        
        if 'auc' in metrics['validation']:
            model_results.update({
                'Validation AUC': metrics['validation']['auc'],
                'Test AUC': metrics['test']['auc']
            })
        
        results.append(model_results)
    
    return pd.DataFrame(results)
