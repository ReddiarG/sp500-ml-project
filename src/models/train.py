"""
Training module for machine learning models.

This module contains functions for:
- Data preprocessing and pipeline creation
- Model training
- Hyperparameter tuning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_preprocessing_pipeline(numeric_features: List[str]) -> ColumnTransformer:
    """
    Create a preprocessing pipeline to handle scaling.
    
    Args:
        numeric_features (List[str]): Columns with numeric features
    
    Returns:
        ColumnTransformer: Preprocessor for scaling numeric features
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', RobustScaler(), numeric_features),
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = 0.3, val_size: float = 0.5,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                               pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        test_size (float): Proportion of data to use for test set
        val_size (float): Proportion of remaining data to use for validation
        random_state (int): Random seed for reproducibility
    
    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Initial split: train and temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Validation set shape: {X_val.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train: pd.DataFrame, y_train: pd.Series, 
                preprocessor: ColumnTransformer, 
                model: BaseEstimator) -> Pipeline:
    """
    Train a single model without hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target variable
        preprocessor (ColumnTransformer): Preprocessing pipeline
        model (BaseEstimator): Machine learning model
    
    Returns:
        Pipeline: Trained pipeline
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def tune_model(X_train: pd.DataFrame, y_train: pd.Series,
               preprocessor: ColumnTransformer,
               model: BaseEstimator,
               param_grid: Dict,
               cv: int = 3,
               scoring: str = 'f1',
               n_jobs: int = -1) -> Tuple[Pipeline, Dict]:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training feature matrix
        y_train (pd.Series): Training target variable
        preprocessor (ColumnTransformer): Preprocessing pipeline
        model (BaseEstimator): Machine learning model
        param_grid (Dict): Parameter grid for tuning
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric
        n_jobs (int): Number of jobs to run in parallel
    
    Returns:
        Tuple: (best_model, best_params)
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best Parameters: {grid_search.best_params_}")
    logger.info(f"Best Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def get_default_param_grids() -> Dict[str, Dict]:
    """
    Get default parameter grids for common models.
    
    Returns:
        Dict: Dictionary of parameter grids for different models
    """
    return {
        'logistic_regression': {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        },
        'svm': {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['auto', 'scale', 0.1, 1, 10]
        },
        'random_forest': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    }