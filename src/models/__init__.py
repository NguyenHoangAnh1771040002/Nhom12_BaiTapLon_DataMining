"""
Models Module
=============

Machine learning models for classification and forecasting.

Classes/Functions:
------------------
- supervised: Classification models (Logistic, Tree, RF, XGBoost)
- semi_supervised: Self-training, Label Propagation
- forecasting: Time series models (ARIMA, SARIMA)
"""

# Import supervised learning functions
from .supervised import (
    # Training functions
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    train_all_models,
    
    # Tuning
    tune_hyperparameters,
    cross_validate_model,
    
    # Prediction
    predict,
    predict_proba,
    
    # Model I/O
    save_model,
    load_model,
    
    # Feature importance
    get_feature_importance,
    
    # Availability flags
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE
)

__all__ = [
    # Training
    'train_logistic_regression',
    'train_decision_tree',
    'train_random_forest',
    'train_xgboost',
    'train_lightgbm',
    'train_all_models',
    
    # Tuning
    'tune_hyperparameters',
    'cross_validate_model',
    
    # Prediction
    'predict',
    'predict_proba',
    
    # Model I/O
    'save_model',
    'load_model',
    
    # Feature importance
    'get_feature_importance',
    
    # Flags
    'XGBOOST_AVAILABLE',
    'LIGHTGBM_AVAILABLE'
]
