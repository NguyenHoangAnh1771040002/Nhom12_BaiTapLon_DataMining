"""
Models Module
=============

Machine learning models for classification and forecasting.

Classes/Functions:
------------------
- supervised: Classification models (Logistic, Tree, RF, XGBoost)
- semi_supervised: Self-training, Label Propagation, Label Spreading
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

# Import semi-supervised learning functions
from .semi_supervised import (
    # Split functions
    create_labeled_unlabeled_split,
    create_multiple_splits,
    
    # Self-Training
    train_self_training,
    train_self_training_rf,
    
    # Label Propagation/Spreading
    train_label_propagation,
    train_label_spreading,
    
    # Analysis
    analyze_pseudo_labels,
    evaluate_semi_supervised,
    
    # Comparison
    compare_semi_supervised_methods,
    run_label_fraction_experiment,
    
    # Visualization
    plot_learning_curve_by_labels,
    plot_pseudo_label_confusion_matrix
)

__all__ = [
    # ============ SUPERVISED ============
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
    'LIGHTGBM_AVAILABLE',
    
    # ============ SEMI-SUPERVISED ============
    # Split functions
    'create_labeled_unlabeled_split',
    'create_multiple_splits',
    
    # Self-Training
    'train_self_training',
    'train_self_training_rf',
    
    # Label Propagation/Spreading
    'train_label_propagation',
    'train_label_spreading',
    
    # Analysis
    'analyze_pseudo_labels',
    'evaluate_semi_supervised',
    
    # Comparison
    'compare_semi_supervised_methods',
    'run_label_fraction_experiment',
    
    # Visualization
    'plot_learning_curve_by_labels',
    'plot_pseudo_label_confusion_matrix'
]
