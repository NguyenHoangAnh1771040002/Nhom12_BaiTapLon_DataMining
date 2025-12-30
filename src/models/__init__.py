"""
Models Module
=============

Machine learning models for classification and forecasting.

Classes/Functions:
------------------
- supervised: Classification models (Logistic, Tree, RF, XGBoost)
- semi_supervised: Self-training, Label Propagation, Label Spreading
- forecasting: Time series models (ARIMA, SARIMA, Exponential Smoothing)
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

# Import time series forecasting functions
from .forecasting import (
    # Data preparation
    create_date_column,
    aggregate_by_period,
    prepare_time_series,
    
    # Analysis
    check_stationarity,
    decompose_time_series,
    
    # Models
    train_arima,
    train_sarima,
    train_exponential_smoothing,
    train_prophet,
    
    # Baselines
    moving_average_forecast,
    naive_forecast,
    
    # Forecasting
    forecast,
    
    # Evaluation
    calculate_mae,
    calculate_rmse,
    calculate_mape,
    evaluate_forecast,
    
    # Train-test split
    train_test_split_ts,
    
    # Visualization
    plot_time_series,
    plot_decomposition,
    plot_forecast,
    plot_model_comparison,
    
    # Pipeline
    run_forecasting_pipeline,
    
    # Availability flags
    STATSMODELS_AVAILABLE,
    PROPHET_AVAILABLE
)

# Extend __all__ with forecasting exports
__all__.extend([
    # ============ TIME SERIES FORECASTING ============
    # Data preparation
    'create_date_column',
    'aggregate_by_period', 
    'prepare_time_series',
    
    # Analysis
    'check_stationarity',
    'decompose_time_series',
    
    # Models
    'train_arima',
    'train_sarima',
    'train_exponential_smoothing',
    'train_prophet',
    
    # Baselines
    'moving_average_forecast',
    'naive_forecast',
    
    # Forecasting
    'forecast',
    
    # Evaluation
    'calculate_mae',
    'calculate_rmse',
    'calculate_mape',
    'evaluate_forecast',
    
    # Train-test split
    'train_test_split_ts',
    
    # Visualization
    'plot_time_series',
    'plot_decomposition',
    'plot_forecast',
    'plot_model_comparison',
    
    # Pipeline
    'run_forecasting_pipeline',
    
    # Flags
    'STATSMODELS_AVAILABLE',
    'PROPHET_AVAILABLE'
])
