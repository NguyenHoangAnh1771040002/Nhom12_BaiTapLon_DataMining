"""
Evaluation Module
=================

Metrics and reporting utilities.

Classes/Functions:
------------------
- metrics: Calculate various evaluation metrics
- report: Generate reports and summaries
"""

# Import metrics functions
from .metrics import (
    # Metrics
    calculate_metrics,
    get_classification_report,
    
    # Confusion Matrix
    plot_confusion_matrix,
    
    # ROC Curves
    plot_roc_curve,
    plot_roc_curves_comparison,
    
    # PR Curves
    plot_pr_curve,
    plot_pr_curves_comparison,
    
    # Feature Importance
    plot_feature_importance,
    
    # Model Comparison
    compare_models,
    plot_model_comparison,
    
    # Threshold Analysis
    find_optimal_threshold,
    plot_threshold_analysis,
    
    # Pipeline
    evaluate_model
)

__all__ = [
    # Metrics
    'calculate_metrics',
    'get_classification_report',
    
    # Confusion Matrix
    'plot_confusion_matrix',
    
    # ROC Curves
    'plot_roc_curve',
    'plot_roc_curves_comparison',
    
    # PR Curves
    'plot_pr_curve',
    'plot_pr_curves_comparison',
    
    # Feature Importance
    'plot_feature_importance',
    
    # Model Comparison
    'compare_models',
    'plot_model_comparison',
    
    # Threshold Analysis
    'find_optimal_threshold',
    'plot_threshold_analysis',
    
    # Pipeline
    'evaluate_model'
]
