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

# Import report functions
from .report import (
    # Summary tables
    create_model_summary_table,
    create_classification_report_table,
    create_comparison_table,
    create_feature_importance_table,
    create_error_analysis_table,
    
    # Business insights
    extract_business_insights,
    format_insights_markdown,
    
    # Export functions
    export_table_csv,
    export_table_latex,
    export_table_excel,
    export_results_json,
    
    # Report generation
    generate_summary_report,
    generate_full_report
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
    'evaluate_model',
    
    # Report - Summary tables
    'create_model_summary_table',
    'create_classification_report_table',
    'create_comparison_table',
    'create_feature_importance_table',
    'create_error_analysis_table',
    
    # Report - Business insights
    'extract_business_insights',
    'format_insights_markdown',
    
    # Report - Export functions
    'export_table_csv',
    'export_table_latex',
    'export_table_excel',
    'export_results_json',
    
    # Report - Report generation
    'generate_summary_report',
    'generate_full_report'
]
