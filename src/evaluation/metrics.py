"""
Evaluation Metrics for Classification Models
============================================

This module provides comprehensive evaluation metrics and visualization
functions for classification model performance.

Metrics included:
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC
- Confusion Matrix
- Classification Report

Visualizations:
- Confusion Matrix heatmap
- ROC Curve
- Precision-Recall Curve
- Feature Importance plot
- Model Comparison chart

Author: Nhom12
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


# =============================================================================
# METRIC CALCULATION
# =============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Predicted probabilities for positive class (for AUC metrics)
    verbose : bool
        Whether to print results
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing all metrics
        
    Examples
    --------
    >>> metrics = calculate_metrics(y_test, y_pred, y_proba)
    >>> print(f"F1 Score: {metrics['f1']:.4f}")
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Add AUC metrics if probabilities provided
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = np.nan
            
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics['pr_auc'] = np.nan
    
    # Calculate specificity (True Negative Rate)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['true_positives'] = tp
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
    
    if verbose:
        print("\n" + "=" * 60)
        print("CLASSIFICATION METRICS")
        print("=" * 60)
        print(f"Accuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1 Score:    {metrics['f1']:.4f}")
        if 'specificity' in metrics:
            print(f"Specificity: {metrics['specificity']:.4f}")
        if y_proba is not None:
            print(f"ROC-AUC:     {metrics.get('roc_auc', 'N/A'):.4f}" if not np.isnan(metrics.get('roc_auc', np.nan)) else "ROC-AUC:     N/A")
            print(f"PR-AUC:      {metrics.get('pr_auc', 'N/A'):.4f}" if not np.isnan(metrics.get('pr_auc', np.nan)) else "PR-AUC:      N/A")
        print("=" * 60)
    
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str] = None,
    output_dict: bool = False,
    verbose: bool = True
) -> Union[str, Dict]:
    """
    Generate detailed classification report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    target_names : list of str, optional
        Names for each class. Default: ['Not Canceled', 'Canceled']
    output_dict : bool
        If True, return dict instead of string
    verbose : bool
        Whether to print report
        
    Returns
    -------
    str or Dict
        Classification report
    """
    if target_names is None:
        target_names = ['Not Canceled', 'Canceled']
    
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0
    )
    
    if verbose and not output_dict:
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(report)
    
    return report


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    normalize: bool = False,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : list of str, optional
        Class names. Default: ['Not Canceled', 'Canceled']
    title : str
        Plot title
    figsize : tuple
        Figure size
    cmap : str
        Colormap name
    normalize : bool
        If True, normalize values (show percentages)
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
        
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    if labels is None:
        labels = ['Not Canceled', 'Canceled']
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap=cmap,
        xticklabels=labels, yticklabels=labels,
        ax=ax, cbar=True,
        annot_kws={'size': 14}
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# ROC CURVE
# =============================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = 'Model',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> Tuple[plt.Figure, float]:
    """
    Plot ROC curve for a single model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    model_name : str
        Name of the model for legend
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
        
    Returns
    -------
    Tuple[plt.Figure, float]
        Figure object and ROC-AUC score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved ROC curve to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, roc_auc


def plot_roc_curves_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot ROC curves for multiple models for comparison.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    predictions : Dict[str, np.ndarray]
        Dictionary of model_name -> y_proba
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
        
    Returns
    -------
    plt.Figure
        Figure object
        
    Examples
    --------
    >>> predictions = {
    ...     'Logistic': y_proba_lr,
    ...     'Random Forest': y_proba_rf
    ... }
    >>> plot_roc_curves_comparison(y_test, predictions)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    
    for (name, y_proba), color in zip(predictions.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, lw=2, color=color, label=f'{name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved ROC curves comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# PRECISION-RECALL CURVE
# =============================================================================

def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = 'Model',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> Tuple[plt.Figure, float]:
    """
    Plot Precision-Recall curve for a single model.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    model_name : str
        Name of the model for legend
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
        
    Returns
    -------
    Tuple[plt.Figure, float]
        Figure object and PR-AUC score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    # Baseline (random classifier)
    baseline = y_true.mean()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, lw=2, label=f'{model_name} (AP = {pr_auc:.4f})')
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1, 
               label=f'Baseline (AP = {baseline:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved PR curve to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, pr_auc


def plot_pr_curves_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple models for comparison.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    predictions : Dict[str, np.ndarray]
        Dictionary of model_name -> y_proba
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    baseline = y_true.mean()
    
    for (name, y_proba), color in zip(predictions.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        ax.plot(recall, precision, lw=2, color=color, label=f'{name} (AP = {pr_auc:.4f})')
    
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1, 
               label=f'Baseline (AP = {baseline:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved PR curves comparison to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# FEATURE IMPORTANCE VISUALIZATION
# =============================================================================

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = 'Feature Importance',
    figsize: Tuple[int, int] = (10, 10),
    color: str = 'steelblue',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
        (output from get_feature_importance)
    top_n : int
        Number of top features to show
    title : str
        Plot title
    figsize : tuple
        Figure size
    color : str
        Bar color
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    df_plot = importance_df.head(top_n).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(df_plot['feature'], df_plot['importance'], color=color, edgecolor='black')
    
    # Add value labels
    for bar, val in zip(bars, df_plot['importance']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved feature importance plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def compare_models(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create a comparison table of multiple models.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Dictionary of model_name -> metrics_dict
    metrics : list of str, optional
        Metrics to include. Default: all available
    verbose : bool
        Whether to print table
        
    Returns
    -------
    pd.DataFrame
        Comparison table
        
    Examples
    --------
    >>> results = {
    ...     'Logistic': {'accuracy': 0.8, 'f1': 0.75},
    ...     'RF': {'accuracy': 0.85, 'f1': 0.82}
    ... }
    >>> df = compare_models(results)
    """
    # Create DataFrame
    df = pd.DataFrame(results).T
    
    # Filter metrics if specified
    if metrics is not None:
        available = [m for m in metrics if m in df.columns]
        df = df[available]
    
    # Sort by F1 score if available
    if 'f1' in df.columns:
        df = df.sort_values('f1', ascending=False)
    
    # Round values
    df = df.round(4)
    
    if verbose:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(df.to_string())
        print("=" * 70)
        
        # Highlight best model
        if 'f1' in df.columns:
            best_model = df['f1'].idxmax()
            print(f"\n🏆 Best model by F1: {best_model} ({df.loc[best_model, 'f1']:.4f})")
    
    return df


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: List[str] = None,
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create bar chart comparing models across metrics.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from compare_models()
    metrics : list of str, optional
        Metrics to plot. Default: all numeric columns
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    if metrics is None:
        metrics = [col for col in comparison_df.columns 
                   if comparison_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    # Filter to available metrics
    metrics = [m for m in metrics if m in comparison_df.columns]
    
    df_plot = comparison_df[metrics]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df_plot.index))
    width = 0.8 / len(metrics)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, df_plot[metric], width, label=metric.upper(), color=color)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot.index, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved model comparison plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# THRESHOLD ANALYSIS
# =============================================================================

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1',
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Find optimal classification threshold for a given metric.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    metric : str
        Metric to optimize: 'f1', 'precision', 'recall', 'youden' (J-statistic)
    verbose : bool
        Whether to print results
        
    Returns
    -------
    Tuple[float, float]
        (optimal_threshold, metric_value)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic = Sensitivity + Specificity - 1
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            else:
                score = 0
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    if verbose:
        print(f"\n📊 Optimal threshold for {metric.upper()}:")
        print(f"   Threshold: {best_threshold:.2f}")
        print(f"   Score: {best_score:.4f}")
    
    return best_threshold, best_score


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot metrics across different thresholds.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
        
    Returns
    -------
    plt.Figure
        Figure object
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    
    precisions = []
    recalls = []
    f1s = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Precision, Recall, F1
    ax1.plot(thresholds, precisions, label='Precision', lw=2)
    ax1.plot(thresholds, recalls, label='Recall', lw=2)
    ax1.plot(thresholds, f1s, label='F1 Score', lw=2)
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default (0.5)')
    
    # Mark optimal F1 threshold
    best_idx = np.argmax(f1s)
    ax1.axvline(x=thresholds[best_idx], color='red', linestyle=':', 
                label=f'Best F1 ({thresholds[best_idx]:.2f})')
    
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Precision-Recall tradeoff
    ax2.plot(recalls, precisions, lw=2)
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved threshold analysis to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = 'Model',
    save_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Complete evaluation pipeline for a single model.
    
    Parameters
    ----------
    model : estimator
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    model_name : str
        Name of the model for saving files
    save_dir : str, optional
        Directory to save figures
    verbose : bool
        Whether to print results
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing metrics, predictions, and probabilities
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba, verbose=verbose)
    
    # Classification report
    report = get_classification_report(y_test, y_pred, output_dict=True, verbose=False)
    
    # Plots
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        plot_confusion_matrix(
            y_test, y_pred,
            title=f'Confusion Matrix - {model_name}',
            save_path=str(save_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'),
            show=False
        )
        
        # ROC curve
        if y_proba is not None:
            plot_roc_curve(
                y_test, y_proba, model_name,
                save_path=str(save_dir / f'roc_curve_{model_name.lower().replace(" ", "_")}.png'),
                show=False
            )
            
            # PR curve
            plot_pr_curve(
                y_test, y_proba, model_name,
                save_path=str(save_dir / f'pr_curve_{model_name.lower().replace(" ", "_")}.png'),
                show=False
            )
    
    return {
        'model_name': model_name,
        'metrics': metrics,
        'report': report,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


# Export all functions
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
