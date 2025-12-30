"""
Visualization Module - Plots
============================

Comprehensive plotting functions for data mining project.

Functions:
----------
- Distribution plots
- Correlation heatmaps
- Model comparison charts
- Learning curves
- Error analysis plots
- Business insights visualizations

Author: Nhom12
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set default style
plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'canceled': '#E74C3C',
    'not_canceled': '#2ECC71'
}

# Color palettes for different purposes
PALETTE_CATEGORICAL = 'Set2'
PALETTE_SEQUENTIAL = 'Blues'
PALETTE_DIVERGING = 'RdYlBu_r'


def set_plot_style(style: str = 'seaborn-v0_8-whitegrid', font_size: int = 11):
    """Set global plot style."""
    plt.style.use(style)
    plt.rcParams['font.size'] = font_size
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.titlesize'] = font_size + 2
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size - 1
    plt.rcParams['ytick.labelsize'] = font_size - 1


# =============================================================================
# DISTRIBUTION PLOTS
# =============================================================================

def plot_distribution(
    data: pd.Series,
    title: str = 'Distribution',
    xlabel: str = None,
    ylabel: str = 'Frequency',
    kind: str = 'hist',
    bins: int = 30,
    kde: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    color: str = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot distribution of a single variable.
    
    Parameters
    ----------
    data : pd.Series
        Data to plot
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    kind : str
        'hist', 'kde', 'box', 'violin'
    bins : int
        Number of bins for histogram
    kde : bool
        Whether to show KDE overlay
    figsize : tuple
        Figure size
    color : str
        Color for the plot
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    color = color or COLORS['primary']
    
    if kind == 'hist':
        ax.hist(data.dropna(), bins=bins, color=color, alpha=0.7, edgecolor='white')
        if kde:
            ax2 = ax.twinx()
            data.dropna().plot.kde(ax=ax2, color=COLORS['secondary'], linewidth=2)
            ax2.set_ylabel('Density')
            ax2.set_ylim(bottom=0)
    elif kind == 'kde':
        data.dropna().plot.kde(ax=ax, color=color, linewidth=2, fill=True, alpha=0.3)
    elif kind == 'box':
        ax.boxplot(data.dropna(), patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7))
    elif kind == 'violin':
        parts = ax.violinplot(data.dropna(), showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel or data.name or 'Value')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_target_distribution(
    data: pd.Series,
    labels: Dict[int, str] = None,
    title: str = 'Target Distribution',
    figsize: Tuple[int, int] = (10, 5),
    colors: List[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot target variable distribution with counts and percentages.
    
    Parameters
    ----------
    data : pd.Series
        Target variable
    labels : dict
        Mapping of values to labels (e.g., {0: 'Not Canceled', 1: 'Canceled'})
    title : str
        Plot title
    figsize : tuple
        Figure size
    colors : list
        Colors for each class
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Default labels
    if labels is None:
        labels = {0: 'Not Canceled', 1: 'Canceled'}
    
    # Default colors
    if colors is None:
        colors = [COLORS['not_canceled'], COLORS['canceled']]
    
    # Count plot
    value_counts = data.value_counts().sort_index()
    ax1 = axes[0]
    bars = ax1.bar([labels.get(i, str(i)) for i in value_counts.index], 
                   value_counts.values, color=colors)
    ax1.set_title('Count', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.bar_label(bars, fmt='%d')
    
    # Percentage plot
    ax2 = axes[1]
    percentages = (value_counts / len(data) * 100).values
    wedges, texts, autotexts = ax2.pie(
        percentages, 
        labels=[labels.get(i, str(i)) for i in value_counts.index],
        colors=colors,
        autopct='%1.1f%%',
        startangle=90
    )
    ax2.set_title('Percentage', fontsize=12, fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_distribution_by_group(
    df: pd.DataFrame,
    column: str,
    group_by: str,
    title: str = None,
    figsize: Tuple[int, int] = (12, 5),
    kind: str = 'hist',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot distribution of a variable split by groups.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
    column : str
        Column to plot
    group_by : str
        Column to group by
    title : str
        Plot title
    figsize : tuple
        Figure size
    kind : str
        'hist', 'kde', 'box'
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    groups = df[group_by].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(groups)))
    
    if kind == 'hist':
        for group, color in zip(groups, colors):
            subset = df[df[group_by] == group][column].dropna()
            ax.hist(subset, bins=30, alpha=0.5, label=str(group), color=color)
    elif kind == 'kde':
        for group, color in zip(groups, colors):
            subset = df[df[group_by] == group][column].dropna()
            subset.plot.kde(ax=ax, label=str(group), color=color, linewidth=2)
    elif kind == 'box':
        data_to_plot = [df[df[group_by] == g][column].dropna() for g in groups]
        bp = ax.boxplot(data_to_plot, labels=[str(g) for g in groups], patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax.set_title(title or f'{column} by {group_by}', fontsize=14, fontweight='bold')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency' if kind != 'box' else column)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# CORRELATION PLOTS
# =============================================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'pearson',
    title: str = 'Correlation Matrix',
    figsize: Tuple[int, int] = (12, 10),
    annot: bool = True,
    fmt: str = '.2f',
    cmap: str = 'RdYlBu_r',
    mask_upper: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot correlation heatmap.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numerical columns
    columns : list
        Columns to include (None for all numerical)
    method : str
        Correlation method: 'pearson', 'spearman', 'kendall'
    title : str
        Plot title
    figsize : tuple
        Figure size
    annot : bool
        Whether to annotate cells
    fmt : str
        Format for annotations
    cmap : str
        Colormap
    mask_upper : bool
        Whether to mask upper triangle
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Select columns
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Calculate correlation
    corr = df[columns].corr(method=method)
    
    # Create mask for upper triangle
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_feature_target_correlation(
    df: pd.DataFrame,
    target: str,
    top_n: int = 15,
    title: str = 'Feature Correlations with Target',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot top features correlated with target.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
    target : str
        Target column name
    top_n : int
        Number of top features to show
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate correlations with target
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numerical_cols].corrwith(df[target]).drop(target, errors='ignore')
    
    # Sort by absolute value and get top n
    top_corr = correlations.abs().sort_values(ascending=False).head(top_n)
    top_corr = correlations[top_corr.index].sort_values()
    
    # Color based on positive/negative
    colors = [COLORS['success'] if x > 0 else COLORS['danger'] for x in top_corr.values]
    
    # Plot
    bars = ax.barh(range(len(top_corr)), top_corr.values, color=colors)
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr.index)
    ax.set_xlabel('Correlation with Target')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, top_corr.values):
        ax.text(val + 0.01 if val >= 0 else val - 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left' if val >= 0 else 'right',
                fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# MODEL COMPARISON PLOTS
# =============================================================================

def plot_model_comparison_bar(
    results: pd.DataFrame,
    metrics: List[str] = None,
    title: str = 'Model Comparison',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot model comparison as grouped bar chart.
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame with models as index and metrics as columns
    metrics : list
        Metrics to plot (None for all)
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    if metrics is None:
        metrics = results.columns.tolist()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    models = results.index.tolist()
    x = np.arange(len(models))
    width = 0.8 / len(metrics)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, results[metric], width, label=metric, color=color)
        ax.bar_label(bars, fmt='%.3f', fontsize=8, rotation=45)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_model_comparison_radar(
    results: pd.DataFrame,
    metrics: List[str] = None,
    title: str = 'Model Comparison (Radar)',
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot model comparison as radar chart.
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame with models as index and metrics as columns
    metrics : list
        Metrics to plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    if metrics is None:
        metrics = results.columns.tolist()
    
    # Number of variables
    num_vars = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for (model, row), color in zip(results.iterrows(), colors):
        values = row[metrics].values.tolist()
        values += values[:1]  # Complete the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title(title, fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_model_ranking(
    results: pd.DataFrame,
    metric: str = 'f1',
    title: str = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot model ranking by a specific metric.
    
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame with models as index and metrics as columns
    metric : str
        Metric to rank by
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by metric
    sorted_results = results.sort_values(metric, ascending=True)
    
    # Color gradient
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_results)))
    
    # Plot
    bars = ax.barh(sorted_results.index, sorted_results[metric], color=colors)
    
    ax.set_xlabel(metric.upper())
    ax.set_title(title or f'Model Ranking by {metric.upper()}', fontsize=14, fontweight='bold')
    ax.bar_label(bars, fmt='%.4f', padding=3)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Highlight best
    ax.get_children()[len(sorted_results)-1].set_edgecolor('gold')
    ax.get_children()[len(sorted_results)-1].set_linewidth(3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# LEARNING CURVES
# =============================================================================

def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    title: str = 'Learning Curve',
    xlabel: str = 'Training Size',
    ylabel: str = 'Score',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot learning curve.
    
    Parameters
    ----------
    train_sizes : np.ndarray
        Array of training sizes
    train_scores : np.ndarray
        Training scores (can be 2D for multiple runs)
    test_scores : np.ndarray
        Test scores (can be 2D for multiple runs)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle 1D or 2D arrays
    if train_scores.ndim == 1:
        train_mean = train_scores
        train_std = np.zeros_like(train_scores)
        test_mean = test_scores
        test_std = np.zeros_like(test_scores)
    else:
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
    
    # Plot training curve
    ax.plot(train_sizes, train_mean, 'o-', color=COLORS['primary'], 
            label='Training Score', linewidth=2)
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color=COLORS['primary'])
    
    # Plot validation curve
    ax.plot(train_sizes, test_mean, 'o-', color=COLORS['secondary'],
            label='Cross-Validation Score', linewidth=2)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color=COLORS['secondary'])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_validation_curve(
    param_range: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    param_name: str = 'Parameter',
    title: str = 'Validation Curve',
    figsize: Tuple[int, int] = (10, 6),
    log_scale: bool = False,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot validation curve for hyperparameter tuning.
    
    Parameters
    ----------
    param_range : np.ndarray
        Parameter values
    train_scores : np.ndarray
        Training scores
    test_scores : np.ndarray
        Validation scores
    param_name : str
        Name of parameter
    title : str
        Plot title
    figsize : tuple
        Figure size
    log_scale : bool
        Whether to use log scale for x-axis
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot
    ax.plot(param_range, train_mean, 'o-', color=COLORS['primary'],
            label='Training Score', linewidth=2)
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color=COLORS['primary'])
    
    ax.plot(param_range, test_mean, 'o-', color=COLORS['secondary'],
            label='Cross-Validation Score', linewidth=2)
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color=COLORS['secondary'])
    
    if log_scale:
        ax.set_xscale('log')
    
    ax.set_xlabel(param_name)
    ax.set_ylabel('Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Mark best parameter
    best_idx = np.argmax(test_mean)
    ax.axvline(x=param_range[best_idx], color=COLORS['success'], linestyle='--',
               label=f'Best: {param_range[best_idx]}')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# ERROR ANALYSIS PLOTS
# =============================================================================

def plot_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_values: np.ndarray,
    feature_name: str,
    title: str = None,
    bins: int = 10,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Analyze prediction errors by feature values.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    feature_values : np.ndarray
        Feature values for grouping
    feature_name : str
        Feature name
    title : str
        Plot title
    bins : int
        Number of bins for continuous features
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate errors
    errors = (y_true != y_pred).astype(int)
    
    # Bin continuous features
    if np.issubdtype(feature_values.dtype, np.number):
        bins_edges = np.percentile(feature_values, np.linspace(0, 100, bins + 1))
        binned = np.digitize(feature_values, bins_edges[1:-1])
        bin_labels = [f'{bins_edges[i]:.1f}-{bins_edges[i+1]:.1f}' for i in range(len(bins_edges)-1)]
    else:
        binned = feature_values
        bin_labels = None
    
    # Error rate by bin
    error_df = pd.DataFrame({'bin': binned, 'error': errors})
    error_rate = error_df.groupby('bin')['error'].mean()
    
    # Plot 1: Error rate
    ax1 = axes[0]
    bars = ax1.bar(range(len(error_rate)), error_rate.values, color=COLORS['danger'], alpha=0.7)
    ax1.set_xlabel(feature_name)
    ax1.set_ylabel('Error Rate')
    ax1.set_title('Error Rate by Feature Value', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(error_rate)))
    if bin_labels:
        ax1.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Count distribution
    ax2 = axes[1]
    count_by_bin = error_df.groupby('bin')['error'].count()
    ax2.bar(range(len(count_by_bin)), count_by_bin.values, color=COLORS['info'], alpha=0.7)
    ax2.set_xlabel(feature_name)
    ax2.set_ylabel('Count')
    ax2.set_title('Sample Distribution', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(count_by_bin)))
    if bin_labels:
        ax2.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(title or f'Error Analysis by {feature_name}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_confusion_matrix_detailed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    title: str = 'Confusion Matrix',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot detailed confusion matrix with counts and percentages.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : list
        Class labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    from sklearn.metrics import confusion_matrix
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = ['Not Canceled', 'Canceled']
    
    # Plot 1: Counts
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=labels, yticklabels=labels)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Counts', fontsize=12, fontweight='bold')
    
    # Plot 2: Percentages (row-normalized)
    ax2 = axes[1]
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                xticklabels=labels, yticklabels=labels)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Percentages (Row-Normalized)', fontsize=12, fontweight='bold')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# BUSINESS INSIGHTS PLOTS
# =============================================================================

def plot_cancellation_by_category(
    df: pd.DataFrame,
    category_col: str,
    target_col: str = 'is_canceled',
    title: str = None,
    top_n: int = 10,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot cancellation rate by category.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
    category_col : str
        Category column
    target_col : str
        Target column
    title : str
        Plot title
    top_n : int
        Top categories to show
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Calculate stats
    stats = df.groupby(category_col)[target_col].agg(['mean', 'count'])
    stats.columns = ['cancel_rate', 'count']
    stats = stats.sort_values('cancel_rate', ascending=False).head(top_n)
    
    # Plot 1: Cancellation rate
    ax1 = axes[0]
    colors = plt.cm.RdYlGn_r(stats['cancel_rate'].values)
    bars = ax1.barh(range(len(stats)), stats['cancel_rate'] * 100, color=colors)
    ax1.set_yticks(range(len(stats)))
    ax1.set_yticklabels(stats.index)
    ax1.set_xlabel('Cancellation Rate (%)')
    ax1.set_title('Cancellation Rate', fontsize=12, fontweight='bold')
    ax1.bar_label(bars, fmt='%.1f%%', padding=3)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Count
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(stats)), stats['count'], color=COLORS['info'], alpha=0.7)
    ax2.set_yticks(range(len(stats)))
    ax2.set_yticklabels(stats.index)
    ax2.set_xlabel('Count')
    ax2.set_title('Booking Count', fontsize=12, fontweight='bold')
    ax2.bar_label(bars2, fmt='%d', padding=3)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle(title or f'Cancellation Analysis by {category_col}', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_monthly_trend(
    df: pd.DataFrame,
    date_col: str = 'arrival_date_month',
    target_col: str = 'is_canceled',
    title: str = 'Monthly Cancellation Trend',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot monthly cancellation trend.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
    date_col : str
        Date/month column
    target_col : str
        Target column
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Order months
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    # Calculate monthly stats
    monthly = df.groupby(date_col)[target_col].agg(['mean', 'count'])
    monthly.columns = ['cancel_rate', 'count']
    
    # Reorder if possible
    if df[date_col].dtype == 'object':
        monthly = monthly.reindex([m for m in month_order if m in monthly.index])
    
    # Plot
    ax2 = ax.twinx()
    
    # Bar plot for count
    bars = ax2.bar(range(len(monthly)), monthly['count'], 
                   color=COLORS['info'], alpha=0.3, label='Booking Count')
    ax2.set_ylabel('Booking Count', color=COLORS['info'])
    
    # Line plot for cancellation rate
    line = ax.plot(range(len(monthly)), monthly['cancel_rate'] * 100, 
                   'o-', color=COLORS['danger'], linewidth=2, markersize=8, 
                   label='Cancellation Rate')
    ax.set_ylabel('Cancellation Rate (%)', color=COLORS['danger'])
    
    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly.index, rotation=45, ha='right')
    ax.set_xlabel('Month')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_lead_time_analysis(
    df: pd.DataFrame,
    lead_time_col: str = 'lead_time',
    target_col: str = 'is_canceled',
    bins: int = 20,
    title: str = 'Cancellation Rate by Lead Time',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot cancellation rate by lead time bins.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame
    lead_time_col : str
        Lead time column
    target_col : str
        Target column
    bins : int
        Number of bins
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bins
    df_temp = df.copy()
    df_temp['lead_time_bin'] = pd.cut(df_temp[lead_time_col], bins=bins)
    
    # Calculate stats
    stats = df_temp.groupby('lead_time_bin')[target_col].agg(['mean', 'count'])
    stats.columns = ['cancel_rate', 'count']
    
    # Plot
    ax2 = ax.twinx()
    
    x = range(len(stats))
    
    # Bar for count
    ax2.bar(x, stats['count'], color=COLORS['info'], alpha=0.3, label='Count')
    ax2.set_ylabel('Booking Count', color=COLORS['info'])
    
    # Line for cancellation rate
    ax.plot(x, stats['cancel_rate'] * 100, 'o-', color=COLORS['danger'],
            linewidth=2, markersize=6, label='Cancellation Rate')
    ax.set_ylabel('Cancellation Rate (%)', color=COLORS['danger'])
    
    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(i.left)}-{int(i.right)}' for i in stats.index], 
                       rotation=45, ha='right')
    ax.set_xlabel('Lead Time (days)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# SUMMARY DASHBOARD
# =============================================================================

def create_summary_dashboard(
    model_results: pd.DataFrame,
    best_model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_importance: pd.DataFrame = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary dashboard.
    
    Parameters
    ----------
    model_results : pd.DataFrame
        Model comparison results
    best_model : str
        Name of best model
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predictions from best model
    feature_importance : pd.DataFrame
        Feature importance (optional)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    from sklearn.metrics import confusion_matrix
    
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Model comparison bar (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['accuracy', 'precision', 'recall', 'f1'] if 'accuracy' in model_results.columns else model_results.columns[:4].tolist()
    available_metrics = [m for m in metrics if m in model_results.columns]
    x = np.arange(len(model_results))
    width = 0.2
    colors = plt.cm.Set2(np.linspace(0, 1, len(available_metrics)))
    for i, (metric, color) in enumerate(zip(available_metrics, colors)):
        offset = (i - len(available_metrics)/2 + 0.5) * width
        ax1.bar(x + offset, model_results[metric], width, label=metric, color=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_results.index, rotation=45, ha='right')
    ax1.set_title('Model Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Best model metrics (top center) - Text summary
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    best_metrics = model_results.loc[best_model]
    text = f"🏆 Best Model: {best_model}\n\n"
    for metric, value in best_metrics.items():
        text += f"  {metric}: {value:.4f}\n"
    ax2.text(0.5, 0.5, text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5),
             family='monospace')
    ax2.set_title('Best Model Summary', fontsize=12, fontweight='bold')
    
    # Plot 3: Confusion Matrix (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Not Canceled', 'Canceled'],
                yticklabels=['Not Canceled', 'Canceled'])
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Plot 4: Model ranking (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])
    rank_metric = 'f1' if 'f1' in model_results.columns else model_results.columns[0]
    sorted_models = model_results.sort_values(rank_metric, ascending=True)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_models)))
    bars = ax4.barh(sorted_models.index, sorted_models[rank_metric], color=colors)
    ax4.set_xlabel(rank_metric.upper())
    ax4.set_title(f'Model Ranking by {rank_metric.upper()}', fontsize=12, fontweight='bold')
    ax4.bar_label(bars, fmt='%.4f', padding=3)
    
    # Plot 5: Feature Importance (bottom center and right)
    if feature_importance is not None:
        ax5 = fig.add_subplot(gs[1, 1:])
        top_features = feature_importance.head(15)
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(top_features)))
        bars = ax5.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax5.set_yticks(range(len(top_features)))
        ax5.set_yticklabels(top_features['feature'])
        ax5.set_xlabel('Importance')
        ax5.set_title('Top 15 Feature Importance', fontsize=12, fontweight='bold')
        ax5.invert_yaxis()
    
    fig.suptitle('Model Evaluation Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    'set_plot_style',
    'COLORS',
    
    # Distribution plots
    'plot_distribution',
    'plot_target_distribution',
    'plot_distribution_by_group',
    
    # Correlation plots
    'plot_correlation_heatmap',
    'plot_feature_target_correlation',
    
    # Model comparison
    'plot_model_comparison_bar',
    'plot_model_comparison_radar',
    'plot_model_ranking',
    
    # Learning curves
    'plot_learning_curve',
    'plot_validation_curve',
    
    # Error analysis
    'plot_error_analysis',
    'plot_confusion_matrix_detailed',
    
    # Business insights
    'plot_cancellation_by_category',
    'plot_monthly_trend',
    'plot_lead_time_analysis',
    
    # Dashboard
    'create_summary_dashboard'
]
