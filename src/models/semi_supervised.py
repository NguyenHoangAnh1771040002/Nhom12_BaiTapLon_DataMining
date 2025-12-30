"""
Mô Hình Học Bán Giám Sát - Dự Đoán Huỷ Đặt Phòng
=====================================================
(Semi-Supervised Learning for Hotel Booking Cancellation Prediction)

Module cung cấp các phương pháp học bán giám sát cho kịch bản
chỉ có một phần nhỏ dữ liệu được gán nhãn.

Các phương pháp:
- Self-Training: Lặp lại gán nhãn dữ liệu chưa gán bằng dự đoán tin cậy
- Label Propagation: Lan truyền nhãn dựa trên đồ thị
- Label Spreading: Phiên bản mềm của Label Propagation

Trường hợp sử dụng:
- Khi việc gán nhãn tốn kém và chỉ có một phần nhãn
- Tận dụng dữ liệu chưa gán nhãn để cải thiện mô hình
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from pathlib import Path

# Scikit-learn imports
from sklearn.semi_supervised import (
    SelfTrainingClassifier,
    LabelPropagation,
    LabelSpreading
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


# =============================================================================
# LABELED/UNLABELED SPLIT
# =============================================================================

def create_labeled_unlabeled_split(
    X: pd.DataFrame,
    y: pd.Series,
    labeled_fraction: float = 0.1,
    stratify: bool = True,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Create a labeled/unlabeled split from fully labeled data.
    
    Simulates a semi-supervised scenario by hiding labels from a portion
    of the data. The unlabeled samples are marked with -1.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels (fully labeled)
    labeled_fraction : float
        Fraction of data to keep labeled (0.0 to 1.0)
        E.g., 0.1 means 10% labeled, 90% unlabeled
    stratify : bool
        Whether to stratify the labeled sample to maintain class balance
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Tuple[X, y_semi, mask_labeled]
        - X: Original features (unchanged)
        - y_semi: Labels where unlabeled samples are marked as -1
        - mask_labeled: Boolean mask of labeled samples
        
    Examples
    --------
    >>> X, y_semi, mask = create_labeled_unlabeled_split(X_train, y_train, labeled_fraction=0.1)
    >>> print(f"Labeled: {mask.sum()}, Unlabeled: {(~mask).sum()}")
    """
    if not 0.0 < labeled_fraction <= 1.0:
        raise ValueError(f"labeled_fraction must be in (0, 1], got {labeled_fraction}")
    
    n_samples = len(X)
    n_labeled = int(n_samples * labeled_fraction)
    
    if verbose:
        print("=" * 60)
        print("CREATING LABELED/UNLABELED SPLIT")
        print("=" * 60)
        print(f"Total samples: {n_samples:,}")
        print(f"Labeled fraction: {labeled_fraction:.1%}")
        print(f"Target labeled samples: {n_labeled:,}")
    
    # Create stratified split
    if stratify and labeled_fraction < 1.0:
        # Get indices of labeled samples
        _, _, _, _, idx_labeled, _ = train_test_split(
            X, y, np.arange(n_samples),
            train_size=labeled_fraction,
            stratify=y,
            random_state=random_state
        )
    else:
        np.random.seed(random_state)
        idx_labeled = np.random.choice(n_samples, size=n_labeled, replace=False)
    
    # Create mask
    mask_labeled = np.zeros(n_samples, dtype=bool)
    mask_labeled[idx_labeled] = True
    
    # Create semi-supervised labels (-1 for unlabeled)
    y_semi = y.copy()
    y_semi = y_semi.values if hasattr(y_semi, 'values') else np.array(y_semi)
    y_semi = y_semi.astype(float)  # Convert to float to handle -1
    y_semi[~mask_labeled] = -1
    
    if verbose:
        n_actual_labeled = mask_labeled.sum()
        n_unlabeled = (~mask_labeled).sum()
        
        print(f"\n📊 Split Result:")
        print(f"   Labeled samples: {n_actual_labeled:,} ({n_actual_labeled/n_samples:.1%})")
        print(f"   Unlabeled samples: {n_unlabeled:,} ({n_unlabeled/n_samples:.1%})")
        
        # Class distribution in labeled set
        y_labeled = y.values[mask_labeled] if hasattr(y, 'values') else np.array(y)[mask_labeled]
        class_counts = pd.Series(y_labeled).value_counts().sort_index()
        print(f"\n   Labeled class distribution:")
        for cls, count in class_counts.items():
            print(f"      Class {int(cls)}: {count:,} ({count/n_actual_labeled:.1%})")
        print("=" * 60)
    
    return X, y_semi, mask_labeled


def create_multiple_splits(
    X: pd.DataFrame,
    y: pd.Series,
    fractions: List[float] = [0.05, 0.10, 0.20],
    random_state: int = 42,
    verbose: bool = True
) -> Dict[float, Tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    """
    Create multiple labeled/unlabeled splits with different label fractions.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    fractions : list of float
        List of labeled fractions to create
    random_state : int
        Random seed
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Dict[float, Tuple]
        Dictionary mapping fraction -> (X, y_semi, mask_labeled)
    """
    splits = {}
    
    if verbose:
        print("=" * 60)
        print("CREATING MULTIPLE SPLITS")
        print("=" * 60)
    
    for frac in fractions:
        X_split, y_semi, mask = create_labeled_unlabeled_split(
            X, y, labeled_fraction=frac,
            random_state=random_state,
            verbose=False
        )
        splits[frac] = (X_split, y_semi, mask)
        
        if verbose:
            n_labeled = mask.sum()
            print(f"   {frac:.0%} labeled: {n_labeled:,} samples")
    
    if verbose:
        print("=" * 60)
    
    return splits


# =============================================================================
# SELF-TRAINING
# =============================================================================

def train_self_training(
    X: pd.DataFrame,
    y_semi: np.ndarray,
    base_estimator: Any = None,
    threshold: float = 0.9,
    max_iter: int = 10,
    verbose: bool = True
) -> Tuple[SelfTrainingClassifier, Dict[str, Any]]:
    """
    Train a Self-Training classifier.
    
    Self-Training iteratively:
    1. Train on labeled data
    2. Predict on unlabeled data
    3. Add high-confidence predictions to labeled set
    4. Repeat until convergence or max iterations
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y_semi : np.ndarray
        Labels with -1 for unlabeled samples
    base_estimator : estimator, optional
        Base classifier. Default: LogisticRegression
    threshold : float
        Confidence threshold for adding pseudo-labels (0.5 to 1.0)
        Higher = more conservative (fewer but more accurate labels)
    max_iter : int
        Maximum number of self-training iterations
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Tuple[model, info_dict]
        Trained model and dictionary with training info
        
    Examples
    --------
    >>> model, info = train_self_training(X, y_semi, threshold=0.9)
    >>> y_pred = model.predict(X_test)
    """
    if verbose:
        print("=" * 60)
        print("TRAINING SELF-TRAINING CLASSIFIER")
        print("=" * 60)
        print(f"Confidence threshold: {threshold}")
        print(f"Max iterations: {max_iter}")
        
        n_labeled = (y_semi != -1).sum()
        n_unlabeled = (y_semi == -1).sum()
        print(f"Initial labeled: {n_labeled:,}")
        print(f"Initial unlabeled: {n_unlabeled:,}")
    
    # Default base estimator
    if base_estimator is None:
        base_estimator = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    
    # Create Self-Training classifier
    model = SelfTrainingClassifier(
        base_estimator=base_estimator,
        threshold=threshold,
        max_iter=max_iter,
        verbose=verbose
    )
    
    # Fit
    model.fit(X, y_semi)
    
    # Collect info
    info = {
        'threshold': threshold,
        'max_iter': max_iter,
        'n_iter': model.n_iter_,
        'termination_condition': model.termination_condition_,
        'base_estimator': type(base_estimator).__name__
    }
    
    # Count labeled after training
    labeled_after = (model.transduction_ != -1).sum()
    pseudo_labeled = labeled_after - n_labeled
    
    info['n_labeled_initial'] = n_labeled
    info['n_labeled_final'] = labeled_after
    info['n_pseudo_labeled'] = pseudo_labeled
    
    if verbose:
        print(f"\n📊 Self-Training Results:")
        print(f"   Iterations: {model.n_iter_}")
        print(f"   Termination: {model.termination_condition_}")
        print(f"   Pseudo-labels added: {pseudo_labeled:,}")
        print(f"   Final labeled: {labeled_after:,} ({labeled_after/len(y_semi):.1%})")
        print("=" * 60)
    
    return model, info


def train_self_training_rf(
    X: pd.DataFrame,
    y_semi: np.ndarray,
    threshold: float = 0.9,
    n_estimators: int = 100,
    max_depth: int = 15,
    verbose: bool = True
) -> Tuple[SelfTrainingClassifier, Dict[str, Any]]:
    """
    Train Self-Training with Random Forest as base estimator.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y_semi : np.ndarray
        Labels with -1 for unlabeled
    threshold : float
        Confidence threshold
    n_estimators : int
        Number of trees
    max_depth : int
        Maximum tree depth
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Tuple[model, info]
    """
    base_rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    return train_self_training(
        X, y_semi,
        base_estimator=base_rf,
        threshold=threshold,
        verbose=verbose
    )


# =============================================================================
# LABEL PROPAGATION
# =============================================================================

def train_label_propagation(
    X: pd.DataFrame,
    y_semi: np.ndarray,
    kernel: str = 'rbf',
    gamma: float = 20,
    n_neighbors: int = 7,
    max_iter: int = 1000,
    verbose: bool = True
) -> Tuple[LabelPropagation, Dict[str, Any]]:
    """
    Train a Label Propagation classifier.
    
    Label Propagation uses graph-based semi-supervised learning:
    - Constructs a similarity graph between samples
    - Propagates labels through the graph
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y_semi : np.ndarray
        Labels with -1 for unlabeled
    kernel : str
        Kernel type: 'rbf' or 'knn'
    gamma : float
        RBF kernel parameter (only for 'rbf' kernel)
    n_neighbors : int
        Number of neighbors (only for 'knn' kernel)
    max_iter : int
        Maximum iterations
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Tuple[model, info]
    """
    if verbose:
        print("=" * 60)
        print("TRAINING LABEL PROPAGATION")
        print("=" * 60)
        print(f"Kernel: {kernel}")
        if kernel == 'rbf':
            print(f"Gamma: {gamma}")
        else:
            print(f"N neighbors: {n_neighbors}")
        
        n_labeled = (y_semi != -1).sum()
        n_unlabeled = (y_semi == -1).sum()
        print(f"Labeled samples: {n_labeled:,}")
        print(f"Unlabeled samples: {n_unlabeled:,}")
    
    # Create model
    if kernel == 'rbf':
        model = LabelPropagation(
            kernel='rbf',
            gamma=gamma,
            max_iter=max_iter,
            n_jobs=-1
        )
    else:
        model = LabelPropagation(
            kernel='knn',
            n_neighbors=n_neighbors,
            max_iter=max_iter,
            n_jobs=-1
        )
    
    # Fit
    model.fit(X, y_semi)
    
    info = {
        'kernel': kernel,
        'gamma': gamma if kernel == 'rbf' else None,
        'n_neighbors': n_neighbors if kernel == 'knn' else None,
        'n_iter': model.n_iter_
    }
    
    if verbose:
        print(f"\n📊 Label Propagation Results:")
        print(f"   Iterations: {model.n_iter_}")
        print("=" * 60)
    
    return model, info


# =============================================================================
# LABEL SPREADING
# =============================================================================

def train_label_spreading(
    X: pd.DataFrame,
    y_semi: np.ndarray,
    kernel: str = 'rbf',
    gamma: float = 20,
    n_neighbors: int = 7,
    alpha: float = 0.2,
    max_iter: int = 30,
    verbose: bool = True
) -> Tuple[LabelSpreading, Dict[str, Any]]:
    """
    Train a Label Spreading classifier.
    
    Label Spreading is a soft version of Label Propagation:
    - Uses a soft clamping technique
    - alpha controls how much initial labels are preserved
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y_semi : np.ndarray
        Labels with -1 for unlabeled
    kernel : str
        Kernel type: 'rbf' or 'knn'
    gamma : float
        RBF kernel parameter
    n_neighbors : int
        Number of neighbors for KNN kernel
    alpha : float
        Clamping factor (0-1). Higher = more weight on initial labels
    max_iter : int
        Maximum iterations
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Tuple[model, info]
    """
    if verbose:
        print("=" * 60)
        print("TRAINING LABEL SPREADING")
        print("=" * 60)
        print(f"Kernel: {kernel}")
        print(f"Alpha (clamping): {alpha}")
        
        n_labeled = (y_semi != -1).sum()
        n_unlabeled = (y_semi == -1).sum()
        print(f"Labeled samples: {n_labeled:,}")
        print(f"Unlabeled samples: {n_unlabeled:,}")
    
    # Create model
    if kernel == 'rbf':
        model = LabelSpreading(
            kernel='rbf',
            gamma=gamma,
            alpha=alpha,
            max_iter=max_iter,
            n_jobs=-1
        )
    else:
        model = LabelSpreading(
            kernel='knn',
            n_neighbors=n_neighbors,
            alpha=alpha,
            max_iter=max_iter,
            n_jobs=-1
        )
    
    # Fit
    model.fit(X, y_semi)
    
    info = {
        'kernel': kernel,
        'gamma': gamma if kernel == 'rbf' else None,
        'n_neighbors': n_neighbors if kernel == 'knn' else None,
        'alpha': alpha,
        'n_iter': model.n_iter_
    }
    
    if verbose:
        print(f"\n📊 Label Spreading Results:")
        print(f"   Iterations: {model.n_iter_}")
        print("=" * 60)
    
    return model, info


# =============================================================================
# PSEUDO-LABEL ANALYSIS
# =============================================================================

def analyze_pseudo_labels(
    y_true: np.ndarray,
    y_pseudo: np.ndarray,
    mask_unlabeled: np.ndarray,
    X: Optional[pd.DataFrame] = None,
    feature_to_analyze: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze the quality of pseudo-labels against true labels.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels (ground truth for all samples)
    y_pseudo : np.ndarray
        Pseudo-labels from semi-supervised model
    mask_unlabeled : np.ndarray
        Boolean mask indicating which samples were originally unlabeled
    X : pd.DataFrame, optional
        Features for detailed analysis
    feature_to_analyze : str, optional
        Feature name to analyze errors by (e.g., 'lead_time')
    verbose : bool
        Whether to print analysis
        
    Returns
    -------
    Dict[str, Any]
        Analysis results including accuracy, confusion matrix, error patterns
    """
    # Get only the originally unlabeled samples
    y_true_unlabeled = y_true[mask_unlabeled]
    y_pseudo_unlabeled = y_pseudo[mask_unlabeled]
    
    # Filter out any remaining -1 labels (samples that were not pseudo-labeled)
    valid_mask = y_pseudo_unlabeled != -1
    y_true_valid = y_true_unlabeled[valid_mask]
    y_pseudo_valid = y_pseudo_unlabeled[valid_mask]
    
    # Convert to int for proper metrics calculation
    y_true_valid = y_true_valid.astype(int)
    y_pseudo_valid = y_pseudo_valid.astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_valid, y_pseudo_valid)
    precision = precision_score(y_true_valid, y_pseudo_valid, zero_division=0)
    recall = recall_score(y_true_valid, y_pseudo_valid, zero_division=0)
    f1 = f1_score(y_true_valid, y_pseudo_valid, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_valid, y_pseudo_valid)
    
    # Error analysis
    errors = y_true_valid != y_pseudo_valid
    n_errors = errors.sum()
    error_rate = n_errors / len(y_true_valid) if len(y_true_valid) > 0 else 0
    
    # False positives and false negatives
    fp_mask = (y_true_valid == 0) & (y_pseudo_valid == 1)
    fn_mask = (y_true_valid == 1) & (y_pseudo_valid == 0)
    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    
    results = {
        'n_unlabeled': len(y_true_unlabeled),
        'n_pseudo_labeled': valid_mask.sum(),
        'n_still_unlabeled': (~valid_mask).sum(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'n_errors': n_errors,
        'error_rate': error_rate,
        'n_false_positives': n_fp,
        'n_false_negatives': n_fn
    }
    
    # Feature-specific analysis
    if X is not None and feature_to_analyze is not None and feature_to_analyze in X.columns:
        X_unlabeled = X.iloc[mask_unlabeled] if hasattr(X, 'iloc') else X[mask_unlabeled]
        X_valid = X_unlabeled.iloc[valid_mask] if hasattr(X_unlabeled, 'iloc') else X_unlabeled[valid_mask]
        feature_vals = X_valid[feature_to_analyze].values
        
        # Error by feature bins
        if np.issubdtype(feature_vals.dtype, np.number) and len(feature_vals) > 0:
            # Numeric feature - create bins
            bins = np.percentile(feature_vals, [0, 25, 50, 75, 100])
            bin_labels = ['Q1', 'Q2', 'Q3', 'Q4']
            feature_bins = pd.cut(feature_vals, bins=bins, labels=bin_labels, include_lowest=True)
            
            error_by_bin = {}
            for bin_label in bin_labels:
                bin_mask = feature_bins == bin_label
                if bin_mask.sum() > 0:
                    bin_errors = errors[bin_mask]
                    error_by_bin[bin_label] = {
                        'n_samples': int(bin_mask.sum()),
                        'n_errors': int(bin_errors.sum()),
                        'error_rate': float(bin_errors.mean())
                    }
            
            results[f'error_by_{feature_to_analyze}'] = error_by_bin
    
    if verbose:
        print("\n" + "=" * 60)
        print("PSEUDO-LABEL ANALYSIS")
        print("=" * 60)
        print(f"\n📊 Pseudo-Label Quality:")
        print(f"   Originally unlabeled: {results['n_unlabeled']:,}")
        print(f"   Pseudo-labeled: {results['n_pseudo_labeled']:,}")
        print(f"   Still unlabeled: {results['n_still_unlabeled']:,}")
        print(f"\n   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        
        print(f"\n❌ Errors (on pseudo-labeled samples):")
        print(f"   Total errors: {n_errors:,} ({error_rate:.2%})")
        print(f"   False Positives: {n_fp:,} (predicted cancel, actually not)")
        print(f"   False Negatives: {n_fn:,} (predicted not cancel, actually cancel)")
        
        if len(cm) == 2:
            print(f"\n📋 Confusion Matrix:")
            print(f"   TN={cm[0,0]:,}  FP={cm[0,1]:,}")
            print(f"   FN={cm[1,0]:,}  TP={cm[1,1]:,}")
        
        if feature_to_analyze and f'error_by_{feature_to_analyze}' in results:
            print(f"\n📈 Error Rate by {feature_to_analyze}:")
            for bin_label, stats in results[f'error_by_{feature_to_analyze}'].items():
                print(f"   {bin_label}: {stats['error_rate']:.2%} "
                      f"({stats['n_errors']}/{stats['n_samples']} errors)")
        
        print("=" * 60)
    
    return results


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def evaluate_semi_supervised(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = 'Model',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate a semi-supervised model on test data.
    
    Parameters
    ----------
    model : estimator
        Trained semi-supervised model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    model_name : str
        Name for display
    verbose : bool
        Whether to print results
        
    Returns
    -------
    Dict[str, float]
        Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Get probabilities if available
    y_proba = None
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except:
            pass
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
            metrics['pr_auc'] = average_precision_score(y_test, y_proba)
        except:
            pass
    
    if verbose:
        print(f"\n📊 {model_name} Evaluation:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        if 'pr_auc' in metrics:
            print(f"   PR-AUC:    {metrics['pr_auc']:.4f}")
    
    return metrics


def compare_semi_supervised_methods(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labeled_fraction: float = 0.1,
    methods: List[str] = None,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare different semi-supervised methods at a given label fraction.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    labeled_fraction : float
        Fraction of training data to keep labeled
    methods : list of str, optional
        Methods to compare. Default: ['supervised', 'self_training', 'label_spreading']
    random_state : int
        Random seed
    verbose : bool
        Whether to print results
        
    Returns
    -------
    pd.DataFrame
        Comparison table with metrics for each method
    """
    if methods is None:
        methods = ['supervised', 'self_training', 'label_spreading']
    
    # Create labeled/unlabeled split
    X, y_semi, mask_labeled = create_labeled_unlabeled_split(
        X_train, y_train,
        labeled_fraction=labeled_fraction,
        random_state=random_state,
        verbose=verbose
    )
    
    results = {}
    
    # Get labeled-only data for supervised baseline
    X_labeled = X.iloc[mask_labeled] if hasattr(X, 'iloc') else X[mask_labeled]
    y_labeled = y_train.iloc[mask_labeled] if hasattr(y_train, 'iloc') else y_train[mask_labeled]
    
    for method in methods:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training: {method.upper()}")
            print('='*60)
        
        try:
            if method == 'supervised':
                # Supervised baseline using only labeled data
                model = LogisticRegression(
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=random_state
                )
                model.fit(X_labeled, y_labeled)
                
            elif method == 'self_training':
                model, _ = train_self_training(
                    X, y_semi,
                    threshold=0.9,
                    verbose=verbose
                )
                
            elif method == 'self_training_95':
                model, _ = train_self_training(
                    X, y_semi,
                    threshold=0.95,
                    verbose=verbose
                )
                
            elif method == 'label_propagation':
                model, _ = train_label_propagation(
                    X, y_semi,
                    kernel='knn',
                    n_neighbors=7,
                    verbose=verbose
                )
                
            elif method == 'label_spreading':
                model, _ = train_label_spreading(
                    X, y_semi,
                    kernel='knn',
                    n_neighbors=7,
                    alpha=0.2,
                    verbose=verbose
                )
            else:
                print(f"Unknown method: {method}")
                continue
            
            # Evaluate
            metrics = evaluate_semi_supervised(
                model, X_test, y_test,
                model_name=method,
                verbose=verbose
            )
            results[method] = metrics
            
        except Exception as e:
            print(f"❌ Error training {method}: {e}")
            results[method] = {'error': str(e)}
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    
    if verbose:
        print("\n" + "=" * 70)
        print(f"COMPARISON SUMMARY ({labeled_fraction:.0%} labeled)")
        print("=" * 70)
        print(comparison_df.round(4).to_string())
        print("=" * 70)
    
    return comparison_df


def run_label_fraction_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fractions: List[float] = [0.05, 0.10, 0.20],
    random_state: int = 42,
    verbose: bool = True
) -> Dict[float, pd.DataFrame]:
    """
    Run semi-supervised experiments across different label fractions.
    
    Parameters
    ----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    fractions : list of float
        Label fractions to test
    random_state : int
        Random seed
    verbose : bool
        Whether to print results
        
    Returns
    -------
    Dict[float, pd.DataFrame]
        Results for each label fraction
    """
    all_results = {}
    
    for frac in fractions:
        if verbose:
            print("\n" + "#" * 70)
            print(f"# EXPERIMENT: {frac:.0%} LABELED DATA")
            print("#" * 70)
        
        results = compare_semi_supervised_methods(
            X_train, y_train, X_test, y_test,
            labeled_fraction=frac,
            methods=['supervised', 'self_training', 'label_spreading'],
            random_state=random_state,
            verbose=verbose
        )
        
        all_results[frac] = results
    
    return all_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_learning_curve_by_labels(
    results: Dict[float, pd.DataFrame],
    metric: str = 'f1',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot learning curve showing performance vs. label fraction.
    
    Parameters
    ----------
    results : Dict[float, pd.DataFrame]
        Output from run_label_fraction_experiment
    metric : str
        Metric to plot ('f1', 'accuracy', 'roc_auc', etc.)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display plot
    """
    import matplotlib.pyplot as plt
    
    fractions = sorted(results.keys())
    methods = results[fractions[0]].index.tolist()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for method in methods:
        scores = []
        for frac in fractions:
            if metric in results[frac].columns and method in results[frac].index:
                scores.append(results[frac].loc[method, metric])
            else:
                scores.append(np.nan)
        
        ax.plot(fractions, scores, 'o-', label=method, markersize=8, linewidth=2)
    
    ax.set_xlabel('Labeled Fraction', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'Learning Curve: {metric.upper()} vs. Labeled Fraction', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis as percentage
    ax.set_xticks(fractions)
    ax.set_xticklabels([f'{f:.0%}' for f in fractions])
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved learning curve to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_pseudo_label_confusion_matrix(
    y_true: np.ndarray,
    y_pseudo: np.ndarray,
    mask_unlabeled: np.ndarray,
    title: str = 'Pseudo-Label Confusion Matrix',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot confusion matrix for pseudo-labels.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pseudo : np.ndarray
        Pseudo-labels
    mask_unlabeled : np.ndarray
        Mask for unlabeled samples
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get only unlabeled samples
    y_true_u = y_true[mask_unlabeled]
    y_pseudo_u = y_pseudo[mask_unlabeled]
    
    # Filter out samples that weren't pseudo-labeled (still -1)
    valid_mask = y_pseudo_u != -1
    y_true_valid = y_true_u[valid_mask].astype(int)
    y_pseudo_valid = y_pseudo_u[valid_mask].astype(int)
    
    cm = confusion_matrix(y_true_valid, y_pseudo_valid)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Not Canceled', 'Canceled'],
        yticklabels=['Not Canceled', 'Canceled'],
        ax=ax, cbar=True,
        annot_kws={'size': 14}
    )
    
    ax.set_xlabel('Pseudo Label', fontsize=12)
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


# Export all functions
__all__ = [
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
