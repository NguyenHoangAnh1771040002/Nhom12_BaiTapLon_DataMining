"""
Mô Hình Học Có Giám Sát - Dự Đoán Huỷ Đặt Phòng
=================================================
(Supervised Learning Models for Hotel Booking Cancellation Prediction)

Module cung cấp các mô hình phân loại dự đoán huỷ đặt phòng khách sạn.

Các mô hình:
- Logistic Regression (Hồi quy Logistic - baseline)
- Decision Tree (Cây quyết định - baseline)
- Random Forest (Rừng ngẫu nhiên)
- XGBoost (Extreme Gradient Boosting)
- LightGBM (Light Gradient Boosting Machine)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import joblib
from pathlib import Path

# Scikit-learn imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler

# Try importing XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not installed. Run: pip install lightgbm")


# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: Optional[Union[str, Dict]] = 'balanced',
    solver: str = 'lbfgs',
    random_state: int = 42,
    verbose: bool = True
) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    C : float
        Inverse of regularization strength (smaller = stronger regularization)
    max_iter : int
        Maximum number of iterations
    class_weight : str or dict, optional
        Class weights. 'balanced' auto-adjusts for imbalanced data
    solver : str
        Optimization algorithm
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Whether to print information
        
    Returns
    -------
    LogisticRegression
        Trained model
        
    Examples
    --------
    >>> model = train_logistic_regression(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    """
    if verbose:
        print("=" * 60)
        print("TRAINING LOGISTIC REGRESSION")
        print("=" * 60)
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"C (regularization): {C}")
        print(f"Class weight: {class_weight}")
    
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        solver=solver,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    if verbose:
        train_score = model.score(X_train, y_train)
        print(f"Training accuracy: {train_score:.4f}")
        print("=" * 60)
    
    return model


# =============================================================================
# DECISION TREE
# =============================================================================

def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: Optional[int] = 10,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    class_weight: Optional[Union[str, Dict]] = 'balanced',
    random_state: int = 42,
    verbose: bool = True
) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    max_depth : int, optional
        Maximum depth of tree (None = unlimited)
    min_samples_split : int
        Minimum samples required to split a node
    min_samples_leaf : int
        Minimum samples required in a leaf node
    class_weight : str or dict, optional
        Class weights
    random_state : int
        Random seed
    verbose : bool
        Whether to print information
        
    Returns
    -------
    DecisionTreeClassifier
        Trained model
    """
    if verbose:
        print("=" * 60)
        print("TRAINING DECISION TREE")
        print("=" * 60)
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Max depth: {max_depth}")
        print(f"Min samples split: {min_samples_split}")
        print(f"Class weight: {class_weight}")
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    if verbose:
        train_score = model.score(X_train, y_train)
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Tree depth: {model.get_depth()}")
        print(f"Number of leaves: {model.get_n_leaves()}")
        print("=" * 60)
    
    return model


# =============================================================================
# RANDOM FOREST
# =============================================================================

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = 15,
    min_samples_split: int = 10,
    min_samples_leaf: int = 5,
    max_features: Union[str, int, float] = 'sqrt',
    class_weight: Optional[Union[str, Dict]] = 'balanced',
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_estimators : int
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of each tree
    min_samples_split : int
        Minimum samples to split a node
    min_samples_leaf : int
        Minimum samples in a leaf
    max_features : str, int, or float
        Number of features to consider for splits
    class_weight : str or dict, optional
        Class weights
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs)
    verbose : bool
        Whether to print information
        
    Returns
    -------
    RandomForestClassifier
        Trained model
    """
    if verbose:
        print("=" * 60)
        print("TRAINING RANDOM FOREST")
        print("=" * 60)
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Number of trees: {n_estimators}")
        print(f"Max depth: {max_depth}")
        print(f"Class weight: {class_weight}")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    model.fit(X_train, y_train)
    
    if verbose:
        train_score = model.score(X_train, y_train)
        print(f"Training accuracy: {train_score:.4f}")
        print("=" * 60)
    
    return model


# =============================================================================
# XGBOOST
# =============================================================================

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True
) -> Any:
    """
    Train an XGBoost classifier.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_estimators : int
        Number of boosting rounds
    max_depth : int
        Maximum tree depth
    learning_rate : float
        Step size shrinkage
    subsample : float
        Fraction of samples per tree
    colsample_bytree : float
        Fraction of features per tree
    scale_pos_weight : float, optional
        Ratio of negative/positive for imbalanced data
        If None, will be calculated automatically
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel threads
    verbose : bool
        Whether to print information
        
    Returns
    -------
    XGBClassifier
        Trained model
        
    Raises
    ------
    ImportError
        If XGBoost is not installed
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    if verbose:
        print("=" * 60)
        print("TRAINING XGBOOST")
        print("=" * 60)
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Number of estimators: {n_estimators}")
        print(f"Max depth: {max_depth}")
        print(f"Learning rate: {learning_rate}")
    
    # Calculate scale_pos_weight for imbalanced data
    if scale_pos_weight is None:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        if verbose:
            print(f"Scale pos weight (auto): {scale_pos_weight:.2f}")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    if verbose:
        train_score = model.score(X_train, y_train)
        print(f"Training accuracy: {train_score:.4f}")
        print("=" * 60)
    
    return model


# =============================================================================
# LIGHTGBM
# =============================================================================

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = -1,
    num_leaves: int = 31,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    class_weight: Optional[Union[str, Dict]] = 'balanced',
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True
) -> Any:
    """
    Train a LightGBM classifier.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_estimators : int
        Number of boosting iterations
    max_depth : int
        Maximum tree depth (-1 = unlimited)
    num_leaves : int
        Maximum number of leaves per tree
    learning_rate : float
        Boosting learning rate
    subsample : float
        Fraction of data for each tree
    colsample_bytree : float
        Fraction of features for each tree
    class_weight : str or dict, optional
        Class weights
    random_state : int
        Random seed
    n_jobs : int
        Number of threads
    verbose : bool
        Whether to print information
        
    Returns
    -------
    LGBMClassifier
        Trained model
        
    Raises
    ------
    ImportError
        If LightGBM is not installed
    """
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not installed. Run: pip install lightgbm")
    
    if verbose:
        print("=" * 60)
        print("TRAINING LIGHTGBM")
        print("=" * 60)
        print(f"Training samples: {len(X_train):,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Number of estimators: {n_estimators}")
        print(f"Max depth: {max_depth}")
        print(f"Num leaves: {num_leaves}")
        print(f"Learning rate: {learning_rate}")
    
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1  # Suppress LightGBM warnings
    )
    
    model.fit(X_train, y_train)
    
    if verbose:
        train_score = model.score(X_train, y_train)
        print(f"Training accuracy: {train_score:.4f}")
        print("=" * 60)
    
    return model


# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

def tune_hyperparameters(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict] = None,
    search_method: str = 'grid',
    cv: int = 5,
    scoring: str = 'f1',
    n_iter: int = 20,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[Any, Dict, float]:
    """
    Perform hyperparameter tuning using GridSearch or RandomizedSearch.
    
    Parameters
    ----------
    model_type : str
        Type of model: 'logistic', 'tree', 'rf', 'xgboost', 'lightgbm'
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    param_grid : dict, optional
        Parameter grid for search. If None, uses default grid.
    search_method : str
        'grid' for GridSearchCV, 'random' for RandomizedSearchCV
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric ('f1', 'roc_auc', 'precision', 'recall', 'accuracy')
    n_iter : int
        Number of iterations for RandomizedSearch
    random_state : int
        Random seed
    n_jobs : int
        Number of parallel jobs
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Tuple[model, best_params, best_score]
        Best model, best parameters, and best CV score
        
    Examples
    --------
    >>> best_model, best_params, best_score = tune_hyperparameters(
    ...     'rf', X_train, y_train, scoring='f1'
    ... )
    """
    if verbose:
        print("=" * 60)
        print(f"HYPERPARAMETER TUNING: {model_type.upper()}")
        print("=" * 60)
        print(f"Search method: {search_method}")
        print(f"CV folds: {cv}")
        print(f"Scoring: {scoring}")
    
    # Default parameter grids
    default_grids = {
        'logistic': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [500, 1000]
        },
        'tree': {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 5, 10]
        },
        'rf': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'max_features': ['sqrt', 'log2', 0.5]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        },
        'lightgbm': {
            'n_estimators': [50, 100, 200],
            'max_depth': [-1, 5, 10, 15],
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9]
        }
    }
    
    # Get base model
    base_models = {
        'logistic': LogisticRegression(class_weight='balanced', random_state=random_state),
        'tree': DecisionTreeClassifier(class_weight='balanced', random_state=random_state),
        'rf': RandomForestClassifier(class_weight='balanced', random_state=random_state, n_jobs=-1)
    }
    
    if model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        base_models['xgboost'] = xgb.XGBClassifier(
            scale_pos_weight=neg/pos if pos > 0 else 1,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    if model_type == 'lightgbm':
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        base_models['lightgbm'] = lgb.LGBMClassifier(
            class_weight='balanced',
            random_state=random_state,
            verbose=-1
        )
    
    if model_type not in base_models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: {list(base_models.keys())}")
    
    base_model = base_models[model_type]
    grid = param_grid if param_grid is not None else default_grids.get(model_type, {})
    
    if verbose:
        print(f"Parameter grid: {grid}")
    
    # Create search object
    if search_method == 'grid':
        search = GridSearchCV(
            base_model,
            grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1 if verbose else 0
        )
    else:
        search = RandomizedSearchCV(
            base_model,
            grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=1 if verbose else 0
        )
    
    # Fit
    search.fit(X_train, y_train)
    
    if verbose:
        print(f"\nBest parameters: {search.best_params_}")
        print(f"Best CV {scoring}: {search.best_score_:.4f}")
        print("=" * 60)
    
    return search.best_estimator_, search.best_params_, search.best_score_


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: List[str] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Perform cross-validation on a model.
    
    Parameters
    ----------
    model : estimator
        Trained or untrained sklearn-compatible model
    X : pd.DataFrame
        Features
    y : pd.Series
        Labels
    cv : int
        Number of folds
    scoring : list of str, optional
        Metrics to evaluate. Default: ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    random_state : int
        Random seed for fold splitting
    verbose : bool
        Whether to print results
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of metric name -> array of scores per fold
    """
    if scoring is None:
        scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    
    if verbose:
        print("=" * 60)
        print("CROSS-VALIDATION")
        print("=" * 60)
        print(f"Model: {type(model).__name__}")
        print(f"Folds: {cv}")
        print(f"Metrics: {scoring}")
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    results = {}
    for metric in scoring:
        scores = cross_val_score(model, X, y, cv=skf, scoring=metric, n_jobs=-1)
        results[metric] = scores
        
        if verbose:
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
            print(f"  Per fold: {[f'{s:.4f}' for s in scores]}")
    
    if verbose:
        print("=" * 60)
    
    return results


# =============================================================================
# PREDICTION
# =============================================================================

def predict(
    model: Any,
    X: pd.DataFrame,
    verbose: bool = True
) -> np.ndarray:
    """
    Make predictions using a trained model.
    
    Parameters
    ----------
    model : estimator
        Trained model
    X : pd.DataFrame
        Features to predict
    verbose : bool
        Whether to print information
        
    Returns
    -------
    np.ndarray
        Predicted labels (0 or 1)
    """
    predictions = model.predict(X)
    
    if verbose:
        n_positive = (predictions == 1).sum()
        n_negative = (predictions == 0).sum()
        print(f"Predictions: {len(predictions):,} total")
        print(f"  Predicted cancellations: {n_positive:,} ({n_positive/len(predictions)*100:.1f}%)")
        print(f"  Predicted non-cancellations: {n_negative:,} ({n_negative/len(predictions)*100:.1f}%)")
    
    return predictions


def predict_proba(
    model: Any,
    X: pd.DataFrame,
    verbose: bool = True
) -> np.ndarray:
    """
    Get prediction probabilities from a trained model.
    
    Parameters
    ----------
    model : estimator
        Trained model with predict_proba method
    X : pd.DataFrame
        Features to predict
    verbose : bool
        Whether to print information
        
    Returns
    -------
    np.ndarray
        Probability of positive class (cancellation)
    """
    if not hasattr(model, 'predict_proba'):
        raise ValueError(f"Model {type(model).__name__} does not support predict_proba")
    
    proba = model.predict_proba(X)[:, 1]  # Probability of class 1
    
    if verbose:
        print(f"Prediction probabilities:")
        print(f"  Min: {proba.min():.4f}")
        print(f"  Max: {proba.max():.4f}")
        print(f"  Mean: {proba.mean():.4f}")
        print(f"  Std: {proba.std():.4f}")
    
    return proba


# =============================================================================
# MODEL I/O
# =============================================================================

def save_model(
    model: Any,
    filepath: Union[str, Path],
    verbose: bool = True
) -> None:
    """
    Save a trained model to disk.
    
    Parameters
    ----------
    model : estimator
        Trained model
    filepath : str or Path
        Path to save the model
    verbose : bool
        Whether to print information
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    
    if verbose:
        print(f"✅ Model saved to {filepath}")


def load_model(
    filepath: Union[str, Path],
    verbose: bool = True
) -> Any:
    """
    Load a trained model from disk.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the saved model
    verbose : bool
        Whether to print information
        
    Returns
    -------
    estimator
        Loaded model
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    
    if verbose:
        print(f"✅ Model loaded from {filepath}")
        print(f"   Model type: {type(model).__name__}")
    
    return model


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(
    model: Any,
    feature_names: List[str],
    importance_type: str = 'default',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Parameters
    ----------
    model : estimator
        Trained model (must have feature_importances_ or coef_ attribute)
    feature_names : list of str
        Names of features
    importance_type : str
        Type of importance: 'default', 'gain', 'weight', 'cover' (for XGBoost/LightGBM)
    verbose : bool
        Whether to print top features
        
    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and importance scores, sorted descending
    """
    # Get importance values
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])  # For logistic regression
    else:
        raise ValueError(f"Model {type(model).__name__} does not have feature importance")
    
    # Create DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Normalize to percentages
    df_importance['importance_pct'] = df_importance['importance'] / df_importance['importance'].sum() * 100
    
    if verbose:
        print("\n📊 TOP 10 IMPORTANT FEATURES:")
        print("-" * 50)
        for i, row in df_importance.head(10).iterrows():
            print(f"{i+1:2d}. {row['feature']:<30} {row['importance_pct']:6.2f}%")
    
    return df_importance


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models_to_train: List[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train multiple models with default parameters.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    models_to_train : list of str, optional
        Which models to train. Default: ['logistic', 'tree', 'rf']
        Options: 'logistic', 'tree', 'rf', 'xgboost', 'lightgbm'
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Dict[str, model]
        Dictionary of model name -> trained model
    """
    if models_to_train is None:
        models_to_train = ['logistic', 'tree', 'rf']
    
    trained_models = {}
    
    for model_name in models_to_train:
        try:
            if model_name == 'logistic':
                trained_models['Logistic Regression'] = train_logistic_regression(
                    X_train, y_train, verbose=verbose
                )
            elif model_name == 'tree':
                trained_models['Decision Tree'] = train_decision_tree(
                    X_train, y_train, verbose=verbose
                )
            elif model_name == 'rf':
                trained_models['Random Forest'] = train_random_forest(
                    X_train, y_train, verbose=verbose
                )
            elif model_name == 'xgboost':
                trained_models['XGBoost'] = train_xgboost(
                    X_train, y_train, verbose=verbose
                )
            elif model_name == 'lightgbm':
                trained_models['LightGBM'] = train_lightgbm(
                    X_train, y_train, verbose=verbose
                )
        except ImportError as e:
            if verbose:
                print(f"⚠️ Skipping {model_name}: {e}")
    
    if verbose:
        print(f"\n✅ Trained {len(trained_models)} models: {list(trained_models.keys())}")
    
    return trained_models


# Export all functions
__all__ = [
    # Training functions
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
    
    # Availability flags
    'XGBOOST_AVAILABLE',
    'LIGHTGBM_AVAILABLE'
]
