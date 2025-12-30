"""
Data Cleaner Module
===================

Utilities for cleaning and preprocessing hotel booking data.

Functions:
----------
- drop_leakage_columns: Remove columns that cause data leakage
- handle_missing_values: Handle missing values with various strategies
- handle_outliers: Detect and handle outliers
- encode_categorical: Encode categorical variables
- scale_numerical: Scale numerical features
- clean_data: Full cleaning pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import yaml
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')


# ====================
# COLUMN DEFINITIONS
# ====================

# Columns that cause data leakage
LEAKAGE_COLUMNS = ['reservation_status', 'reservation_status_date']

# Columns with known missing values
MISSING_COLUMNS = {
    'company': {'pct': 94.31, 'strategy': 'drop_column'},  # Too many missing
    'agent': {'pct': 13.69, 'strategy': 'fill_zero'},       # 0 = no agent
    'country': {'pct': 0.41, 'strategy': 'fill_mode'},
    'children': {'pct': 0.00, 'strategy': 'fill_zero'}      # 4 rows only
}

# Target column
TARGET_COLUMN = 'is_canceled'

# Numerical columns for scaling
NUMERICAL_FOR_SCALING = [
    'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes',
    'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
    'total_of_special_requests'
]

# Categorical columns for encoding
CATEGORICAL_FOR_ENCODING = [
    'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
    'distribution_channel', 'reserved_room_type', 'assigned_room_type',
    'deposit_type', 'customer_type'
]

# ID-like columns to exclude from encoding
ID_COLUMNS = ['agent', 'company']


# ====================
# CONFIG LOADING
# ====================

def load_config(config_path: str = 'configs/params.yaml') -> dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Return default config if file not found
        return {
            'preprocessing': {
                'numerical_missing': 'median',
                'categorical_missing': 'mode',
                'outlier_method': 'iqr',
                'outlier_threshold': 1.5,
                'scaling_method': 'standard',
                'encoding_method': 'onehot'
            },
            'leakage_columns': LEAKAGE_COLUMNS
        }


# ====================
# DROP LEAKAGE COLUMNS
# ====================

def drop_leakage_columns(
    df: pd.DataFrame,
    leakage_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Remove columns that cause data leakage.
    
    These columns contain information about the outcome (cancellation)
    that wouldn't be available at prediction time.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    leakage_cols : list, optional
        List of columns to drop. If None, uses default LEAKAGE_COLUMNS
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame without leakage columns
        
    Examples
    --------
    >>> df_clean = drop_leakage_columns(df)
    >>> df_clean = drop_leakage_columns(df, leakage_cols=['reservation_status'])
    """
    df_clean = df.copy()
    
    if leakage_cols is None:
        leakage_cols = LEAKAGE_COLUMNS
    
    # Find columns that exist in dataframe
    cols_to_drop = [col for col in leakage_cols if col in df_clean.columns]
    
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        if verbose:
            print(f"✓ Dropped {len(cols_to_drop)} leakage column(s): {cols_to_drop}")
    else:
        if verbose:
            print("→ No leakage columns found to drop")
    
    return df_clean


# ====================
# MISSING VALUES
# ====================

def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'auto',
    numerical_strategy: str = 'median',
    categorical_strategy: str = 'mode',
    drop_columns: Optional[List[str]] = None,
    fill_zero_columns: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Overall strategy: 'auto' (uses predefined rules), 'fill', or 'drop'
    numerical_strategy : str
        Strategy for numerical columns: 'mean', 'median', 'zero'
    categorical_strategy : str
        Strategy for categorical columns: 'mode', 'unknown', 'drop'
    drop_columns : list, optional
        Columns to drop entirely (e.g., 'company' with 94% missing)
    fill_zero_columns : list, optional
        Columns to fill with 0 (e.g., 'agent' - no agent)
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
        
    Examples
    --------
    >>> df_clean = handle_missing_values(df, strategy='auto')
    >>> df_clean = handle_missing_values(df, numerical_strategy='mean')
    """
    df_clean = df.copy()
    
    # Get initial missing count
    initial_missing = df_clean.isnull().sum().sum()
    
    if verbose:
        print("=" * 60)
        print("HANDLING MISSING VALUES")
        print("=" * 60)
        print(f"Initial missing values: {initial_missing:,}")
    
    if strategy == 'auto':
        # Use predefined rules for this dataset
        
        # 1. Drop 'company' column (94% missing)
        if 'company' in df_clean.columns:
            df_clean = df_clean.drop(columns=['company'])
            if verbose:
                print("  → Dropped 'company' column (94% missing)")
        
        # 2. Fill 'agent' with 0 (no agent)
        if 'agent' in df_clean.columns:
            df_clean['agent'] = df_clean['agent'].fillna(0)
            if verbose:
                print("  → Filled 'agent' with 0 (no agent)")
        
        # 3. Fill 'country' with mode
        if 'country' in df_clean.columns and df_clean['country'].isnull().any():
            mode_country = df_clean['country'].mode()[0]
            df_clean['country'] = df_clean['country'].fillna(mode_country)
            if verbose:
                print(f"  → Filled 'country' with mode: '{mode_country}'")
        
        # 4. Fill 'children' with 0
        if 'children' in df_clean.columns and df_clean['children'].isnull().any():
            df_clean['children'] = df_clean['children'].fillna(0)
            if verbose:
                print("  → Filled 'children' with 0")
        
    else:
        # Custom handling
        
        # Drop specified columns
        if drop_columns:
            cols_exist = [c for c in drop_columns if c in df_clean.columns]
            if cols_exist:
                df_clean = df_clean.drop(columns=cols_exist)
                if verbose:
                    print(f"  → Dropped columns: {cols_exist}")
        
        # Fill specified columns with 0
        if fill_zero_columns:
            for col in fill_zero_columns:
                if col in df_clean.columns:
                    df_clean[col] = df_clean[col].fillna(0)
                    if verbose:
                        print(f"  → Filled '{col}' with 0")
        
        # Handle remaining numerical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df_clean[col].isnull().any():
                if numerical_strategy == 'mean':
                    fill_val = df_clean[col].mean()
                elif numerical_strategy == 'median':
                    fill_val = df_clean[col].median()
                elif numerical_strategy == 'zero':
                    fill_val = 0
                else:
                    fill_val = df_clean[col].median()
                
                df_clean[col] = df_clean[col].fillna(fill_val)
                if verbose:
                    print(f"  → Filled '{col}' with {numerical_strategy}: {fill_val:.2f}")
        
        # Handle remaining categorical columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                if categorical_strategy == 'mode':
                    fill_val = df_clean[col].mode()[0]
                elif categorical_strategy == 'unknown':
                    fill_val = 'Unknown'
                elif categorical_strategy == 'drop':
                    df_clean = df_clean.dropna(subset=[col])
                    if verbose:
                        print(f"  → Dropped rows with missing '{col}'")
                    continue
                else:
                    fill_val = df_clean[col].mode()[0]
                
                df_clean[col] = df_clean[col].fillna(fill_val)
                if verbose:
                    print(f"  → Filled '{col}' with {categorical_strategy}: '{fill_val}'")
    
    # Final check
    final_missing = df_clean.isnull().sum().sum()
    
    if verbose:
        print("-" * 60)
        print(f"Final missing values: {final_missing:,}")
        print(f"Rows: {len(df)} → {len(df_clean)}")
        print("=" * 60)
    
    return df_clean


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of missing values in dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Summary of missing values
    """
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    
    summary = pd.DataFrame({
        'Column': missing_count.index,
        'Missing_Count': missing_count.values,
        'Missing_Pct': missing_pct.values,
        'Dtype': df.dtypes.values
    })
    
    summary = summary[summary['Missing_Count'] > 0]
    summary = summary.sort_values('Missing_Count', ascending=False)
    
    return summary


# ====================
# OUTLIER HANDLING
# ====================

def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 1.5
) -> Dict[str, Tuple[float, float, int]]:
    """
    Detect outliers using IQR method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to check. If None, checks all numerical columns.
    threshold : float
        IQR multiplier (typically 1.5 for outliers, 3 for extreme outliers)
        
    Returns
    -------
    dict
        Dictionary with column: (lower_bound, upper_bound, outlier_count)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = {}
    
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            
            outliers = df[(df[col] < lower) | (df[col] > upper)]
            outlier_info[col] = (lower, upper, len(outliers))
    
    return outlier_info


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 3.0
) -> Dict[str, int]:
    """
    Detect outliers using Z-score method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to check
    threshold : float
        Z-score threshold (typically 3)
        
    Returns
    -------
    dict
        Dictionary with column: outlier_count
    """
    from scipy import stats
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_info = {}
    
    for col in columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_count = (z_scores > threshold).sum()
            outlier_info[col] = outlier_count
    
    return outlier_info


def handle_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'iqr',
    threshold: float = 1.5,
    strategy: str = 'cap',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Handle outliers in numerical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to process. If None, processes specific columns.
    method : str
        Detection method: 'iqr' or 'zscore'
    threshold : float
        Threshold for detection (1.5 for IQR, 3 for zscore)
    strategy : str
        How to handle outliers: 'cap' (winsorize), 'remove', or 'keep'
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with outliers handled
        
    Examples
    --------
    >>> df_clean = handle_outliers(df, method='iqr', strategy='cap')
    >>> df_clean = handle_outliers(df, columns=['adr', 'lead_time'])
    """
    df_clean = df.copy()
    
    # Default columns to check for outliers
    if columns is None:
        columns = ['lead_time', 'adr', 'stays_in_weekend_nights', 
                   'stays_in_week_nights', 'adults', 'children', 'babies',
                   'days_in_waiting_list']
    
    if verbose:
        print("=" * 60)
        print("HANDLING OUTLIERS")
        print(f"Method: {method.upper()}, Strategy: {strategy.upper()}")
        print("=" * 60)
    
    outliers_handled = 0
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
        else:  # zscore
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower = mean - threshold * std
            upper = mean + threshold * std
        
        # Count outliers
        outlier_mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        n_outliers = outlier_mask.sum()
        
        if n_outliers == 0:
            continue
        
        if strategy == 'cap':
            # Winsorize: cap values at bounds
            df_clean.loc[df_clean[col] < lower, col] = lower
            df_clean.loc[df_clean[col] > upper, col] = upper
            if verbose:
                print(f"  → {col}: Capped {n_outliers} outliers [{lower:.2f}, {upper:.2f}]")
        
        elif strategy == 'remove':
            # Remove rows with outliers
            df_clean = df_clean[~outlier_mask]
            if verbose:
                print(f"  → {col}: Removed {n_outliers} rows with outliers")
        
        elif strategy == 'keep':
            if verbose:
                print(f"  → {col}: Found {n_outliers} outliers (kept)")
        
        outliers_handled += n_outliers
    
    if verbose:
        print("-" * 60)
        print(f"Total outliers handled: {outliers_handled:,}")
        print(f"Rows: {len(df)} → {len(df_clean)}")
        print("=" * 60)
    
    return df_clean


def handle_adr_outliers(
    df: pd.DataFrame,
    min_adr: float = 0,
    max_adr: float = 5000,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Special handling for ADR (Average Daily Rate) outliers.
    
    ADR can have:
    - Negative values (invalid)
    - Extremely high values (may be valid but extreme)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    min_adr : float
        Minimum valid ADR (default: 0)
    max_adr : float
        Maximum valid ADR (default: 5000)
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with ADR outliers handled
    """
    df_clean = df.copy()
    
    if 'adr' not in df_clean.columns:
        return df_clean
    
    # Count problematic values
    negative_count = (df_clean['adr'] < min_adr).sum()
    extreme_count = (df_clean['adr'] > max_adr).sum()
    
    if verbose:
        print(f"ADR Outliers: {negative_count} negative, {extreme_count} extreme (>{max_adr})")
    
    # Handle negative ADR: Replace with median or 0
    if negative_count > 0:
        median_adr = df_clean.loc[df_clean['adr'] >= 0, 'adr'].median()
        df_clean.loc[df_clean['adr'] < min_adr, 'adr'] = median_adr
        if verbose:
            print(f"  → Replaced {negative_count} negative ADR with median: {median_adr:.2f}")
    
    # Handle extreme ADR: Cap at max_adr
    if extreme_count > 0:
        df_clean.loc[df_clean['adr'] > max_adr, 'adr'] = max_adr
        if verbose:
            print(f"  → Capped {extreme_count} extreme ADR at {max_adr}")
    
    return df_clean


# ====================
# ENCODING
# ====================

def encode_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'onehot',
    drop_first: bool = True,
    handle_unknown: str = 'ignore',
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to encode. If None, uses default CATEGORICAL_FOR_ENCODING
    method : str
        Encoding method: 'onehot', 'label', or 'target'
    drop_first : bool
        For one-hot encoding, whether to drop first category (avoid multicollinearity)
    handle_unknown : str
        How to handle unknown categories in transform: 'ignore' or 'error'
    verbose : bool
        Whether to print information
        
    Returns
    -------
    tuple
        (encoded_df, encoders_dict)
        
    Examples
    --------
    >>> df_encoded, encoders = encode_categorical(df, method='onehot')
    >>> df_encoded, encoders = encode_categorical(df, columns=['hotel', 'meal'])
    """
    df_clean = df.copy()
    encoders = {}
    
    if columns is None:
        # Use only columns that exist in dataframe
        columns = [col for col in CATEGORICAL_FOR_ENCODING if col in df_clean.columns]
    
    if verbose:
        print("=" * 60)
        print(f"ENCODING CATEGORICAL VARIABLES (Method: {method.upper()})")
        print("=" * 60)
    
    if method == 'onehot':
        # One-Hot Encoding using pandas get_dummies
        df_encoded = pd.get_dummies(
            df_clean, 
            columns=columns, 
            drop_first=drop_first,
            dtype=int
        )
        
        if verbose:
            new_cols = len(df_encoded.columns) - len(df_clean.columns) + len(columns)
            print(f"  → Created {new_cols} new columns from {len(columns)} categorical columns")
            print(f"  → Columns: {len(df_clean.columns)} → {len(df_encoded.columns)}")
        
        # Store column mapping for later use
        encoders['method'] = 'onehot'
        encoders['original_columns'] = columns
        encoders['new_columns'] = [c for c in df_encoded.columns if c not in df_clean.columns or c in columns]
        
        return df_encoded, encoders
    
    elif method == 'label':
        # Label Encoding
        for col in columns:
            if col in df_clean.columns:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                encoders[col] = le
                
                if verbose:
                    print(f"  → {col}: {len(le.classes_)} unique values encoded")
        
        encoders['method'] = 'label'
        return df_clean, encoders
    
    elif method == 'target':
        # Target encoding - requires target column
        if TARGET_COLUMN not in df_clean.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' required for target encoding")
        
        for col in columns:
            if col in df_clean.columns:
                # Calculate mean target for each category
                target_means = df_clean.groupby(col)[TARGET_COLUMN].mean()
                df_clean[col + '_target_enc'] = df_clean[col].map(target_means)
                encoders[col] = target_means.to_dict()
                
                if verbose:
                    print(f"  → {col}: Target encoded (mean cancellation rate per category)")
        
        encoders['method'] = 'target'
        return df_clean, encoders
    
    else:
        raise ValueError(f"Unknown encoding method: {method}")


def encode_categorical_with_sklearn(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    encoder: Optional[OneHotEncoder] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, OneHotEncoder]:
    """
    Encode categorical variables using sklearn OneHotEncoder.
    
    Useful for consistent encoding between train and test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to encode
    encoder : OneHotEncoder, optional
        Pre-fitted encoder for transform
    fit : bool
        Whether to fit the encoder
        
    Returns
    -------
    tuple
        (encoded_df, encoder)
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = [col for col in CATEGORICAL_FOR_ENCODING if col in df_clean.columns]
    
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    
    if fit:
        encoded_array = encoder.fit_transform(df_clean[columns])
    else:
        encoded_array = encoder.transform(df_clean[columns])
    
    # Get feature names
    feature_names = encoder.get_feature_names_out(columns)
    
    # Create encoded dataframe
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=feature_names,
        index=df_clean.index
    )
    
    # Combine with non-encoded columns
    df_result = pd.concat([
        df_clean.drop(columns=columns),
        encoded_df
    ], axis=1)
    
    return df_result, encoder


# ====================
# SCALING
# ====================

def scale_numerical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'standard',
    scaler: Optional[object] = None,
    fit: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, object]:
    """
    Scale numerical features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to scale. If None, uses default NUMERICAL_FOR_SCALING
    method : str
        Scaling method: 'standard', 'minmax', or 'robust'
    scaler : object, optional
        Pre-fitted scaler for transform
    fit : bool
        Whether to fit the scaler
    verbose : bool
        Whether to print information
        
    Returns
    -------
    tuple
        (scaled_df, scaler)
        
    Examples
    --------
    >>> df_scaled, scaler = scale_numerical(df, method='standard')
    >>> df_test_scaled, _ = scale_numerical(df_test, scaler=scaler, fit=False)
    """
    df_clean = df.copy()
    
    if columns is None:
        # Use only columns that exist in dataframe
        columns = [col for col in NUMERICAL_FOR_SCALING if col in df_clean.columns]
    
    if verbose:
        print("=" * 60)
        print(f"SCALING NUMERICAL FEATURES (Method: {method.upper()})")
        print("=" * 60)
    
    # Initialize scaler if not provided
    if scaler is None:
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    # Fit and/or transform
    if fit:
        scaled_values = scaler.fit_transform(df_clean[columns])
        if verbose:
            print(f"  → Fitted and transformed {len(columns)} columns")
    else:
        scaled_values = scaler.transform(df_clean[columns])
        if verbose:
            print(f"  → Transformed {len(columns)} columns (using pre-fitted scaler)")
    
    # Replace values in dataframe
    df_clean[columns] = scaled_values
    
    if verbose:
        print(f"  → Columns scaled: {columns}")
        print("=" * 60)
    
    return df_clean, scaler


# ====================
# FULL PIPELINE
# ====================

def clean_data(
    df: pd.DataFrame,
    drop_leakage: bool = True,
    handle_missing: bool = True,
    handle_outliers_flag: bool = True,
    encode: bool = False,
    scale: bool = False,
    encoding_method: str = 'onehot',
    scaling_method: str = 'standard',
    config_path: str = 'configs/params.yaml',
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Full data cleaning pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    drop_leakage : bool
        Whether to drop leakage columns
    handle_missing : bool
        Whether to handle missing values
    handle_outliers_flag : bool
        Whether to handle outliers
    encode : bool
        Whether to encode categorical variables
    scale : bool
        Whether to scale numerical features
    encoding_method : str
        Encoding method if encode=True
    scaling_method : str
        Scaling method if scale=True
    config_path : str
        Path to configuration file
    verbose : bool
        Whether to print information
        
    Returns
    -------
    tuple
        (cleaned_df, artifacts_dict)
        
    Examples
    --------
    >>> df_clean, artifacts = clean_data(df, encode=True, scale=True)
    >>> # Later, for test data:
    >>> df_test_clean, _ = clean_data(df_test, 
    ...     scaler=artifacts['scaler'], 
    ...     encoder=artifacts['encoder'])
    """
    if verbose:
        print("\n" + "=" * 70)
        print("🧹 DATA CLEANING PIPELINE")
        print("=" * 70 + "\n")
    
    df_clean = df.copy()
    artifacts = {}
    
    # Load config
    config = load_config(config_path)
    
    # Step 1: Drop leakage columns
    if drop_leakage:
        leakage_cols = config.get('leakage_columns', LEAKAGE_COLUMNS)
        df_clean = drop_leakage_columns(df_clean, leakage_cols, verbose)
        print()
    
    # Step 2: Handle missing values
    if handle_missing:
        df_clean = handle_missing_values(df_clean, strategy='auto', verbose=verbose)
        print()
    
    # Step 3: Handle outliers
    if handle_outliers_flag:
        # Special handling for ADR
        df_clean = handle_adr_outliers(df_clean, verbose=verbose)
        
        # General outlier handling
        outlier_config = config.get('preprocessing', {})
        method = outlier_config.get('outlier_method', 'iqr')
        threshold = outlier_config.get('outlier_threshold', 1.5)
        
        df_clean = handle_outliers(
            df_clean, 
            method=method, 
            threshold=threshold,
            strategy='cap',
            verbose=verbose
        )
        print()
    
    # Step 4: Encode categorical
    if encode:
        df_clean, encoders = encode_categorical(
            df_clean, 
            method=encoding_method,
            verbose=verbose
        )
        artifacts['encoders'] = encoders
        print()
    
    # Step 5: Scale numerical
    if scale:
        df_clean, scaler = scale_numerical(
            df_clean,
            method=scaling_method,
            verbose=verbose
        )
        artifacts['scaler'] = scaler
        print()
    
    if verbose:
        print("=" * 70)
        print("✅ DATA CLEANING COMPLETE")
        print(f"   Original shape: {df.shape}")
        print(f"   Cleaned shape: {df_clean.shape}")
        print("=" * 70 + "\n")
    
    return df_clean, artifacts


def save_artifacts(
    artifacts: Dict,
    output_dir: str = 'outputs/models/',
    prefix: str = 'cleaner'
) -> None:
    """
    Save cleaning artifacts (scaler, encoders) for later use.
    
    Parameters
    ----------
    artifacts : dict
        Dictionary containing scaler and encoders
    output_dir : str
        Output directory
    prefix : str
        Prefix for file names
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if 'scaler' in artifacts:
        joblib.dump(artifacts['scaler'], output_path / f'{prefix}_scaler.joblib')
        print(f"✓ Saved scaler to {output_path / f'{prefix}_scaler.joblib'}")
    
    if 'encoders' in artifacts:
        joblib.dump(artifacts['encoders'], output_path / f'{prefix}_encoders.joblib')
        print(f"✓ Saved encoders to {output_path / f'{prefix}_encoders.joblib'}")


def load_artifacts(
    output_dir: str = 'outputs/models/',
    prefix: str = 'cleaner'
) -> Dict:
    """
    Load cleaning artifacts.
    
    Parameters
    ----------
    output_dir : str
        Output directory
    prefix : str
        Prefix for file names
        
    Returns
    -------
    dict
        Dictionary containing scaler and encoders
    """
    output_path = Path(output_dir)
    artifacts = {}
    
    scaler_path = output_path / f'{prefix}_scaler.joblib'
    if scaler_path.exists():
        artifacts['scaler'] = joblib.load(scaler_path)
        print(f"✓ Loaded scaler from {scaler_path}")
    
    encoders_path = output_path / f'{prefix}_encoders.joblib'
    if encoders_path.exists():
        artifacts['encoders'] = joblib.load(encoders_path)
        print(f"✓ Loaded encoders from {encoders_path}")
    
    return artifacts


# ====================
# MAIN (Testing)
# ====================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.data.loader import load_raw_data
    
    print("Testing cleaner module...")
    print("=" * 70)
    
    # Load data
    df = load_raw_data()
    print(f"\nOriginal data shape: {df.shape}")
    
    # Test cleaning pipeline
    df_clean, artifacts = clean_data(
        df,
        drop_leakage=True,
        handle_missing=True,
        handle_outliers_flag=True,
        encode=False,  # Don't encode for this test
        scale=False,   # Don't scale for this test
        verbose=True
    )
    
    print(f"\nCleaned data shape: {df_clean.shape}")
    print(f"\nRemaining missing values: {df_clean.isnull().sum().sum()}")
    print(f"\nColumns removed: {set(df.columns) - set(df_clean.columns)}")
    
    print("\n✅ Cleaner module test passed!")
