"""
Data Loader Module
==================

Utilities for loading and validating hotel booking data.

Functions:
----------
- load_raw_data: Load raw CSV data from file
- load_processed_data: Load processed data
- validate_schema: Validate dataframe schema
- get_data_info: Get summary information about data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import yaml
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


# ====================
# SCHEMA DEFINITION
# ====================

EXPECTED_COLUMNS = [
    'hotel', 'is_canceled', 'lead_time', 'arrival_date_year', 
    'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
    'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 
    'babies', 'meal', 'country', 'market_segment', 'distribution_channel',
    'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
    'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type',
    'agent', 'company', 'days_in_waiting_list', 'customer_type', 'adr',
    'required_car_parking_spaces', 'total_of_special_requests',
    'reservation_status', 'reservation_status_date'
]

NUMERICAL_COLUMNS = [
    'lead_time', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'booking_changes', 'agent', 'company',
    'days_in_waiting_list', 'adr', 'required_car_parking_spaces',
    'total_of_special_requests'
]

CATEGORICAL_COLUMNS = [
    'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
    'distribution_channel', 'reserved_room_type', 'assigned_room_type',
    'deposit_type', 'customer_type', 'reservation_status'
]

TARGET_COLUMN = 'is_canceled'

# Columns that cause data leakage (contain information after booking)
LEAKAGE_COLUMNS = ['reservation_status', 'reservation_status_date']


# ====================
# LOADING FUNCTIONS
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
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_raw_data(
    file_path: Optional[str] = None,
    config_path: str = 'configs/params.yaml'
) -> pd.DataFrame:
    """
    Load raw hotel booking data from CSV file.
    
    Parameters
    ----------
    file_path : str, optional
        Path to the CSV file. If None, uses path from config.
    config_path : str
        Path to configuration file
        
    Returns
    -------
    pd.DataFrame
        Raw hotel booking data
        
    Raises
    ------
    FileNotFoundError
        If the data file doesn't exist
    ValueError
        If the data doesn't have expected columns
    """
    # Get file path from config if not provided
    if file_path is None:
        config = load_config(config_path)
        file_path = config['paths']['data_raw']
    
    # Check if file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load data
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    # Validate schema
    is_valid, missing_cols = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Missing expected columns: {missing_cols}")
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def load_processed_data(
    file_path: Optional[str] = None,
    config_path: str = 'configs/params.yaml'
) -> pd.DataFrame:
    """
    Load processed data from parquet or CSV.
    
    Parameters
    ----------
    file_path : str, optional
        Path to the processed data file
    config_path : str
        Path to configuration file
        
    Returns
    -------
    pd.DataFrame
        Processed data
    """
    if file_path is None:
        config = load_config(config_path)
        processed_dir = config['paths']['data_processed']
        # Try parquet first, then CSV
        parquet_path = Path(processed_dir) / 'hotel_bookings_processed.parquet'
        csv_path = Path(processed_dir) / 'hotel_bookings_processed.csv'
        
        if parquet_path.exists():
            file_path = str(parquet_path)
        elif csv_path.exists():
            file_path = str(csv_path)
        else:
            raise FileNotFoundError(
                f"No processed data found in {processed_dir}. "
                "Run preprocessing first."
            )
    
    # Load based on file extension
    if str(file_path).endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)
    
    print(f"✓ Loaded processed data: {len(df):,} rows")
    return df


# ====================
# VALIDATION FUNCTIONS
# ====================

def validate_schema(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate that dataframe has expected columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate
    expected_columns : list, optional
        List of expected column names
        
    Returns
    -------
    tuple
        (is_valid, missing_columns)
    """
    if expected_columns is None:
        expected_columns = EXPECTED_COLUMNS
    
    current_columns = set(df.columns)
    expected_set = set(expected_columns)
    missing = expected_set - current_columns
    
    return len(missing) == 0, list(missing)


def check_target_exists(df: pd.DataFrame) -> bool:
    """Check if target column exists in dataframe."""
    return TARGET_COLUMN in df.columns


# ====================
# INFO FUNCTIONS
# ====================

def get_data_info(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive information about the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    dict
        Dictionary containing data information
    """
    info = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
    }
    
    # Target info if exists
    if check_target_exists(df):
        target_counts = df[TARGET_COLUMN].value_counts().to_dict()
        info['target_distribution'] = target_counts
        info['target_imbalance_ratio'] = target_counts.get(1, 0) / len(df)
    
    return info


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print a formatted summary of the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    """
    info = get_data_info(df)
    
    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Rows: {info['n_rows']:,}")
    print(f"Columns: {info['n_columns']}")
    print(f"Memory: {info['memory_mb']:.2f} MB")
    
    print("\n--- Missing Values ---")
    missing = {k: v for k, v in info['missing_values'].items() if v > 0}
    if missing:
        for col, count in sorted(missing.items(), key=lambda x: -x[1]):
            pct = info['missing_percentage'][col]
            print(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        print("  No missing values")
    
    if 'target_distribution' in info:
        print("\n--- Target Distribution ---")
        for label, count in info['target_distribution'].items():
            pct = count / info['n_rows'] * 100
            label_name = "Canceled" if label == 1 else "Not Canceled"
            print(f"  {label_name} ({label}): {count:,} ({pct:.2f}%)")
    
    print("=" * 50)


def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize columns by their types.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    dict
        Dictionary with 'numerical', 'categorical', 'target', 'leakage' keys
    """
    numerical = [col for col in NUMERICAL_COLUMNS if col in df.columns]
    categorical = [col for col in CATEGORICAL_COLUMNS if col in df.columns]
    leakage = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    
    return {
        'numerical': numerical,
        'categorical': categorical,
        'target': TARGET_COLUMN if TARGET_COLUMN in df.columns else None,
        'leakage': leakage
    }


# ====================
# EXPORT FUNCTIONS
# ====================

__all__ = [
    # Constants
    'EXPECTED_COLUMNS',
    'NUMERICAL_COLUMNS', 
    'CATEGORICAL_COLUMNS',
    'TARGET_COLUMN',
    'LEAKAGE_COLUMNS',
    # Functions
    'load_config',
    'load_raw_data',
    'load_processed_data',
    'validate_schema',
    'check_target_exists',
    'get_data_info',
    'print_data_summary',
    'get_column_types',
]


# ====================
# MAIN (for testing)
# ====================

if __name__ == "__main__":
    # Test loading
    try:
        df = load_raw_data()
        print_data_summary(df)
        
        col_types = get_column_types(df)
        print(f"\nNumerical columns: {len(col_types['numerical'])}")
        print(f"Categorical columns: {len(col_types['categorical'])}")
        print(f"Leakage columns: {col_types['leakage']}")
        
    except Exception as e:
        print(f"Error: {e}")
