"""
Module Xây Dựng Đặc Trưng (Feature Builder)
==========================================

Các hàm kỹ thuật đặc trưng (feature engineering) trên dữ liệu đặt phòng.

Các hàm chính:
--------------
- create_total_guests: Tạo đặc trưng tổng số khách
- create_total_nights: Tạo đặc trưng tổng số đêm
- discretize_lead_time: Phân nhóm lead_time thành khoảng
- discretize_country: Nhóm quốc gia thành top N + Khác
- create_season_features: Tạo đặc trưng mùa
- create_guest_history_features: Tạo đặc trưng lịch sử khách
- create_room_features: Tạo đặc trưng phòng
- create_booking_features: Tạo đặc trưng đặt phòng
- create_all_features: Áp dụng tất cả kỹ thuật đặc trưng
- prepare_for_association_rules: Chuẩn bị dữ liệu cho luật kết hợp
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import yaml
import warnings

warnings.filterwarnings('ignore')


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
        return get_default_config()


def get_default_config() -> dict:
    """Return default configuration for feature engineering."""
    return {
        'features': {
            'lead_time_bins': [0, 7, 30, 90, 180, 365, 1000],
            'lead_time_labels': ['0-7days', '7-30days', '30-90days', 
                                  '90-180days', '180-365days', '365+days'],
            'top_n_countries': 10
        }
    }


# ====================
# BASIC FEATURES
# ====================

def create_total_guests(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create total_guests feature = adults + children + babies.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with total_guests feature
        
    Examples
    --------
    >>> df = create_total_guests(df)
    """
    df_new = df.copy()
    
    # Handle missing values in children
    adults = df_new['adults'].fillna(0) if 'adults' in df_new.columns else 0
    children = df_new['children'].fillna(0) if 'children' in df_new.columns else 0
    babies = df_new['babies'].fillna(0) if 'babies' in df_new.columns else 0
    
    df_new['total_guests'] = adults + children + babies
    
    if verbose:
        print(f"✓ Created 'total_guests' = adults + children + babies")
        print(f"   Range: [{df_new['total_guests'].min()}, {df_new['total_guests'].max()}]")
    
    return df_new


def create_total_nights(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create total_nights feature = stays_in_weekend_nights + stays_in_week_nights.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with total_nights feature
    """
    df_new = df.copy()
    
    weekend = df_new['stays_in_weekend_nights'].fillna(0) if 'stays_in_weekend_nights' in df_new.columns else 0
    week = df_new['stays_in_week_nights'].fillna(0) if 'stays_in_week_nights' in df_new.columns else 0
    
    df_new['total_nights'] = weekend + week
    
    if verbose:
        print(f"✓ Created 'total_nights' = weekend_nights + week_nights")
        print(f"   Range: [{df_new['total_nights'].min()}, {df_new['total_nights'].max()}]")
    
    return df_new


def create_is_family(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create is_family feature (has children or babies).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with is_family feature
    """
    df_new = df.copy()
    
    children = df_new['children'].fillna(0) if 'children' in df_new.columns else 0
    babies = df_new['babies'].fillna(0) if 'babies' in df_new.columns else 0
    
    df_new['is_family'] = ((children > 0) | (babies > 0)).astype(int)
    
    if verbose:
        family_pct = df_new['is_family'].mean() * 100
        print(f"✓ Created 'is_family' (has children/babies)")
        print(f"   Family bookings: {family_pct:.2f}%")
    
    return df_new


# ====================
# DISCRETIZATION
# ====================

def discretize_lead_time(
    df: pd.DataFrame,
    bins: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
    config_path: str = 'configs/params.yaml',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Discretize lead_time into categorical bins.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    bins : list, optional
        Bin edges. If None, uses config.
    labels : list, optional
        Labels for bins. If None, uses config.
    config_path : str
        Path to configuration file
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with lead_time_category feature
        
    Examples
    --------
    >>> df = discretize_lead_time(df)
    >>> df = discretize_lead_time(df, bins=[0, 7, 30, 90, 365, 1000])
    """
    df_new = df.copy()
    
    if 'lead_time' not in df_new.columns:
        if verbose:
            print("⚠️ 'lead_time' column not found")
        return df_new
    
    # Load config if bins/labels not provided
    if bins is None or labels is None:
        config = load_config(config_path)
        feature_config = config.get('features', {})
        bins = bins or feature_config.get('lead_time_bins', [0, 7, 30, 90, 180, 365, 1000])
        labels = labels or feature_config.get('lead_time_labels', 
                                               ['0-7days', '7-30days', '30-90days', 
                                                '90-180days', '180-365days', '365+days'])
    
    # Ensure bins cover the full range
    max_lead_time = df_new['lead_time'].max()
    if bins[-1] < max_lead_time:
        bins[-1] = max_lead_time + 1
    
    df_new['lead_time_category'] = pd.cut(
        df_new['lead_time'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    
    if verbose:
        print(f"✓ Created 'lead_time_category' with {len(labels)} bins")
        print(f"   Bins: {bins}")
        print(f"   Distribution:")
        for label in labels:
            count = (df_new['lead_time_category'] == label).sum()
            pct = count / len(df_new) * 100
            print(f"     {label}: {count:,} ({pct:.1f}%)")
    
    return df_new


def discretize_country(
    df: pd.DataFrame,
    top_n: Optional[int] = None,
    config_path: str = 'configs/params.yaml',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Group countries into top N + 'Other'.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    top_n : int, optional
        Number of top countries to keep. If None, uses config.
    config_path : str
        Path to configuration file
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with country_grouped feature
    """
    df_new = df.copy()
    
    if 'country' not in df_new.columns:
        if verbose:
            print("⚠️ 'country' column not found")
        return df_new
    
    # Load config if top_n not provided
    if top_n is None:
        config = load_config(config_path)
        top_n = config.get('features', {}).get('top_n_countries', 10)
    
    # Get top N countries
    top_countries = df_new['country'].value_counts().head(top_n).index.tolist()
    
    # Create grouped column
    df_new['country_grouped'] = df_new['country'].apply(
        lambda x: x if x in top_countries else 'Other'
    )
    
    if verbose:
        print(f"✓ Created 'country_grouped' (top {top_n} + Other)")
        print(f"   Top countries: {top_countries}")
        other_pct = (df_new['country_grouped'] == 'Other').mean() * 100
        print(f"   'Other' category: {other_pct:.2f}%")
    
    return df_new


def discretize_adr(
    df: pd.DataFrame,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Discretize ADR (Average Daily Rate) into price categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    bins : list, optional
        Bin edges
    labels : list, optional
        Labels for bins
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with adr_category feature
    """
    df_new = df.copy()
    
    if 'adr' not in df_new.columns:
        if verbose:
            print("⚠️ 'adr' column not found")
        return df_new
    
    # Default bins based on quartiles
    if bins is None:
        bins = [0, 50, 100, 150, 200, float('inf')]
        labels = ['Budget', 'Economy', 'Standard', 'Premium', 'Luxury']
    
    df_new['adr_category'] = pd.cut(
        df_new['adr'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    
    if verbose:
        print(f"✓ Created 'adr_category' (price category)")
        print(f"   Bins: {bins}")
    
    return df_new


# ====================
# TEMPORAL FEATURES
# ====================

# Month to season mapping
MONTH_TO_SEASON = {
    'January': 'Winter', 'February': 'Winter', 'March': 'Spring',
    'April': 'Spring', 'May': 'Spring', 'June': 'Summer',
    'July': 'Summer', 'August': 'Summer', 'September': 'Fall',
    'October': 'Fall', 'November': 'Fall', 'December': 'Winter'
}

# Month to number mapping
MONTH_TO_NUM = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}


def create_season_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create seasonal features from arrival_date_month.
    
    Creates:
    - arrival_season: Winter/Spring/Summer/Fall
    - arrival_month_num: 1-12
    - is_summer: 1 if June/July/August
    - is_peak_season: 1 if July/August (high season)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with seasonal features
    """
    df_new = df.copy()
    
    if 'arrival_date_month' not in df_new.columns:
        if verbose:
            print("⚠️ 'arrival_date_month' column not found")
        return df_new
    
    # Map month to season
    df_new['arrival_season'] = df_new['arrival_date_month'].map(MONTH_TO_SEASON)
    
    # Map month to number
    df_new['arrival_month_num'] = df_new['arrival_date_month'].map(MONTH_TO_NUM)
    
    # Is summer (June, July, August)
    df_new['is_summer'] = df_new['arrival_date_month'].isin(
        ['June', 'July', 'August']
    ).astype(int)
    
    # Is peak season (July, August)
    df_new['is_peak_season'] = df_new['arrival_date_month'].isin(
        ['July', 'August']
    ).astype(int)
    
    # Is weekend arrival (approximate based on day of month)
    if 'arrival_date_day_of_month' in df_new.columns:
        # This is an approximation - real weekend detection would need full date
        pass
    
    if verbose:
        print(f"✓ Created seasonal features:")
        print(f"   - arrival_season: {df_new['arrival_season'].unique().tolist()}")
        summer_pct = df_new['is_summer'].mean() * 100
        peak_pct = df_new['is_peak_season'].mean() * 100
        print(f"   - is_summer: {summer_pct:.1f}%")
        print(f"   - is_peak_season: {peak_pct:.1f}%")
    
    return df_new


def create_arrival_date(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create full arrival date from components.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with arrival_date feature
    """
    df_new = df.copy()
    
    required_cols = ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']
    if not all(col in df_new.columns for col in required_cols):
        if verbose:
            print("⚠️ Required date columns not found")
        return df_new
    
    # Create month number if not exists
    if 'arrival_month_num' not in df_new.columns:
        df_new['arrival_month_num'] = df_new['arrival_date_month'].map(MONTH_TO_NUM)
    
    # Create date string and parse
    df_new['arrival_date'] = pd.to_datetime(
        df_new['arrival_date_year'].astype(str) + '-' +
        df_new['arrival_month_num'].astype(str).str.zfill(2) + '-' +
        df_new['arrival_date_day_of_month'].astype(str).str.zfill(2),
        errors='coerce'
    )
    
    # Extract day of week
    df_new['arrival_day_of_week'] = df_new['arrival_date'].dt.dayofweek
    df_new['is_weekend_arrival'] = df_new['arrival_day_of_week'].isin([4, 5, 6]).astype(int)
    
    if verbose:
        print(f"✓ Created arrival date features:")
        print(f"   - arrival_date: {df_new['arrival_date'].min()} to {df_new['arrival_date'].max()}")
        weekend_pct = df_new['is_weekend_arrival'].mean() * 100
        print(f"   - is_weekend_arrival: {weekend_pct:.1f}%")
    
    return df_new


# ====================
# GUEST HISTORY FEATURES
# ====================

def create_guest_history_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create features based on guest booking history.
    
    Creates:
    - has_canceled_before: 1 if previous_cancellations > 0
    - is_returning_customer: 1 if is_repeated_guest = 1 or previous_bookings_not_canceled > 0
    - canceled_before_and_repeated: Interaction feature
    - total_previous_bookings: Sum of previous cancellations and non-cancellations
    - cancellation_ratio: Ratio of cancellations to total previous bookings
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with guest history features
    """
    df_new = df.copy()
    
    # Has canceled before
    if 'previous_cancellations' in df_new.columns:
        df_new['has_canceled_before'] = (df_new['previous_cancellations'] > 0).astype(int)
    
    # Is returning customer
    is_repeated = df_new['is_repeated_guest'].fillna(0) if 'is_repeated_guest' in df_new.columns else 0
    prev_not_canceled = df_new['previous_bookings_not_canceled'].fillna(0) if 'previous_bookings_not_canceled' in df_new.columns else 0
    df_new['is_returning_customer'] = ((is_repeated > 0) | (prev_not_canceled > 0)).astype(int)
    
    # Interaction: Repeated guest who has canceled before (high risk)
    if 'has_canceled_before' in df_new.columns and 'is_repeated_guest' in df_new.columns:
        df_new['repeated_and_canceled_before'] = (
            (df_new['is_repeated_guest'] == 1) & (df_new['has_canceled_before'] == 1)
        ).astype(int)
    
    # Total previous bookings
    prev_canceled = df_new['previous_cancellations'].fillna(0) if 'previous_cancellations' in df_new.columns else 0
    df_new['total_previous_bookings'] = prev_canceled + prev_not_canceled
    
    # Cancellation ratio (for returning customers)
    df_new['cancellation_ratio'] = np.where(
        df_new['total_previous_bookings'] > 0,
        prev_canceled / df_new['total_previous_bookings'],
        0
    )
    
    if verbose:
        print(f"✓ Created guest history features:")
        if 'has_canceled_before' in df_new.columns:
            pct = df_new['has_canceled_before'].mean() * 100
            print(f"   - has_canceled_before: {pct:.2f}%")
        pct = df_new['is_returning_customer'].mean() * 100
        print(f"   - is_returning_customer: {pct:.2f}%")
        if 'repeated_and_canceled_before' in df_new.columns:
            pct = df_new['repeated_and_canceled_before'].mean() * 100
            print(f"   - repeated_and_canceled_before: {pct:.2f}%")
    
    return df_new


# ====================
# ROOM FEATURES
# ====================

def create_room_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create room-related features.
    
    Creates:
    - room_type_changed: 1 if assigned_room_type != reserved_room_type
    - room_type_upgraded: 1 if room changed and possibly upgraded
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with room features
    """
    df_new = df.copy()
    
    if 'reserved_room_type' in df_new.columns and 'assigned_room_type' in df_new.columns:
        # Room type changed
        df_new['room_type_changed'] = (
            df_new['reserved_room_type'] != df_new['assigned_room_type']
        ).astype(int)
        
        if verbose:
            pct = df_new['room_type_changed'].mean() * 100
            print(f"✓ Created 'room_type_changed': {pct:.2f}% of bookings")
    
    return df_new


# ====================
# BOOKING FEATURES
# ====================

def create_booking_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create booking-related features.
    
    Creates:
    - has_special_requests: 1 if total_of_special_requests > 0
    - has_booking_changes: 1 if booking_changes > 0
    - has_agent: 1 if booked through agent
    - is_direct_booking: 1 if distribution_channel is Direct
    - deposit_required: 1 if deposit_type is not 'No Deposit'
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with booking features
    """
    df_new = df.copy()
    
    # Has special requests
    if 'total_of_special_requests' in df_new.columns:
        df_new['has_special_requests'] = (df_new['total_of_special_requests'] > 0).astype(int)
    
    # Has booking changes
    if 'booking_changes' in df_new.columns:
        df_new['has_booking_changes'] = (df_new['booking_changes'] > 0).astype(int)
    
    # Has agent
    if 'agent' in df_new.columns:
        df_new['has_agent'] = (df_new['agent'].fillna(0) > 0).astype(int)
    
    # Is direct booking
    if 'distribution_channel' in df_new.columns:
        df_new['is_direct_booking'] = (df_new['distribution_channel'] == 'Direct').astype(int)
    
    # Deposit required
    if 'deposit_type' in df_new.columns:
        df_new['deposit_required'] = (df_new['deposit_type'] != 'No Deposit').astype(int)
    
    # Requires parking
    if 'required_car_parking_spaces' in df_new.columns:
        df_new['requires_parking'] = (df_new['required_car_parking_spaces'] > 0).astype(int)
    
    if verbose:
        print(f"✓ Created booking features:")
        for col in ['has_special_requests', 'has_booking_changes', 'has_agent', 
                    'is_direct_booking', 'deposit_required', 'requires_parking']:
            if col in df_new.columns:
                pct = df_new[col].mean() * 100
                print(f"   - {col}: {pct:.2f}%")
    
    return df_new


# ====================
# REVENUE FEATURES
# ====================

def create_revenue_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create revenue-related features.
    
    Creates:
    - total_revenue: adr * total_nights
    - revenue_per_guest: total_revenue / total_guests
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with revenue features
    """
    df_new = df.copy()
    
    # Ensure total_nights exists
    if 'total_nights' not in df_new.columns:
        df_new = create_total_nights(df_new, verbose=False)
    
    # Ensure total_guests exists
    if 'total_guests' not in df_new.columns:
        df_new = create_total_guests(df_new, verbose=False)
    
    # Total revenue estimate
    if 'adr' in df_new.columns and 'total_nights' in df_new.columns:
        df_new['total_revenue'] = df_new['adr'] * df_new['total_nights']
        
        # Revenue per guest
        df_new['revenue_per_guest'] = np.where(
            df_new['total_guests'] > 0,
            df_new['total_revenue'] / df_new['total_guests'],
            df_new['total_revenue']
        )
    
    if verbose:
        if 'total_revenue' in df_new.columns:
            print(f"✓ Created revenue features:")
            print(f"   - total_revenue: mean={df_new['total_revenue'].mean():.2f}")
            print(f"   - revenue_per_guest: mean={df_new['revenue_per_guest'].mean():.2f}")
    
    return df_new


# ====================
# ASSOCIATION RULES PREPARATION
# ====================

def prepare_for_association_rules(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    include_target: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Prepare data for association rule mining.
    
    Converts selected categorical columns to one-hot encoded format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list, optional
        Columns to include. If None, uses default columns.
    include_target : bool
        Whether to include target column (is_canceled)
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        One-hot encoded DataFrame for association rules
        
    Examples
    --------
    >>> df_assoc = prepare_for_association_rules(df)
    >>> # Run Apriori on df_assoc
    """
    df_new = df.copy()
    
    # Default columns for association rules
    if columns is None:
        columns = [
            'hotel', 'arrival_season', 'meal', 'country_grouped',
            'market_segment', 'deposit_type', 'customer_type',
            'lead_time_category', 'is_family', 'is_returning_customer'
        ]
    
    # Filter to existing columns
    columns = [col for col in columns if col in df_new.columns]
    
    # Add target if requested
    if include_target and 'is_canceled' in df_new.columns:
        # Convert target to categorical
        df_new['canceled'] = df_new['is_canceled'].map({0: 'Not_Canceled', 1: 'Canceled'})
        columns.append('canceled')
    
    # Select only these columns
    df_subset = df_new[columns]
    
    # One-hot encode
    df_encoded = pd.get_dummies(df_subset, prefix_sep='=')
    
    # Convert to boolean for association rules
    df_encoded = df_encoded.astype(bool)
    
    if verbose:
        print(f"✓ Prepared data for association rules:")
        print(f"   - Input columns: {len(columns)}")
        print(f"   - Output features: {len(df_encoded.columns)}")
        print(f"   - Rows: {len(df_encoded)}")
    
    return df_encoded


# ====================
# FULL PIPELINE
# ====================

def create_all_features(
    df: pd.DataFrame,
    config_path: str = 'configs/params.yaml',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    config_path : str
        Path to configuration file
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all engineered features
        
    Examples
    --------
    >>> df_features = create_all_features(df)
    """
    if verbose:
        print("\n" + "=" * 70)
        print("🔧 FEATURE ENGINEERING PIPELINE")
        print("=" * 70 + "\n")
    
    df_new = df.copy()
    initial_cols = len(df_new.columns)
    
    # 1. Basic features
    if verbose:
        print("📊 Basic Features:")
    df_new = create_total_guests(df_new, verbose=verbose)
    df_new = create_total_nights(df_new, verbose=verbose)
    df_new = create_is_family(df_new, verbose=verbose)
    if verbose:
        print()
    
    # 2. Discretization
    if verbose:
        print("📊 Discretization:")
    df_new = discretize_lead_time(df_new, config_path=config_path, verbose=verbose)
    df_new = discretize_country(df_new, config_path=config_path, verbose=verbose)
    df_new = discretize_adr(df_new, verbose=verbose)
    if verbose:
        print()
    
    # 3. Temporal features
    if verbose:
        print("📊 Temporal Features:")
    df_new = create_season_features(df_new, verbose=verbose)
    df_new = create_arrival_date(df_new, verbose=verbose)
    if verbose:
        print()
    
    # 4. Guest history features
    if verbose:
        print("📊 Guest History Features:")
    df_new = create_guest_history_features(df_new, verbose=verbose)
    if verbose:
        print()
    
    # 5. Room features
    if verbose:
        print("📊 Room Features:")
    df_new = create_room_features(df_new, verbose=verbose)
    if verbose:
        print()
    
    # 6. Booking features
    if verbose:
        print("📊 Booking Features:")
    df_new = create_booking_features(df_new, verbose=verbose)
    if verbose:
        print()
    
    # 7. Revenue features
    if verbose:
        print("📊 Revenue Features:")
    df_new = create_revenue_features(df_new, verbose=verbose)
    if verbose:
        print()
    
    # Summary
    final_cols = len(df_new.columns)
    new_features = final_cols - initial_cols
    
    if verbose:
        print("=" * 70)
        print(f"✅ FEATURE ENGINEERING COMPLETE")
        print(f"   Original columns: {initial_cols}")
        print(f"   Final columns: {final_cols}")
        print(f"   New features created: {new_features}")
        print("=" * 70 + "\n")
    
    return df_new


def get_feature_list() -> Dict[str, List[str]]:
    """
    Get list of all features created by this module.
    
    Returns
    -------
    dict
        Dictionary with feature categories and their names
    """
    return {
        'basic': ['total_guests', 'total_nights', 'is_family'],
        'discretized': ['lead_time_category', 'country_grouped', 'adr_category'],
        'temporal': ['arrival_season', 'arrival_month_num', 'is_summer', 
                     'is_peak_season', 'arrival_date', 'arrival_day_of_week', 
                     'is_weekend_arrival'],
        'guest_history': ['has_canceled_before', 'is_returning_customer',
                          'repeated_and_canceled_before', 'total_previous_bookings',
                          'cancellation_ratio'],
        'room': ['room_type_changed'],
        'booking': ['has_special_requests', 'has_booking_changes', 'has_agent',
                    'is_direct_booking', 'deposit_required', 'requires_parking'],
        'revenue': ['total_revenue', 'revenue_per_guest']
    }


# ====================
# MAIN (Testing)
# ====================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.data.loader import load_raw_data
    from src.data.cleaner import clean_data
    
    print("Testing feature builder module...")
    print("=" * 70)
    
    # Load and clean data
    df = load_raw_data()
    df_clean, _ = clean_data(df, encode=False, scale=False, verbose=False)
    print(f"\nCleaned data shape: {df_clean.shape}")
    
    # Apply feature engineering
    df_features = create_all_features(df_clean, verbose=True)
    
    print(f"\nFinal shape: {df_features.shape}")
    print(f"\nNew columns created:")
    new_cols = set(df_features.columns) - set(df_clean.columns)
    for col in sorted(new_cols):
        print(f"  - {col}")
    
    # Test association rules preparation
    print("\n" + "=" * 70)
    print("Testing association rules preparation...")
    df_assoc = prepare_for_association_rules(df_features, verbose=True)
    print(f"\nAssociation rules data shape: {df_assoc.shape}")
    print(f"Sample columns: {list(df_assoc.columns)[:10]}")
    
    print("\n✅ Feature builder module test passed!")
