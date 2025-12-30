"""
Time Series Forecasting for Hotel Booking Cancellation Rate
===========================================================

This module provides time series analysis and forecasting methods
for predicting cancellation rates over time.

Methods included:
- Data aggregation by time period (month, week)
- ARIMA/SARIMA models
- Simple forecasting baselines (moving average, exponential smoothing)
- Prophet (optional, if installed)
- Evaluation metrics (MAE, RMSE, MAPE)

Author: Nhom12
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from pathlib import Path
from datetime import datetime

# Scikit-learn for metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Check for statsmodels availability
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not installed. ARIMA/SARIMA will not be available.")

# Check for Prophet availability
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


# =============================================================================
# DATA AGGREGATION
# =============================================================================

def create_date_column(
    df: pd.DataFrame,
    year_col: str = 'arrival_date_year',
    month_col: str = 'arrival_date_month',
    day_col: str = 'arrival_date_day_of_month',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create a datetime column from year, month, day columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date components
    year_col, month_col, day_col : str
        Column names for year, month, day
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'arrival_date' column added
    """
    df = df.copy()
    
    # Month mapping
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    # Convert month names to numbers if needed
    if df[month_col].dtype == 'object':
        df['month_num'] = df[month_col].map(month_map)
    else:
        df['month_num'] = df[month_col]
    
    # Create date column
    df['arrival_date'] = pd.to_datetime(
        df[[year_col, 'month_num', day_col]].rename(
            columns={year_col: 'year', 'month_num': 'month', day_col: 'day'}
        )
    )
    
    # Drop temporary column
    df = df.drop(columns=['month_num'])
    
    if verbose:
        print(f"✅ Created 'arrival_date' column")
        print(f"   Date range: {df['arrival_date'].min()} to {df['arrival_date'].max()}")
    
    return df


def aggregate_by_period(
    df: pd.DataFrame,
    date_col: str = 'arrival_date',
    target_col: str = 'is_canceled',
    period: str = 'M',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Aggregate booking data by time period.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with bookings
    date_col : str
        Date column name
    target_col : str
        Target column (cancellation indicator)
    period : str
        Aggregation period: 'M' (month), 'W' (week), 'D' (day)
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        Aggregated time series with columns:
        - period: Period end date
        - total_bookings: Total bookings in period
        - cancellations: Number of cancellations
        - cancellation_rate: Cancellation rate (0-1)
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    
    # Set date as index
    df = df.set_index(date_col)
    
    # Aggregate
    agg_df = df.groupby(pd.Grouper(freq=period)).agg({
        target_col: ['count', 'sum']
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = ['period', 'total_bookings', 'cancellations']
    
    # Calculate cancellation rate
    agg_df['cancellation_rate'] = agg_df['cancellations'] / agg_df['total_bookings']
    
    # Handle division by zero
    agg_df['cancellation_rate'] = agg_df['cancellation_rate'].fillna(0)
    
    # Remove periods with no data
    agg_df = agg_df[agg_df['total_bookings'] > 0].reset_index(drop=True)
    
    if verbose:
        period_names = {'M': 'Monthly', 'W': 'Weekly', 'D': 'Daily'}
        print(f"\n📊 {period_names.get(period, period)} Aggregation:")
        print(f"   Periods: {len(agg_df)}")
        print(f"   Date range: {agg_df['period'].min()} to {agg_df['period'].max()}")
        print(f"   Avg cancellation rate: {agg_df['cancellation_rate'].mean():.2%}")
        print(f"   Min cancellation rate: {agg_df['cancellation_rate'].min():.2%}")
        print(f"   Max cancellation rate: {agg_df['cancellation_rate'].max():.2%}")
    
    return agg_df


def prepare_time_series(
    df: pd.DataFrame,
    year_col: str = 'arrival_date_year',
    month_col: str = 'arrival_date_month',
    day_col: str = 'arrival_date_day_of_month',
    target_col: str = 'is_canceled',
    period: str = 'M',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Complete pipeline to prepare time series data from raw bookings.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw booking data
    year_col, month_col, day_col : str
        Date component columns
    target_col : str
        Target column
    period : str
        Aggregation period ('M', 'W', 'D')
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        Time series DataFrame ready for forecasting
    """
    if verbose:
        print("=" * 60)
        print("PREPARING TIME SERIES DATA")
        print("=" * 60)
    
    # Create date column
    df_with_date = create_date_column(
        df, year_col, month_col, day_col, verbose=verbose
    )
    
    # Aggregate by period
    ts_df = aggregate_by_period(
        df_with_date, 
        date_col='arrival_date',
        target_col=target_col,
        period=period,
        verbose=verbose
    )
    
    if verbose:
        print("=" * 60)
    
    return ts_df


# =============================================================================
# TIME SERIES ANALYSIS
# =============================================================================

def check_stationarity(
    series: pd.Series,
    significance_level: float = 0.05,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Check if time series is stationary using Augmented Dickey-Fuller test.
    
    Parameters
    ----------
    series : pd.Series
        Time series to test
    significance_level : float
        Significance level for the test
    verbose : bool
        Whether to print results
        
    Returns
    -------
    Dict[str, Any]
        Test results including statistic, p-value, and conclusion
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for stationarity test")
    
    # Drop NaN values
    series = series.dropna()
    
    # Run ADF test
    result = adfuller(series, autolag='AIC')
    
    adf_stat = result[0]
    p_value = result[1]
    used_lags = result[2]
    n_obs = result[3]
    critical_values = result[4]
    
    is_stationary = p_value < significance_level
    
    results = {
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'used_lags': used_lags,
        'n_observations': n_obs,
        'critical_values': critical_values,
        'is_stationary': is_stationary
    }
    
    if verbose:
        print("\n📊 Augmented Dickey-Fuller Test:")
        print(f"   ADF Statistic: {adf_stat:.4f}")
        print(f"   P-Value: {p_value:.4f}")
        print(f"   Used Lags: {used_lags}")
        print(f"   Critical Values:")
        for key, value in critical_values.items():
            print(f"      {key}: {value:.4f}")
        print(f"\n   Conclusion: Series is {'STATIONARY' if is_stationary else 'NON-STATIONARY'}")
    
    return results


def decompose_time_series(
    series: pd.Series,
    period: int = 12,
    model: str = 'additive',
    verbose: bool = True
) -> Any:
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Parameters
    ----------
    series : pd.Series
        Time series to decompose
    period : int
        Seasonal period (12 for monthly data)
    model : str
        'additive' or 'multiplicative'
    verbose : bool
        Whether to print information
        
    Returns
    -------
    DecomposeResult
        Decomposition result with trend, seasonal, resid attributes
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for decomposition")
    
    # Need enough data points
    if len(series) < 2 * period:
        if verbose:
            print(f"⚠️ Not enough data for decomposition (need at least {2*period} points)")
        return None
    
    result = seasonal_decompose(series, model=model, period=period)
    
    if verbose:
        print(f"\n📊 Time Series Decomposition ({model}):")
        print(f"   Period: {period}")
        print(f"   Trend range: {result.trend.min():.4f} to {result.trend.max():.4f}")
        print(f"   Seasonal range: {result.seasonal.min():.4f} to {result.seasonal.max():.4f}")
    
    return result


# =============================================================================
# FORECASTING MODELS
# =============================================================================

def train_arima(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train ARIMA model.
    
    Parameters
    ----------
    series : pd.Series
        Time series to model
    order : tuple
        (p, d, q) order for ARIMA
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Tuple[model, info_dict]
        Fitted model and information dictionary
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for ARIMA")
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"TRAINING ARIMA{order}")
        print("=" * 60)
    
    # Fit model
    model = ARIMA(series, order=order)
    fitted = model.fit()
    
    info = {
        'order': order,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'n_obs': fitted.nobs
    }
    
    if verbose:
        print(f"   AIC: {fitted.aic:.2f}")
        print(f"   BIC: {fitted.bic:.2f}")
        print(f"   Observations: {fitted.nobs}")
        print("=" * 60)
    
    return fitted, info


def train_sarima(
    series: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12),
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train SARIMA model (Seasonal ARIMA).
    
    Parameters
    ----------
    series : pd.Series
        Time series to model
    order : tuple
        (p, d, q) non-seasonal order
    seasonal_order : tuple
        (P, D, Q, s) seasonal order
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Tuple[model, info_dict]
        Fitted model and information dictionary
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for SARIMA")
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"TRAINING SARIMA{order}x{seasonal_order}")
        print("=" * 60)
    
    # Fit model
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)
    
    info = {
        'order': order,
        'seasonal_order': seasonal_order,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'n_obs': fitted.nobs
    }
    
    if verbose:
        print(f"   AIC: {fitted.aic:.2f}")
        print(f"   BIC: {fitted.bic:.2f}")
        print(f"   Observations: {fitted.nobs}")
        print("=" * 60)
    
    return fitted, info


def train_exponential_smoothing(
    series: pd.Series,
    trend: str = 'add',
    seasonal: str = 'add',
    seasonal_periods: int = 12,
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train Exponential Smoothing (Holt-Winters) model.
    
    Parameters
    ----------
    series : pd.Series
        Time series to model
    trend : str
        Trend component: 'add', 'mul', or None
    seasonal : str
        Seasonal component: 'add', 'mul', or None
    seasonal_periods : int
        Number of periods in a season
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Tuple[model, info_dict]
        Fitted model and information dictionary
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for Exponential Smoothing")
    
    if verbose:
        print("\n" + "=" * 60)
        print(f"TRAINING EXPONENTIAL SMOOTHING (trend={trend}, seasonal={seasonal})")
        print("=" * 60)
    
    # Fit model
    model = ExponentialSmoothing(
        series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )
    fitted = model.fit()
    
    info = {
        'trend': trend,
        'seasonal': seasonal,
        'seasonal_periods': seasonal_periods,
        'aic': fitted.aic,
        'bic': fitted.bic
    }
    
    if verbose:
        print(f"   AIC: {fitted.aic:.2f}")
        print(f"   BIC: {fitted.bic:.2f}")
        print("=" * 60)
    
    return fitted, info


def train_prophet(
    df: pd.DataFrame,
    date_col: str = 'period',
    target_col: str = 'cancellation_rate',
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = False,
    daily_seasonality: bool = False,
    verbose: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train Facebook Prophet model.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date and target columns
    date_col : str
        Date column name
    target_col : str
        Target column name
    yearly_seasonality : bool
        Include yearly seasonality
    weekly_seasonality : bool
        Include weekly seasonality
    daily_seasonality : bool
        Include daily seasonality
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Tuple[model, info_dict]
        Fitted model and information dictionary
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("prophet is not installed. Install with: pip install prophet")
    
    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING PROPHET")
        print("=" * 60)
    
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = df[[date_col, target_col]].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Create and fit model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality
    )
    
    # Suppress output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prophet_df)
    
    info = {
        'yearly_seasonality': yearly_seasonality,
        'weekly_seasonality': weekly_seasonality,
        'daily_seasonality': daily_seasonality,
        'n_changepoints': len(model.changepoints)
    }
    
    if verbose:
        print(f"   Changepoints detected: {len(model.changepoints)}")
        print("=" * 60)
    
    return model, info


# =============================================================================
# SIMPLE BASELINES
# =============================================================================

def moving_average_forecast(
    series: pd.Series,
    window: int = 3,
    forecast_periods: int = 6,
    verbose: bool = True
) -> pd.Series:
    """
    Simple moving average forecast.
    
    Parameters
    ----------
    series : pd.Series
        Historical time series
    window : int
        Moving average window
    forecast_periods : int
        Number of periods to forecast
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.Series
        Forecasted values
    """
    # Calculate last moving average
    last_ma = series.tail(window).mean()
    
    # Create forecast (constant based on last MA)
    last_date = series.index[-1] if hasattr(series.index, '__len__') else len(series)
    forecast = pd.Series([last_ma] * forecast_periods)
    
    if verbose:
        print(f"\n📊 Moving Average Forecast (window={window}):")
        print(f"   Forecast value: {last_ma:.4f}")
        print(f"   Periods: {forecast_periods}")
    
    return forecast


def naive_forecast(
    series: pd.Series,
    forecast_periods: int = 6,
    seasonal_period: int = None,
    verbose: bool = True
) -> pd.Series:
    """
    Naive forecast (last value or seasonal naive).
    
    Parameters
    ----------
    series : pd.Series
        Historical time series
    forecast_periods : int
        Number of periods to forecast
    seasonal_period : int, optional
        If provided, use seasonal naive (repeat last season)
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.Series
        Forecasted values
    """
    if seasonal_period:
        # Seasonal naive: repeat last season
        last_season = series.tail(seasonal_period).values
        repeats = (forecast_periods // seasonal_period) + 1
        forecast = np.tile(last_season, repeats)[:forecast_periods]
    else:
        # Simple naive: repeat last value
        forecast = np.repeat(series.iloc[-1], forecast_periods)
    
    forecast = pd.Series(forecast)
    
    if verbose:
        method = f"Seasonal Naive (period={seasonal_period})" if seasonal_period else "Naive"
        print(f"\n📊 {method} Forecast:")
        print(f"   Periods: {forecast_periods}")
    
    return forecast


# =============================================================================
# FORECASTING
# =============================================================================

def forecast(
    model: Any,
    steps: int = 6,
    model_type: str = 'arima',
    prophet_df: pd.DataFrame = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate forecasts from a fitted model.
    
    Parameters
    ----------
    model : fitted model
        Fitted forecasting model
    steps : int
        Number of steps to forecast
    model_type : str
        Type of model: 'arima', 'sarima', 'exp_smoothing', 'prophet'
    prophet_df : pd.DataFrame, optional
        Future dataframe for Prophet (if not provided, will be created)
    verbose : bool
        Whether to print information
        
    Returns
    -------
    pd.DataFrame
        Forecast with columns: forecast, lower, upper (confidence intervals)
    """
    if verbose:
        print(f"\n📈 Generating {steps}-step forecast...")
    
    if model_type in ['arima', 'sarima']:
        # Get forecast with confidence intervals
        forecast_result = model.get_forecast(steps=steps)
        forecast_df = pd.DataFrame({
            'forecast': forecast_result.predicted_mean,
            'lower': forecast_result.conf_int().iloc[:, 0],
            'upper': forecast_result.conf_int().iloc[:, 1]
        })
    
    elif model_type == 'exp_smoothing':
        # Exponential Smoothing uses forecast() method
        forecast_values = model.forecast(steps=steps)
        # Estimate confidence intervals (simple approximation)
        std_resid = np.std(model.resid)
        forecast_df = pd.DataFrame({
            'forecast': forecast_values.values,
            'lower': forecast_values.values - 1.96 * std_resid,
            'upper': forecast_values.values + 1.96 * std_resid
        })
        
    elif model_type == 'prophet':
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is not installed")
        
        # Create future dataframe
        if prophet_df is None:
            prophet_df = model.make_future_dataframe(periods=steps, freq='M')
        
        # Predict
        prophet_forecast = model.predict(prophet_df)
        
        # Get only future predictions
        forecast_df = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(steps)
        forecast_df.columns = ['date', 'forecast', 'lower', 'upper']
        forecast_df = forecast_df.reset_index(drop=True)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if verbose:
        print(f"   Forecast mean: {forecast_df['forecast'].mean():.4f}")
        print(f"   Forecast range: {forecast_df['forecast'].min():.4f} to {forecast_df['forecast'].max():.4f}")
    
    return forecast_df


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model',
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate forecast performance.
    
    Parameters
    ----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name for display
    verbose : bool
        Whether to print results
        
    Returns
    -------
    Dict[str, float]
        Dictionary with MAE, RMSE, MAPE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }
    
    if verbose:
        print(f"\n📊 {model_name} Forecast Evaluation:")
        print(f"   MAE:  {mae:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAPE: {mape:.2f}%")
    
    return metrics


# =============================================================================
# TRAIN-TEST SPLIT FOR TIME SERIES
# =============================================================================

def train_test_split_ts(
    series: pd.Series,
    test_size: int = 6,
    verbose: bool = True
) -> Tuple[pd.Series, pd.Series]:
    """
    Split time series into train and test sets.
    
    Parameters
    ----------
    series : pd.Series
        Time series to split
    test_size : int
        Number of periods for test set
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Tuple[train, test]
        Training and test series
    """
    train = series[:-test_size]
    test = series[-test_size:]
    
    if verbose:
        print(f"\n📊 Train-Test Split:")
        print(f"   Train size: {len(train)}")
        print(f"   Test size: {len(test)}")
    
    return train, test


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_time_series(
    df: pd.DataFrame,
    date_col: str = 'period',
    value_col: str = 'cancellation_rate',
    title: str = 'Cancellation Rate Over Time',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        Time series data
    date_col : str
        Date column
    value_col : str
        Value column
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(df[date_col], df[value_col], 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cancellation Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    # Rotate x-labels
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved time series plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_decomposition(
    decomposition_result,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot time series decomposition.
    
    Parameters
    ----------
    decomposition_result : DecomposeResult
        Result from seasonal_decompose
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 1, figsize=figsize)
    
    decomposition_result.observed.plot(ax=axes[0], title='Observed')
    decomposition_result.trend.plot(ax=axes[1], title='Trend')
    decomposition_result.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition_result.resid.plot(ax=axes[3], title='Residual')
    
    for ax in axes:
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved decomposition plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_forecast(
    train: pd.Series,
    test: pd.Series,
    forecast_df: pd.DataFrame,
    title: str = 'Forecast vs Actual',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot forecast against actual values.
    
    Parameters
    ----------
    train : pd.Series
        Training data
    test : pd.Series
        Test data (actual values)
    forecast_df : pd.DataFrame
        Forecast with 'forecast', 'lower', 'upper' columns
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
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training data
    ax.plot(range(len(train)), train.values, 'b-', linewidth=2, label='Training')
    
    # Plot test data
    test_idx = range(len(train), len(train) + len(test))
    ax.plot(test_idx, test.values, 'g-o', linewidth=2, markersize=6, label='Actual')
    
    # Plot forecast
    ax.plot(test_idx, forecast_df['forecast'].values, 'r--', linewidth=2, label='Forecast')
    
    # Plot confidence interval
    if 'lower' in forecast_df.columns and 'upper' in forecast_df.columns:
        ax.fill_between(
            test_idx,
            forecast_df['lower'].values,
            forecast_df['upper'].values,
            color='red', alpha=0.2, label='95% CI'
        )
    
    ax.set_xlabel('Time Period', fontsize=12)
    ax.set_ylabel('Cancellation Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved forecast plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = 'rmse',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot model comparison bar chart.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with models as index and metrics as columns
    metric : str
        Metric to plot
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save
    show : bool
        Whether to display
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    
    models = comparison_df.index.tolist()
    values = comparison_df[metric].values
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    bars = ax.bar(models, values, color=colors)
    
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f'Model Comparison: {metric.upper()}', fontsize=14, fontweight='bold')
    ax.bar_label(bars, fmt='%.4f')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


# =============================================================================
# COMPREHENSIVE PIPELINE
# =============================================================================

def run_forecasting_pipeline(
    df: pd.DataFrame,
    year_col: str = 'arrival_date_year',
    month_col: str = 'arrival_date_month',
    day_col: str = 'arrival_date_day_of_month',
    target_col: str = 'is_canceled',
    test_periods: int = 6,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run complete forecasting pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw booking data
    year_col, month_col, day_col : str
        Date columns
    target_col : str
        Target column
    test_periods : int
        Number of periods for test set
    verbose : bool
        Whether to print information
        
    Returns
    -------
    Dict[str, Any]
        Results including models, forecasts, and metrics
    """
    results = {}
    
    # 1. Prepare time series
    ts_df = prepare_time_series(
        df, year_col, month_col, day_col, target_col,
        period='M', verbose=verbose
    )
    results['time_series'] = ts_df
    
    # 2. Get cancellation rate series
    series = ts_df.set_index('period')['cancellation_rate']
    
    # 3. Train-test split
    train, test = train_test_split_ts(series, test_size=test_periods, verbose=verbose)
    results['train'] = train
    results['test'] = test
    
    # 4. Check stationarity
    if STATSMODELS_AVAILABLE:
        stationarity = check_stationarity(train, verbose=verbose)
        results['stationarity'] = stationarity
    
    # 5. Train models and evaluate
    model_results = {}
    
    # Naive baseline
    naive_pred = naive_forecast(train, forecast_periods=test_periods, verbose=verbose)
    naive_metrics = evaluate_forecast(test.values, naive_pred.values, 'Naive', verbose=verbose)
    model_results['naive'] = {'forecast': naive_pred, 'metrics': naive_metrics}
    
    # Moving Average
    ma_pred = moving_average_forecast(train, window=3, forecast_periods=test_periods, verbose=verbose)
    ma_metrics = evaluate_forecast(test.values, ma_pred.values, 'Moving Average', verbose=verbose)
    model_results['moving_average'] = {'forecast': ma_pred, 'metrics': ma_metrics}
    
    # ARIMA
    if STATSMODELS_AVAILABLE:
        try:
            arima_model, arima_info = train_arima(train, order=(1, 1, 1), verbose=verbose)
            arima_forecast = forecast(arima_model, steps=test_periods, model_type='arima', verbose=verbose)
            arima_metrics = evaluate_forecast(test.values, arima_forecast['forecast'].values, 'ARIMA(1,1,1)', verbose=verbose)
            model_results['arima'] = {'model': arima_model, 'info': arima_info, 'forecast': arima_forecast, 'metrics': arima_metrics}
        except Exception as e:
            if verbose:
                print(f"⚠️ ARIMA failed: {e}")
    
    results['models'] = model_results
    
    # 6. Create comparison table
    comparison_data = {}
    for name, result in model_results.items():
        comparison_data[name] = result['metrics']
    
    comparison_df = pd.DataFrame(comparison_data).T
    results['comparison'] = comparison_df
    
    if verbose:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 60)
        print(comparison_df.round(4).to_string())
        print("=" * 60)
    
    return results


# Export all functions
__all__ = [
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
    
    # Availability flags
    'STATSMODELS_AVAILABLE',
    'PROPHET_AVAILABLE'
]
