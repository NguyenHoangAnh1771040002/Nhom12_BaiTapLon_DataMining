"""
Module Tạo Báo Cáo (Đánh Giá)
==============================
(Evaluation Module - Report Generation)

Các hàm tạo báo cáo toàn diện cho dự án khai phá dữ liệu.

Các hàm chính:
--------------
- Tạo bảng tổng hợp (Summary tables)
- Báo cáo so sánh mô hình
- Xuất biểu đồ và bảng
- Tạo báo cáo Markdown/LaTeX
- Trích xuất business insights
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# SUMMARY TABLE GENERATION
# =============================================================================

def create_model_summary_table(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    round_digits: int = 4
) -> pd.DataFrame:
    """
    Create a summary table from model results.
    
    Parameters
    ----------
    results : dict
        Dictionary of model_name -> metrics dict
        e.g., {'LogisticRegression': {'accuracy': 0.85, 'f1': 0.80}}
    metrics : list
        List of metrics to include (None for all)
    round_digits : int
        Number of decimal places
        
    Returns
    -------
    pd.DataFrame
        Summary table with models as index
    """
    df = pd.DataFrame(results).T
    
    if metrics is not None:
        df = df[[m for m in metrics if m in df.columns]]
    
    df = df.round(round_digits)
    
    # Add rank column based on F1 score if available
    if 'f1' in df.columns:
        df['rank'] = df['f1'].rank(ascending=False).astype(int)
        df = df.sort_values('rank')
    
    return df


def create_classification_report_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    round_digits: int = 4
) -> pd.DataFrame:
    """
    Create classification report as a DataFrame.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : list
        Class labels
    round_digits : int
        Decimal places
        
    Returns
    -------
    pd.DataFrame
        Classification report table
    """
    from sklearn.metrics import classification_report
    
    if labels is None:
        labels = ['Not Canceled', 'Canceled']
    
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    
    df = pd.DataFrame(report).T
    df = df.round(round_digits)
    
    return df


def create_comparison_table(
    supervised_results: pd.DataFrame,
    semi_supervised_results: pd.DataFrame = None,
    time_series_results: pd.DataFrame = None,
    metric_col: str = 'f1'
) -> pd.DataFrame:
    """
    Create comprehensive comparison table across all phases.
    
    Parameters
    ----------
    supervised_results : pd.DataFrame
        Supervised learning results
    semi_supervised_results : pd.DataFrame
        Semi-supervised learning results (optional)
    time_series_results : pd.DataFrame
        Time series results (optional)
        
    Returns
    -------
    pd.DataFrame
        Combined comparison table
    """
    all_results = []
    
    # Add supervised results
    if supervised_results is not None:
        sup_df = supervised_results.copy()
        sup_df['phase'] = 'Supervised'
        sup_df['model'] = sup_df.index
        all_results.append(sup_df)
    
    # Add semi-supervised results
    if semi_supervised_results is not None:
        semi_df = semi_supervised_results.copy()
        semi_df['phase'] = 'Semi-Supervised'
        semi_df['model'] = semi_df.index
        all_results.append(semi_df)
    
    # Add time series results
    if time_series_results is not None:
        ts_df = time_series_results.copy()
        ts_df['phase'] = 'Time Series'
        ts_df['model'] = ts_df.index
        all_results.append(ts_df)
    
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.set_index(['phase', 'model'])
        return combined
    
    return pd.DataFrame()


def create_feature_importance_table(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    round_digits: int = 4
) -> pd.DataFrame:
    """
    Create feature importance table.
    
    Parameters
    ----------
    feature_names : list
        Names of features
    importances : np.ndarray
        Importance values
    top_n : int
        Number of top features
    round_digits : int
        Decimal places
        
    Returns
    -------
    pd.DataFrame
        Feature importance table
    """
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    df = df.sort_values('importance', ascending=False).head(top_n)
    df['rank'] = range(1, len(df) + 1)
    df['importance'] = df['importance'].round(round_digits)
    df['cumulative'] = df['importance'].cumsum().round(round_digits)
    
    return df.reset_index(drop=True)


def create_error_analysis_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Create error analysis table showing characteristics of errors.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    df : pd.DataFrame
        Original DataFrame
    feature_cols : list
        Feature columns to analyze
        
    Returns
    -------
    pd.DataFrame
        Error analysis summary
    """
    # Add predictions to DataFrame
    df_analysis = df.copy()
    df_analysis['y_true'] = y_true
    df_analysis['y_pred'] = y_pred
    df_analysis['error_type'] = 'Correct'
    df_analysis.loc[(y_true == 0) & (y_pred == 1), 'error_type'] = 'False Positive'
    df_analysis.loc[(y_true == 1) & (y_pred == 0), 'error_type'] = 'False Negative'
    
    # Calculate statistics for each error type
    results = []
    
    for error_type in ['Correct', 'False Positive', 'False Negative']:
        subset = df_analysis[df_analysis['error_type'] == error_type]
        
        if len(subset) == 0:
            continue
            
        stats = {'error_type': error_type, 'count': len(subset)}
        
        for col in feature_cols:
            if col in subset.columns:
                if pd.api.types.is_numeric_dtype(subset[col]):
                    stats[f'{col}_mean'] = subset[col].mean()
                    stats[f'{col}_std'] = subset[col].std()
                else:
                    # Get mode for categorical
                    mode_val = subset[col].mode()
                    stats[f'{col}_mode'] = mode_val.iloc[0] if len(mode_val) > 0 else None
        
        results.append(stats)
    
    return pd.DataFrame(results)


# =============================================================================
# BUSINESS INSIGHTS
# =============================================================================

def extract_business_insights(
    df: pd.DataFrame,
    target_col: str = 'is_canceled',
    feature_importance: pd.DataFrame = None,
    model_results: pd.DataFrame = None,
    top_n_features: int = 5
) -> List[Dict[str, str]]:
    """
    Extract actionable business insights from analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Target column
    feature_importance : pd.DataFrame
        Feature importance table
    model_results : pd.DataFrame
        Model comparison results
    top_n_features : int
        Number of top features to analyze
        
    Returns
    -------
    list
        List of insight dictionaries
    """
    insights = []
    
    # Overall cancellation rate
    cancel_rate = df[target_col].mean() * 100
    insights.append({
        'category': 'Overview',
        'title': 'Overall Cancellation Rate',
        'insight': f'Tỷ lệ hủy đặt phòng tổng thể là {cancel_rate:.1f}%.',
        'recommendation': 'Cần có chiến lược chủ động để giảm tỷ lệ hủy đặt phòng.'
    })
    
    # Lead time analysis
    if 'lead_time' in df.columns:
        high_lead_time = df[df['lead_time'] > 100][target_col].mean() * 100
        low_lead_time = df[df['lead_time'] <= 30][target_col].mean() * 100
        insights.append({
            'category': 'Lead Time',
            'title': 'Impact of Lead Time',
            'insight': f'Đặt phòng với lead time > 100 ngày có tỷ lệ hủy {high_lead_time:.1f}%, '
                      f'trong khi lead time <= 30 ngày chỉ có {low_lead_time:.1f}%.',
            'recommendation': 'Áp dụng chính sách đặt cọc cao hơn cho đặt phòng có lead time dài.'
        })
    
    # Deposit type analysis
    if 'deposit_type' in df.columns:
        deposit_cancel = df.groupby('deposit_type')[target_col].mean() * 100
        insights.append({
            'category': 'Deposit Policy',
            'title': 'Deposit Type Impact',
            'insight': f"Tỷ lệ hủy theo loại deposit: {', '.join([f'{k}: {v:.1f}%' for k, v in deposit_cancel.items()])}.",
            'recommendation': 'Khuyến khích khách hàng đặt cọc không hoàn lại để giảm tỷ lệ hủy.'
        })
    
    # Customer type analysis
    if 'customer_type' in df.columns:
        customer_cancel = df.groupby('customer_type')[target_col].mean() * 100
        highest = customer_cancel.idxmax()
        insights.append({
            'category': 'Customer Segment',
            'title': 'Customer Type Risk',
            'insight': f'Nhóm khách hàng "{highest}" có tỷ lệ hủy cao nhất ({customer_cancel[highest]:.1f}%).',
            'recommendation': f'Tập trung chương trình loyalty cho nhóm "{highest}" để giữ chân khách.'
        })
    
    # Market segment analysis
    if 'market_segment' in df.columns:
        segment_cancel = df.groupby('market_segment')[target_col].mean() * 100
        high_risk = segment_cancel[segment_cancel > cancel_rate].sort_values(ascending=False)
        if len(high_risk) > 0:
            insights.append({
                'category': 'Market Segment',
                'title': 'High-Risk Segments',
                'insight': f"Các phân khúc có rủi ro cao: {', '.join([f'{k} ({v:.1f}%)' for k, v in high_risk.head(3).items()])}.",
                'recommendation': 'Xem xét yêu cầu đặt cọc hoặc xác nhận bổ sung cho các phân khúc rủi ro cao.'
            })
    
    # Previous cancellations
    if 'previous_cancellations' in df.columns:
        has_prev_cancel = df[df['previous_cancellations'] > 0][target_col].mean() * 100
        no_prev_cancel = df[df['previous_cancellations'] == 0][target_col].mean() * 100
        insights.append({
            'category': 'Booking History',
            'title': 'Previous Cancellation Pattern',
            'insight': f'Khách có lịch sử hủy trước đó có tỷ lệ hủy {has_prev_cancel:.1f}%, '
                      f'so với {no_prev_cancel:.1f}% cho khách không có lịch sử hủy.',
            'recommendation': 'Áp dụng chính sách đặt phòng nghiêm ngặt hơn với khách có lịch sử hủy.'
        })
    
    # Feature importance insights
    if feature_importance is not None and len(feature_importance) > 0:
        top_features = feature_importance.head(top_n_features)['feature'].tolist()
        insights.append({
            'category': 'Predictive Features',
            'title': 'Key Predictive Factors',
            'insight': f"Các yếu tố dự đoán hủy quan trọng nhất: {', '.join(top_features)}.",
            'recommendation': 'Tập trung thu thập và phân tích các yếu tố này để cải thiện dự đoán.'
        })
    
    # Model performance insight
    if model_results is not None and len(model_results) > 0:
        best_model = model_results['f1'].idxmax() if 'f1' in model_results.columns else model_results.index[0]
        best_f1 = model_results.loc[best_model, 'f1'] if 'f1' in model_results.columns else None
        insights.append({
            'category': 'Model Performance',
            'title': 'Best Prediction Model',
            'insight': f'Mô hình {best_model} đạt hiệu suất cao nhất với F1-score = {best_f1:.4f}.' if best_f1 else f'Mô hình {best_model} được khuyến nghị.',
            'recommendation': 'Deploy mô hình này vào hệ thống để dự đoán và can thiệp sớm.'
        })
    
    # Seasonal patterns
    if 'arrival_date_month' in df.columns:
        monthly_cancel = df.groupby('arrival_date_month')[target_col].mean() * 100
        high_months = monthly_cancel[monthly_cancel > cancel_rate].sort_values(ascending=False)
        if len(high_months) > 0:
            insights.append({
                'category': 'Seasonality',
                'title': 'Seasonal Cancellation Patterns',
                'insight': f"Các tháng có tỷ lệ hủy cao: {', '.join([f'{k} ({v:.1f}%)' for k, v in high_months.head(3).items()])}.",
                'recommendation': 'Điều chỉnh chính sách đặt phòng và overbooking theo mùa.'
            })
    
    return insights


def format_insights_markdown(insights: List[Dict[str, str]]) -> str:
    """
    Format insights as Markdown text.
    
    Parameters
    ----------
    insights : list
        List of insight dictionaries
        
    Returns
    -------
    str
        Markdown formatted text
    """
    md_text = "# 📊 Business Insights & Recommendations\n\n"
    
    current_category = None
    
    for i, insight in enumerate(insights, 1):
        if insight['category'] != current_category:
            md_text += f"\n## {insight['category']}\n\n"
            current_category = insight['category']
        
        md_text += f"### {i}. {insight['title']}\n\n"
        md_text += f"**Insight:** {insight['insight']}\n\n"
        md_text += f"**Recommendation:** {insight['recommendation']}\n\n"
        md_text += "---\n\n"
    
    return md_text


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_table_csv(
    df: pd.DataFrame,
    filepath: str,
    index: bool = True
) -> str:
    """
    Export DataFrame to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export
    filepath : str
        Output path
    index : bool
        Include index
        
    Returns
    -------
    str
        Path to saved file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(path, index=index)
    print(f"✅ Table saved to {path}")
    
    return str(path)


def export_table_latex(
    df: pd.DataFrame,
    filepath: str,
    caption: str = None,
    label: str = None,
    index: bool = True
) -> str:
    """
    Export DataFrame to LaTeX.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export
    filepath : str
        Output path
    caption : str
        Table caption
    label : str
        LaTeX label
    index : bool
        Include index
        
    Returns
    -------
    str
        Path to saved file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    latex_str = df.to_latex(
        index=index,
        caption=caption,
        label=label,
        float_format='%.4f'
    )
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    
    print(f"✅ LaTeX table saved to {path}")
    
    return str(path)


def export_table_excel(
    tables: Dict[str, pd.DataFrame],
    filepath: str
) -> str:
    """
    Export multiple tables to Excel (multiple sheets).
    
    Parameters
    ----------
    tables : dict
        Dictionary of sheet_name -> DataFrame
    filepath : str
        Output path
        
    Returns
    -------
    str
        Path to saved file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet_name, df in tables.items():
            df.to_excel(writer, sheet_name=sheet_name[:31])  # Excel limit
    
    print(f"✅ Excel file saved to {path}")
    
    return str(path)


def export_results_json(
    results: Dict[str, Any],
    filepath: str
) -> str:
    """
    Export results to JSON.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    filepath : str
        Output path
        
    Returns
    -------
    str
        Path to saved file
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return obj
    
    converted = {k: convert(v) for k, v in results.items()}
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON results saved to {path}")
    
    return str(path)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_summary_report(
    model_results: pd.DataFrame,
    best_model_name: str,
    insights: List[Dict[str, str]],
    feature_importance: pd.DataFrame = None,
    output_dir: str = 'outputs/reports'
) -> str:
    """
    Generate comprehensive summary report in Markdown.
    
    Parameters
    ----------
    model_results : pd.DataFrame
        Model comparison results
    best_model_name : str
        Name of best model
    insights : list
        Business insights
    feature_importance : pd.DataFrame
        Feature importance (optional)
    output_dir : str
        Output directory
        
    Returns
    -------
    str
        Path to saved report
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    report = f"""# 📊 Hotel Booking Cancellation Prediction - Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Executive Summary

This report summarizes the results of the Hotel Booking Cancellation Prediction project, 
including model comparisons, feature analysis, and actionable business insights.

**Best Model:** {best_model_name}

---

## 2. Model Comparison

### Performance Metrics

{model_results.to_markdown()}

### Key Findings

- Best performing model: **{best_model_name}**
"""
    
    if 'f1' in model_results.columns:
        best_f1 = model_results.loc[best_model_name, 'f1']
        report += f"- F1-Score: **{best_f1:.4f}**\n"
    
    if 'accuracy' in model_results.columns:
        best_acc = model_results.loc[best_model_name, 'accuracy']
        report += f"- Accuracy: **{best_acc:.4f}**\n"
    
    # Feature importance section
    if feature_importance is not None:
        report += f"""

---

## 3. Feature Importance

Top 10 Most Important Features:

{feature_importance.head(10).to_markdown(index=False)}

"""
    
    # Business insights section
    report += """

---

## 4. Business Insights & Recommendations

"""
    
    for i, insight in enumerate(insights, 1):
        report += f"""
### {i}. {insight['title']}

**Category:** {insight['category']}

**Insight:** {insight['insight']}

**Recommendation:** {insight['recommendation']}

"""
    
    # Conclusion
    report += f"""

---

## 5. Conclusion

The analysis demonstrates that hotel booking cancellations can be effectively predicted 
using machine learning models. The **{best_model_name}** model achieved the best performance 
and is recommended for deployment.

### Next Steps

1. Deploy the prediction model in production environment
2. Implement automated intervention system for high-risk bookings
3. Monitor model performance and retrain periodically
4. A/B test different intervention strategies

---

*Report generated by Nhom12 Data Mining Project*
"""
    
    # Save report
    report_path = output_path / 'summary_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Summary report saved to {report_path}")
    
    return str(report_path)


def generate_full_report(
    project_name: str = "Hotel Booking Cancellation Prediction",
    supervised_results: pd.DataFrame = None,
    semi_supervised_results: pd.DataFrame = None,
    time_series_results: pd.DataFrame = None,
    best_model_name: str = None,
    insights: List[Dict[str, str]] = None,
    feature_importance: pd.DataFrame = None,
    output_dir: str = 'outputs/reports'
) -> Dict[str, str]:
    """
    Generate full project report with all components.
    
    Parameters
    ----------
    project_name : str
        Project name
    supervised_results : pd.DataFrame
        Supervised learning results
    semi_supervised_results : pd.DataFrame
        Semi-supervised results
    time_series_results : pd.DataFrame
        Time series results
    best_model_name : str
        Best model name
    insights : list
        Business insights
    feature_importance : pd.DataFrame
        Feature importance
    output_dir : str
        Output directory
        
    Returns
    -------
    dict
        Paths to generated files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = {}
    
    # Generate main report
    report = f"""# {project_name}
## Final Report

**Date:** {datetime.now().strftime('%Y-%m-%d')}
**Team:** Nhom12

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Overview](#2-data-overview)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Business Insights](#5-business-insights)
6. [Conclusion](#6-conclusion)

---

## 1. Introduction

This project aims to predict hotel booking cancellations using various 
machine learning and data mining techniques. Accurate prediction of 
cancellations can help hotels optimize their revenue management and 
reduce losses from no-shows.

---

## 2. Data Overview

The dataset contains hotel booking records with features including:
- Customer information (type, country, etc.)
- Booking details (lead time, deposit, room type)
- Stay information (duration, special requests)
- Historical data (previous cancellations, bookings)

---

## 3. Methodology

### 3.1 Data Preprocessing
- Missing value handling
- Feature engineering
- Categorical encoding
- Feature scaling

### 3.2 Modeling Approaches

"""
    
    if supervised_results is not None:
        report += """
#### Supervised Learning
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

"""
    
    if semi_supervised_results is not None:
        report += """
#### Semi-Supervised Learning
- Label Spreading
- Label Propagation
- Self-Training

"""
    
    if time_series_results is not None:
        report += """
#### Time Series Forecasting
- ARIMA / SARIMA
- Exponential Smoothing
- Moving Average

"""
    
    # Results section
    report += """
---

## 4. Results

"""
    
    if supervised_results is not None:
        report += f"""
### 4.1 Supervised Learning Results

{supervised_results.to_markdown()}

"""
    
    if semi_supervised_results is not None:
        report += f"""
### 4.2 Semi-Supervised Learning Results

{semi_supervised_results.to_markdown()}

"""
    
    if time_series_results is not None:
        report += f"""
### 4.3 Time Series Forecasting Results

{time_series_results.to_markdown()}

"""
    
    if best_model_name:
        report += f"""
### 4.4 Best Model

The best performing model is **{best_model_name}**.

"""
    
    # Feature importance
    if feature_importance is not None:
        report += f"""
### 4.5 Feature Importance

{feature_importance.head(15).to_markdown(index=False)}

"""
    
    # Business insights
    report += """
---

## 5. Business Insights

"""
    
    if insights:
        for i, insight in enumerate(insights, 1):
            report += f"""
### 5.{i} {insight['title']}

**{insight['insight']}**

*Recommendation:* {insight['recommendation']}

"""
    
    # Conclusion
    report += f"""
---

## 6. Conclusion

This project successfully developed a predictive model for hotel booking 
cancellations. The {best_model_name or 'best'} model achieved strong 
performance and provides actionable insights for hotel management.

### Key Takeaways

1. Lead time is a strong predictor of cancellation
2. Deposit type significantly affects cancellation rates
3. Customer booking history provides valuable signals
4. Seasonal patterns exist in cancellation rates

### Recommendations

1. Implement risk-based deposit policies
2. Use predictive scoring for early intervention
3. Monitor and retrain models regularly
4. A/B test intervention strategies

---

*Report generated by Nhom12 Data Mining Project*
"""
    
    # Save report
    report_path = output_path / 'full_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    generated_files['report'] = str(report_path)
    
    # Export tables
    if supervised_results is not None:
        csv_path = export_table_csv(supervised_results, output_path / 'supervised_results.csv')
        generated_files['supervised_csv'] = csv_path
    
    if semi_supervised_results is not None:
        csv_path = export_table_csv(semi_supervised_results, output_path / 'semi_supervised_results.csv')
        generated_files['semi_supervised_csv'] = csv_path
    
    if time_series_results is not None:
        csv_path = export_table_csv(time_series_results, output_path / 'time_series_results.csv')
        generated_files['time_series_csv'] = csv_path
    
    if feature_importance is not None:
        csv_path = export_table_csv(feature_importance, output_path / 'feature_importance.csv', index=False)
        generated_files['feature_importance_csv'] = csv_path
    
    print(f"\n✅ Full report generated with {len(generated_files)} files")
    
    return generated_files


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Summary tables
    'create_model_summary_table',
    'create_classification_report_table',
    'create_comparison_table',
    'create_feature_importance_table',
    'create_error_analysis_table',
    
    # Business insights
    'extract_business_insights',
    'format_insights_markdown',
    
    # Export functions
    'export_table_csv',
    'export_table_latex',
    'export_table_excel',
    'export_results_json',
    
    # Report generation
    'generate_summary_report',
    'generate_full_report'
]
