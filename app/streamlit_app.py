"""
Hotel Booking Cancellation Prediction - Streamlit Demo App
============================================================

Ứng dụng web demo dự đoán khả năng huỷ đặt phòng khách sạn.

Usage:
    streamlit run app/streamlit_app.py

Features:
    - Nhập thông tin booking
    - Dự đoán xác suất huỷ
    - Giải thích feature importance
    - Recommendations cho khách sạn
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page config
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .low-risk {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .medium-risk {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    .high-risk {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .feature-importance {
        background-color: #e9ecef;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    """Load trained model."""
    model_paths = [
        PROJECT_ROOT / 'outputs' / 'models' / 'random_forest_tuned.joblib',
        PROJECT_ROOT / 'outputs' / 'models' / 'best_model.pkl',
        PROJECT_ROOT / 'outputs' / 'models' / 'random_forest.joblib',
        PROJECT_ROOT / 'outputs' / 'models' / 'xgboost.joblib'
    ]
    
    for path in model_paths:
        if path.exists():
            try:
                model = joblib.load(path)
                return model, path.name
            except:
                continue
    
    return None, None


@st.cache_data
def load_sample_data():
    """Load sample data for reference."""
    data_path = PROJECT_ROOT / 'data' / 'raw' / 'hotel_bookings.csv'
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    return None


@st.cache_data
def get_feature_stats(_df):
    """Get feature statistics for input validation."""
    if _df is None:
        return {}
    
    stats = {
        'lead_time': {'min': 0, 'max': int(_df['lead_time'].max()), 'median': int(_df['lead_time'].median())},
        'adr': {'min': 0, 'max': float(_df['adr'].quantile(0.99)), 'median': float(_df['adr'].median())},
        'countries': sorted(_df['country'].dropna().unique().tolist())[:50],
        'market_segments': _df['market_segment'].unique().tolist(),
        'customer_types': _df['customer_type'].unique().tolist(),
        'deposit_types': _df['deposit_type'].unique().tolist(),
        'meal_types': _df['meal'].unique().tolist(),
        'room_types': sorted(_df['reserved_room_type'].unique().tolist()),
        'hotels': _df['hotel'].unique().tolist(),
    }
    return stats


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def prepare_features(input_data: dict) -> pd.DataFrame:
    """Prepare features for prediction."""
    
    # Create base dataframe
    df = pd.DataFrame([input_data])
    
    # Feature engineering
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['has_special_requests'] = (df['total_of_special_requests'] > 0).astype(int)
    df['has_booking_changes'] = (df['booking_changes'] > 0).astype(int)
    df['is_company_booking'] = 0  # Simplified
    df['has_agent'] = 1 if input_data.get('agent', 0) > 0 else 0
    
    # Deposit required
    df['deposit_required'] = (df['deposit_type'] != 'No Deposit').astype(int)
    
    # Room type changed
    df['room_type_changed'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)
    
    # Season from month
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    month_num = month_map.get(input_data['arrival_date_month'], 1)
    
    if month_num in [12, 1, 2]:
        df['season'] = 'Winter'
    elif month_num in [3, 4, 5]:
        df['season'] = 'Spring'
    elif month_num in [6, 7, 8]:
        df['season'] = 'Summer'
    else:
        df['season'] = 'Fall'
    
    # Lead time category
    lead_time = input_data['lead_time']
    if lead_time <= 7:
        df['lead_time_category'] = 'Short'
    elif lead_time <= 30:
        df['lead_time_category'] = 'Medium'
    elif lead_time <= 90:
        df['lead_time_category'] = 'Long'
    else:
        df['lead_time_category'] = 'Very Long'
    
    return df


def get_model_features(model, df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the features the model expects."""
    
    # Get expected features from model
    try:
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        elif hasattr(model, 'n_features_in_'):
            # If no names, use numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            expected_features = numeric_cols[:model.n_features_in_]
        else:
            expected_features = df.select_dtypes(include=[np.number]).columns.tolist()
    except:
        expected_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create feature dataframe
    feature_df = pd.DataFrame()
    
    for feat in expected_features:
        if feat in df.columns:
            feature_df[feat] = df[feat]
        else:
            # Try to create missing features
            feature_df[feat] = 0
    
    return feature_df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical features."""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_cancellation(model, features: pd.DataFrame):
    """Make prediction with probability."""
    try:
        # Get prediction
        prediction = model.predict(features)[0]
        
        # Get probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            cancel_prob = proba[1] if len(proba) > 1 else proba[0]
        else:
            cancel_prob = prediction
        
        return prediction, cancel_prob
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None


def get_risk_level(probability: float) -> tuple:
    """Determine risk level from probability."""
    if probability < 0.3:
        return "LOW RISK", "low-risk", "🟢"
    elif probability < 0.6:
        return "MEDIUM RISK", "medium-risk", "🟡"
    else:
        return "HIGH RISK", "high-risk", "🔴"


def get_recommendations(input_data: dict, probability: float) -> list:
    """Generate recommendations based on input and prediction."""
    recommendations = []
    
    if probability >= 0.5:
        # High risk booking
        if input_data['deposit_type'] == 'No Deposit':
            recommendations.append("💰 **Yêu cầu đặt cọc** để giảm rủi ro huỷ")
        
        if input_data['lead_time'] > 100:
            recommendations.append("📞 **Liên hệ xác nhận** 48-72 giờ trước ngày đến")
        
        if input_data['total_of_special_requests'] == 0:
            recommendations.append("🎁 **Đề xuất ưu đãi** (nâng cấp phòng, check-in sớm) để khách cam kết")
        
        if input_data['market_segment'] in ['Groups', 'Online TA']:
            recommendations.append("📋 **Áp dụng chính sách huỷ nghiêm ngặt** cho phân khúc này")
    
    if probability >= 0.3 and probability < 0.5:
        recommendations.append("👀 **Theo dõi booking** này trong danh sách cần chú ý")
        recommendations.append("📧 **Gửi email nhắc nhở** và ưu đãi đặc biệt")
    
    if probability < 0.3:
        recommendations.append("✅ **Booking ổn định** - xử lý bình thường")
        if input_data.get('is_repeated_guest', 0) == 1:
            recommendations.append("🌟 **Khách quen** - chuẩn bị welcome gift")
    
    # Always suggest
    if input_data['lead_time'] > 180:
        recommendations.append("📅 **Lead time dài** - cân nhắc overbooking strategy")
    
    return recommendations


def get_key_factors(input_data: dict, probability: float) -> list:
    """Identify key factors affecting the prediction."""
    factors = []
    
    # Lead time
    if input_data['lead_time'] > 100:
        factors.append(("Lead Time", f"{input_data['lead_time']} ngày", "⬆️ Rủi ro cao", "#dc3545"))
    elif input_data['lead_time'] < 7:
        factors.append(("Lead Time", f"{input_data['lead_time']} ngày", "⬇️ Rủi ro thấp", "#28a745"))
    
    # Deposit
    if input_data['deposit_type'] == 'No Deposit':
        factors.append(("Đặt cọc", "Không", "⬆️ Rủi ro cao", "#dc3545"))
    elif input_data['deposit_type'] == 'Non Refund':
        factors.append(("Đặt cọc", "Không hoàn", "⬇️ Rủi ro thấp", "#28a745"))
    
    # Special requests
    if input_data['total_of_special_requests'] > 0:
        factors.append(("Yêu cầu đặc biệt", str(input_data['total_of_special_requests']), "⬇️ Rủi ro thấp", "#28a745"))
    else:
        factors.append(("Yêu cầu đặc biệt", "0", "⬆️ Rủi ro cao", "#ffc107"))
    
    # Repeated guest
    if input_data.get('is_repeated_guest', 0) == 1:
        factors.append(("Khách quen", "Có", "⬇️ Rủi ro thấp", "#28a745"))
    
    # Market segment
    if input_data['market_segment'] in ['Groups', 'Online TA']:
        factors.append(("Phân khúc", input_data['market_segment'], "⬆️ Rủi ro cao", "#ffc107"))
    elif input_data['market_segment'] == 'Direct':
        factors.append(("Phân khúc", "Direct", "⬇️ Rủi ro thấp", "#28a745"))
    
    return factors


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🏨 Hotel Booking Cancellation Prediction</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 1.1rem;">
        Dự đoán khả năng huỷ đặt phòng và nhận khuyến nghị quản lý rủi ro
    </p>
    """, unsafe_allow_html=True)
    
    # Load model
    model, model_name = load_model()
    
    if model is None:
        st.error("❌ Không tìm thấy model đã train. Vui lòng chạy pipeline training trước.")
        st.info("Chạy: `python scripts/run_pipeline.py --modeling`")
        return
    
    # Load sample data for reference
    sample_df = load_sample_data()
    stats = get_feature_stats(sample_df)
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("ℹ️ Thông tin Model")
        st.success(f"**Model:** {model_name}")
        
        if hasattr(model, 'n_estimators'):
            st.info(f"**Trees:** {model.n_estimators}")
        
        st.markdown("---")
        st.header("📊 Thống kê Dataset")
        if sample_df is not None:
            st.metric("Tổng bookings", f"{len(sample_df):,}")
            cancel_rate = sample_df['is_canceled'].mean() * 100
            st.metric("Tỷ lệ huỷ", f"{cancel_rate:.1f}%")
        
        st.markdown("---")
        st.header("🎯 Model Performance")
        st.markdown("""
        - **F1-Score:** 0.8010
        - **Accuracy:** 85.7%
        - **ROC-AUC:** 0.9268
        """)
    
    # Main content
    st.markdown("---")
    
    # Input form
    st.header("📝 Nhập Thông Tin Booking")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏨 Thông tin khách sạn")
        
        hotel = st.selectbox(
            "Loại khách sạn",
            options=stats.get('hotels', ['City Hotel', 'Resort Hotel']),
            help="City Hotel hoặc Resort Hotel"
        )
        
        arrival_month = st.selectbox(
            "Tháng đến",
            options=['January', 'February', 'March', 'April', 'May', 'June',
                    'July', 'August', 'September', 'October', 'November', 'December']
        )
        
        lead_time = st.slider(
            "Lead Time (ngày)",
            min_value=0,
            max_value=500,
            value=50,
            help="Số ngày từ khi đặt đến ngày nhận phòng"
        )
        
        stays_weekend = st.number_input(
            "Đêm cuối tuần",
            min_value=0,
            max_value=10,
            value=1
        )
        
        stays_week = st.number_input(
            "Đêm trong tuần",
            min_value=0,
            max_value=20,
            value=2
        )
    
    with col2:
        st.subheader("👥 Thông tin khách")
        
        adults = st.number_input(
            "Số người lớn",
            min_value=1,
            max_value=10,
            value=2
        )
        
        children = st.number_input(
            "Số trẻ em",
            min_value=0,
            max_value=10,
            value=0
        )
        
        babies = st.number_input(
            "Số em bé",
            min_value=0,
            max_value=5,
            value=0
        )
        
        is_repeated_guest = st.checkbox("Khách quen (đã đặt trước đây)")
        
        customer_type = st.selectbox(
            "Loại khách hàng",
            options=stats.get('customer_types', ['Transient', 'Contract', 'Transient-Party', 'Group'])
        )
        
        country = st.selectbox(
            "Quốc gia",
            options=['PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'IRL', 'BEL', 'BRA', 'NLD', 'USA', 'Other'],
            index=0
        )
    
    with col3:
        st.subheader("💳 Thông tin đặt phòng")
        
        market_segment = st.selectbox(
            "Phân khúc thị trường",
            options=stats.get('market_segments', ['Online TA', 'Offline TA/TO', 'Direct', 'Corporate', 'Groups'])
        )
        
        deposit_type = st.selectbox(
            "Loại đặt cọc",
            options=stats.get('deposit_types', ['No Deposit', 'Non Refund', 'Refundable']),
            help="No Deposit = rủi ro cao hơn"
        )
        
        meal = st.selectbox(
            "Loại bữa ăn",
            options=stats.get('meal_types', ['BB', 'HB', 'FB', 'SC', 'Undefined'])
        )
        
        reserved_room_type = st.selectbox(
            "Loại phòng đặt",
            options=stats.get('room_types', ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        )
        
        assigned_room_type = st.selectbox(
            "Loại phòng được gán",
            options=stats.get('room_types', ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']),
            index=0
        )
        
        adr = st.slider(
            "Giá phòng/đêm (€)",
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            step=5.0
        )
        
        special_requests = st.number_input(
            "Số yêu cầu đặc biệt",
            min_value=0,
            max_value=5,
            value=0,
            help="Nhiều yêu cầu = ít khả năng huỷ"
        )
        
        booking_changes = st.number_input(
            "Số lần thay đổi booking",
            min_value=0,
            max_value=10,
            value=0
        )
    
    st.markdown("---")
    
    # Predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button(
            "🔮 DỰ ĐOÁN CANCELLATION",
            use_container_width=True,
            type="primary"
        )
    
    if predict_button:
        # Prepare input data
        input_data = {
            'hotel': hotel,
            'lead_time': lead_time,
            'arrival_date_month': arrival_month,
            'stays_in_weekend_nights': stays_weekend,
            'stays_in_week_nights': stays_week,
            'adults': adults,
            'children': children,
            'babies': babies,
            'meal': meal,
            'country': country,
            'market_segment': market_segment,
            'is_repeated_guest': 1 if is_repeated_guest else 0,
            'previous_cancellations': 0,
            'previous_bookings_not_canceled': 1 if is_repeated_guest else 0,
            'reserved_room_type': reserved_room_type,
            'assigned_room_type': assigned_room_type,
            'booking_changes': booking_changes,
            'deposit_type': deposit_type,
            'agent': 1,
            'days_in_waiting_list': 0,
            'customer_type': customer_type,
            'adr': adr,
            'required_car_parking_spaces': 0,
            'total_of_special_requests': special_requests,
        }
        
        # Prepare features
        with st.spinner("Đang phân tích..."):
            features_df = prepare_features(input_data)
            features_encoded = encode_categorical(features_df)
            
            # Get model features
            try:
                model_features = get_model_features(model, features_encoded)
                prediction, probability = predict_cancellation(model, model_features)
            except Exception as e:
                # Fallback: use only numeric features
                numeric_features = features_encoded.select_dtypes(include=[np.number])
                
                # Match number of features
                if hasattr(model, 'n_features_in_'):
                    n_expected = model.n_features_in_
                    if len(numeric_features.columns) < n_expected:
                        # Pad with zeros
                        for i in range(len(numeric_features.columns), n_expected):
                            numeric_features[f'feature_{i}'] = 0
                    elif len(numeric_features.columns) > n_expected:
                        numeric_features = numeric_features.iloc[:, :n_expected]
                
                prediction, probability = predict_cancellation(model, numeric_features)
        
        if probability is not None:
            st.markdown("---")
            st.header("📊 Kết Quả Dự Đoán")
            
            risk_level, risk_class, risk_emoji = get_risk_level(probability)
            
            # Main prediction display
            col_result1, col_result2 = st.columns([2, 1])
            
            with col_result1:
                st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    <h1>{risk_emoji} {probability*100:.1f}%</h1>
                    <h3>Xác suất huỷ booking</h3>
                    <h2 style="margin-top: 1rem;">{risk_level}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_result2:
                st.markdown("### 📈 Metrics")
                st.metric("Xác suất huỷ", f"{probability*100:.1f}%")
                st.metric("Xác suất giữ", f"{(1-probability)*100:.1f}%")
                st.metric("Mức độ rủi ro", risk_level)
            
            # Key factors
            st.markdown("---")
            st.header("🔍 Các Yếu Tố Ảnh Hưởng")
            
            factors = get_key_factors(input_data, probability)
            
            cols = st.columns(len(factors))
            for i, (name, value, impact, color) in enumerate(factors):
                with cols[i]:
                    st.markdown(f"""
                    <div style="background-color: {color}20; padding: 1rem; border-radius: 8px; 
                                border-left: 4px solid {color}; text-align: center;">
                        <strong>{name}</strong><br>
                        <span style="font-size: 1.5rem;">{value}</span><br>
                        <small>{impact}</small>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown("---")
            st.header("💡 Khuyến Nghị")
            
            recommendations = get_recommendations(input_data, probability)
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Summary table
            st.markdown("---")
            st.header("📋 Tóm Tắt Booking")
            
            summary_data = {
                'Thông tin': ['Khách sạn', 'Lead Time', 'Số đêm', 'Số khách', 'Giá/đêm', 
                             'Phân khúc', 'Đặt cọc', 'Yêu cầu đặc biệt'],
                'Giá trị': [
                    str(hotel),
                    f"{lead_time} ngày",
                    f"{stays_weekend + stays_week} đêm ({stays_weekend} cuối tuần)",
                    f"{adults} người lớn, {children} trẻ em, {babies} em bé",
                    f"€{adr:.2f}",
                    str(market_segment),
                    str(deposit_type),
                    str(special_requests)
                ]
            }
            
            st.table(pd.DataFrame(summary_data))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p>🎓 Hotel Booking Cancellation Prediction - Data Mining Project</p>
        <p>Model: Random Forest (Tuned) | F1-Score: 0.8010 | Accuracy: 85.7%</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
