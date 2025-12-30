# 🏨 Hotel Booking Cancellation Prediction

> **Đề tài 12:** Dự đoán huỷ đặt phòng khách sạn  
> **Học phần:** Khai phá dữ liệu (Data Mining)  
> **GVHD:** ThS. Lê Thị Thùy Trang  
> **Nhóm:** 12

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](#)

---

## 📋 Mục lục

- [Mô tả dự án](#-mô-tả-dự-án)
- [Dataset](#-dataset)
- [Kết quả](#-kết-quả)
- [Cấu trúc thư mục](#️-cấu-trúc-thư-mục)
- [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Demo App](#-demo-app)
- [Thành viên nhóm](#-thành-viên-nhóm)

---

## 📖 Mô tả dự án

Dự án xây dựng **hệ thống khai phá dữ liệu toàn diện** để dự đoán và phân tích hành vi huỷ đặt phòng khách sạn:

| # | Phương pháp | Mô tả |
|---|-------------|-------|
| 1 | **Luật kết hợp (Association Rules)** | Tìm các combo thuộc tính liên quan đến huỷ booking |
| 2 | **Phân cụm (Clustering)** | Nhóm bookings theo hành vi, xác định cụm rủi ro cao |
| 3 | **Phân lớp (Classification)** | Dự đoán khách có huỷ phòng hay không |
| 4 | **Bán giám sát (Semi-supervised)** | Thử nghiệm với kịch bản thiếu nhãn (5%, 10%, 20%) |
| 5 | **Chuỗi thời gian (Time Series)** | Dự báo tỷ lệ huỷ theo tháng |
| 6 | **Demo App** | Ứng dụng Streamlit dự đoán cancellation |

---

## 📊 Dataset

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn** | [Hotel Booking Demand - Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |
| **File** | `data/raw/hotel_bookings.csv` |
| **Số dòng** | 119,390 bookings |
| **Số cột** | 32 features |
| **Target** | `is_canceled` (0: Không huỷ, 1: Huỷ) |
| **Tỷ lệ huỷ** | 37.04% (Imbalanced) |

### Các features quan trọng:
- `lead_time`: Số ngày từ khi đặt đến ngày nhận phòng
- `deposit_type`: Loại đặt cọc (No Deposit, Non Refund, Refundable)
- `market_segment`: Phân khúc thị trường
- `total_of_special_requests`: Số yêu cầu đặc biệt
- `previous_cancellations`: Số lần huỷ trước đây

---

## 📈 Kết quả

### 🏆 Model Performance Summary

| Phase | Best Model | Metric | Score |
|-------|------------|--------|-------|
| **Supervised Learning** | Random Forest (Tuned) | F1-Score | **0.8010** |
| **Supervised Learning** | Random Forest (Tuned) | ROC-AUC | **0.9268** |
| **Supervised Learning** | Random Forest (Tuned) | Accuracy | **85.7%** |
| **Semi-Supervised** | Supervised (10% labeled) | F1-Score | 0.6817 |
| **Time Series** | Moving Average (6) | MAPE | **10.39%** |

### 📊 Supervised Models Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest (Tuned)** | **0.857** | **0.815** | **0.788** | **0.801** | **0.927** |
| XGBoost | 0.848 | 0.791 | 0.777 | 0.784 | 0.921 |
| LightGBM | 0.846 | 0.785 | 0.775 | 0.780 | 0.919 |
| Random Forest | 0.845 | 0.793 | 0.764 | 0.778 | 0.917 |
| Decision Tree | 0.791 | 0.717 | 0.691 | 0.704 | 0.775 |
| Logistic Regression | 0.789 | 0.691 | 0.738 | 0.714 | 0.860 |

### 🔑 Top 5 Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `deposit_required` | 19.7% |
| 2 | `lead_time` | 11.6% |
| 3 | `agent` | 11.1% |
| 4 | `has_special_requests` | 7.7% |
| 5 | `room_type_changed` | 7.2% |

---

## 🗂️ Cấu trúc thư mục

```
Nhom12_BaiTapLon_DataMining/
│
├── 📁 app/                          # Demo Application
│   ├── __init__.py
│   ├── streamlit_app.py             # Streamlit web app
│   └── README.md                    # App documentation
│
├── 📁 configs/
│   └── params.yaml                  # Tham số cấu hình (seed, paths, hyperparams)
│
├── 📁 data/
│   ├── raw/                         # Dữ liệu gốc
│   │   └── hotel_bookings.csv       # Dataset từ Kaggle
│   └── processed/                   # Dữ liệu đã xử lý
│
├── 📁 notebooks/                    # Jupyter Notebooks
│   ├── 01_eda.ipynb                 # Khám phá dữ liệu
│   ├── 02_preprocess_feature.ipynb  # Tiền xử lý & Feature Engineering
│   ├── 03_mining_clustering.ipynb   # Association Rules & Clustering
│   ├── 04_modeling.ipynb            # Supervised Learning
│   ├── 04b_semi_supervised.ipynb    # Semi-supervised Learning
│   ├── 05_time_series.ipynb         # Time Series Forecasting
│   └── 06_evaluation_report.ipynb   # Tổng hợp & Báo cáo
│
├── 📁 src/                          # Source Code Modules
│   ├── __init__.py
│   ├── data/                        # Data loading & cleaning
│   │   ├── loader.py                # Load dataset
│   │   └── cleaner.py               # Handle missing, outliers, encoding
│   ├── features/                    # Feature engineering
│   │   └── builder.py               # Create new features
│   ├── mining/                      # Data mining algorithms
│   │   ├── association.py           # Apriori, FP-Growth
│   │   └── clustering.py            # KMeans, DBSCAN, Hierarchical
│   ├── models/                      # Machine learning models
│   │   ├── supervised.py            # LR, DT, RF, XGBoost, LightGBM
│   │   ├── semi_supervised.py       # Self-training, Label Propagation
│   │   └── forecasting.py           # ARIMA, Exponential Smoothing
│   ├── evaluation/                  # Evaluation metrics & reports
│   │   ├── metrics.py               # Accuracy, F1, ROC-AUC, etc.
│   │   └── report.py                # Generate reports
│   └── visualization/               # Plotting functions
│       └── plots.py                 # Various plot utilities
│
├── 📁 scripts/                      # Automation Scripts
│   ├── __init__.py
│   ├── run_pipeline.py              # Run full pipeline
│   ├── run_papermill.py             # Run notebooks programmatically
│   └── verify_reproducibility.py    # Verify reproducibility
│
├── 📁 outputs/                      # Generated Outputs
│   ├── figures/                     # 47 visualization files
│   ├── tables/                      # CSV result tables
│   ├── models/                      # 7 trained models (.joblib, .pkl)
│   └── reports/                     # Markdown & JSON reports
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── TODO.md                          # Project task tracking
└── .gitignore                       # Git ignore rules
```

---

## 🚀 Hướng dẫn cài đặt

### 1. Clone repository
```bash
git clone https://github.com/NguyenHoangAnh1771040002/Nhom12_BaiTapLon_DataMining.git
cd Nhom12_BaiTapLon_DataMining
```

### 2. Tạo môi trường ảo

**Cách 1: Sử dụng Conda (Khuyến nghị)**
```bash
conda create -n hotel-booking python=3.10 -y
conda activate hotel-booking
```

**Cách 2: Sử dụng venv**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị dữ liệu
1. Tải dataset từ [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
2. Đặt file `hotel_bookings.csv` vào thư mục `data/raw/`

---

## 📖 Hướng dẫn sử dụng

### 🔹 Cách 1: Chạy từng Notebook

Mở và chạy các notebooks theo thứ tự trong Jupyter/VS Code:

```
1. notebooks/01_eda.ipynb                 # Khám phá dữ liệu
2. notebooks/02_preprocess_feature.ipynb  # Tiền xử lý
3. notebooks/03_mining_clustering.ipynb   # Khai phá tri thức
4. notebooks/04_modeling.ipynb            # Train models
5. notebooks/04b_semi_supervised.ipynb    # Semi-supervised
6. notebooks/05_time_series.ipynb         # Time series
7. notebooks/06_evaluation_report.ipynb   # Tổng hợp kết quả
```

### 🔹 Cách 2: Chạy Pipeline tự động

```bash
# Chạy toàn bộ pipeline
python scripts/run_pipeline.py --all

# Chạy từng phase riêng biệt
python scripts/run_pipeline.py --eda
python scripts/run_pipeline.py --modeling
python scripts/run_pipeline.py --timeseries

# Chạy với seed cụ thể
python scripts/run_pipeline.py --all --seed 42
```

### 🔹 Cách 3: Chạy Notebooks bằng Papermill

```bash
# Xem danh sách notebooks
python scripts/run_papermill.py --list

# Chạy notebook cụ thể
python scripts/run_papermill.py --notebook 01

# Chạy tất cả notebooks
python scripts/run_papermill.py --all
```

### 🔹 Verify Reproducibility

```bash
python scripts/verify_reproducibility.py --full
```

---

## 🎯 Demo App

Dự án bao gồm ứng dụng web Streamlit để dự đoán khả năng huỷ đặt phòng.

### Khởi chạy App

```bash
# Activate environment
conda activate hotel-booking

# Run Streamlit app
streamlit run app/streamlit_app.py
```

App sẽ mở tại: **http://localhost:8501**

### Tính năng

| Feature | Mô tả |
|---------|-------|
| **Input Form** | Nhập thông tin booking (hotel, lead_time, guests, deposit, etc.) |
| **Prediction** | Xác suất huỷ (%) với risk level (LOW/MEDIUM/HIGH) |
| **Key Factors** | Phân tích các yếu tố ảnh hưởng chính |
| **Recommendations** | Khuyến nghị hành động cho khách sạn |

---

## 📂 Output Files

### 📊 Figures (47 files)
- `outputs/figures/target_distribution.png` - Phân phối target
- `outputs/figures/feature_importance_rf.png` - Feature importance
- `outputs/figures/confusion_matrix_best_model.png` - Confusion matrix
- `outputs/figures/roc_curves_comparison.png` - ROC curves
- `outputs/figures/ts_all_forecasts.png` - Time series forecasts

### 🤖 Models (7 files)
- `outputs/models/random_forest_tuned.joblib` - **Best model**
- `outputs/models/xgboost.joblib`
- `outputs/models/lightgbm.joblib`
- `outputs/models/decision_tree.joblib`
- `outputs/models/logistic_regression.joblib`

### 📝 Reports
- `outputs/reports/full_report.md` - Báo cáo đầy đủ
- `outputs/reports/summary_report.md` - Báo cáo tóm tắt
- `outputs/reports/business_insights.md` - Business insights

---

## 💡 Business Insights

Dự án đã rút ra **9 insights quan trọng** cho khách sạn:

1. **Deposit Policy**: Yêu cầu đặt cọc giảm 60%+ rủi ro huỷ
2. **Lead Time**: Booking >100 ngày trước có tỷ lệ huỷ cao nhất
3. **Special Requests**: Khách có yêu cầu đặc biệt ít huỷ hơn 50%
4. **Market Segment**: Groups và Online TA có rủi ro cao nhất
5. **Room Changes**: Thay đổi loại phòng tăng rủi ro huỷ
6. **Repeated Guests**: Khách quen ít huỷ hơn đáng kể
7. **Seasonality**: Tỷ lệ huỷ cao vào mùa hè
8. **Agent Bookings**: Booking qua agent có pattern khác biệt
9. **Prediction**: Model đạt 80% F1-score, hỗ trợ quyết định hiệu quả

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.10 |
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM |
| **Visualization** | matplotlib, seaborn |
| **Time Series** | statsmodels |
| **Association Rules** | mlxtend |
| **Web App** | Streamlit |
| **Notebooks** | Jupyter, papermill |

---

## 👥 Thành viên nhóm

| STT | Họ tên | MSSV | Vai trò |
|-----|--------|------|---------|
| 1 | Nguyễn Hoàng Anh | 1771040002 | Team Leader |
| 2 | Nguyễn Trung Thành | 1771040022 | Developer |
| 3 | Trần Việt Vinh | 1771040030 | Developer |
| 4 | Nguyễn Minh Phượng | 1677030156 | Developer |

---

## 📚 References

- [Hotel Booking Demand Dataset - Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📝 License

Dự án này được thực hiện cho **mục đích học tập** tại môn Khai phá dữ liệu.

---

<div align="center">

**⭐ Nếu dự án hữu ích, hãy cho chúng tôi một star! ⭐**

Made with ❤️ by **Nhóm 12**

</div>
