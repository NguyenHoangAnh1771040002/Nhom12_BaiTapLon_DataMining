# 🏨 Dự Đoán Huỷ Đặt Phòng Khách Sạn
# Hotel Booking Cancellation Prediction

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
- [Tập dữ liệu](#-tập-dữ-liệu)
- [Kết quả](#-kết-quả)
- [Cấu trúc thư mục](#️-cấu-trúc-thư-mục)
- [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Ứng dụng Demo](#-ứng-dụng-demo)
- [Thành viên nhóm](#-thành-viên-nhóm)

---

## 📖 Mô tả dự án

Dự án xây dựng **hệ thống khai phá dữ liệu toàn diện** để dự đoán và phân tích hành vi huỷ đặt phòng khách sạn:

| # | Phương pháp | Mô tả |
|---|-------------|-------|
| 1 | **Luật kết hợp (Association Rules)** | Tìm các combo thuộc tính liên quan đến huỷ đặt phòng |
| 2 | **Phân cụm (Clustering)** | Nhóm đặt phòng theo hành vi, xác định cụm rủi ro cao |
| 3 | **Phân lớp (Classification)** | Dự đoán khách có huỷ phòng hay không |
| 4 | **Bán giám sát (Semi-supervised)** | Thử nghiệm với kịch bản thiếu nhãn (5%, 10%, 20%) |
| 5 | **Chuỗi thời gian (Time Series)** | Dự báo tỷ lệ huỷ theo tháng |
| 6 | **Ứng dụng Demo** | Ứng dụng Streamlit dự đoán huỷ đặt phòng |

---

## 📊 Tập dữ liệu (Dataset)

| Thuộc tính | Giá trị |
|------------|---------|
| **Nguồn** | [Hotel Booking Demand - Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |
| **File** | `data/raw/hotel_bookings.csv` |
| **Số dòng** | 119,390 lượt đặt phòng |
| **Số cột** | 32 đặc trưng (features) |
| **Biến mục tiêu** | `is_canceled` (0: Không huỷ, 1: Huỷ) |
| **Tỷ lệ huỷ** | 37.04% (Mất cân bằng - Imbalanced) |

### Các đặc trưng quan trọng:
- `lead_time`: Số ngày từ khi đặt đến ngày nhận phòng (Thời gian đặt trước)
- `deposit_type`: Loại đặt cọc (No Deposit, Non Refund, Refundable)
- `market_segment`: Phân khúc thị trường
- `total_of_special_requests`: Số yêu cầu đặc biệt
- `previous_cancellations`: Số lần huỷ trước đây

---

## 📈 Kết quả

### 🏆 Tóm tắt hiệu suất mô hình (Model Performance Summary)

| Giai đoạn | Mô hình tốt nhất | Chỉ số | Điểm |
|-----------|------------------|--------|------|
| **Học có giám sát** | Random Forest (Đã tinh chỉnh) | F1-Score | **0.8010** |
| **Học có giám sát** | Random Forest (Đã tinh chỉnh) | ROC-AUC | **0.9268** |
| **Học có giám sát** | Random Forest (Đã tinh chỉnh) | Độ chính xác | **85.7%** |
| **Học bán giám sát** | Supervised (10% có nhãn) | F1-Score | 0.6817 |
| **Chuỗi thời gian** | Trung bình trượt MA(6) | MAPE | **10.39%** |

### 📊 So sánh các mô hình học có giám sát (Supervised Models Comparison)

| Mô hình | Độ chính xác | Precision | Recall | F1-Score | ROC-AUC |
|---------|--------------|-----------|--------|----------|---------|
| **Random Forest (Đã tinh chỉnh)** | **0.857** | **0.815** | **0.788** | **0.801** | **0.927** |
| XGBoost | 0.848 | 0.791 | 0.777 | 0.784 | 0.921 |
| LightGBM | 0.846 | 0.785 | 0.775 | 0.780 | 0.919 |
| Random Forest | 0.845 | 0.793 | 0.764 | 0.778 | 0.917 |
| Cây quyết định (Decision Tree) | 0.791 | 0.717 | 0.691 | 0.704 | 0.775 |
| Hồi quy Logistic | 0.789 | 0.691 | 0.738 | 0.714 | 0.860 |

### 🔑 Top 5 đặc trưng quan trọng nhất (Top 5 Important Features)

| Hạng | Đặc trưng | Mức độ quan trọng |
|------|-----------|-------------------|
| 1 | `deposit_required` (Yêu cầu đặt cọc) | 19.7% |
| 2 | `lead_time` (Thời gian đặt trước) | 11.6% |
| 3 | `agent` (Đại lý đặt phòng) | 11.1% |
| 4 | `has_special_requests` (Có yêu cầu đặc biệt) | 7.7% |
| 5 | `room_type_changed` (Thay đổi loại phòng) | 7.2% |

---

## 🗂️ Cấu trúc thư mục

```
Nhom12_BaiTapLon_DataMining/
│
├── 📁 app/                          # Ứng dụng Demo
│   ├── __init__.py
│   └── streamlit_app.py             # Ứng dụng web Streamlit
│
├── 📁 configs/
│   └── params.yaml                  # Tham số cấu hình (seed, paths, hyperparams)
│
├── 📁 data/
│   ├── raw/                         # Dữ liệu gốc
│   │   └── hotel_bookings.csv       # Tập dữ liệu từ Kaggle
│   └── processed/                   # Dữ liệu đã xử lý
│
├── 📁 notebooks/                    # Jupyter Notebooks
│   ├── 01_eda.ipynb                 # Khám phá dữ liệu (EDA)
│   ├── 02_preprocess_feature.ipynb  # Tiền xử lý & Kỹ thuật đặc trưng
│   ├── 03_mining_clustering.ipynb   # Luật kết hợp & Phân cụm
│   ├── 04_modeling.ipynb            # Học có giám sát
│   ├── 04b_semi_supervised.ipynb    # Học bán giám sát
│   ├── 05_time_series.ipynb         # Dự báo chuỗi thời gian
│   └── 06_evaluation_report.ipynb   # Tổng hợp & Báo cáo
│
├── 📁 src/                          # Mã nguồn (Source Code)
│   ├── __init__.py
│   ├── data/                        # Đọc & làm sạch dữ liệu
│   │   ├── loader.py                # Đọc tập dữ liệu
│   │   └── cleaner.py               # Xử lý thiếu, ngoại lai, mã hóa
│   ├── features/                    # Kỹ thuật đặc trưng
│   │   └── builder.py               # Tạo đặc trưng mới
│   ├── mining/                      # Thuật toán khai phá dữ liệu
│   │   ├── association.py           # Apriori, FP-Growth
│   │   └── clustering.py            # KMeans, DBSCAN, Phân cấp
│   ├── models/                      # Mô hình học máy
│   │   ├── supervised.py            # LR, DT, RF, XGBoost, LightGBM
│   │   ├── semi_supervised.py       # Self-training, Label Propagation
│   │   └── forecasting.py           # ARIMA, Exponential Smoothing
│   ├── evaluation/                  # Đánh giá & báo cáo
│   │   ├── metrics.py               # Accuracy, F1, ROC-AUC, v.v.
│   │   └── report.py                # Tạo báo cáo
│   └── visualization/               # Trực quan hóa
│       └── plots.py                 # Các hàm vẽ đồ thị
│
├── 📁 scripts/                      # Script tự động hóa
│   ├── __init__.py
│   ├── run_pipeline.py              # Chạy toàn bộ pipeline
│   ├── run_papermill.py             # Chạy notebooks tự động
│   └── verify_reproducibility.py    # Kiểm tra tính tái lập
│
├── 📁 outputs/                      # Kết quả đầu ra
│   ├── figures/                     # 47 file hình ảnh
│   ├── tables/                      # Bảng kết quả CSV
│   ├── models/                      # 7 mô hình đã huấn luyện (.joblib, .pkl)
│   └── reports/                     # Báo cáo Markdown & JSON
│
├── README.md                        # File này
├── requirements.txt                 # Thư viện Python cần thiết
├── TODO.md                          # Theo dõi tiến độ dự án
└── .gitignore                       # Quy tắc Git ignore
```

---

## 🚀 Hướng dẫn cài đặt

### 1. Clone repository (Sao chép kho mã nguồn)
```bash
git clone https://github.com/NguyenHoangAnh1771040002/Nhom12_BaiTapLon_DataMining.git
cd Nhom12_BaiTapLon_DataMining
```

### 2. Tạo môi trường ảo (Virtual Environment)

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
1. Tải tập dữ liệu từ [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
2. Đặt file `hotel_bookings.csv` vào thư mục `data/raw/`

---

## 📖 Hướng dẫn sử dụng

### 🔹 Cách 1: Chạy từng Notebook

Mở và chạy các notebooks theo thứ tự trong Jupyter/VS Code:

```
1. notebooks/01_eda.ipynb                 # Khám phá dữ liệu
2. notebooks/02_preprocess_feature.ipynb  # Tiền xử lý
3. notebooks/03_mining_clustering.ipynb   # Khai phá tri thức
4. notebooks/04_modeling.ipynb            # Huấn luyện mô hình
5. notebooks/04b_semi_supervised.ipynb    # Học bán giám sát
6. notebooks/05_time_series.ipynb         # Chuỗi thời gian
7. notebooks/06_evaluation_report.ipynb   # Tổng hợp kết quả
```

### 🔹 Cách 2: Chạy Pipeline tự động

```bash
# Chạy toàn bộ pipeline
python scripts/run_pipeline.py --all

# Chạy từng giai đoạn riêng biệt
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

### 🔹 Kiểm tra tính tái lập (Verify Reproducibility)

```bash
python scripts/verify_reproducibility.py --full
```

---

## 🎯 Ứng dụng Demo

Dự án bao gồm ứng dụng web Streamlit để dự đoán khả năng huỷ đặt phòng.

### Khởi chạy ứng dụng

```bash
# Kích hoạt môi trường
conda activate hotel-booking

# Chạy ứng dụng Streamlit
streamlit run app/streamlit_app.py
```

Ứng dụng sẽ mở tại: **http://localhost:8501**

### Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| **Form nhập liệu** | Nhập thông tin đặt phòng (khách sạn, thời gian đặt trước, số khách, đặt cọc, v.v.) |
| **Dự đoán** | Xác suất huỷ (%) với mức độ rủi ro (THẤP/TRUNG BÌNH/CAO) |
| **Yếu tố chính** | Phân tích các yếu tố ảnh hưởng chính |
| **Khuyến nghị** | Đề xuất hành động cho khách sạn |

---

## 📂 Các file đầu ra (Output Files)

### 📊 Hình ảnh (47 files)
- `outputs/figures/target_distribution.png` - Phân phối biến mục tiêu
- `outputs/figures/feature_importance_rf.png` - Độ quan trọng đặc trưng
- `outputs/figures/confusion_matrix_best_model.png` - Ma trận nhầm lẫn
- `outputs/figures/roc_curves_comparison.png` - Đường cong ROC
- `outputs/figures/ts_all_forecasts.png` - Dự báo chuỗi thời gian

### 🤖 Mô hình (7 files)
- `outputs/models/random_forest_tuned.joblib` - **Mô hình tốt nhất**
- `outputs/models/xgboost.joblib`
- `outputs/models/lightgbm.joblib`
- `outputs/models/decision_tree.joblib`
- `outputs/models/logistic_regression.joblib`

### 📝 Báo cáo
- `outputs/reports/final_report.md` - Báo cáo đầy đủ
- `outputs/reports/business_insights.json` - Thông tin kinh doanh (JSON)

---

## 💡 Thông tin kinh doanh (Business Insights)

Dự án đã rút ra **9 insights quan trọng** cho khách sạn:

1. **Chính sách đặt cọc**: Yêu cầu đặt cọc giảm 60%+ rủi ro huỷ
2. **Thời gian đặt trước**: Đặt phòng >100 ngày trước có tỷ lệ huỷ cao nhất
3. **Yêu cầu đặc biệt**: Khách có yêu cầu đặc biệt ít huỷ hơn 50%
4. **Phân khúc thị trường**: Groups và Online TA có rủi ro cao nhất
5. **Thay đổi phòng**: Thay đổi loại phòng tăng rủi ro huỷ
6. **Khách quen**: Khách quay lại ít huỷ hơn đáng kể
7. **Mùa vụ**: Tỷ lệ huỷ cao vào mùa hè
8. **Đặt qua đại lý**: Booking qua agent có pattern khác biệt
9. **Dự đoán**: Mô hình đạt 80% F1-score, hỗ trợ quyết định hiệu quả

---

## 🛠️ Công nghệ sử dụng (Tech Stack)

| Danh mục | Công nghệ |
|----------|-----------|
| **Ngôn ngữ** | Python 3.10 |
| **Xử lý dữ liệu** | pandas, numpy |
| **Học máy** | scikit-learn, XGBoost, LightGBM |
| **Trực quan hóa** | matplotlib, seaborn |
| **Chuỗi thời gian** | statsmodels |
| **Luật kết hợp** | mlxtend |
| **Ứng dụng web** | Streamlit |
| **Notebooks** | Jupyter, papermill |

---

## 👥 Thành viên nhóm

| STT | Họ tên | MSSV |
|-----|--------|------|
| 1 | Nguyễn Hoàng Anh | 1771040002 |
| 2 | Nguyễn Trung Thành | 1771040022 |
| 3 | Trần Việt Vinh | 1771040030 |
| 4 | Nguyễn Minh Phượng | 1677030156 |

---

## 📚 Tài liệu tham khảo (References)

- [Hotel Booking Demand Dataset - Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- [Tài liệu scikit-learn](https://scikit-learn.org/stable/)
- [Tài liệu XGBoost](https://xgboost.readthedocs.io/)
- [Tài liệu Streamlit](https://docs.streamlit.io/)

---

## 📝 Giấy phép (License)

Dự án này được thực hiện cho **mục đích học tập** tại môn Khai phá dữ liệu.

---

<div align="center">

**⭐ Nếu dự án hữu ích, hãy cho chúng tôi một star! ⭐**

Made with ❤️ by **Nhóm 12**

</div>
