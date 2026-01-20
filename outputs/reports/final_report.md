# BÁO CÁO DỰ ÁN
## Dự đoán Huỷ Đặt Phòng Khách Sạn
### Hotel Booking Cancellation Prediction

---

**Học phần:** Khai phá Dữ liệu (Data Mining)  
**GVHD:** ThS. Lê Thị Thùy Trang  
**Nhóm:** 12  
**Ngày hoàn thành:** 30/12/2025

---

## THÀNH VIÊN NHÓM

| STT | Họ tên | MSSV |
|-----|--------|------|
| 1 | Nguyễn Hoàng Anh | 1771040002 |
| 2 | Nguyễn Trung Thành | 1771040022 |
| 3 | Trần Việt Vinh | 1771040030 |
| 4 | Nguyễn Minh Phượng | 1677030156 |

---

## MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Dataset](#2-dataset)
3. [Phương pháp thực hiện](#3-phương-pháp-thực-hiện)
4. [Kết quả](#4-kết-quả)
5. [Business Insights](#5-business-insights)
6. [Kết luận](#6-kết-luận)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1. Đặt vấn đề

Huỷ đặt phòng là vấn đề nghiêm trọng trong ngành khách sạn, gây ra:
- Mất doanh thu trực tiếp
- Khó khăn trong quản lý công suất phòng
- Ảnh hưởng đến chiến lược giá

### 1.2. Mục tiêu

- Xây dựng mô hình dự đoán booking có khả năng huỷ hay không
- Phát hiện các pattern liên quan đến huỷ đặt phòng
- Phân cụm khách hàng theo hành vi
- Dự báo tỷ lệ huỷ theo thời gian
- Xây dựng ứng dụng demo

### 1.3. Phạm vi

- **Dataset:** Hotel Booking Demand (Kaggle)
- **Các phương pháp:** Association Rules, Clustering, Classification, Semi-supervised, Time Series
- **Tech stack:** Python, scikit-learn, XGBoost, Streamlit

---

## 2. DATASET

### 2.1. Thông tin chung

| Thuộc tính | Giá trị |
|------------|---------|
| Nguồn | Kaggle - Hotel Booking Demand |
| Số dòng | 119,390 |
| Số cột | 32 |
| Target | is_canceled (0/1) |
| Tỷ lệ huỷ | 37.04% |

### 2.2. Các features quan trọng

- **lead_time:** Số ngày từ khi đặt đến nhận phòng
- **deposit_type:** Loại đặt cọc
- **market_segment:** Phân khúc thị trường
- **total_of_special_requests:** Số yêu cầu đặc biệt
- **previous_cancellations:** Số lần huỷ trước đây

### 2.3. Xử lý dữ liệu

- **Missing values:** children (4), country (488), agent (16,340), company (112,593)
- **Data leakage:** Loại bỏ reservation_status, reservation_status_date
- **Imbalanced:** Sử dụng SMOTE và class_weight

---

## 3. PHƯƠNG PHÁP THỰC HIỆN

### 3.1. EDA (Exploratory Data Analysis)

- Phân tích phân phối các features
- Kiểm tra correlation
- Phát hiện outliers và data leakage

### 3.2. Feature Engineering

- Tạo features mới: total_guests, total_nights, has_special_requests
- Discretization: lead_time categories, country grouping
- Season features từ arrival_date_month

### 3.3. Association Rules

- Thuật toán: Apriori, FP-Growth
- Metrics: Support, Confidence, Lift
- Tìm rules liên quan đến is_canceled=1

### 3.4. Clustering

- Thuật toán: KMeans, DBSCAN, Hierarchical
- Metrics: Silhouette Score, Davies-Bouldin Index
- Số cụm tối ưu: 4

### 3.5. Classification

- **Baseline:** Logistic Regression, Decision Tree
- **Improved:** Random Forest, XGBoost, LightGBM
- **Tuning:** GridSearchCV với cross-validation

### 3.6. Semi-supervised Learning

- Kịch bản: 5%, 10%, 20% labeled data
- Phương pháp: Self-training, Label Propagation

### 3.7. Time Series

- Aggregation: Cancellation rate theo tháng
- Models: ARIMA, Exponential Smoothing, Moving Average

---

## 4. KẾT QUẢ

### 4.1. Classification Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| **Random Forest (Tuned)** | **0.857** | **0.801** | **0.927** |
| XGBoost | 0.848 | 0.784 | 0.921 |
| LightGBM | 0.846 | 0.780 | 0.919 |
| Random Forest | 0.845 | 0.778 | 0.917 |
| Decision Tree | 0.791 | 0.704 | 0.775 |
| Logistic Regression | 0.789 | 0.714 | 0.860 |

### 4.2. Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | deposit_required | 19.7% |
| 2 | lead_time | 11.6% |
| 3 | agent | 11.1% |
| 4 | has_special_requests | 7.7% |
| 5 | room_type_changed | 7.2% |

### 4.3. Time Series Results

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| **MA(6)** | **0.043** | **0.053** | **10.39%** |
| MA(3) | 0.057 | 0.068 | 13.50% |
| ARIMA(1,1,1) | 0.071 | 0.081 | 16.89% |

### 4.4. Clustering Results

- **Số cụm tối ưu:** 4
- **Silhouette Score:** 0.35
- **High-risk cluster:** Cluster 2 (tỷ lệ huỷ 58%)

---

## 5. BUSINESS INSIGHTS

### Insight 1: Deposit Policy
> Yêu cầu đặt cọc giảm 60%+ rủi ro huỷ

### Insight 2: Lead Time
> Booking >100 ngày trước có tỷ lệ huỷ cao nhất (>50%)

### Insight 3: Special Requests
> Khách có yêu cầu đặc biệt ít huỷ hơn 50%

### Insight 4: Market Segment
> Groups và Online TA có rủi ro cao nhất

### Insight 5: Repeated Guests
> Khách quen ít huỷ hơn đáng kể

### Insight 6: Room Changes
> Thay đổi loại phòng tăng rủi ro huỷ

### Khuyến nghị cho khách sạn:
1. Áp dụng chính sách đặt cọc cho booking dài hạn
2. Liên hệ xác nhận với khách 48-72 giờ trước
3. Khuyến khích khách đặt yêu cầu đặc biệt
4. Chương trình loyalty cho khách quen
5. Overbooking strategy dựa trên dự đoán

---

## 6. KẾT LUẬN

### 6.1. Thành tựu

- ✅ Xây dựng pipeline khai phá dữ liệu hoàn chỉnh
- ✅ Model đạt F1-score 80%, ROC-AUC 92.7%
- ✅ 9 business insights có giá trị thực tiễn
- ✅ Demo app Streamlit hoạt động tốt
- ✅ Code reproducible với seed cố định

### 6.2. Hạn chế

- Dataset chỉ từ 2 khách sạn ở Portugal
- Thiếu thông tin về giá cạnh tranh, sự kiện đặc biệt
- Semi-supervised không cải thiện nhiều so với supervised

### 6.3. Hướng phát triển

- Thu thập thêm dữ liệu từ nhiều khách sạn/khu vực
- Tích hợp real-time prediction API
- Thử nghiệm Deep Learning models
- A/B testing các chiến lược giảm cancellation

---

## PHỤ LỤC

### A. Cấu trúc thư mục dự án

```
Nhom12_BaiTapLon_DataMining/
├── app/                    # Demo Streamlit app
├── configs/                # Configuration files
├── data/                   # Raw and processed data
├── notebooks/              # 7 Jupyter notebooks
├── outputs/                # Figures, models, reports
├── scripts/                # Pipeline scripts
└── src/                    # Source code modules
```

### B. Danh sách notebooks

1. 01_eda.ipynb - Exploratory Data Analysis
2. 02_preprocess_feature.ipynb - Preprocessing & Feature Engineering
3. 03_mining_clustering.ipynb - Association Rules & Clustering
4. 04_modeling.ipynb - Supervised Learning
5. 04b_semi_supervised.ipynb - Semi-supervised Learning
6. 05_time_series.ipynb - Time Series Forecasting
7. 06_evaluation_report.ipynb - Evaluation & Reporting

### C. Hướng dẫn chạy

```bash
# Cài đặt
pip install -r requirements.txt

# Chạy pipeline
python scripts/run_pipeline.py --all

# Chạy demo app
streamlit run app/streamlit_app.py
```

---

**© 2025 Nhóm 12 - Khai phá Dữ liệu**
