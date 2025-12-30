# 📋 DANH SÁCH CÔNG VIỆC - DỰ ÁN DỰ ĐOÁN HUỶ ĐẶT PHÒNG

> **Đề tài:** Dự đoán huỷ đặt phòng khách sạn (Hotel Booking Cancellation Prediction)  
> **Dataset:** [Hotel Booking Demand - Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)  
---

## 🔴 PHASE 1: THIẾT LẬP DỰ ÁN

### 1.1. Tạo cấu trúc thư mục
- [x] Tạo thư mục `configs/`
- [x] Tạo thư mục `data/processed/`
- [x] Tạo thư mục `notebooks/`
- [x] Tạo thư mục `src/` với các submodule:
  - [x] `src/data/`
  - [x] `src/features/`
  - [x] `src/mining/`
  - [x] `src/models/`
  - [x] `src/evaluation/`
  - [x] `src/visualization/`
- [x] Tạo thư mục `scripts/`
- [x] Tạo thư mục `outputs/` (figures, tables, models, reports)

### 1.2. Tạo các file cấu hình
- [x] Tạo `README.md` - Mô tả dự án, hướng dẫn cài đặt và chạy
- [x] Tạo `requirements.txt` - Danh sách thư viện cần thiết
- [x] Tạo `.gitignore` - Loại trừ data lớn, cache, outputs
- [x] Tạo `configs/params.yaml` - Tham số: seed, split ratio, paths, hyperparams
- [x] Tạo `src/__init__.py` và các `__init__.py` cho submodules
- [x] Tạo `outputs/` subfolders (figures, tables, models, reports)

### 1.3. Chuẩn bị dữ liệu
- [x] Tải dataset từ Kaggle (nếu chưa có)
- [x] Đặt file `hotel_bookings.csv` vào `data/raw/`
- [x] Kiểm tra file có đọc được không

---

## � THÔNG TIN DATASET

| Thuộc tính | Giá trị |
|------------|---------|
| **Số dòng** | 119,390 |
| **Số cột** | 32 |
| **Dung lượng** | ~94 MB |
| **Target** | `is_canceled` (0: Không huỷ, 1: Huỷ) |
| **Tỷ lệ huỷ** | 37.04% (44,224 / 119,390) → **Imbalanced** |

### Các cột trong dataset:

| # | Cột | Kiểu | Missing | Ghi chú |
|---|-----|------|---------|---------|
| 1 | `hotel` | object | 0 | Resort Hotel / City Hotel |
| 2 | `is_canceled` | int64 | 0 | **TARGET** (0/1) |
| 3 | `lead_time` | int64 | 0 | Số ngày từ đặt đến nhận phòng |
| 4 | `arrival_date_year` | int64 | 0 | Năm (2015-2017) |
| 5 | `arrival_date_month` | object | 0 | Tháng |
| 6 | `arrival_date_week_number` | int64 | 0 | Tuần trong năm |
| 7 | `arrival_date_day_of_month` | int64 | 0 | Ngày trong tháng |
| 8 | `stays_in_weekend_nights` | int64 | 0 | Số đêm cuối tuần |
| 9 | `stays_in_week_nights` | int64 | 0 | Số đêm trong tuần |
| 10 | `adults` | int64 | 0 | Số người lớn |
| 11 | `children` | float64 | **4** | Số trẻ em |
| 12 | `babies` | int64 | 0 | Số em bé |
| 13 | `meal` | object | 0 | Loại bữa ăn |
| 14 | `country` | object | **488** | Quốc gia |
| 15 | `market_segment` | object | 0 | Phân khúc thị trường |
| 16 | `distribution_channel` | object | 0 | Kênh phân phối |
| 17 | `is_repeated_guest` | int64 | 0 | Khách quay lại (0/1) |
| 18 | `previous_cancellations` | int64 | 0 | Số lần huỷ trước |
| 19 | `previous_bookings_not_canceled` | int64 | 0 | Số lần đặt không huỷ |
| 20 | `reserved_room_type` | object | 0 | Loại phòng đặt |
| 21 | `assigned_room_type` | object | 0 | Loại phòng được gán |
| 22 | `booking_changes` | int64 | 0 | Số lần thay đổi |
| 23 | `deposit_type` | object | 0 | Loại đặt cọc |
| 24 | `agent` | float64 | **16,340** | ID đại lý |
| 25 | `company` | float64 | **112,593** | ID công ty (94% missing!) |
| 26 | `days_in_waiting_list` | int64 | 0 | Số ngày chờ |
| 27 | `customer_type` | object | 0 | Loại khách hàng |
| 28 | `adr` | float64 | 0 | Giá phòng trung bình/đêm |
| 29 | `required_car_parking_spaces` | int64 | 0 | Số chỗ đỗ xe |
| 30 | `total_of_special_requests` | int64 | 0 | Số yêu cầu đặc biệt |
| 31 | `reservation_status` | object | 0 | ⚠️ **DATA LEAKAGE** |
| 32 | `reservation_status_date` | object | 0 | ⚠️ **DATA LEAKAGE** |

### ⚠️ Vấn đề cần xử lý:
1. **Data Leakage**: `reservation_status` chứa kết quả (Check-Out/Canceled/No-Show) → PHẢI DROP
2. **Missing Values**: `children` (4), `country` (488), `agent` (16,340), `company` (112,593)
3. **Imbalanced**: 37% huỷ vs 63% không huỷ → Cần SMOTE/class_weight
4. **Cột `company`**: 94% missing → Xem xét DROP

---

## 🟠 PHASE 2: KHÁM PHÁ DỮ LIỆU (EDA)

### 2.1. Tạo module loader
- [x] `src/data/__init__.py`
- [x] `src/data/loader.py` - Hàm đọc dữ liệu, kiểm tra schema

### 2.2. Notebook 01_eda.ipynb ✅
- [x] Tạo notebook `notebooks/01_eda.ipynb`
- [x] **Thống kê tổng quan:**
  - [x] Shape, dtypes, memory usage
  - [x] Số lượng missing values mỗi cột
  - [x] Thống kê mô tả (describe)
- [x] **Data Dictionary:**
  - [x] Giải thích ý nghĩa từng cột
  - [x] Xác định biến target: `is_canceled`
  - [x] Phân loại: numerical vs categorical
- [x] **Phân tích phân phối:**
  - [x] Biểu đồ 1: Phân phối target (is_canceled) - Kiểm tra imbalance
  - [x] Biểu đồ 2: Phân phối lead_time
  - [x] Biểu đồ 3: Tỷ lệ huỷ theo hotel type
  - [x] Biểu đồ 4: Tỷ lệ huỷ theo tháng/mùa
  - [x] Biểu đồ 5: Tỷ lệ huỷ theo market_segment
  - [x] Biểu đồ 6: Tỷ lệ huỷ theo country (top 10)
- [x] **Phân tích tương quan:**
  - [x] Correlation matrix cho numerical features
  - [x] Chi-square test cho categorical vs target
- [x] **Phát hiện vấn đề:**
  - [x] Xác định các cột có DATA LEAKAGE (reservation_status, etc.)
  - [x] Xác định outliers
  - [x] Xác định các cột cần drop/transform

---

## 🟡 PHASE 3: TIỀN XỬ LÝ & FEATURE ENGINEERING

### 3.1. Tạo module cleaner ✅
- [x] `src/data/cleaner.py`
  - [x] Hàm xử lý missing values (`handle_missing_values()`)
  - [x] Hàm xử lý outliers (`handle_outliers()`, `handle_adr_outliers()`)
  - [x] Hàm loại bỏ cột leakage (`drop_leakage_columns()`)
  - [x] Hàm encoding categorical variables (`encode_categorical()`)
  - [x] Hàm scaling numerical features (`scale_numerical()`)
  - [x] Pipeline hoàn chỉnh (`clean_data()`)
  - [x] Save/load artifacts (`save_artifacts()`, `load_artifacts()`)

### 3.2. Tạo module features
- [ ] `src/features/__init__.py`
- [ ] `src/features/builder.py`
  - [ ] Rời rạc hoá `lead_time` (bins: short/medium/long)
  - [ ] Rời rạc hoá `country` (top countries + others)
  - [ ] Tạo feature `total_guests` = adults + children + babies
  - [ ] Tạo feature `total_nights` = stays_in_weekend_nights + stays_in_week_nights
  - [ ] Tạo feature `is_repeated_guest_and_canceled_before`
  - [ ] Tạo feature theo mùa từ arrival_date_month
  - [ ] Feature cho association rules (one-hot encoding)

### 3.3. Notebook 02_preprocess_feature.ipynb
- [ ] Tạo notebook `notebooks/02_preprocess_feature.ipynb`
- [ ] Gọi cleaner để xử lý missing/outliers
- [ ] Gọi builder để tạo features
- [ ] Lưu dữ liệu đã xử lý vào `data/processed/`
- [ ] Thống kê trước-sau tiền xử lý
- [ ] Train/Test split (80/20 hoặc theo params.yaml)
- [ ] Xử lý imbalance: SMOTE / class_weight / undersampling

---

## 🟢 PHASE 4: KHAI PHÁ TRI THỨC (DATA MINING)

### 4.1. Luật kết hợp (Association Rules)
- [ ] `src/mining/__init__.py`
- [ ] `src/mining/association.py`
  - [ ] Hàm chuyển đổi data sang dạng transaction
  - [ ] Hàm chạy Apriori/FP-Growth
  - [ ] Hàm trích xuất rules với support/confidence/lift

### 4.2. Phân cụm (Clustering)
- [ ] `src/mining/clustering.py`
  - [ ] Hàm chuẩn hoá features cho clustering
  - [ ] Hàm KMeans với Elbow method
  - [ ] Hàm DBSCAN
  - [ ] Hàm Hierarchical Clustering
  - [ ] Hàm đánh giá: Silhouette Score, Davies-Bouldin Index
  - [ ] Hàm profiling cụm (cluster interpretation)

### 4.3. Notebook 03_mining_or_clustering.ipynb
- [ ] Tạo notebook `notebooks/03_mining_or_clustering.ipynb`
- [ ] **Luật kết hợp:**
  - [ ] Tìm rules liên quan đến `is_canceled=1`
  - [ ] So sánh rules theo mùa (summer vs winter)
  - [ ] So sánh rules theo quốc gia (top countries)
  - [ ] Visualize top rules (heatmap, network graph)
- [ ] **Phân cụm:**
  - [ ] Chọn features phù hợp (lead_time, total_nights, adr, etc.)
  - [ ] Tìm số cụm tối ưu (Elbow + Silhouette)
  - [ ] Chạy KMeans với k tối ưu
  - [ ] Profiling từng cụm
  - [ ] Xác định cụm có rủi ro huỷ cao
  - [ ] Visualize clusters (PCA/t-SNE 2D)

---

## 🔵 PHASE 5: MÔ HÌNH PHÂN LỚP (CLASSIFICATION)

### 5.1. Tạo module models
- [ ] `src/models/__init__.py`
- [ ] `src/models/supervised.py`
  - [ ] Hàm train Logistic Regression (baseline 1)
  - [ ] Hàm train Decision Tree (baseline 2)
  - [ ] Hàm train Random Forest
  - [ ] Hàm train XGBoost/LightGBM
  - [ ] Hàm hyperparameter tuning (GridSearch/RandomSearch)
  - [ ] Hàm predict và predict_proba

### 5.2. Tạo module evaluation
- [ ] `src/evaluation/__init__.py`
- [ ] `src/evaluation/metrics.py`
  - [ ] Hàm tính Accuracy, Precision, Recall, F1
  - [ ] Hàm tính PR-AUC, ROC-AUC
  - [ ] Hàm vẽ Confusion Matrix
  - [ ] Hàm vẽ ROC Curve, PR Curve
  - [ ] Hàm vẽ Feature Importance

### 5.3. Notebook 04_modeling.ipynb
- [ ] Tạo notebook `notebooks/04_modeling.ipynb`
- [ ] **Baseline models:**
  - [ ] Train Logistic Regression
  - [ ] Train Decision Tree
- [ ] **Improved models:**
  - [ ] Train Random Forest với tuning
  - [ ] Train XGBoost/LightGBM với tuning
- [ ] **Đánh giá:**
  - [ ] Bảng so sánh metrics (Accuracy, F1, PR-AUC, ROC-AUC)
  - [ ] Confusion matrix cho mỗi model
  - [ ] Feature importance analysis
  - [ ] Cross-validation (5-fold)
- [ ] **Kiểm tra leakage:**
  - [ ] Verify không dùng cột reservation_status
  - [ ] Verify split đúng (không data leak từ test)

---

## 🟣 PHASE 6: BÁN GIÁM SÁT (SEMI-SUPERVISED)

### 6.1. Tạo module semi-supervised
- [ ] `src/models/semi_supervised.py`
  - [ ] Hàm tạo labeled/unlabeled split (5%, 10%, 20% labeled)
  - [ ] Hàm Self-Training với threshold cao
  - [ ] Hàm Label Propagation
  - [ ] Hàm Label Spreading
  - [ ] Hàm phân tích pseudo-label errors

### 6.2. Notebook 04b_semi_supervised.ipynb
- [ ] Tạo notebook `notebooks/04b_semi_supervised.ipynb`
- [ ] **Kịch bản thiếu nhãn:**
  - [ ] Giữ 5% labeled → train supervised vs semi-supervised
  - [ ] Giữ 10% labeled → train supervised vs semi-supervised
  - [ ] Giữ 20% labeled → train supervised vs semi-supervised
- [ ] **So sánh:**
  - [ ] Supervised-only với ít nhãn
  - [ ] Self-training (ngưỡng confidence 0.9, 0.95)
  - [ ] Label Spreading
- [ ] **Phân tích:**
  - [ ] Learning curve theo % nhãn
  - [ ] Phân tích pseudo-label sai theo lead_time dài
  - [ ] Confusion matrix của pseudo-labels
  - [ ] Bảng so sánh F1/PR-AUC

---

## ⚫ PHASE 7: CHUỖI THỜI GIAN (TIME SERIES)

### 7.1. Tạo module forecasting
- [ ] `src/models/forecasting.py`
  - [ ] Hàm aggregate cancellation rate theo tháng
  - [ ] Hàm train ARIMA/SARIMA
  - [ ] Hàm train Prophet (optional)
  - [ ] Hàm đánh giá MAE, RMSE

### 7.2. Thêm vào Notebook hoặc tạo riêng
- [ ] Aggregate data theo tháng: cancellation_rate = canceled/total
- [ ] Visualize time series của cancellation rate
- [ ] Train model dự báo
- [ ] Đánh giá MAE/RMSE
- [ ] Visualize forecast vs actual

---

## 🔶 PHASE 8: TỔNG HỢP & BÁO CÁO

### 8.1. Tạo module visualization
- [ ] `src/visualization/__init__.py`
- [ ] `src/visualization/plots.py`
  - [ ] Hàm vẽ distribution plot
  - [ ] Hàm vẽ correlation heatmap
  - [ ] Hàm vẽ model comparison bar chart
  - [ ] Hàm vẽ learning curve

### 8.2. Tạo module report
- [ ] `src/evaluation/report.py`
  - [ ] Hàm tạo bảng tổng hợp kết quả
  - [ ] Hàm export figures
  - [ ] Hàm export tables (CSV/LaTeX)

### 8.3. Notebook 05_evaluation_report.ipynb
- [ ] Tạo notebook `notebooks/05_evaluation_report.ipynb`
- [ ] **Tổng hợp kết quả:**
  - [ ] Bảng so sánh tất cả models
  - [ ] Best model selection với justification
- [ ] **Phân tích lỗi:**
  - [ ] Error analysis của best model
  - [ ] Các trường hợp FP/FN phổ biến
- [ ] **Insights (≥5 actionable insights):**
  - [ ] Insight 1: Đặc điểm booking dễ huỷ
  - [ ] Insight 2: Thời điểm rủi ro cao
  - [ ] Insight 3: Phân khúc khách hàng rủi ro
  - [ ] Insight 4: Khuyến nghị cho khách sạn
  - [ ] Insight 5: Chiến lược giảm tỷ lệ huỷ
- [ ] **Export outputs:**
  - [ ] Lưu figures vào `outputs/figures/`
  - [ ] Lưu tables vào `outputs/tables/`
  - [ ] Lưu trained models vào `outputs/models/`

---

## 🔷 PHASE 9: PIPELINE & REPRODUCIBILITY

### 9.1. Tạo scripts
- [ ] `scripts/run_pipeline.py` - Chạy toàn bộ pipeline
- [ ] `scripts/run_papermill.py` - Chạy notebooks bằng papermill (optional)

### 9.2. Kiểm tra reproducibility
- [ ] Chạy lại từ đầu với seed cố định
- [ ] Verify outputs giống nhau
- [ ] Test trên máy khác (nếu có)

---

## 🌟 PHASE 10: ĐIỂM THƯỞNG (OPTIONAL)

### 10.1. Demo App
- [ ] Tạo `app/` hoặc `demo/` folder
- [ ] Streamlit app để predict cancellation
- [ ] Input: Thông tin booking
- [ ] Output: Xác suất huỷ + giải thích

---

## 📝 PHASE 11: BÁO CÁO CUỐI CÙNG

### 11.1. Viết báo cáo
- [ ] **Phần 1:** Đặt vấn đề và phân tích yêu cầu
- [ ] **Phần 2:** Thiết kế giải pháp và quy trình khai phá
- [ ] **Phần 3:** Phân tích mã nguồn và chức năng
- [ ] **Phần 4:** Thử nghiệm và kết quả
- [ ] **Phần 5:** Thảo luận và so sánh
- [ ] **Phần 6:** Tổng kết và hướng phát triển

### 11.2. Hoàn thiện
- [ ] Review toàn bộ code
- [ ] Clean up notebooks (remove unnecessary outputs)
- [ ] Update README.md
- [ ] Final commit và push to GitHub
- [ ] Export báo cáo PDF vào `outputs/reports/`

---

## ⚠️ LƯU Ý QUAN TRỌNG

1. **Data Leakage**: KHÔNG được sử dụng các cột sau để train:
   - `reservation_status` (chứa thông tin huỷ)
   - `reservation_status_date`
   - Các cột có thông tin sau khi đặt phòng

2. **Imbalanced Data**: Dataset có thể không cân bằng, cần:
   - Kiểm tra tỷ lệ is_canceled
   - Sử dụng SMOTE/class_weight nếu cần
   - Dùng PR-AUC thay vì ROC-AUC

3. **Reproducibility**: 
   - Luôn set random seed
   - Ghi rõ hyperparameters
   - Lưu model và kết quả

4. **Code Quality**:
   - Notebook chỉ gọi hàm từ src/
   - Comment đầy đủ
   - Docstring cho mỗi function

---
