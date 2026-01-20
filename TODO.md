# 📋 DANH SÁCH CÔNG VIỆC - DỰ ÁN DỰ ĐOÁN HUỶ ĐẶT PHÒNG

> **Đề tài:** Dự đoán huỷ đặt phòng khách sạn
> **Dataset:** [Hotel Booking Demand - Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)  
> **Nhóm:** 12  

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

### Output figures
- `target_distribution.png`, `missing_values.png`
- `hotel_type_cancellation.png`, `lead_time_analysis.png`
- `cancellation_by_deposit.png`, `cancellation_by_segment.png`
- `monthly_trend.png`, `leakage_detection.png`
- `correlation_matrix.png`, `chi_square_results.png`

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

### 3.2. Tạo module features ✅
- [x] `src/features/__init__.py`
- [x] `src/features/builder.py`
  - [x] Rời rạc hoá `lead_time` (`discretize_lead_time()`)
  - [x] Rời rạc hoá `country` (`discretize_country()`)
  - [x] Tạo feature `total_guests` = adults + children + babies
  - [x] Tạo feature `total_nights` = stays_in_weekend_nights + stays_in_week_nights
  - [x] Tạo feature `is_repeated_guest_and_canceled_before` (`repeated_and_canceled_before`)
  - [x] Tạo feature theo mùa từ arrival_date_month (`create_season_features()`)
  - [x] Feature cho association rules (`prepare_for_association_rules()`)
  - [x] Thêm nhiều features khác: revenue, booking, room, guest history

### 3.3. Notebook 02_preprocess_feature.ipynb ✅
- [x] Tạo notebook `notebooks/02_preprocess_feature.ipynb`
- [x] Gọi cleaner để xử lý missing/outliers
- [x] Gọi builder để tạo features
- [x] Lưu dữ liệu đã xử lý vào `data/processed/`
- [x] Thống kê trước-sau tiền xử lý
- [x] Train/Test split (80/20 hoặc theo params.yaml)
- [x] Xử lý imbalance: SMOTE / class_weight / undersampling

---

## 🟢 PHASE 4: KHAI PHÁ TRI THỨC (DATA MINING)

### 4.1. Luật kết hợp (Association Rules) ✅
- [x] `src/mining/__init__.py`
- [x] `src/mining/association.py`
  - [x] Hàm chuyển đổi data sang dạng transaction (`prepare_transactions()`)
  - [x] Hàm chạy Apriori/FP-Growth (`run_apriori()`, `run_fpgrowth()`)
  - [x] Hàm trích xuất rules với support/confidence/lift (`extract_rules()`)
  - [x] Hàm lọc rules theo consequent (`filter_rules_by_consequent()`)
  - [x] Hàm so sánh rules theo nhóm (`compare_rules_by_group()`)
  - [x] Pipeline hoàn chỉnh (`mine_association_rules()`)
  - [x] Visualization functions (`plot_rules_heatmap()`, `plot_support_confidence_scatter()`)
### 4.2. Phân cụm (Clustering) ✅
- [x] `src/mining/clustering.py`
  - [x] Hàm chuẩn hoá features cho clustering (`prepare_clustering_data()`)
  - [x] Hàm KMeans với Elbow method (`run_kmeans()`, `find_optimal_k()`)
  - [x] Hàm DBSCAN (`run_dbscan()`)
  - [x] Hàm Hierarchical Clustering (`run_hierarchical()`)
  - [x] Hàm đánh giá: Silhouette Score, Davies-Bouldin Index (`evaluate_clustering()`)
  - [x] Hàm profiling cụm (`profile_clusters()`, `identify_high_risk_clusters()`)
  - [x] Visualization functions (`plot_clusters_2d()`, `plot_cluster_profiles()`, `plot_cancellation_by_cluster()`)
  - [x] Pipeline hoàn chỉnh (`cluster_bookings()`)

### 4.3. Notebook 03_mining_clustering.ipynb ✅
- [x] Tạo notebook `notebooks/03_mining_clustering.ipynb`
- [x] **Luật kết hợp:**
  - [x] Tìm rules liên quan đến `is_canceled=1`
  - [x] So sánh rules theo mùa (summer vs winter)
  - [x] So sánh rules theo quốc gia (top countries)
  - [x] Visualize top rules (heatmap, network graph)
- [x] **Phân cụm:**
  - [x] Chọn features phù hợp (lead_time, total_nights, adr, etc.)
  - [x] Tìm số cụm tối ưu (Elbow + Silhouette)
  - [x] Chạy KMeans với k tối ưu
  - [x] Profiling từng cụm
  - [x] Xác định cụm có rủi ro huỷ cao
  - [x] Visualize clusters (PCA/t-SNE 2D)

### Output
- `association_rules_scatter.png`, `cancellation_rules_heatmap.png`
- `clustering_optimal_k.png`, `kmeans_clusters_pca.png`
- `kmeans_cluster_profiles.png`, `hierarchical_clusters_pca.png`
- `association_rules_cancellation.csv`, `clustering_comparison.csv`

---

## 🔵 PHASE 5: MÔ HÌNH PHÂN LỚP (CLASSIFICATION) ✅

### 5.1. Tạo module models ✅
- [x] `src/models/__init__.py`
- [x] `src/models/supervised.py`
  - [x] Hàm train Logistic Regression (baseline 1)
  - [x] Hàm train Decision Tree (baseline 2)
  - [x] Hàm train Random Forest
  - [x] Hàm train XGBoost/LightGBM
  - [x] Hàm hyperparameter tuning (GridSearch/RandomSearch)
  - [x] Hàm predict và predict_proba

### 5.2. Tạo module evaluation ✅
- [x] `src/evaluation/__init__.py`
- [x] `src/evaluation/metrics.py`
  - [x] Hàm tính Accuracy, Precision, Recall, F1
  - [x] Hàm tính PR-AUC, ROC-AUC
  - [x] Hàm vẽ Confusion Matrix
  - [x] Hàm vẽ ROC Curve, PR Curve
  - [x] Hàm vẽ Feature Importance

### 5.3. Notebook 04_modeling.ipynb ✅
- [x] Tạo notebook `notebooks/04_modeling.ipynb`
- [x] **Baseline models:**
  - [x] Train Logistic Regression
  - [x] Train Decision Tree
- [x] **Improved models:**
  - [x] Train Random Forest với tuning
  - [x] Train XGBoost/LightGBM với tuning
- [x] **Đánh giá:**
  - [x] Bảng so sánh metrics (Accuracy, F1, PR-AUC, ROC-AUC)
  - [x] Confusion matrix cho mỗi model
  - [x] Feature importance analysis
  - [x] Cross-validation (5-fold)
- [x] **Kiểm tra leakage:**
  - [x] Verify không dùng cột reservation_status
  - [x] Verify split đúng (không data leak từ test)

---

## 🟣 PHASE 6: BÁN GIÁM SÁT (SEMI-SUPERVISED) ✅

### 6.1. Tạo module semi-supervised ✅
- [x] `src/models/semi_supervised.py`
  - [x] Hàm tạo labeled/unlabeled split (5%, 10%, 20% labeled)
  - [x] Hàm Self-Training với threshold cao (0.9, 0.95)
  - [x] Hàm Label Propagation
  - [x] Hàm Label Spreading
  - [x] Hàm phân tích pseudo-label errors

### 6.2. Notebook 04b_semi_supervised.ipynb ✅
- [x] Tạo notebook `notebooks/04b_semi_supervised.ipynb`
- [x] **Kịch bản thiếu nhãn:**
  - [x] Giữ 5% labeled → train supervised vs semi-supervised
  - [x] Giữ 10% labeled → train supervised vs semi-supervised
  - [x] Giữ 20% labeled → train supervised vs semi-supervised
- [x] **So sánh:**
  - [x] Supervised-only với ít nhãn
  - [x] Self-training (ngưỡng confidence 0.9, 0.95)
  - [x] Label Spreading
- [x] **Phân tích:**
  - [x] Learning curve theo % nhãn
  - [x] Phân tích pseudo-label sai theo lead_time dài
  - [x] Confusion matrix của pseudo-labels
  - [x] Bảng so sánh F1/PR-AUC

---

## ✅ PHASE 7: CHUỖI THỜI GIAN (TIME SERIES)

### 7.1. Tạo module forecasting
- [x] `src/models/forecasting.py`
  - [x] Hàm aggregate cancellation rate theo tháng (`prepare_time_series`)
  - [x] Hàm train ARIMA/SARIMA (`train_arima`, `train_sarima`)
  - [x] Hàm train Exponential Smoothing (`train_exponential_smoothing`)
  - [x] Hàm train Prophet (optional) (`train_prophet`)
  - [x] Hàm đánh giá MAE, RMSE, MAPE (`evaluate_forecast`)
  - [x] Hàm baseline forecasts (`naive_forecast`, `moving_average_forecast`)
  - [x] Hàm phân tích time series (`check_stationarity`, `decompose_time_series`)
  - [x] Hàm visualization (`plot_time_series`, `plot_forecast`, `plot_decomposition`)

### 7.2. Notebook Time Series Analysis
- [x] Tạo `notebooks/05_time_series.ipynb`
- [x] Aggregate data theo tháng: cancellation_rate = canceled/total
- [x] Visualize time series của cancellation rate
- [x] Phân tích stationarity (ADF test) - Series is NON-STATIONARY
- [x] Decomposition: Trend, Seasonal, Residual components
- [x] ACF/PACF analysis
- [x] Train-Test split (20 train, 6 test months)
- [x] Train models:
  - [x] Baseline: Naive, MA(3), MA(6)
  - [x] ARIMA(1,1,1), ARIMA(2,1,2)
  - [x] Exponential Smoothing
- [x] Đánh giá MAE/RMSE/MAPE
- [x] Visualize forecast vs actual
- [x] So sánh tất cả models
### 7.3. Output files
- [x] `outputs/figures/ts_cancellation_rate.png`
- [x] `outputs/figures/ts_bookings_cancellations.png`
- [x] `outputs/figures/ts_decomposition.png`
- [x] `outputs/figures/ts_acf_pacf.png`
- [x] `outputs/figures/ts_train_test_split.png`
- [x] `outputs/figures/ts_model_comparison.png`
- [x] `outputs/figures/ts_best_forecast.png`
- [x] `outputs/figures/ts_all_forecasts.png`
- [x] `outputs/tables/ts_model_comparison.csv`
- [x] `outputs/tables/ts_summary_report.txt`

---

## ✅ PHASE 8: TỔNG HỢP & BÁO CÁO - HOÀN THÀNH!

### 8.1. Tạo module visualization ✅
- [x] `src/visualization/__init__.py`
- [x] `src/visualization/plots.py`
  - [x] Hàm vẽ distribution plot
  - [x] Hàm vẽ correlation heatmap
  - [x] Hàm vẽ model comparison bar chart
  - [x] Hàm vẽ radar chart
  - [x] Hàm vẽ confusion matrix detailed
  - [x] Hàm vẽ feature importance bar
  - [x] Hàm vẽ cumulative importance
  - [x] Hàm vẽ learning curve

### 8.2. Tạo module report ✅
- [x] `src/evaluation/report.py`
  - [x] Hàm tạo bảng tổng hợp kết quả
  - [x] Hàm export figures
  - [x] Hàm export tables (CSV/JSON)
  - [x] Hàm generate_summary_report
  - [x] Hàm generate_full_report
  - [x] Hàm extract_business_insights
### 8.3. Notebook 06_evaluation_report.ipynb ✅
- [x] Tạo notebook `notebooks/06_evaluation_report.ipynb`
- [x] **Tổng hợp kết quả:**
  - [x] Bảng so sánh tất cả models (supervised, semi-supervised, time series)
  - [x] Model comparison bar chart & radar chart
  - [x] Best model selection với justification
- [x] **Phân tích lỗi:**
  - [x] Error analysis của best model
  - [x] Confusion matrix detailed
  - [x] Classification report
- [x] **Insights (9 actionable insights):**
  - [x] Insight 1: Đặc điểm booking dễ huỷ (Top 5 Features)
  - [x] Insight 2: Thời điểm rủi ro cao (Lead Time Analysis)
  - [x] Insight 3: Phân khúc khách hàng rủi ro
  - [x] Insight 4: Deposit Type Analysis
  - [x] Insight 5: Market Segment Analysis
  - [x] Insight 6: Customer Type Analysis
  - [x] Insight 7: Model Performance Insights
  - [x] Insight 8: Booking Trend Analysis
  - [x] Insight 9: Special Requests Impact
- [x] **Export outputs:**
  - [x] Lưu figures vào `outputs/figures/`
  - [x] Lưu tables vào `outputs/tables/`
  - [x] Lưu reports vào `outputs/reports/`
### 8.4. Output Files ✅
- [x] `outputs/figures/supervised_comparison_bar.png`
- [x] `outputs/figures/supervised_comparison_radar.png`
- [x] `outputs/figures/model_ranking_f1.png`
- [x] `outputs/figures/confusion_matrix_best_model.png`
- [x] `outputs/figures/error_distribution.png`
- [x] `outputs/figures/feature_importance_top15.png`
- [x] `outputs/figures/cumulative_importance.png`
- [x] `outputs/figures/lead_time_analysis.png`
- [x] `outputs/figures/monthly_trend.png`
- [x] `outputs/figures/cancellation_by_deposit.png`
- [x] `outputs/figures/cancellation_by_segment.png`
- [x] `outputs/figures/cancellation_by_customer.png`
- [x] `outputs/figures/summary_dashboard.png`
- [x] `outputs/tables/project_summary.csv`
- [x] `outputs/reports/business_insights.json`
- [x] `outputs/reports/business_insights.md`
- [x] `outputs/reports/summary_report.md`
- [x] `outputs/reports/full_report.md`
- [x] `outputs/reports/supervised_results.csv`
- [x] `outputs/reports/semi_supervised_results.csv`
- [x] `outputs/reports/time_series_results.csv`
- [x] `outputs/reports/feature_importance.csv`
---

## ✅ PHASE 9: PIPELINE & REPRODUCIBILITY

### 9.1. Tạo scripts ✅
- [x] `scripts/__init__.py` - Module init
- [x] `scripts/run_pipeline.py` - Chạy toàn bộ pipeline
  - [x] Support CLI arguments: --all, --eda, --preprocess, --mining, --modeling, --semi, --timeseries, --report
  - [x] Support --seed argument để override random seed
  - [x] Logging đầy đủ vào outputs/logs/
  - [x] Summary report sau khi chạy
- [x] `scripts/run_papermill.py` - Chạy notebooks bằng papermill
  - [x] List notebooks available
  - [x] Run specific notebook
  - [x] Run all notebooks in order
  - [x] Verify reproducibility
- [x] `scripts/verify_reproducibility.py` - Kiểm tra reproducibility
  - [x] Verify random operations
  - [x] Verify model training
  - [x] Run mini pipeline và so sánh results
  - [x] Check output file hashes

### 9.2. Kiểm tra reproducibility ✅
- [x] Chạy lại từ đầu với seed cố định (seed=42)
- [x] Verify outputs giống nhau (F1: 0.803497, Accuracy: 0.864436)
- [x] Random operations consistent (numpy, sklearn, pandas)
- [x] Model training consistent
### 9.3. Usage Examples
```bash
# Run complete pipeline
python scripts/run_pipeline.py --all --seed 42

# Run specific phase
python scripts/run_pipeline.py --modeling
python scripts/run_pipeline.py --timeseries

# Verify reproducibility
python scripts/verify_reproducibility.py --full

# Run notebooks with papermill
python scripts/run_papermill.py --list
python scripts/run_papermill.py --notebook 01
```

---

## ✅ PHASE 10: DEMO APP

### 10.1. Demo App ✅
- [x] Tạo `app/` folder với cấu trúc module
- [x] `app/__init__.py` - Module init
- [x] `app/streamlit_app.py` - Streamlit demo app
  - [x] Input form: Thông tin khách sạn, khách hàng, đặt phòng
  - [x] Load model Random Forest (Tuned)
  - [x] Dự đoán xác suất huỷ với color-coded risk level
  - [x] Hiển thị các yếu tố ảnh hưởng chính
  - [x] Khuyến nghị hành động cho khách sạn
- [x] `app/README.md` - Hướng dẫn sử dụng app

### 10.2. App Features
- **Input Form**: 3 cột với các trường thông tin booking
  - Thông tin khách sạn: Hotel type, tháng đến, lead time, số đêm
  - Thông tin khách: Số người, khách quen, loại khách hàng, quốc gia
  - Thông tin đặt phòng: Phân khúc, đặt cọc, meal, phòng, giá, yêu cầu đặc biệt
- **Output**:
  - Xác suất huỷ (%) với màu theo mức rủi ro
  - Risk Level: LOW/MEDIUM/HIGH với icons
  - Key Factors: Phân tích các yếu tố ảnh hưởng
  - Recommendations: Khuyến nghị cho khách sạn
  - Booking Summary: Tóm tắt thông tin đặt phòng
### 10.3. Run App
```bash
# Run Streamlit app
streamlit run app/streamlit_app.py
```

### 10.4. Output Files
- [x] `app/__init__.py`
- [x] `app/streamlit_app.py`
---

## ✅ PHASE 11: BÁO CÁO CUỐI CÙNG - HOÀN THÀNH!

### 11.1. Viết báo cáo
- [ ] **Phần 1:** Đặt vấn đề và phân tích yêu cầu
- [ ] **Phần 2:** Thiết kế giải pháp và quy trình khai phá
- [ ] **Phần 3:** Phân tích mã nguồn và chức năng
- [ ] **Phần 4:** Thử nghiệm và kết quả
- [ ] **Phần 5:** Thảo luận và so sánh
- [ ] **Phần 6:** Tổng kết và hướng phát triển

### 11.2. Hoàn thiện ✅
- [x] Review toàn bộ code
- [x] Clean up notebooks
- [x] Update README.md
  - [x] Cập nhật cấu trúc thư mục đầy đủ
  - [x] Thêm bảng kết quả model performance
  - [x] Thêm hướng dẫn sử dụng chi tiết
  - [x] Thêm hướng dẫn demo app
  - [x] Thêm business insights
  - [x] Thêm tech stack
  - [x] Format chuyên nghiệp với badges
- [x] Final commit và push to GitHub (ready)
- [x] Export báo cáo vào `outputs/reports/`
  - [x] `outputs/reports/final_report.md` - Báo cáo cuối cùng

## 🚀 PHASE 12: PHÁT TRIỂN MỞ RỘNG

> **Ghi chú:** Các tính năng dưới đây độc lập với nhau, có thể thực hiện theo thứ tự bất kỳ.  
> **Tập trung:** Các kỹ thuật Data Mining nâng cao và ứng dụng thực tế.

---

### 12.1. 🔍 Giải thích mô hình (SHAP/LIME)
**Mục tiêu:** Giải thích chi tiết tại sao mô hình đưa ra dự đoán - Interpretable ML

- [ ] Cài đặt `shap`, `lime`
- [ ] Tạo `src/evaluation/explainability.py`
  - [ ] `compute_shap_values()` - Tính SHAP values toàn cục & cục bộ
  - [ ] `plot_shap_summary()` - Biểu đồ tổng hợp SHAP
  - [ ] `plot_shap_waterfall()` - Biểu đồ thác nước cho từng dự đoán
  - [ ] `plot_shap_dependence()` - Biểu đồ phụ thuộc feature
  - [ ] `lime_explain_instance()` - LIME cho dự đoán đơn lẻ
- [ ] Tạo notebook `07_explainability.ipynb`
  - [ ] Tầm quan trọng đặc trưng toàn cục với SHAP
  - [ ] Giải thích cục bộ cho các trường hợp thú vị (FP, FN)
  - [ ] So sánh SHAP vs Feature Importance truyền thống
  - [ ] Phân tích tương tác giữa các đặc trưng
- [ ] Tích hợp SHAP vào Streamlit app (giải thích dự đoán)

**Đầu ra:** `shap_summary.png`, `shap_dependence_*.png`, `shap_interaction.png`

---

### 12.2. 🧠 Mô hình Deep Learning (Mạng nơ-ron)
**Mục tiêu:** Thử nghiệm Neural Network để so sánh với ML truyền thống

- [ ] Cài đặt `tensorflow` hoặc `pytorch`
- [ ] Tạo `src/models/deep_learning.py`:
  - [ ] `build_mlp_model()` - Mạng Perceptron đa tầng
  - [ ] `build_embedding_model()` - Mô hình với categorical embeddings
  - [ ] `train_nn_model()` - Vòng lặp huấn luyện
  - [ ] `evaluate_nn_model()` - Đánh giá mô hình
- [ ] Tạo notebook `08_deep_learning.ipynb`:
  - [ ] Tiền xử lý dữ liệu cho Neural Network
  - [ ] Thiết kế kiến trúc mô hình
  - [ ] Huấn luyện với early stopping
  - [ ] So sánh với Random Forest (accuracy, F1, thời gian)
  - [ ] Phân tích overfitting/underfitting
- [ ] Hyperparameter tuning với Keras Tuner

**Đầu ra:** `neural_network.h5`, `nn_training_history.png`, `nn_comparison.csv`

---

### 12.3. ⚡ Tối ưu siêu tham số (Optuna)
**Mục tiêu:** Tìm hyperparameters tối ưu một cách tự động với Bayesian Optimization

- [ ] Cài đặt `optuna`
- [ ] Tạo `src/optimization/optuna_tuner.py`:
  - [ ] `create_objective()` - Hàm mục tiêu
  - [ ] `run_optimization()` - Chạy study với nhiều trials
  - [ ] `visualize_optimization()` - Biểu đồ kết quả
  - [ ] `get_best_params()` - Lấy tham số tốt nhất
- [ ] Tạo notebook `09_hyperparameter_optimization.ipynb`:
  - [ ] Định nghĩa không gian tìm kiếm (search space)
  - [ ] Chạy 100+ trials cho Random Forest, XGBoost
  - [ ] Visualization: importance, history, contour, parallel coordinate
  - [ ] So sánh với GridSearchCV (hiệu quả, thời gian)
  - [ ] Phân tích convergence
- [ ] Pruning để tiết kiệm thời gian tính toán

**Đầu ra:** `optuna_study.db`, `optuna_importance.png`, `optuna_history.png`

---

### 12.4. 🔔 Phát hiện Data Drift & Giám sát mô hình
**Mục tiêu:** Phát hiện khi dữ liệu thay đổi và mô hình cần huấn luyện lại

- [ ] Cài đặt `evidently`, `alibi-detect`
- [ ] Tạo `src/monitoring/drift_detection.py`:
  - [ ] `detect_data_drift()` - Kiểm định thống kê (KS, Chi-square)
  - [ ] `detect_concept_drift()` - Giám sát hiệu suất theo thời gian
  - [ ] `detect_feature_drift()` - Drift từng đặc trưng
  - [ ] `generate_drift_report()` - Báo cáo HTML
- [ ] Tạo notebook `10_drift_monitoring.ipynb`:
  - [ ] Mô phỏng data drift (thay đổi phân phối)
  - [ ] Giám sát các đặc trưng quan trọng
  - [ ] Thiết lập ngưỡng cảnh báo
  - [ ] Phân tích ảnh hưởng drift đến hiệu suất mô hình
- [ ] Lên lịch kiểm tra định kỳ

**Đầu ra:** `drift_report.html`, `feature_drift.png`, `performance_over_time.png`

---

### 12.5. 🎯 Kỹ thuật xử lý mất cân bằng nâng cao
**Mục tiêu:** Thử nghiệm các kỹ thuật sampling để xử lý imbalanced data

- [ ] Cài đặt `imbalanced-learn`
- [ ] Tạo `src/data/sampling.py`:
  - [ ] `apply_smote()` - SMOTE oversampling
  - [ ] `apply_adasyn()` - ADASYN adaptive sampling
  - [ ] `apply_smoteenn()` - Kết hợp SMOTE + ENN
  - [ ] `apply_undersampling()` - Random undersampling
  - [ ] `compare_sampling_methods()` - So sánh các phương pháp
- [ ] Tạo notebook `11_imbalanced_learning.ipynb`:
  - [ ] So sánh: No sampling vs SMOTE vs ADASYN vs SMOTEENN
  - [ ] Ảnh hưởng đến Precision, Recall, F1
  - [ ] Visualization: phân phối trước-sau sampling
  - [ ] Tìm tỷ lệ sampling tối ưu

**Đầu ra:** `sampling_comparison.png`, `sampling_results.csv`

---

### 12.6. 🔗 Stacking & Voting Ensemble
**Mục tiêu:** Kết hợp nhiều mô hình để cải thiện hiệu suất

- [ ] Tạo `src/models/ensemble.py`:
  - [ ] `build_voting_classifier()` - Hard/Soft voting
  - [ ] `build_stacking_classifier()` - Stacking với meta-learner
  - [ ] `build_blending_classifier()` - Blending ensemble
  - [ ] `evaluate_ensemble()` - Đánh giá ensemble
- [ ] Tạo notebook `12_ensemble_methods.ipynb`:
  - [ ] Voting Ensemble: RF + XGBoost + LightGBM
  - [ ] Stacking với Logistic Regression làm meta-learner
  - [ ] So sánh với single best model
  - [ ] Phân tích diversity của base models
  - [ ] Cross-validation cho ensemble

**Đầu ra:** `ensemble_comparison.png`, `stacking_model.joblib`

---

### 12.7. 📊 Phân tích cụm nâng cao
**Mục tiêu:** Áp dụng thêm các thuật toán clustering và đánh giá

- [ ] Tạo `src/mining/advanced_clustering.py`:
  - [ ] `apply_gaussian_mixture()` - GMM clustering
  - [ ] `apply_spectral_clustering()` - Spectral Clustering
  - [ ] `apply_optics()` - OPTICS (density-based)
  - [ ] `find_optimal_clusters()` - Elbow, Silhouette, Gap statistic
  - [ ] `cluster_stability_analysis()` - Phân tích ổn định cụm
- [ ] Tạo notebook `13_advanced_clustering.ipynb`:
  - [ ] So sánh: KMeans vs GMM vs DBSCAN vs Spectral
  - [ ] Đánh giá: Silhouette, Calinski-Harabasz, Davies-Bouldin
  - [ ] Phân tích ổn định cụm với bootstrap
  - [ ] Profiling chi tiết từng cụm

**Đầu ra:** `clustering_comparison_advanced.png`, `cluster_stability.csv`

---

### 12.8. 🔀 Feature Selection nâng cao
**Mục tiêu:** Tìm tập đặc trưng tối ưu với các phương pháp khác nhau

- [ ] Tạo `src/features/selection.py`:
  - [ ] `recursive_feature_elimination()` - RFE
  - [ ] `boruta_selection()` - Boruta algorithm
  - [ ] `genetic_algorithm_selection()` - GA-based selection
  - [ ] `mutual_information_selection()` - MI-based
  - [ ] `compare_selection_methods()` - So sánh
- [ ] Tạo notebook `14_feature_selection.ipynb`:
  - [ ] Filter methods: Chi-square, Mutual Information
  - [ ] Wrapper methods: RFE, Forward/Backward selection
  - [ ] Embedded methods: LASSO, Tree-based importance
  - [ ] So sánh số lượng features vs hiệu suất
  - [ ] Stability của feature selection

**Đầu ra:** `feature_selection_comparison.png`, `selected_features.csv`

---

### 12.9. 📈 Phân tích chuỗi thời gian nâng cao
**Mục tiêu:** Áp dụng các mô hình time series phức tạp hơn

- [ ] Cài đặt `prophet`, `neuralprophet`
- [ ] Tạo `src/models/advanced_forecasting.py`:
  - [ ] `prophet_forecast()` - Facebook Prophet
  - [ ] `neural_prophet_forecast()` - NeuralProphet
  - [ ] `ensemble_forecast()` - Ensemble of forecasters
  - [ ] `detect_anomalies()` - Phát hiện điểm bất thường
- [ ] Tạo notebook `15_advanced_time_series.ipynb`:
  - [ ] Prophet với seasonality, holidays
  - [ ] So sánh: ARIMA vs Prophet vs NeuralProphet
  - [ ] Phát hiện anomaly trong cancellation rate
  - [ ] Dự báo dài hạn với confidence intervals
  - [ ] What-if analysis

**Đầu ra:** `prophet_forecast.png`, `anomaly_detection.png`, `forecast_comparison.csv`

---

### 12.10. 🎲 Phân tích Bayesian
**Mục tiêu:** Áp dụng Bayesian inference cho uncertainty quantification

- [ ] Cài đặt `pymc`, `arviz`
- [ ] Tạo `src/models/bayesian.py`:
  - [ ] `bayesian_logistic_regression()` - Bayesian LR
  - [ ] `posterior_predictive_check()` - Kiểm tra posterior
  - [ ] `credible_intervals()` - Khoảng tin cậy Bayesian
- [ ] Tạo notebook `16_bayesian_analysis.ipynb`:
  - [ ] Prior selection cho các tham số
  - [ ] MCMC sampling với PyMC
  - [ ] So sánh: Frequentist vs Bayesian
  - [ ] Uncertainty quantification cho predictions
  - [ ] Visualization với ArviZ

**Đầu ra:** `posterior_distribution.png`, `credible_intervals.png`

---

### 12.11. 🌐 REST API triển khai (FastAPI)
**Mục tiêu:** Expose mô hình qua REST API để tích hợp với hệ thống khác

- [ ] Cài đặt `fastapi`, `uvicorn`, `pydantic`
- [ ] Tạo `api/` folder:
  ```
  api/
  ├── main.py           # FastAPI app
  ├── schemas.py        # Pydantic models
  ├── routes/
  │   ├── predict.py    # /predict endpoint
  │   └── health.py     # /health endpoint
  └── utils.py          # Hàm hỗ trợ
  ```
- [ ] Endpoints:
  - [ ] `POST /predict` - Dự đoán đơn booking
  - [ ] `POST /predict/batch` - Dự đoán hàng loạt
  - [ ] `GET /model/info` - Thông tin mô hình
  - [ ] `GET /health` - Kiểm tra trạng thái
- [ ] API documentation với Swagger UI
- [ ] Unit tests cho API endpoints

**Chạy:** `uvicorn api.main:app --reload`

---

### 12.12. 🐳 Đóng gói Docker
**Mục tiêu:** Đóng gói ứng dụng để triển khai dễ dàng

- [ ] Tạo `Dockerfile` cho Streamlit app
- [ ] Tạo `Dockerfile.api` cho FastAPI (nếu có)
- [ ] Tạo `docker-compose.yml`:
  - [ ] Service: streamlit-app
  - [ ] Service: fastapi (tùy chọn)
  - [ ] Volume mount cho models
- [ ] Tạo `.dockerignore`
- [ ] Test build và chạy locally
- [ ] Tài liệu hướng dẫn triển khai

**Chạy:** `docker-compose up -d`

---

### 12.13. 🧪 Kiểm thử tự động (pytest)
**Mục tiêu:** Đảm bảo chất lượng code với automated tests

- [ ] Cài đặt `pytest`, `pytest-cov`
- [ ] Tạo `tests/` folder:
  ```
  tests/
  ├── __init__.py
  ├── conftest.py           # Fixtures
  ├── test_cleaner.py       # Test tiền xử lý
  ├── test_builder.py       # Test feature engineering
  ├── test_models.py        # Test huấn luyện/dự đoán
  └── test_evaluation.py    # Test metrics
  ```
- [ ] Viết tests cho các modules (≥80% coverage)
- [ ] Tạo `pytest.ini` configuration
- [ ] Coverage report ≥70%

**Chạy:** `pytest tests/ -v --cov=src --cov-report=html`

---

### 12.14. 📊 Theo dõi thí nghiệm (MLflow)
**Mục tiêu:** Theo dõi experiments, parameters, metrics có hệ thống

- [ ] Cài đặt `mlflow`
- [ ] Tạo `src/tracking/mlflow_utils.py`:
  - [ ] `log_experiment()` - Ghi params, metrics, artifacts
  - [ ] `register_model()` - Đăng ký mô hình
  - [ ] `load_production_model()` - Tải mô hình từ registry
- [ ] Tích hợp vào training notebooks:
  - [ ] Ghi hyperparameters
  - [ ] Ghi metrics (F1, ROC-AUC, v.v.)
  - [ ] Ghi confusion matrix như artifact
  - [ ] Ghi model artifacts
- [ ] MLflow UI để so sánh experiments

**Chạy:** `mlflow ui --port 5000`

---

## 📁 CẤU TRÚC THƯ MỤC HIỆN TẠI

```
Nhom12_BaiTapLon_DataMining/
├── README.md                    # Hướng dẫn
├── requirements.txt             # Dependencies
├── TODO.md                      # File này
├── blog_post.md                 # Blog kết quả
├── BaiTapLonToDo.txt           # Yêu cầu đề bài
│
├── configs/
│   └── params.yaml              # Tham số cấu hình
│
├── data/
│   └── raw/
│       └── hotel_bookings.csv   # Dataset gốc (119,390 rows)
│
├── notebooks/                   # 7 Jupyter Notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_clustering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 04b_semi_supervised.ipynb
│   ├── 05_time_series.ipynb
│   └── 06_evaluation_report.ipynb
│
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py
│   │   └── cleaner.py
│   ├── features/
│   │   └── builder.py
│   ├── mining/
│   │   ├── association.py
│   │   └── clustering.py
│   ├── models/
│   │   ├── supervised.py
│   │   ├── semi_supervised.py
│   │   └── forecasting.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── report.py
│   └── visualization/
│       └── plots.py
│
├── scripts/
│   ├── run_pipeline.py
│   ├── run_papermill.py
│   └── verify_reproducibility.py
│
├── app/
│   └── streamlit_app.py         # Demo app
│
└── outputs/
    ├── figures/                 # 52 biểu đồ PNG
    ├── tables/                  # 13 CSV files
    ├── models/                  # 7 trained models
    └── reports/                 # Reports (MD, JSON, CSV)
```

---

## ⚠️ LƯU Ý QUAN TRỌNG

1. **Data Leakage**: Đã loại bỏ `reservation_status`, `reservation_status_date`
2. **Imbalanced Data**: Sử dụng class_weight, đánh giá bằng F1/PR-AUC
3. **Reproducibility**: Seed=42, kết quả consistent
4. **Code Quality**: Notebooks gọi hàm từ src/, có docstrings

---

## 🎯 CHECKLIST TRƯỚC KHI NỘP

- [x] Tất cả notebooks chạy thành công với `run_papermill.py --all`
- [x] README.md đầy đủ hướng dẫn
- [x] requirements.txt cập nhật
- [x] Outputs đầy đủ (figures, tables, models, reports)
- [x] Demo app hoạt động
- [x] Code có comments/docstrings
- [ ] Export BaoCao.pdf
- [ ] Push to GitHub

---
