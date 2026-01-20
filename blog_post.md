# 🏨 Dự Đoán Huỷ Đặt Phòng Khách Sạn Với Học Máy
## Ứng Dụng Khai Phá Dữ Liệu Trong Ngành Khách Sạn

![Bảng tổng hợp kết quả](outputs/figures/summary_dashboard.png)

---
**👥 Tác giả:** Nhóm 12 - Lớp Khai phá Dữ liệu  
---

## 📌 Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Khám phá dữ liệu](#2-khám-phá-dữ-liệu)
3. [Khai phá luật kết hợp](#3-khai-phá-luật-kết-hợp)
4. [Phân cụm khách hàng](#4-phân-cụm-khách-hàng)
5. [Xây dựng mô hình dự đoán](#5-xây-dựng-mô-hình-dự-đoán)
6. [Học bán giám sát](#6-học-bán-giám-sát)
7. [Dự báo chuỗi thời gian](#7-dự-báo-chuỗi-thời-gian)
8. [Kết luận và khuyến nghị](#8-kết-luận-và-khuyến-nghị)

---

## 1. Giới thiệu

### 🎯 Bài toán

Trong ngành khách sạn, **huỷ đặt phòng** là một vấn đề nan giải gây ra nhiều hệ lụy:

- 💸 **Mất doanh thu trực tiếp** khi phòng trống không có khách
- 📊 **Khó quản lý công suất** do không biết chính xác số lượng đặt phòng thực tế
- 💰 **Ảnh hưởng chiến lược giá** và việc đặt quá số phòng

> **Mục tiêu:** Xây dựng mô hình Học máy dự đoán đặt phòng nào có khả năng bị huỷ, từ đó đưa ra các chiến lược phòng ngừa hiệu quả.

### 📊 Bộ dữ liệu

Chúng tôi sử dụng bộ dữ liệu **Nhu cầu đặt phòng khách sạn** từ Kaggle với:

| Thông tin | Giá trị |
|-----------|---------|
| 📁 Số bản ghi | 119.390 lượt đặt phòng |
| 📋 Số thuộc tính | 32 đặc trưng |
| 🎯 Biến mục tiêu | is_canceled (0: không huỷ / 1: huỷ) |
| ⚖️ Tỷ lệ huỷ | 37,04% |
| 🏨 Loại khách sạn | Khách sạn thành phố & Khu nghỉ dưỡng |

---

## 2. Khám phá dữ liệu

*Phân tích dữ liệu khám phá (Exploratory Data Analysis - EDA)*

### 2.1. Phân bố nhãn mục tiêu

![Phân bố biến mục tiêu](outputs/figures/target_distribution.png)

Bộ dữ liệu có tỷ lệ huỷ **37,04%** - đây là tỷ lệ khá cao và cũng tương đối cân bằng (không quá mất cân bằng), giúp việc huấn luyện mô hình thuận lợi hơn.

### 2.2. Giá trị thiếu

![Giá trị thiếu](outputs/figures/missing_values.png)

Các cột có giá trị thiếu đáng kể:
- `company` (công ty): 94,3% thiếu → chuyển thành danh mục "không có công ty"
- `agent` (đại lý): 13,7% thiếu → điền bằng giá trị trung vị
- `country` (quốc gia): 0,4% thiếu → điền bằng giá trị phổ biến nhất

### 2.3. Phân tích theo loại khách sạn

![Tỷ lệ huỷ theo loại khách sạn](outputs/figures/hotel_type_cancellation.png)

**Khách sạn thành phố** có tỷ lệ huỷ cao hơn đáng kể so với **Khu nghỉ dưỡng**:
- Khách sạn thành phố: ~42% tỷ lệ huỷ
- Khu nghỉ dưỡng: ~28% tỷ lệ huỷ

### 2.4. Thời gian đặt trước - Yếu tố quan trọng nhất

![Phân tích thời gian đặt trước](outputs/figures/lead_time_analysis.png)

**Thời gian đặt trước** - số ngày từ khi đặt đến ngày nhận phòng - là một trong những đặc trưng quan trọng nhất:

- Đặt phòng **>100 ngày** trước có tỷ lệ huỷ **>50%**
- Đặt phòng **<7 ngày** trước có tỷ lệ huỷ thấp nhất (~20%)

> 💡 **Phát hiện:** Đặt phòng càng xa ngày nhận phòng, khả năng huỷ càng cao

### 2.5. Ảnh hưởng của loại đặt cọc

![Tỷ lệ huỷ theo loại đặt cọc](outputs/figures/cancellation_by_deposit.png)

**Loại đặt cọc** có ảnh hưởng mạnh nhất đến quyết định huỷ:

| Loại đặt cọc | Tỷ lệ huỷ |
|--------------|-----------|
| Không đặt cọc (No Deposit) | ~30% |
| Không hoàn tiền (Non Refund) | ~99% |
| Có thể hoàn tiền (Refundable) | ~22% |

> ⚠️ **Lưu ý:** Loại "Không hoàn tiền" có tỷ lệ huỷ cao bất thường - có thể do cách ghi nhận dữ liệu

### 2.6. Phân khúc thị trường

![Tỷ lệ huỷ theo phân khúc](outputs/figures/cancellation_by_segment.png)

- **Đại lý du lịch trực tuyến (Online TA)**: Tỷ lệ huỷ cao nhất
- **Đặt trực tiếp (Direct)**: Tỷ lệ huỷ thấp hơn
- **Nhóm (Groups)**: Số lượng ít nhưng tỷ lệ huỷ cao

### 2.7. Loại khách hàng

![Tỷ lệ huỷ theo loại khách](outputs/figures/cancellation_by_customer.png)

**Khách quen** có tỷ lệ huỷ thấp hơn đáng kể so với **khách mới**.

### 2.8. Xu hướng theo tháng

![Xu hướng theo tháng](outputs/figures/monthly_trend.png)

Tỷ lệ huỷ có xu hướng biến động theo mùa:
- **Mùa cao điểm** (hè): Số đặt phòng tăng, tỷ lệ huỷ cũng tăng
- **Mùa thấp điểm** (đông): Ổn định hơn

### 2.9. Phát hiện rò rỉ dữ liệu

![Phát hiện rò rỉ dữ liệu](outputs/figures/leakage_detection.png)

Chúng tôi phát hiện và loại bỏ các đặc trưng gây **rò rỉ dữ liệu (data leakage)**:
- `reservation_status` (trạng thái đặt phòng): Trực tiếp tiết lộ kết quả (Đã huỷ/Đã nhận phòng)
- `reservation_status_date` (ngày cập nhật trạng thái): Ngày cập nhật trạng thái

---

## 3. Khai phá luật kết hợp

*Tìm kiếm các quy luật ẩn trong dữ liệu (Association Rules Mining)*

### 3.1. Khai phá luật kết hợp

Sử dụng thuật toán **Apriori** và **FP-Growth** để tìm các luật kết hợp liên quan đến việc huỷ đặt phòng.

![Biểu đồ phân tán luật kết hợp](outputs/figures/association_rules_scatter.png)

Biểu đồ thể hiện mối quan hệ giữa **Độ hỗ trợ (Support)**, **Độ tin cậy (Confidence)** và **Độ nâng (Lift)** của các luật được phát hiện.

### 3.2. Bản đồ nhiệt các luật quan trọng

![Bản đồ nhiệt luật huỷ phòng](outputs/figures/cancellation_rules_heatmap.png)

**Các luật kết hợp hàng đầu:**

| Luật | Độ tin cậy | Độ nâng |
|------|------------|---------|
| Không đặt cọc + Đại lý trực tuyến → Huỷ | 85% | 2,3 |
| Đặt trước >90 ngày + Không đặt cọc → Huỷ | 78% | 2,1 |
| Khách sạn thành phố + Không có yêu cầu đặc biệt → Huỷ | 65% | 1,8 |

> 💡 **Phát hiện:** Kết hợp nhiều yếu tố rủi ro làm tăng đáng kể khả năng huỷ

---

## 4. Phân cụm khách hàng

*Nhóm khách hàng theo hành vi đặt phòng (Customer Clustering)*

### 4.1. Xác định số cụm tối ưu

![Số cụm tối ưu](outputs/figures/clustering_optimal_k.png)

Sử dụng phương pháp **Khuỷu tay (Elbow)** và **Điểm Silhouette** để xác định số cụm tối ưu: **K = 4**

### 4.2. Kết quả phân cụm KMeans

![Phân cụm KMeans trên PCA](outputs/figures/kmeans_clusters_pca.png)

Trực quan hóa 4 cụm khách hàng trên không gian PCA 2 chiều:

### 4.3. Hồ sơ các cụm

![Hồ sơ các cụm](outputs/figures/kmeans_cluster_profiles.png)

**Đặc điểm từng cụm:**

| Cụm | Mô tả | Tỷ lệ huỷ |
|-----|-------|-----------|
| **0** | Khách đặt ngắn hạn, có yêu cầu đặc biệt | ~25% |
| **1** | Khách đặt trung hạn, đặt trực tiếp | ~32% |
| **2** | Khách đặt dài hạn, qua đại lý trực tuyến | **~58%** ⚠️ |
| **3** | Khách quen, có đặt cọc | ~18% |

![Tỷ lệ huỷ theo cụm](outputs/figures/kmeans_cancellation_by_cluster.png)

> 💡 **Phát hiện:** Cụm 2 là nhóm khách hàng rủi ro cao nhất - cần có chiến lược đặc biệt

### 4.4. Phân cụm phân cấp

![Phân cụm phân cấp](outputs/figures/hierarchical_clusters_pca.png)

So sánh với **Phân cụm phân cấp (Hierarchical Clustering)** cho kết quả tương tự, khẳng định tính ổn định của phân cụm.

---

## 5. Xây dựng mô hình dự đoán

*Học có giám sát - Phân loại nhị phân (Supervised Learning - Binary Classification)*

### 5.1. So sánh các mô hình

![So sánh mô hình](outputs/figures/model_comparison.png)

Chúng tôi thử nghiệm 6 mô hình phân loại:

| Mô hình | Độ chính xác | Độ chính xác dương | Độ nhạy | Điểm F1 | ROC-AUC |
|---------|--------------|-------------------|---------|---------|---------|
| Hồi quy Logistic | 0,789 | 0,724 | 0,705 | 0,714 | 0,860 |
| Cây quyết định | 0,791 | 0,711 | 0,698 | 0,704 | 0,775 |
| Rừng ngẫu nhiên | 0,845 | 0,802 | 0,756 | 0,778 | 0,917 |
| XGBoost | 0,848 | 0,812 | 0,758 | 0,784 | 0,921 |
| LightGBM | 0,846 | 0,809 | 0,753 | 0,780 | 0,919 |
| **Rừng ngẫu nhiên (Tinh chỉnh)** | **0,857** | **0,833** | **0,772** | **0,801** | **0,927** |

### 5.2. Xếp hạng mô hình

![Xếp hạng theo F1](outputs/figures/model_ranking_f1.png)

**🏆 Mô hình tốt nhất: Rừng ngẫu nhiên (Random Forest) đã tinh chỉnh**
- Điểm F1: **0,801**
- ROC-AUC: **0,927**

### 5.3. Biểu đồ radar so sánh

![Biểu đồ radar so sánh](outputs/figures/supervised_comparison_radar.png)

### 5.4. Ma trận nhầm lẫn

#### Mô hình tốt nhất - Rừng ngẫu nhiên (Tinh chỉnh)

![Ma trận nhầm lẫn mô hình tốt nhất](outputs/figures/confusion_matrix_best_model.png)

#### Các mô hình khác

| Hồi quy Logistic | Cây quyết định |
|:----------------:|:--------------:|
| ![Ma trận LR](outputs/figures/cm_logistic_regression.png) | ![Ma trận DT](outputs/figures/cm_decision_tree.png) |

| Rừng ngẫu nhiên | XGBoost | LightGBM |
|:---------------:|:-------:|:--------:|
| ![Ma trận RF](outputs/figures/cm_random_forest.png) | ![Ma trận XGB](outputs/figures/cm_xgboost.png) | ![Ma trận LGB](outputs/figures/cm_lightgbm.png) |

### 5.5. Đường cong ROC

![So sánh đường cong ROC](outputs/figures/roc_curves_comparison.png)

Tất cả mô hình tổ hợp (Rừng ngẫu nhiên, XGBoost, LightGBM) đều có ROC-AUC > 0,9, thể hiện khả năng phân loại tốt.

### 5.6. Đường cong Precision-Recall

![So sánh đường cong PR](outputs/figures/pr_curves_comparison.png)

Đường cong PR quan trọng với bài toán mất cân bằng - Rừng ngẫu nhiên tinh chỉnh cho kết quả tốt nhất.

### 5.7. Độ quan trọng đặc trưng

![Top 15 đặc trưng quan trọng](outputs/figures/feature_importance_top15.png)

**Top 5 đặc trưng quan trọng nhất:**

1. **deposit_required** (19,7%) - Yêu cầu đặt cọc
2. **lead_time** (11,6%) - Thời gian đặt trước
3. **agent** (11,1%) - Đại lý đặt phòng
4. **has_special_requests** (7,7%) - Có yêu cầu đặc biệt
5. **room_type_changed** (7,2%) - Thay đổi loại phòng

#### Độ quan trọng đặc trưng - Rừng ngẫu nhiên

![Độ quan trọng đặc trưng RF](outputs/figures/feature_importance_rf.png)

#### Độ quan trọng tích luỹ

![Độ quan trọng tích luỹ](outputs/figures/cumulative_importance.png)

> 💡 **Phát hiện:** Top 10 đặc trưng đóng góp ~75% sức mạnh dự đoán

### 5.8. Phân tích ngưỡng quyết định

![Phân tích ngưỡng](outputs/figures/threshold_analysis.png)

Phân tích ngưỡng quyết định để tối ưu sự đánh đổi giữa Độ chính xác dương và Độ nhạy theo nhu cầu kinh doanh.

### 5.9. Phân tích lỗi

![Phân bố lỗi](outputs/figures/error_distribution.png)

Phân tích các trường hợp dự đoán sai để hiểu hạn chế của mô hình.

---

## 6. Học bán giám sát

*Tận dụng dữ liệu chưa gán nhãn (Semi-supervised Learning)*

### 6.1. Tại sao cần học bán giám sát?

Trong thực tế, việc gán nhãn dữ liệu tốn kém về thời gian và chi phí. Học bán giám sát giúp tận dụng dữ liệu chưa gán nhãn để cải thiện mô hình.

### 6.2. Thử nghiệm

Chúng tôi thử nghiệm với các kịch bản:
- **5%** dữ liệu có nhãn
- **10%** dữ liệu có nhãn  
- **20%** dữ liệu có nhãn

Phương pháp: **Tự huấn luyện (Self-Training)** và **Lan truyền nhãn (Label Propagation)**

### 6.3. Kết quả

![So sánh học bán giám sát](outputs/figures/semi_supervised_comparison.png)

### 6.4. Đường cong học tập

![Đường cong học tập bán giám sát](outputs/figures/semi_supervised_learning_curve.png)

### 6.5. Ma trận nhầm lẫn - Tự huấn luyện

![Ma trận nhầm lẫn tự huấn luyện](outputs/figures/pseudo_label_cm_self_training.png)

**Nhận xét:**
- Tự huấn luyện với 20% dữ liệu có nhãn đạt **Điểm F1 ~0,75**
- Còn cách khá xa học có giám sát (Điểm F1 = 0,80)
- Lan truyền nhãn không hiệu quả với bộ dữ liệu lớn

> 💡 **Phát hiện:** Với bộ dữ liệu này, học có giám sát vẫn là lựa chọn tối ưu khi có đủ dữ liệu có nhãn

---

## 7. Dự báo chuỗi thời gian

*Dự báo tỷ lệ huỷ theo thời gian (Time Series Forecasting)*

### 7.1. Mục tiêu

Dự báo **tỷ lệ huỷ đặt phòng theo tháng** để hỗ trợ lập kế hoạch kinh doanh.

### 7.2. Dữ liệu chuỗi thời gian

![Số lượng đặt phòng và huỷ phòng](outputs/figures/ts_bookings_cancellations.png)

Số lượng đặt phòng và huỷ phòng theo tháng từ 2015-2017.

### 7.3. Tỷ lệ huỷ theo thời gian

![Tỷ lệ huỷ theo thời gian](outputs/figures/ts_cancellation_rate.png)

Tỷ lệ huỷ dao động từ ~25% đến ~45% theo từng tháng.

### 7.4. Phân tách xu hướng

![Phân tách chuỗi thời gian](outputs/figures/ts_decomposition.png)

Phân tách thành 3 thành phần:
- **Xu hướng (Trend):** Xu hướng tăng nhẹ
- **Mùa vụ (Seasonal):** Biến động theo mùa rõ rệt
- **Nhiễu (Residual):** Nhiễu ngẫu nhiên

### 7.5. Phân tích ACF & PACF

![ACF và PACF](outputs/figures/ts_acf_pacf.png)

Phân tích hàm tự tương quan (ACF) và tự tương quan riêng phần (PACF) để xác định tham số ARIMA.

### 7.6. Chia tập huấn luyện và kiểm tra

![Chia tập dữ liệu](outputs/figures/ts_train_test_split.png)

Chia dữ liệu: 80% huấn luyện, 20% kiểm tra (theo thời gian).

### 7.7. So sánh các mô hình

![So sánh mô hình chuỗi thời gian](outputs/figures/ts_model_comparison.png)

| Mô hình | MAE | RMSE | MAPE |
|---------|-----|------|------|
| **Trung bình trượt (6 tháng)** | **0,043** | **0,053** | **10,4%** |
| Trung bình trượt (3 tháng) | 0,057 | 0,068 | 13,5% |
| ARIMA(1,1,1) | 0,071 | 0,081 | 16,9% |
| Làm mượt hàm mũ | 0,065 | 0,078 | 15,2% |

### 7.8. Kết quả dự báo

![Tất cả các dự báo](outputs/figures/ts_all_forecasts.png)

So sánh dự báo của tất cả các mô hình.

### 7.9. Dự báo tốt nhất

![Dự báo tốt nhất](outputs/figures/ts_best_forecast.png)

**🏆 Mô hình tốt nhất: Trung bình trượt 6 tháng (Moving Average - MA(6))**
- MAPE: **10,4%** (sai số dưới 11%)
- Phù hợp với dữ liệu có quy luật mùa vụ

---

## 8. Kết luận và khuyến nghị

### 🎯 Tóm tắt kết quả

| Phương pháp | Mô hình tốt nhất | Chỉ số đánh giá |
|-------------|------------------|-----------------|
| **Phân loại** | Rừng ngẫu nhiên (Tinh chỉnh) | F1 = 0,801, AUC = 0,927 |
| **Phân cụm** | KMeans (K=4) | Silhouette = 0,35 |
| **Chuỗi thời gian** | MA(6) | MAPE = 10,4% |

### 💡 Các phát hiện kinh doanh quan trọng

#### 1️⃣ Chính sách đặt cọc là vũ khí mạnh nhất

> Yêu cầu đặt cọc có thể **giảm hơn 60% rủi ro huỷ**

**Khuyến nghị:** Áp dụng đặt cọc không hoàn tiền cho:
- Đặt phòng trước hơn 60 ngày
- Đặt qua đại lý du lịch trực tuyến
- Mùa cao điểm du lịch

#### 2️⃣ Đặt trước càng lâu, rủi ro càng cao

> Đặt phòng trước hơn 100 ngày có **hơn 50% khả năng huỷ**

**Khuyến nghị:**
- Gửi email nhắc nhở 30 ngày trước
- Gọi điện xác nhận 7 ngày trước
- Áp dụng chính sách huỷ linh hoạt cho đặt phòng ngắn hạn

#### 3️⃣ Yêu cầu đặc biệt = Khách cam kết

> Khách có yêu cầu đặc biệt **ít huỷ hơn 50%**

**Khuyến nghị:**
- Khuyến khích khách ghi chú sở thích
- Hỏi về yêu cầu ăn uống, phòng ưa thích
- Cá nhân hoá giao tiếp với khách

#### 4️⃣ Phân khúc khác nhau có rủi ro khác nhau

| Phân khúc | Mức rủi ro | Chiến lược |
|-----------|------------|------------|
| Đại lý trực tuyến | 🔴 Cao | Chính sách đặt cọc nghiêm ngặt |
| Nhóm | 🔴 Cao | Hợp đồng + thanh toán trước |
| Doanh nghiệp | 🟢 Thấp | Chính sách linh hoạt |
| Đặt trực tiếp | 🟡 Trung bình | Chương trình khách hàng thân thiết |

#### 5️⃣ Khách quen là tài sản quý

> Khách quen có tỷ lệ huỷ **thấp hơn 40%**

**Khuyến nghị:**
- Chương trình tích điểm thưởng
- Ưu đãi cho đặt phòng quay lại
- Cung cấp dịch vụ cá nhân hoá

#### 6️⃣ Lập kế hoạch theo mùa

> Tỷ lệ huỷ biến động **25-45%** theo mùa

**Khuyến nghị:**
- Chiến lược đặt quá phòng linh hoạt
- Điều chỉnh giá theo mùa
- Lập kế hoạch nhân sự theo dự báo

### 🚀 Ứng dụng Demo

Chúng tôi đã xây dựng **Ứng dụng Streamlit** cho phép:
- ✅ Nhập thông tin đặt phòng mới
- ✅ Dự đoán khả năng huỷ theo thời gian thực
- ✅ Hiển thị các yếu tố rủi ro
- ✅ Đề xuất hành động phù hợp

**Chạy ứng dụng demo:**
```bash
streamlit run app/streamlit_app.py
```

### 📈 Hiệu quả ước tính

Nếu áp dụng mô hình với tỷ lệ huỷ trung bình 37%:

| Chỉ số | Trước | Sau |
|--------|-------|-----|
| Tỷ lệ huỷ không dự báo được | 37% | ~7% |
| Mất doanh thu do phòng trống | $X | $0,2X |
| Hiệu quả đặt quá phòng | 50% | 85% |

> **Lợi tức đầu tư ước tính: Giảm 20-30% tổn thất doanh thu từ huỷ phòng**

---

## 📚 Tài liệu tham khảo

1. Bộ dữ liệu Hotel Booking Demand - Kaggle
2. Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets
3. Tài liệu scikit-learn
4. Tài liệu XGBoost

---

## 🔗 Liên kết

- **📁 Kho mã nguồn:** [Nhom12_BaiTapLon_DataMining](https://github.com/nhom12/datamining-hotel-booking)
- **🖥️ Ứng dụng Demo:** Ứng dụng Streamlit
- **📊 Bộ dữ liệu:** [Kaggle - Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)