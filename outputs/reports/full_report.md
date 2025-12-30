# Hotel Booking Cancellation Prediction
## Final Report

**Date:** 2025-12-30
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


#### Supervised Learning
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM


#### Semi-Supervised Learning
- Label Spreading
- Label Propagation
- Self-Training


#### Time Series Forecasting
- ARIMA / SARIMA
- Exponential Smoothing
- Moving Average


---

## 4. Results


### 4.1 Supervised Learning Results

|                       |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |
|:----------------------|-----------:|------------:|---------:|-------:|----------:|---------:|
| Random Forest (Tuned) |     0.8573 |      0.8287 |   0.775  | 0.801  |    0.9268 |   0.9033 |
| LightGBM              |     0.8385 |      0.8171 |   0.7267 | 0.7693 |    0.9075 |   0.8797 |
| XGBoost               |     0.832  |      0.8013 |   0.7267 | 0.7622 |    0.9015 |   0.8727 |
| Random Forest         |     0.8162 |      0.778  |   0.7049 | 0.7397 |    0.8974 |   0.8667 |
| Decision Tree         |     0.8097 |      0.7793 |   0.6786 | 0.7254 |    0.8773 |   0.8286 |
| Logistic Regression   |     0.7599 |      0.6734 |   0.6833 | 0.6783 |    0.8404 |   0.8    |


### 4.2 Semi-Supervised Learning Results

|                 |   5% labeled |   10% labeled |   20% labeled |
|:----------------|-------------:|--------------:|--------------:|
| supervised      |     0.67766  |      0.681739 |      0.681102 |
| self_training   |     0.664519 |      0.678387 |      0.679098 |
| label_spreading |     0.461682 |      0.500565 |      0.543048 |


### 4.3 Time Series Forecasting Results

|                |       mae |      rmse |    mape |
|:---------------|----------:|----------:|--------:|
| MA(6)          | 0.0434209 | 0.0525686 | 10.3887 |
| MA(3)          | 0.0567929 | 0.0675267 | 13.5033 |
| ARIMA(1,1,1)   | 0.0704806 | 0.0809364 | 16.8906 |
| Naive          | 0.0718262 | 0.0819441 | 17.2422 |
| ARIMA(2,1,2)   | 0.0722673 | 0.0828468 | 17.3259 |
| Exp. Smoothing | 0.0819577 | 0.0912376 | 19.8135 |


### 4.4 Best Model

The best performing model is **Random Forest (Tuned)**.


### 4.5 Feature Importance

| feature                   |   importance |   importance_pct |   cumulative_pct |
|:--------------------------|-------------:|-----------------:|-----------------:|
| deposit_required          |    0.197074  |         19.7074  |          19.7074 |
| lead_time                 |    0.115845  |         11.5845  |          31.292  |
| agent                     |    0.11138   |         11.138   |          42.43   |
| has_special_requests      |    0.0766938 |          7.66938 |          50.0993 |
| room_type_changed         |    0.0716322 |          7.16322 |          57.2626 |
| adr                       |    0.054494  |          5.4494  |          62.712  |
| has_booking_changes       |    0.0358343 |          3.58343 |          66.2954 |
| total_of_special_requests |    0.0344372 |          3.44372 |          69.7391 |
| total_revenue             |    0.0321918 |          3.21918 |          72.9583 |
| revenue_per_guest         |    0.0318334 |          3.18334 |          76.1416 |
| arrival_date_week_number  |    0.0305368 |          3.05368 |          79.1953 |
| cancellation_ratio        |    0.0297674 |          2.97674 |          82.1721 |
| arrival_date_day_of_month |    0.0294743 |          2.94743 |          85.1195 |
| requires_parking          |    0.0170034 |          1.70034 |          86.8198 |
| arrival_day_of_week       |    0.0168042 |          1.68042 |          88.5003 |


---

## 5. Business Insights


### 5.1 Overall Cancellation Rate

**Tỷ lệ hủy đặt phòng tổng thể là 37.0%.**

*Recommendation:* Cần có chiến lược chủ động để giảm tỷ lệ hủy đặt phòng.


### 5.2 Impact of Lead Time

**Đặt phòng với lead time > 100 ngày có tỷ lệ hủy 51.1%, trong khi lead time <= 30 ngày chỉ có 18.6%.**

*Recommendation:* Áp dụng chính sách đặt cọc cao hơn cho đặt phòng có lead time dài.


### 5.3 Deposit Type Impact

**Tỷ lệ hủy theo loại deposit: No Deposit: 28.4%, Non Refund: 99.4%, Refundable: 22.2%.**

*Recommendation:* Khuyến khích khách hàng đặt cọc không hoàn lại để giảm tỷ lệ hủy.


### 5.4 Customer Type Risk

**Nhóm khách hàng "Transient" có tỷ lệ hủy cao nhất (40.7%).**

*Recommendation:* Tập trung chương trình loyalty cho nhóm "Transient" để giữ chân khách.


### 5.5 High-Risk Segments

**Các phân khúc có rủi ro cao: Undefined (100.0%), Groups (61.1%).**

*Recommendation:* Xem xét yêu cầu đặt cọc hoặc xác nhận bổ sung cho các phân khúc rủi ro cao.


### 5.6 Previous Cancellation Pattern

**Khách có lịch sử hủy trước đó có tỷ lệ hủy 91.6%, so với 33.9% cho khách không có lịch sử hủy.**

*Recommendation:* Áp dụng chính sách đặt phòng nghiêm ngặt hơn với khách có lịch sử hủy.


### 5.7 Key Predictive Factors

**Các yếu tố dự đoán hủy quan trọng nhất: deposit_required, lead_time, agent, has_special_requests, room_type_changed.**

*Recommendation:* Tập trung thu thập và phân tích các yếu tố này để cải thiện dự đoán.


### 5.8 Best Prediction Model

**Mô hình Random Forest (Tuned) đạt hiệu suất cao nhất với F1-score = 0.8010.**

*Recommendation:* Deploy mô hình này vào hệ thống để dự đoán và can thiệp sớm.


### 5.9 Seasonal Cancellation Patterns

**Các tháng có tỷ lệ hủy cao: June (41.5%), April (40.8%), May (39.7%).**

*Recommendation:* Điều chỉnh chính sách đặt phòng và overbooking theo mùa.


---

## 6. Conclusion

This project successfully developed a predictive model for hotel booking 
cancellations. The Random Forest (Tuned) model achieved strong 
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
