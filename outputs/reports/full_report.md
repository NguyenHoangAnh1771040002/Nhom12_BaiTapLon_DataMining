# Hotel Booking Cancellation Prediction
## Final Report

**Date:** 2026-01-20
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
| Random Forest (Tuned) |     0.8569 |      0.8275 |   0.7752 | 0.8005 |    0.9266 |   0.9029 |
| LightGBM              |     0.838  |      0.8154 |   0.7273 | 0.7689 |    0.9074 |   0.8797 |
| XGBoost               |     0.8321 |      0.8013 |   0.727  | 0.7623 |    0.9028 |   0.874  |
| Random Forest         |     0.8182 |      0.782  |   0.7063 | 0.7422 |    0.8963 |   0.8654 |
| Decision Tree         |     0.8098 |      0.7794 |   0.6786 | 0.7255 |    0.8773 |   0.8283 |
| Logistic Regression   |     0.7645 |      0.6777 |   0.6946 | 0.6861 |    0.8391 |   0.8018 |


### 4.2 Semi-Supervised Learning Results

|                 |   5% labeled |   10% labeled |   20% labeled |
|:----------------|-------------:|--------------:|--------------:|
| supervised      |     0.683049 |      0.679441 |      0.67982  |
| self_training   |     0.679998 |      0.677886 |      0.679727 |
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
| deposit_required          |    0.203367  |         20.3367  |          20.3367 |
| lead_time                 |    0.113319  |         11.3319  |          31.6686 |
| agent                     |    0.109021  |         10.9021  |          42.5706 |
| has_special_requests      |    0.0783969 |          7.83969 |          50.4103 |
| room_type_changed         |    0.0728536 |          7.28536 |          57.6957 |
| adr                       |    0.055301  |          5.5301  |          63.2258 |
| has_booking_changes       |    0.0335905 |          3.35905 |          66.5848 |
| total_revenue             |    0.0330593 |          3.30593 |          69.8908 |
| revenue_per_guest         |    0.0314245 |          3.14245 |          73.0332 |
| arrival_date_week_number  |    0.0306377 |          3.06377 |          76.097  |
| cancellation_ratio        |    0.0306289 |          3.06289 |          79.1599 |
| arrival_date_day_of_month |    0.0293418 |          2.93418 |          82.094  |
| total_of_special_requests |    0.0278204 |          2.78204 |          84.8761 |
| arrival_day_of_week       |    0.0166176 |          1.66176 |          86.5379 |
| requires_parking          |    0.0161978 |          1.61978 |          88.1576 |


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

**Mô hình Random Forest (Tuned) đạt hiệu suất cao nhất với F1-score = 0.8005.**

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
