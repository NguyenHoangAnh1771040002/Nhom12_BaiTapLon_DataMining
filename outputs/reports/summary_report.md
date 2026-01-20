# 📊 Hotel Booking Cancellation Prediction - Summary Report

**Generated:** 2026-01-20 10:07:21

---

## 1. Executive Summary

This report summarizes the results of the Hotel Booking Cancellation Prediction project, 
including model comparisons, feature analysis, and actionable business insights.

**Best Model:** Random Forest (Tuned)

---

## 2. Model Comparison

### Performance Metrics

|                       |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |
|:----------------------|-----------:|------------:|---------:|-------:|----------:|---------:|
| Random Forest (Tuned) |     0.8569 |      0.8275 |   0.7752 | 0.8005 |    0.9266 |   0.9029 |
| LightGBM              |     0.838  |      0.8154 |   0.7273 | 0.7689 |    0.9074 |   0.8797 |
| XGBoost               |     0.8321 |      0.8013 |   0.727  | 0.7623 |    0.9028 |   0.874  |
| Random Forest         |     0.8182 |      0.782  |   0.7063 | 0.7422 |    0.8963 |   0.8654 |
| Decision Tree         |     0.8098 |      0.7794 |   0.6786 | 0.7255 |    0.8773 |   0.8283 |
| Logistic Regression   |     0.7645 |      0.6777 |   0.6946 | 0.6861 |    0.8391 |   0.8018 |

### Key Findings

- Best performing model: **Random Forest (Tuned)**
- F1-Score: **0.8005**
- Accuracy: **0.8569**


---

## 3. Feature Importance

Top 10 Most Important Features:

| feature                  |   importance |   importance_pct |   cumulative_pct |
|:-------------------------|-------------:|-----------------:|-----------------:|
| deposit_required         |    0.203367  |         20.3367  |          20.3367 |
| lead_time                |    0.113319  |         11.3319  |          31.6686 |
| agent                    |    0.109021  |         10.9021  |          42.5706 |
| has_special_requests     |    0.0783969 |          7.83969 |          50.4103 |
| room_type_changed        |    0.0728536 |          7.28536 |          57.6957 |
| adr                      |    0.055301  |          5.5301  |          63.2258 |
| has_booking_changes      |    0.0335905 |          3.35905 |          66.5848 |
| total_revenue            |    0.0330593 |          3.30593 |          69.8908 |
| revenue_per_guest        |    0.0314245 |          3.14245 |          73.0332 |
| arrival_date_week_number |    0.0306377 |          3.06377 |          76.097  |



---

## 4. Business Insights & Recommendations


### 1. Overall Cancellation Rate

**Category:** Overview

**Insight:** Tỷ lệ hủy đặt phòng tổng thể là 37.0%.

**Recommendation:** Cần có chiến lược chủ động để giảm tỷ lệ hủy đặt phòng.


### 2. Impact of Lead Time

**Category:** Lead Time

**Insight:** Đặt phòng với lead time > 100 ngày có tỷ lệ hủy 51.1%, trong khi lead time <= 30 ngày chỉ có 18.6%.

**Recommendation:** Áp dụng chính sách đặt cọc cao hơn cho đặt phòng có lead time dài.


### 3. Deposit Type Impact

**Category:** Deposit Policy

**Insight:** Tỷ lệ hủy theo loại deposit: No Deposit: 28.4%, Non Refund: 99.4%, Refundable: 22.2%.

**Recommendation:** Khuyến khích khách hàng đặt cọc không hoàn lại để giảm tỷ lệ hủy.


### 4. Customer Type Risk

**Category:** Customer Segment

**Insight:** Nhóm khách hàng "Transient" có tỷ lệ hủy cao nhất (40.7%).

**Recommendation:** Tập trung chương trình loyalty cho nhóm "Transient" để giữ chân khách.


### 5. High-Risk Segments

**Category:** Market Segment

**Insight:** Các phân khúc có rủi ro cao: Undefined (100.0%), Groups (61.1%).

**Recommendation:** Xem xét yêu cầu đặt cọc hoặc xác nhận bổ sung cho các phân khúc rủi ro cao.


### 6. Previous Cancellation Pattern

**Category:** Booking History

**Insight:** Khách có lịch sử hủy trước đó có tỷ lệ hủy 91.6%, so với 33.9% cho khách không có lịch sử hủy.

**Recommendation:** Áp dụng chính sách đặt phòng nghiêm ngặt hơn với khách có lịch sử hủy.


### 7. Key Predictive Factors

**Category:** Predictive Features

**Insight:** Các yếu tố dự đoán hủy quan trọng nhất: deposit_required, lead_time, agent, has_special_requests, room_type_changed.

**Recommendation:** Tập trung thu thập và phân tích các yếu tố này để cải thiện dự đoán.


### 8. Best Prediction Model

**Category:** Model Performance

**Insight:** Mô hình Random Forest (Tuned) đạt hiệu suất cao nhất với F1-score = 0.8005.

**Recommendation:** Deploy mô hình này vào hệ thống để dự đoán và can thiệp sớm.


### 9. Seasonal Cancellation Patterns

**Category:** Seasonality

**Insight:** Các tháng có tỷ lệ hủy cao: June (41.5%), April (40.8%), May (39.7%).

**Recommendation:** Điều chỉnh chính sách đặt phòng và overbooking theo mùa.



---

## 5. Conclusion

The analysis demonstrates that hotel booking cancellations can be effectively predicted 
using machine learning models. The **Random Forest (Tuned)** model achieved the best performance 
and is recommended for deployment.

### Next Steps

1. Deploy the prediction model in production environment
2. Implement automated intervention system for high-risk bookings
3. Monitor model performance and retrain periodically
4. A/B test different intervention strategies

---

*Report generated by Nhom12 Data Mining Project*
