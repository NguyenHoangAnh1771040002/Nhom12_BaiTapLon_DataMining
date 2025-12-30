# 📊 Hotel Booking Cancellation Prediction - Summary Report

**Generated:** 2025-12-30 18:24:28

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
| Random Forest (Tuned) |     0.8573 |      0.8287 |   0.775  | 0.801  |    0.9268 |   0.9033 |
| LightGBM              |     0.8385 |      0.8171 |   0.7267 | 0.7693 |    0.9075 |   0.8797 |
| XGBoost               |     0.832  |      0.8013 |   0.7267 | 0.7622 |    0.9015 |   0.8727 |
| Random Forest         |     0.8162 |      0.778  |   0.7049 | 0.7397 |    0.8974 |   0.8667 |
| Decision Tree         |     0.8097 |      0.7793 |   0.6786 | 0.7254 |    0.8773 |   0.8286 |
| Logistic Regression   |     0.7599 |      0.6734 |   0.6833 | 0.6783 |    0.8404 |   0.8    |

### Key Findings

- Best performing model: **Random Forest (Tuned)**
- F1-Score: **0.8010**
- Accuracy: **0.8573**


---

## 3. Feature Importance

Top 10 Most Important Features:

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

**Insight:** Mô hình Random Forest (Tuned) đạt hiệu suất cao nhất với F1-score = 0.8010.

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
