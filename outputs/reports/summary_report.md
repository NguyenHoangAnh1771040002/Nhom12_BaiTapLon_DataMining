# 📊 Dự Đoán Hủy Đặt Phòng Khách Sạn - Báo Cáo Tóm Tắt

**Ngày tạo:** 2026-01-20 10:07:21

---

## 1. Tóm Tắt Điều Hành

Báo cáo này tóm tắt kết quả của dự án Dự Đoán Hủy Đặt Phòng Khách Sạn, bao gồm so sánh mô hình, phân tích đặc trưng và các thông tin kinh doanh có thể hành động.

**Mô hình tốt nhất:** Rừng Ngẫu Nhiên (Tuned)

---

## 2. So Sánh Mô Hình

### Các Chỉ Số Hiệu Suất

|                       |   Độ chính xác |   Độ chính xác (precision) |   Độ nhạy (recall) |     F1 |   ROC-AUC |   PR-AUC |
|:----------------------|-----------:|------------:|---------:|-------:|----------:|---------:|
| Rừng Ngẫu Nhiên (Tuned) |     0.8569 |      0.8275 |   0.7752 | 0.8005 |    0.9266 |   0.9029 |
| LightGBM              |     0.838  |      0.8154 |   0.7273 | 0.7689 |    0.9074 |   0.8797 |
| XGBoost               |     0.8321 |      0.8013 |   0.727  | 0.7623 |    0.9028 |   0.874  |
| Rừng Ngẫu Nhiên       |     0.8182 |      0.782  |   0.7063 | 0.7422 |    0.8963 |   0.8654 |
| Cây Quyết Định        |     0.8098 |      0.7794 |   0.6786 | 0.7255 |    0.8773 |   0.8283 |
| Hồi Quy Logistic      |     0.7645 |      0.6777 |   0.6946 | 0.6861 |    0.8391 |   0.8018 |

### Phát Hiện Chính

- Mô hình có hiệu suất tốt nhất: **Rừng Ngẫu Nhiên (Tuned)**
- F1-Score: **0.8005**
- Độ chính xác: **0.8569**


---

## 3. Tầm Quan Trọng Của Đặc Trưng

10 Đặc Trưng Quan Trọng Nhất:

| Đặc trưng                  |   Tầm quan trọng |   Tỷ lệ phần trăm |   Tích lũy |
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

## 4. Thông Tin Kinh Doanh & Khuyến Nghị


### 1. Tỷ Lệ Hủy Đặt Phòng Tổng Thể

**Danh mục:** Tổng Quan

**Thông tin:** Tỷ lệ hủy đặt phòng tổng thể là 37.0%.

**Khuyến nghị:** Cần có chiến lược chủ động để giảm tỷ lệ hủy đặt phòng.


### 2. Ảnh Hưởng Của Lead Time

**Danh mục:** Lead Time

**Thông tin:** Đặt phòng với lead time > 100 ngày có tỷ lệ hủy 51.1%, trong khi lead time <= 30 ngày chỉ có 18.6%.

**Khuyến nghị:** Áp dụng chính sách đặt cọc cao hơn cho đặt phòng có lead time dài.


### 3. Ảnh Hưởng Của Loại Tiền Đặt Cọc

**Danh mục:** Chính Sách Đặt Cọc

**Thông tin:** Tỷ lệ hủy theo loại deposit: Không đặt cọc: 28.4%, Không hoàn lại: 99.4%, Hoàn lại: 22.2%.

**Khuyến nghị:** Khuyến khích khách hàng đặt cọc không hoàn lại để giảm tỷ lệ hủy.


### 4. Rủi Ro Theo Loại Khách Hàng

**Danh mục:** Phân Khúc Khách Hàng

**Thông tin:** Nhóm khách hàng "Transient" có tỷ lệ hủy cao nhất (40.7%).

**Khuyến nghị:** Tập trung chương trình loyalty cho nhóm "Transient" để giữ chân khách.


### 5. Phân Khúc Rủi Ro Cao

**Danh mục:** Phân Khúc Thị Trường

**Thông tin:** Các phân khúc có rủi ro cao: Undefined (100.0%), Groups (61.1%).

**Khuyến nghị:** Xem xét yêu cầu đặt cọc hoặc xác nhận bổ sung cho các phân khúc rủi ro cao.


### 6. Mẫu Hủy Trước Đó

**Danh mục:** Lịch Sử Đặt Phòng

**Thông tin:** Khách có lịch sử hủy trước đó có tỷ lệ hủy 91.6%, so với 33.9% cho khách không có lịch sử hủy.

**Khuyến nghị:** Áp dụng chính sách đặt phòng nghiêm ngặt hơn với khách có lịch sử hủy.


### 7. Các Yếu Tố Dự Đoán Quan Trọng

**Danh mục:** Đặc Trưng Dự Đoán

**Thông tin:** Các yếu tố dự đoán hủy quan trọng nhất: deposit_required, lead_time, agent, has_special_requests, room_type_changed.

**Khuyến nghị:** Tập trung thu thập và phân tích các yếu tố này để cải thiện dự đoán.


### 8. Mô Hình Dự Đoán Tốt Nhất

**Danh mục:** Hiệu Suất Mô Hình

**Thông tin:** Mô hình Rừng Ngẫu Nhiên (Tuned) đạt hiệu suất cao nhất với F1-score = 0.8005.

**Khuyến nghị:** Deploy mô hình này vào hệ thống để dự đoán và can thiệp sớm.


### 9. Mẫu Hủy Theo Mùa

**Danh mục:** Tính Thời Vụ

**Thông tin:** Các tháng có tỷ lệ hủy cao: Tháng 6 (41.5%), Tháng 4 (40.8%), Tháng 5 (39.7%).

**Khuyến nghị:** Điều chỉnh chính sách đặt phòng và overbooking theo mùa.



---

## 5. Kết Luận

Phân tích cho thấy việc hủy đặt phòng khách sạn có thể được dự đoán hiệu quả bằng các mô hình học máy. Mô hình **Rừng Ngẫu Nhiên (Tuned)** đạt hiệu suất tốt nhất và được khuyến nghị triển khai.

### Các Bước Tiếp Theo

1. Triển khai mô hình dự đoán vào môi trường sản xuất
2. Thực hiện hệ thống can thiệp tự động cho các đặt phòng có rủi ro cao
3. Theo dõi hiệu suất mô hình và huấn luyện lại định kỳ
4. Thử nghiệm A/B các chiến lược can thiệp khác nhau

---

*Báo cáo được tạo bởi Nhóm 12 Dự Án Khai Phá Dữ Liệu*
