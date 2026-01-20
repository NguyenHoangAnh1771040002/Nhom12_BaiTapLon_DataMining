# Dự đoán Hủy Đặt Phòng Khách Sạn
## Báo Cáo Cuối Cùng

**Ngày:** 2026-01-20
**Nhóm:** Nhóm 12

---

## Mục Lục

1. [Giới Thiệu](#1-gioi-thieu)
2. [Tổng Quan Dữ Liệu](#2-tong-quan-du-lieu)
3. [Phương Pháp](#3-phuong-phap)
4. [Kết Quả](#4-ket-qua)
5. [Thông Tin Kinh Doanh](#5-thong-tin-kinh-doanh)
6. [Kết Luận](#6-ket-luan)

---

## 1. Giới Thiệu

Dự án này nhằm dự đoán việc hủy đặt phòng khách sạn bằng cách sử dụng các kỹ thuật học máy và khai phá dữ liệu khác nhau. Dự đoán chính xác việc hủy đặt phòng có thể giúp khách sạn tối ưu hóa quản lý doanh thu và giảm thiểu tổn thất từ việc khách không đến.

---

## 2. Tổng Quan Dữ Liệu

Bộ dữ liệu chứa các bản ghi đặt phòng khách sạn với các đặc điểm bao gồm:
- Thông tin khách hàng (loại, quốc gia, v.v.)
- Chi tiết đặt phòng (thời gian đặt trước, tiền đặt cọc, loại phòng)
- Thông tin lưu trú (thời gian lưu trú, yêu cầu đặc biệt)
- Dữ liệu lịch sử (hủy trước đó, số lần đặt phòng)

---

## 3. Phương Pháp

### 3.1 Tiền Xử Lý Dữ Liệu
- Xử lý giá trị thiếu
- Kỹ thuật tạo đặc trưng
- Mã hóa dữ liệu phân loại
- Chuẩn hóa đặc trưng

### 3.2 Các Phương Pháp Mô Hình Hóa


#### Học Có Giám Sát
- Hồi Quy Logistic
- Cây Quyết Định
- Rừng Ngẫu Nhiên
- XGBoost
- LightGBM


#### Học Bán Giám Sát
- Lan Truyền Nhãn
- Lan Truyền Nhãn (Label Propagation)
- Tự Huấn Luyện


#### Dự Báo Chuỗi Thời Gian
- ARIMA / SARIMA
- Làm Mịn Số Mũ
- Trung Bình Di Động


---

## 4. Kết Quả


### 4.1 Kết Quả Học Có Giám Sát

|                       |   Độ chính xác |   Độ chính xác (precision) |   Độ nhạy (recall) |     F1 |   ROC-AUC |   PR-AUC |
|:----------------------|-----------:|------------:|---------:|-------:|----------:|---------:|
| Rừng Ngẫu Nhiên (Tuned) |     0.8569 |      0.8275 |   0.7752 | 0.8005 |    0.9266 |   0.9029 |
| LightGBM              |     0.838  |      0.8154 |   0.7273 | 0.7689 |    0.9074 |   0.8797 |
| XGBoost               |     0.8321 |      0.8013 |   0.727  | 0.7623 |    0.9028 |   0.874  |
| Rừng Ngẫu Nhiên       |     0.8182 |      0.782  |   0.7063 | 0.7422 |    0.8963 |   0.8654 |
| Cây Quyết Định        |     0.8098 |      0.7794 |   0.6786 | 0.7255 |    0.8773 |   0.8283 |
| Hồi Quy Logistic      |     0.7645 |      0.6777 |   0.6946 | 0.6861 |    0.8391 |   0.8018 |


### 4.2 Kết Quả Học Bán Giám Sát

|                 |   5% nhãn |   10% nhãn |   20% nhãn |
|:----------------|-------------:|--------------:|--------------:|
| supervised      |     0.683049 |      0.679441 |      0.67982  |
| self_training   |     0.679998 |      0.677886 |      0.679727 |
| label_spreading |     0.461682 |      0.500565 |      0.543048 |


### 4.3 Kết Quả Dự Báo Chuỗi Thời Gian

|                |       MAE |      RMSE |    MAPE |
|:---------------|----------:|----------:|--------:|
| MA(6)          | 0.0434209 | 0.0525686 | 10.3887 |
| MA(3)          | 0.0567929 | 0.0675267 | 13.5033 |
| ARIMA(1,1,1)   | 0.0704806 | 0.0809364 | 16.8906 |
| Naive          | 0.0718262 | 0.0819441 | 17.2422 |
| ARIMA(2,1,2)   | 0.0722673 | 0.0828468 | 17.3259 |
| Làm Mịn Số Mũ  | 0.0819577 | 0.0912376 | 19.8135 |


### 4.4 Mô Hình Tốt Nhất

Mô hình có hiệu suất tốt nhất là **Rừng Ngẫu Nhiên (Tuned)**.


### 4.5 Tầm Quan Trọng Của Đặc Trưng

| Đặc trưng                  |   Tầm quan trọng |   Tỷ lệ phần trăm |   Tích lũy |
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

## 5. Thông Tin Kinh Doanh


### 5.1 Tỷ Lệ Hủy Đặt Phòng Tổng Thể

**Tỷ lệ hủy đặt phòng tổng thể là 37.0%.**

*Khuyến nghị:* Cần có chiến lược chủ động để giảm tỷ lệ hủy đặt phòng.


### 5.2 Ảnh Hưởng Của Lead Time

**Đặt phòng với lead time > 100 ngày có tỷ lệ hủy 51.1%, trong khi lead time <= 30 ngày chỉ có 18.6%.**

*Khuyến nghị:* Áp dụng chính sách đặt cọc cao hơn cho đặt phòng có lead time dài.


### 5.3 Ảnh Hưởng Của Loại Tiền Đặt Cọc

**Tỷ lệ hủy theo loại deposit: Không đặt cọc: 28.4%, Không hoàn lại: 99.4%, Hoàn lại: 22.2%.**

*Khuyến nghị:* Khuyến khích khách hàng đặt cọc không hoàn lại để giảm tỷ lệ hủy.


### 5.4 Rủi Ro Theo Loại Khách Hàng

**Nhóm khách hàng "Transient" có tỷ lệ hủy cao nhất (40.7%).**

*Khuyến nghị:* Tập trung chương trình loyalty cho nhóm "Transient" để giữ chân khách.


### 5.5 Phân Khúc Rủi Ro Cao

**Các phân khúc có rủi ro cao: Undefined (100.0%), Groups (61.1%).**

*Khuyến nghị:* Xem xét yêu cầu đặt cọc hoặc xác nhận bổ sung cho các phân khúc rủi ro cao.


### 5.6 Mẫu Hủy Trước Đó

**Khách có lịch sử hủy trước đó có tỷ lệ hủy 91.6%, so với 33.9% cho khách không có lịch sử hủy.**

*Khuyến nghị:* Áp dụng chính sách đặt phòng nghiêm ngặt hơn với khách có lịch sử hủy.


### 5.7 Các Yếu Tố Dự Đoán Quan Trọng

**Các yếu tố dự đoán hủy quan trọng nhất: deposit_required, lead_time, agent, has_special_requests, room_type_changed.**

*Khuyến nghị:* Tập trung thu thập và phân tích các yếu tố này để cải thiện dự đoán.


### 5.8 Mô Hình Dự Đoán Tốt Nhất

**Mô hình Rừng Ngẫu Nhiên (Tuned) đạt hiệu suất cao nhất với F1-score = 0.8005.**

*Khuyến nghị:* Deploy mô hình này vào hệ thống để dự đoán và can thiệp sớm.


### 5.9 Mẫu Hủy Theo Mùa

**Các tháng có tỷ lệ hủy cao: Tháng 6 (41.5%), Tháng 4 (40.8%), Tháng 5 (39.7%).**

*Khuyến nghị:* Điều chỉnh chính sách đặt phòng và overbooking theo mùa.


---

## 6. Kết Luận

Dự án này đã phát triển thành công một mô hình dự đoán việc hủy đặt phòng khách sạn. Mô hình Rừng Ngẫu Nhiên (Tuned) đạt hiệu suất cao và cung cấp các thông tin hữu ích cho quản lý khách sạn.

### Điểm Chính

1. Lead time là yếu tố dự đoán mạnh mẽ cho việc hủy đặt phòng
2. Loại tiền đặt cọc ảnh hưởng đáng kể đến tỷ lệ hủy
3. Lịch sử đặt phòng của khách hàng cung cấp tín hiệu giá trị
4. Có các mẫu hủy theo mùa

### Khuyến Nghị

1. Thực hiện các chính sách đặt cọc dựa trên rủi ro
2. Sử dụng điểm dự đoán để can thiệp sớm
3. Theo dõi và huấn luyện lại mô hình thường xuyên
4. Thử nghiệm A/B các chiến lược can thiệp

---

*Báo cáo được tạo bởi Nhóm 12 Dự Án Khai Phá Dữ Liệu*
