# Dự Đoán Huỷ Đặt Phòng Khách Sạn - Ứng Dụng Demo
# Hotel Booking Cancellation Prediction - Demo App

## 📖 Giới thiệu

Ứng dụng web demo sử dụng Streamlit để dự đoán khả năng huỷ đặt phòng khách sạn dựa trên mô hình Machine Learning đã huấn luyện.

## 🚀 Cách chạy

### Yêu cầu
- Python 3.8+
- Các thư viện trong requirements.txt đã cài đặt
- Mô hình đã được huấn luyện (file trong outputs/models/)

### Chạy ứng dụng

```bash
# Kích hoạt môi trường (Activate environment)
conda activate lab

# Chạy ứng dụng Streamlit
streamlit run app/streamlit_app.py

# Hoặc với cổng cụ thể (Or with specific port)
streamlit run app/streamlit_app.py --server.port 8501
```

Ứng dụng sẽ mở tại: http://localhost:8501

## 🎯 Tính năng

### 1. Nhập thông tin đặt phòng
- **Thông tin khách sạn**: Loại khách sạn, tháng đến, thời gian đặt trước, số đêm
- **Thông tin khách**: Số người lớn/trẻ em, khách quen, loại khách hàng
- **Thông tin đặt phòng**: Phân khúc, đặt cọc, loại phòng, giá

### 2. Kết quả dự đoán
- **Xác suất huỷ**: Hiển thị phần trăm dự đoán
- **Mức độ rủi ro**: THẤP/TRUNG BÌNH/CAO (LOW/MEDIUM/HIGH)
- **Các yếu tố ảnh hưởng**: Phân tích các factor chính

### 3. Khuyến nghị
- Đề xuất hành động dựa trên mức rủi ro
- Chiến lược cho khách sạn để giảm tỷ lệ huỷ

## 📊 Thông Tin Mô Hình (Model Information)

| Chỉ số | Giá trị |
|--------|---------|
| Mô hình | Random Forest (Đã tinh chỉnh) |
| F1-Score | 0.8010 |
| Độ chính xác | 85.7% |
| ROC-AUC | 0.9268 |

### Đặc Trưng Quan Trọng (Top Features)
1. **deposit_required** (19.7%) - Yêu cầu đặt cọc
2. **lead_time** (11.6%) - Thời gian đặt trước
3. **agent** (11.1%) - Đại lý đặt phòng
4. **has_special_requests** (7.7%) - Có yêu cầu đặc biệt
5. **room_type_changed** (7.2%) - Thay đổi loại phòng

## 🛠️ Cấu trúc thư mục

```
app/
├── __init__.py           # Khởi tạo module (Module init)
├── streamlit_app.py      # Ứng dụng Streamlit chính (Main Streamlit app)
└── README.md             # Tài liệu (Documentation - file này)
```

## 🔧 Xử Lý Sự Cố (Troubleshooting)

### Mô hình không tải được
```bash
# Huấn luyện lại mô hình (Retrain model)
python scripts/run_pipeline.py --modeling
```

### Cổng đã được sử dụng
```bash
# Dùng cổng khác (Use different port)
streamlit run app/streamlit_app.py --server.port 8502
```

### Cảnh báo về phiên bản sklearn
- Đây là cảnh báo không nghiêm trọng do khác phiên bản sklearn khi huấn luyện và tải mô hình
- Kết quả vẫn chính xác

## 📝 Triển khai trong Production

### Triển khai với Streamlit Cloud
1. Đẩy code lên GitHub
2. Kết nối với streamlit.io/cloud
3. Triển khai ứng dụng

### Triển khai với Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/streamlit_app.py"]
```

## 🎓 Thông Tin Nhóm

- **Dự án**: Dự Đoán Huỷ Đặt Phòng Khách Sạn (Hotel Booking Cancellation Prediction)
- **Học phần**: Khai Phá Dữ Liệu (Data Mining)
- **Nhóm**: 12
