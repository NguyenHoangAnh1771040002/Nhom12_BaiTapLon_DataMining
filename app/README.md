# Hotel Booking Cancellation Prediction - Demo App

## 📖 Giới thiệu

Ứng dụng web demo sử dụng Streamlit để dự đoán khả năng huỷ đặt phòng khách sạn dựa trên mô hình Machine Learning đã train.

## 🚀 Cách chạy

### Yêu cầu
- Python 3.8+
- Các packages trong requirements.txt đã cài đặt
- Model đã được train (file trong outputs/models/)

### Chạy app

```bash
# Activate environment
conda activate lab

# Chạy Streamlit app
streamlit run app/streamlit_app.py

# Hoặc với port cụ thể
streamlit run app/streamlit_app.py --server.port 8501
```

App sẽ mở tại: http://localhost:8501

## 🎯 Tính năng

### 1. Nhập thông tin Booking
- **Thông tin khách sạn**: Loại khách sạn, tháng đến, lead time, số đêm
- **Thông tin khách**: Số người lớn/trẻ em, khách quen, loại khách hàng
- **Thông tin đặt phòng**: Phân khúc, đặt cọc, loại phòng, giá

### 2. Kết quả dự đoán
- **Xác suất huỷ**: Hiển thị phần trăm dự đoán
- **Mức độ rủi ro**: LOW/MEDIUM/HIGH
- **Các yếu tố ảnh hưởng**: Phân tích các factor chính

### 3. Khuyến nghị
- Đề xuất hành động dựa trên mức rủi ro
- Strategies cho khách sạn để giảm cancellation

## 📊 Model Information

| Metric | Value |
|--------|-------|
| Model | Random Forest (Tuned) |
| F1-Score | 0.8010 |
| Accuracy | 85.7% |
| ROC-AUC | 0.9268 |

### Top Features
1. **deposit_required** (19.7%) - Đặt cọc
2. **lead_time** (11.6%) - Thời gian đặt trước
3. **agent** (11.1%) - Đại lý
4. **has_special_requests** (7.7%) - Yêu cầu đặc biệt
5. **room_type_changed** (7.2%) - Thay đổi phòng

## 🛠️ Cấu trúc thư mục

```
app/
├── __init__.py           # Module init
├── streamlit_app.py      # Main Streamlit app
└── README.md             # Documentation (this file)
```

## 🔧 Troubleshooting

### Model không load được
```bash
# Train lại model
python scripts/run_pipeline.py --modeling
```

### Port đã được sử dụng
```bash
# Dùng port khác
streamlit run app/streamlit_app.py --server.port 8502
```

### Warning về sklearn version
- Đây là warning không nghiêm trọng do khác version sklearn khi train và load
- Kết quả vẫn chính xác

## 📝 Sử dụng trong Production

### Deploy với Streamlit Cloud
1. Push code lên GitHub
2. Kết nối với streamlit.io/cloud
3. Deploy app

### Deploy với Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/streamlit_app.py"]
```

## 🎓 Team

- **Project**: Hotel Booking Cancellation Prediction
- **Course**: Data Mining
- **Group**: Nhom12
