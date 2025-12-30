# 🏨 Hotel Booking Cancellation Prediction

> **Đề tài 12:** Dự đoán huỷ đặt phòng khách sạn  
> **Học phần:** Khai phá dữ liệu - HK2 2025-2026  
> **GVHD:** ThS. Lê Thị Thùy Trang

---

## 📋 Mô tả dự án

Dự án này xây dựng pipeline khai phá dữ liệu để:
1. **Phân tích luật kết hợp** - Tìm các combo thuộc tính liên quan đến huỷ đặt phòng
2. **Phân cụm** - Nhóm các booking theo hành vi, xác định cụm rủi ro cao
3. **Phân lớp** - Dự đoán khách có huỷ phòng hay không
4. **Bán giám sát** - Thử nghiệm với dữ liệu thiếu nhãn
5. **Chuỗi thời gian** - Dự báo tỷ lệ huỷ theo tháng

## 📊 Dataset

- **Nguồn:** [Hotel Booking Demand - Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- **File:** `data/raw/hotel_bookings.csv`
- **Target:** `is_canceled` (0: Không huỷ, 1: Huỷ)

## 🗂️ Cấu trúc thư mục

```
BaiTapLon/
├── README.md                 # File này
├── requirements.txt          # Thư viện Python cần thiết
├── .gitignore               # Loại trừ files khỏi git
├── configs/
│   └── params.yaml          # Tham số cấu hình
├── data/
│   ├── raw/                 # Dữ liệu gốc
│   │   └── hotel_bookings.csv
│   └── processed/           # Dữ liệu đã xử lý
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_or_clustering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
├── src/
│   ├── data/                # Module đọc và làm sạch dữ liệu
│   ├── features/            # Module tạo đặc trưng
│   ├── mining/              # Module khai phá (association, clustering)
│   ├── models/              # Module mô hình (supervised, semi-supervised)
│   ├── evaluation/          # Module đánh giá
│   └── visualization/       # Module vẽ biểu đồ
├── scripts/
│   └── run_pipeline.py      # Script chạy toàn bộ pipeline
└── outputs/
    ├── figures/             # Biểu đồ xuất ra
    ├── tables/              # Bảng kết quả
    ├── models/              # Model đã train
    └── reports/             # Báo cáo PDF
```

## 🚀 Hướng dẫn cài đặt

### 1. Clone repository
```bash
git clone <repository-url>
cd BaiTapLon
```

### 2. Tạo virtual environment (khuyến nghị)
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 4. Chuẩn bị dữ liệu
- Tải dataset từ [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
- Đặt file `hotel_bookings.csv` vào thư mục `data/raw/`

### 5. Cấu hình tham số
- Chỉnh sửa file `configs/params.yaml` nếu cần

## 📖 Hướng dẫn chạy

### Chạy từng notebook theo thứ tự:
1. `01_eda.ipynb` - Khám phá dữ liệu
2. `02_preprocess_feature.ipynb` - Tiền xử lý và tạo đặc trưng
3. `03_mining_or_clustering.ipynb` - Luật kết hợp và phân cụm
4. `04_modeling.ipynb` - Huấn luyện mô hình phân lớp
5. `04b_semi_supervised.ipynb` - Thử nghiệm bán giám sát
6. `05_evaluation_report.ipynb` - Tổng hợp và đánh giá

### Hoặc chạy pipeline tự động:
```bash
python scripts/run_pipeline.py
```

## 📈 Kết quả

*(Sẽ cập nhật sau khi hoàn thành)*

| Model | Accuracy | F1 | PR-AUC | ROC-AUC |
|-------|----------|-----|--------|---------|
| Logistic Regression | - | - | - | - |
| Decision Tree | - | - | - | - |
| Random Forest | - | - | - | - |
| XGBoost | - | - | - | - |

## 👥 Thành viên nhóm

| STT | Họ tên | MSSV | Vai trò |
|-----|--------|------|---------|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |

## 📝 License

Dự án này được thực hiện cho mục đích học tập.

---
*Cập nhật lần cuối: 30/12/2025*
