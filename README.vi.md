# Photobooth với AI - Điều khiển bằng cử chỉ tay

Hệ thống photobooth sử dụng AI để nhận diện cử chỉ tay và điều khiển camera tự động.

## Tính năng

- **Nhận diện cử chỉ tay**: Real-time, chính xác và ổn định
- **Điều khiển Zoom**:
  - **Nắm tay** → Zoom Out (1x-3x)
  - **Mở tay** → Zoom In (1x-3x)
- **Tự động chụp**:
  - **Dấu OK/V sign** → Tự động chuyển sang Mode ON và chụp 6 ảnh
- **Chuyển chế độ (Mode Toggle)**:
  - **MODE: OFF** → Chỉ nhận diện cử chỉ, không chụp
  - **MODE: ON** → Tự động chụp ảnh
- **Giao diện thời gian thực**: Hiển thị mức Zoom, Mode và Gesture hiện tại

## Cài đặt

### Backend (AI Service)

```bash
cd ai-service
pip install -r requirements.txt
python main.py
```

Backend chạy tại: `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend chạy tại: `http://localhost:5173`

## Cách sử dụng

1. **Khởi động hệ thống**:

   - Chạy backend trước: `python ai-service/main.py`
   - Chạy frontend: `npm run dev` (trong thư mục `frontend`)

2. **Điều khiển bằng cử chỉ tay**:

   - **Nắm tay**: Giảm zoom (tối thiểu 1x)
   - **Mở tay**: Tăng zoom (tối đa 3x)
   - **Dấu OK/V**: Tự động chuyển Mode ON và chụp 6 ảnh

3. **Điều khiển thủ công**:
   - Nhấn nút "MODE: ON/OFF" để chuyển đổi chế độ
   - **Mode OFF**: Chỉ nhận diện, không chụp ảnh
   - **Mode ON**: Tự động chụp ảnh với đồng hồ đếm ngược

## Luồng hoạt động

1. **Khởi tạo**: Camera bắt đầu nhận diện tay, Mode mặc định là OFF
2. **Điều chỉnh Zoom**: Dùng nắm tay/mở tay để điều chỉnh zoom
3. **Chụp ảnh**:
   - Cách 1: Làm dấu OK/V → Tự động chuyển Mode ON và chụp 6 ảnh
   - Cách 2: Nhấn nút MODE ON → Chụp thủ công
4. **Chọn ảnh**: Chọn 3 ảnh đẹp nhất trong 6 ảnh đã chụp
5. **Ghép ảnh**: Tạo ảnh tổng hợp kèm filter
6. **Tải xuống**: Tải về kết quả cuối cùng

## Cấu trúc dự án

```
photobooth-with-ai/
├── ai-service/
│   ├── main.py              # FastAPI backend với WebSocket
│   ├── requirements.txt     # Python dependencies
│   └── hands-demo.py        # Demo gốc (không dùng)
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Ứng dụng React chính
│   │   └── ...
│   └── package.json
└── README.md
```

## Hiệu suất

- **Real-time**: Kết nối WebSocket độ trễ thấp
- **Mượt mà**: Stream camera 30 FPS
- **Chính xác**: Nhận diện cử chỉ đáng tin cậy với MediaPipe
- **Ổn định**: Tự động kết nối lại khi mất kết nối

## Khắc phục sự cố

1. **Camera không hoạt động**: Kiểm tra quyền truy cập camera
2. **Lỗi WebSocket**: Đảm bảo backend chạy tại cổng 8000
3. **Không nhận diện được cử chỉ**: Đảm bảo tay trong khung hình và đủ ánh sáng
4. **Hiệu năng chậm**: Giảm chất lượng xử lý ảnh ở backend hoặc nâng cấp phần cứng

## Lưu ý

- Cần camera để hoạt động
- Đảm bảo đủ ánh sáng để nhận diện tốt
- Ưu tiên dùng tay phải (có thể điều chỉnh trong code)
- Ảnh được tự động lưu tại thư mục `captured_images/`
