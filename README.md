# Photobooth with AI - Hand Gesture Control

Hệ thống photobooth sử dụng AI để nhận diện cử chỉ tay và điều khiển camera tự động.

## Tính năng

- **Hand Gesture Recognition**: Nhận diện cử chỉ tay real-time
- **Zoom Control**: 
  - Nắm tay → Zoom Out (1x-3x)
  - Mở tay → Zoom In (1x-3x)
- **Auto Capture**: 
  - OK Sign → Tự động chuyển Mode ON và chụp 6 ảnh
- **Mode Toggle**: 
  - MODE: OFF → Chỉ nhận diện cử chỉ, không chụp
  - MODE: ON → Chụp ảnh tự động
- **Real-time UI**: Hiển thị Zoom level, Mode, Gesture hiện tại

## Cài đặt

### Backend (AI Service)

```bash
cd ai-service
pip install -r requirements.txt
python main.py
```

Backend sẽ chạy tại: `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend sẽ chạy tại: `http://localhost:5173`

## Cách sử dụng

1. **Khởi động hệ thống**:
   - Chạy backend trước: `python ai-service/main.py`
   - Chạy frontend: `npm run dev` (trong thư mục frontend)

2. **Điều khiển bằng cử chỉ tay**:
   - **Nắm tay**: Giảm zoom (tối thiểu 1x)
   - **Mở tay**: Tăng zoom (tối đa 3x)
   - **OK Sign**: Tự động chuyển Mode ON và chụp 6 ảnh

3. **Điều khiển thủ công**:
   - Nhấn button "MODE: ON/OFF" để chuyển đổi chế độ
   - Mode OFF: Chỉ nhận diện cử chỉ, không chụp ảnh
   - Mode ON: Tự động chụp ảnh với countdown

## Luồng hoạt động

1. **Khởi tạo**: Camera bắt đầu nhận diện tay, Mode mặc định là OFF
2. **Zoom Control**: Sử dụng nắm tay/mở tay để điều chỉnh zoom
3. **Capture**: 
   - Cách 1: Làm OK sign → Tự động chuyển Mode ON và chụp 6 ảnh
   - Cách 2: Nhấn button MODE ON → Chụp ảnh thủ công
4. **Selection**: Chọn 3 ảnh đẹp nhất từ 6 ảnh đã chụp
5. **Composition**: Tạo ảnh tổng hợp với filter
6. **Download**: Tải về kết quả cuối cùng

## Cấu trúc dự án

```
photobooth-with-ai/
├── ai-service/
│   ├── main.py              # FastAPI backend với WebSocket
│   ├── requirements.txt     # Python dependencies
│   └── hands-demo.py        # Demo gốc (không dùng)
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # React frontend chính
│   │   └── ...
│   └── package.json
└── README.md
```

## Hiệu suất

- **Real-time**: WebSocket connection cho độ trễ thấp
- **Smooth**: Camera stream mượt mà 30 FPS
- **Accurate**: Nhận diện cử chỉ chính xác với MediaPipe
- **Stable**: Tự động reconnect khi mất kết nối

## Troubleshooting

1. **Camera không hoạt động**: Kiểm tra quyền truy cập camera
2. **WebSocket lỗi**: Đảm bảo backend đang chạy tại port 8000
3. **Gesture không nhận diện**: Đảm bảo tay trong khung hình và đủ ánh sáng
4. **Performance chậm**: Giảm quality trong backend hoặc tăng hardware

## Lưu ý

- Cần camera để hoạt động
- Đảm bảo ánh sáng đủ để nhận diện tay
- Tốt nhất sử dụng tay phải (có thể điều chỉnh trong code)
- Hệ thống tự động lưu ảnh vào thư mục `captured_images/`