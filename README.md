# Photobooth with AI - Hand Gesture Control

An AI-powered photobooth that recognizes hand gestures and controls the camera automatically.

## Features

- **Hand Gesture Recognition**: Real-time hand gesture detection
- **Zoom Control**:
  - Fist → Zoom Out (1x-3x)
  - Open palm → Zoom In (1x-3x)
- **Auto Capture**:
  - OK/Peace Sign → Automatically switch Mode ON and capture 6 photos
- **Mode Toggle**:
  - MODE: OFF → Only recognizes gestures, no capture
  - MODE: ON → Automatically captures photos
- **Real-time UI**: Displays Zoom level, Mode, and current Gesture

## Setup

### Backend (AI Service)

```bash
cd ai-service
pip install -r requirements.txt
python main.py
```

Backend runs at: `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:5173`

## Usage

1. **Start the system**:

   - Start backend first: `python ai-service/main.py`
   - Start frontend: `npm run dev` (inside the frontend directory)

2. **Control with hand gestures**:

   - **Fist**: Decrease zoom (min 1x)
   - **Open palm**: Increase zoom (max 3x)
   - **OK/Peace Sign**: Auto switch Mode ON and capture 6 photos

3. **Manual control**:
   - Click the "MODE: ON/OFF" button to toggle mode
   - Mode OFF: Only detection, no photo capture
   - Mode ON: Auto capture with countdown

## Workflow

1. **Initialization**: Camera starts detecting hands, default Mode is OFF
2. **Zoom Control**: Use fist/open palm to adjust zoom
3. **Capture**:
   - Method 1: Make OK/peace sign → auto switch to Mode ON and capture 6 photos
   - Method 2: Click MODE ON button → manual capture flow
4. **Selection**: Pick the best 3 photos from the 6 captured
5. **Composition**: Create a composite image with filters
6. **Download**: Download the final result

## Project Structure

```
photobooth-with-ai/
├── ai-service/
│   ├── main.py              # FastAPI backend with WebSocket
│   ├── requirements.txt     # Python dependencies
│   └── hands-demo.py        # Original demo (not used)
├── frontend/
│   ├── src/
│   │   ├── App.tsx         # Main React frontend
│   │   └── ...
│   └── package.json
└── README.md
```

## Performance

- **Real-time**: Low-latency WebSocket connection
- **Smooth**: 30 FPS camera stream
- **Accurate**: Reliable gesture detection with MediaPipe
- **Stable**: Auto-reconnect on connection loss

## Troubleshooting

1. **Camera not working**: Check camera permissions
2. **WebSocket error**: Ensure backend is running on port 8000
3. **Gestures not recognized**: Keep your hand in frame with good lighting
4. **Slow performance**: Lower image quality in backend or upgrade hardware

## Notes

- A camera is required
- Ensure adequate lighting for gesture detection
- Prefer using the right hand (adjustable in code)
- Photos are automatically saved to the `captured_images/` directory
