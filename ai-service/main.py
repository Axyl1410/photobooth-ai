from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import asyncio
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List
import json

app = FastAPI()

# CORS middleware để frontend có thể kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HandGestureRecognizer:
    def __init__(self):
        # Khởi tạo MediaPipe với settings tối ưu
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Chỉ nhận diện 1 tay
            min_detection_confidence=0.8,  # Tăng confidence để chính xác hơn
            min_tracking_confidence=0.7,  # Tăng confidence để chính xác hơn
        )
        
        # Các biến điều khiển
        self.zoom_level = 1.0
        self.min_zoom = 1.0
        self.max_zoom = 3.0
        self.zoom_step = 0.2
        
        # Biến mode và capture
        self.mode = "OFF"  # OFF hoặc ON
        self.is_capturing = False
        self.captured_photos: List[dict] = []
        self.max_photos = 6
        self.countdown = 0
        self.last_ok_detection = 0.0
        self.ok_cooldown = 1.0
        self.last_countdown_update = 0.0
        self.countdown_interval = 1.0
        self.requires_ok_continuous = False

        # Peace sign stability
        self.peace_sign_count = 0
        self.required_peace_count = 3
        self.last_gesture = "unknown"
        self.gesture_stability_count = 0

        # Gesture filtering
        self.gesture_history: List[str] = []
        self.gesture_history_size = 3
        self.confidence_threshold = 0.5

        # Peace-specific filtering
        self.peace_history: List[str] = []
        self.peace_history_size = 2
        self.peace_confidence_threshold = 0.5

        # Manual stop protection
        self.manual_stop_time = 0.0
        self.manual_stop_cooldown = 3.0

        # Đồng bộ ảnh WS
        self.last_reported_photos_count = 0

        # Retry capture state
        self.retry_captures_remaining = 0
        self.capture_retry_deadline = 0.0
        self.last_frame = None

        # Thư mục lưu ảnh
        if not os.path.exists("captured_images"):
            os.makedirs("captured_images")

    def calculate_distance(self, point1, point2) -> float:
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def is_finger_up(self, landmarks, tip_id: int, pip_id: int, mcp_id: int | None = None) -> bool:
        basic_check = landmarks[tip_id].y < landmarks[pip_id].y
        if mcp_id is None:
            return basic_check
        advanced_check = landmarks[tip_id].y < landmarks[mcp_id].y
        distance_check = self.calculate_distance(landmarks[tip_id], landmarks[pip_id]) > 0.02
        return basic_check and advanced_check and distance_check

    def is_thumb_up(self, landmarks) -> bool:
        thumb_tip = 4
        thumb_ip = 3
        thumb_mcp = 2
        thumb_cmc = 1
        tip_to_ip = self.calculate_distance(landmarks[thumb_tip], landmarks[thumb_ip])
        horizontal_extended = abs(landmarks[thumb_tip].x - landmarks[thumb_mcp].x) > abs(
            landmarks[thumb_ip].x - landmarks[thumb_mcp].x
        )
        vertical_extended = landmarks[thumb_tip].y < landmarks[thumb_ip].y
        distance_check = tip_to_ip > 0.02
        angle_check = landmarks[thumb_tip].y < landmarks[thumb_cmc].y
        return horizontal_extended and vertical_extended and distance_check and angle_check

    def recognize_gesture(self, landmarks) -> str:
        try:
            # Landmarks indices
            thumb_tip, thumb_ip, thumb_mcp = 4, 3, 2
            index_tip, index_pip, index_mcp = 8, 6, 5
            middle_tip, middle_pip, middle_mcp = 12, 10, 9
            ring_tip, ring_pip, ring_mcp = 16, 14, 13
            pinky_tip, pinky_pip, pinky_mcp = 20, 18, 17

            # Fingers up
            fingers_up: List[bool] = []
            fingers_up.append(self.is_thumb_up(landmarks))
            for tip, pip, mcp in [
                (index_tip, index_pip, index_mcp),
                (middle_tip, middle_pip, middle_mcp),
                (ring_tip, ring_pip, ring_mcp),
                (pinky_tip, pinky_pip, pinky_mcp),
            ]:
                fingers_up.append(self.is_finger_up(landmarks, tip, pip, mcp))

            fingers_up_count = fingers_up.count(True)

            # Peace sign (less strict)
            peace_basic = fingers_up[1] and fingers_up[2]
            peace_condition_1 = False
            peace_condition_2 = False
            index_middle_distance = 0.0
            peace_spacing = False

            if peace_basic:
                index_middle_distance = self.calculate_distance(landmarks[index_tip], landmarks[middle_tip])
                peace_spacing = 0.02 < index_middle_distance < 0.15
                height_separation = (
                    landmarks[index_tip].y < landmarks[ring_pip].y - 0.01
                    and landmarks[middle_tip].y < landmarks[ring_pip].y - 0.01
                )
                wrist = landmarks[0]
                index_angle_x = landmarks[index_tip].x - wrist.x
                middle_angle_x = landmarks[middle_tip].x - wrist.x
                angle_spread = abs(index_angle_x - middle_angle_x)
                proper_angle = 0.01 < angle_spread < 0.20
                ring_folded = landmarks[ring_tip].y > landmarks[ring_mcp].y - 0.01
                pinky_folded = landmarks[pinky_tip].y > landmarks[pinky_mcp].y - 0.01
                index_straight = landmarks[index_tip].y < landmarks[index_pip].y - 0.01
                middle_straight = landmarks[middle_tip].y < landmarks[middle_pip].y - 0.01
                not_gun = fingers_up[2]
                not_rock = not fingers_up[4]
                palm_orientation = abs(landmarks[9].x - landmarks[13].x) < 0.15

                failed_conditions: List[str] = []
                if not peace_spacing:
                    failed_conditions.append(f"spacing({index_middle_distance:.3f})")
                if not height_separation:
                    failed_conditions.append("height")
                if not proper_angle:
                    failed_conditions.append(f"angle({angle_spread:.3f})")
                if not ring_folded:
                    failed_conditions.append("ring")
                if not pinky_folded:
                    failed_conditions.append("pinky")
                if not index_straight:
                    failed_conditions.append("index_straight")
                if not middle_straight:
                    failed_conditions.append("middle_straight")
                if not not_gun:
                    failed_conditions.append("not_gun")
                if not not_rock:
                    failed_conditions.append("not_rock")
                if not palm_orientation:
                    failed_conditions.append(f"palm({abs(landmarks[9].x - landmarks[13].x):.3f})")
                if 0 < len(failed_conditions) <= 3:
                    print(f"✌️ [DEBUG] Peace basic OK but failed: {', '.join(failed_conditions)}")

                peace_condition_1 = (
                    fingers_up_count == 2
                    and not fingers_up[0]
                    and peace_basic
                    and peace_spacing
                    and height_separation
                    and proper_angle
                    and ring_folded
                    and pinky_folded
                    and index_straight
                    and middle_straight
                    and not_gun
                    and not_rock
                    and palm_orientation
                )
                peace_condition_2 = (
                    fingers_up_count == 3
                    and fingers_up[0]
                    and peace_basic
                    and peace_spacing
                    and height_separation
                    and proper_angle
                    and ring_folded
                    and pinky_folded
                    and index_straight
                    and middle_straight
                    and not_gun
                    and not_rock
                    and palm_orientation
                )

            if peace_condition_1 or peace_condition_2:
                return "peace"

            if peace_basic and peace_spacing and fingers_up_count in [2, 3]:
                print(
                    f"✌️ [DEBUG] Backup peace detected - fingers_up: {fingers_up}, count: {fingers_up_count}, spacing: {index_middle_distance:.3f}"
                )
                return "peace"

            if (
                fingers_up[1]
                and fingers_up[2]
                and 0.01 < index_middle_distance < 0.20
                and fingers_up_count in [2, 3]
            ):
                print(
                    f"✌️ [DEBUG] Ultra simple peace detected - spacing: {index_middle_distance:.3f}"
                )
                return "peace"

            # Fist
            if fingers_up_count <= 1:
                palm_center_x = (
                    landmarks[0].x + landmarks[9].x + landmarks[13].x + landmarks[17].x
                ) / 4
                palm_center_y = (
                    landmarks[0].y + landmarks[9].y + landmarks[13].y + landmarks[17].y
                ) / 4
                fingertips_near_palm = all(
                    self.calculate_distance(
                        landmarks[tip],
                        type("obj", (object,), {"x": palm_center_x, "y": palm_center_y})(),
                    )
                    < 0.15
                    for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
                )
                if fingertips_near_palm:
                    return "fist"

            # Open
            if fingers_up_count >= 4:
                finger_separation = (
                    self.calculate_distance(landmarks[index_tip], landmarks[middle_tip]) > 0.03
                    and self.calculate_distance(landmarks[middle_tip], landmarks[ring_tip]) > 0.03
                    and self.calculate_distance(landmarks[ring_tip], landmarks[pinky_tip]) > 0.03
                )
                if finger_separation:
                    return "open"

            return "unknown"
        except Exception as e:
            print(f"❌ [ERROR] Gesture recognition failed: {e}")
            print(f"❌ [DEBUG] Error occurred in recognize_gesture function")
            import traceback
            traceback.print_exc()
            return "unknown"

    def filter_gesture(self, raw_gesture: str) -> str:
        # Thêm gesture mới vào lịch sử chung
        self.gesture_history.append(raw_gesture)
        if len(self.gesture_history) > self.gesture_history_size:
            self.gesture_history.pop(0)

        # Peace filtering
        if raw_gesture == "peace":
            self.peace_history.append(raw_gesture)
            if len(self.peace_history) > self.peace_history_size:
                self.peace_history.pop(0)
        else:
            self.peace_history = []

        if len(self.gesture_history) < 2:
            return "unknown"

        gesture_counts: dict[str, int] = {}
        for g in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        gesture_name, count = max(gesture_counts.items(), key=lambda x: x[1])
        confidence = count / len(self.gesture_history)

        if gesture_name == "peace":
            peace_confidence = (
                len(self.peace_history) / self.peace_history_size if self.peace_history else 0
            )
            if raw_gesture == "peace":
                print(
                    f"✌️ [DEBUG] Peace filter - confidence: {confidence:.2f}/{self.confidence_threshold}, "
                    f"peace_conf: {peace_confidence:.2f}/{self.peace_confidence_threshold}, "
                    f"peace_frames: {len(self.peace_history)}/{self.peace_history_size}"
                )
            if (
                confidence >= self.confidence_threshold
                and peace_confidence >= self.peace_confidence_threshold
                and len(self.peace_history) >= 1
            ):
                return "peace"
            return "unknown"

        if confidence >= self.confidence_threshold and gesture_name != "unknown":
            recent_gestures = self.gesture_history[-3:]
            stability = recent_gestures.count(gesture_name) / len(recent_gestures)
            if stability >= 0.6:
                return gesture_name

        return "unknown"

    def apply_zoom(self, frame):
        if self.zoom_level == 1.0:
            return frame
        h, w = frame.shape[:2]
        crop_h = int(h / self.zoom_level)
        crop_w = int(w / self.zoom_level)
        start_x = (w - crop_w) // 2
        start_y = (h - crop_h) // 2
        cropped = frame[start_y : start_y + crop_h, start_x : start_x + crop_w]
        zoomed = cv2.resize(cropped, (w, h))
        return zoomed

    def capture_image(self, frame) -> bool:
        print(
            f"🔍 [DEBUG] Attempting to capture image. Current photos: {len(self.captured_photos)}/{self.max_photos}"
        )
        if len(self.captured_photos) >= self.max_photos:
            print(
                f"❌ [DEBUG] Max photos reached: {len(self.captured_photos)}/{self.max_photos}"
            )
            return False
        try:
            h, w = frame.shape[:2]
            print(f"🔍 [DEBUG] Original frame size: {w}x{h}")
            target_ratio = 9 / 6
            if w / h > target_ratio:
                new_width = int(h * target_ratio)
                start_x = (w - new_width) // 2
                cropped_frame = frame[:, start_x : start_x + new_width]
            else:
                new_height = int(w / target_ratio)
                start_y = (h - new_height) // 2
                cropped_frame = frame[start_y : start_y + new_height, :]
            print(
                f"🔍 [DEBUG] Cropped frame size: {cropped_frame.shape[1]}x{cropped_frame.shape[0]}"
            )
            timestamp = int(time.time() * 1000)
            filename = f"captured_images/capture_{timestamp}.jpg"
            success = cv2.imwrite(filename, cropped_frame)
            if not success:
                print(f"❌ [DEBUG] Failed to save image to {filename}")
                return False
            print(f"✅ [DEBUG] Image saved to {filename}")
            _, buffer = cv2.imencode(".jpg", cropped_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            img_base64 = base64.b64encode(buffer).decode("utf-8")
            photo_data = {
                "id": str(timestamp),
                "dataUrl": f"data:image/jpeg;base64,{img_base64}",
                "timestamp": timestamp,
            }
            self.captured_photos.append(photo_data)
            print(
                f"📸 [SUCCESS] Photo captured! Total photos: {len(self.captured_photos)}/{self.max_photos}"
            )
            return True
        except Exception as e:
            print(f"❌ [ERROR] Failed to capture image: {e}")
            return False

    def process_frame(self, frame):
        # Lật frame theo chiều ngang để có cảm giác như nhìn gương
        frame = cv2.flip(frame, 1)
        self.last_frame = frame
        # Resize để tăng tốc độ xử lý
        small_frame = cv2.resize(frame, (320, 240))
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        # Nhận diện tay
        results = self.hands.process(rgb_frame)
        raw_gesture = "unknown"
        current_time = time.time()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                raw_gesture = self.recognize_gesture(hand_landmarks.landmark)
        # Lọc gesture
        gesture = self.filter_gesture(raw_gesture)
        # Xử lý theo gesture
        if gesture == "fist":
            if self.zoom_level > self.min_zoom:
                self.zoom_level = max(self.min_zoom, self.zoom_level - self.zoom_step)
        elif gesture == "open":
            if self.zoom_level < self.max_zoom:
                self.zoom_level = min(self.max_zoom, self.zoom_level + self.zoom_step)
        elif gesture == "peace":
            self._handle_peace_sign(current_time)
        # Reset counters khi gesture thay đổi
        if gesture != "peace" and gesture != self.last_gesture:
            self.peace_sign_count = 0
            self.gesture_stability_count = 0
        self.last_gesture = gesture
        # Countdown + Capture
        self._handle_countdown_and_capture(current_time, frame)
        # Render frame để gửi về frontend
        display_frame = self.apply_zoom(frame)
        _, buffer = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode("utf-8")
        response = {
            "frame": f"data:image/jpeg;base64,{frame_base64}",
            "gesture": gesture,
            "zoom_level": round(self.zoom_level, 1),
            "mode": self.mode,
            "is_capturing": self.is_capturing,
            "countdown": self.countdown,
            "photos_count": len(self.captured_photos),
            "peace_sign_count": self.peace_sign_count,
            "required_peace_count": self.required_peace_count,
            "gesture_stability_count": self.gesture_stability_count,
            "gesture_stability_required": self.required_peace_count,
            "retrying": self.retry_captures_remaining > 0 and self.countdown == 0 and self.is_capturing,
            "retry_attempts_remaining": self.retry_captures_remaining,
        }
        current_count = len(self.captured_photos)
        if current_count != self.last_reported_photos_count:
            response["photos_updated"] = True
            response["photos"] = self.captured_photos
            self.last_reported_photos_count = current_count
        else:
            response["photos_updated"] = False
        return response

    def _handle_peace_sign(self, current_time: float) -> None:
        if self.last_gesture == "peace":
            self.gesture_stability_count += 1
        else:
            self.gesture_stability_count = 1
        print(
            f"✌️ [DEBUG] Peace stability: {self.gesture_stability_count}/{self.required_peace_count}"
        )
        if (
            self.gesture_stability_count >= self.required_peace_count
            and current_time - self.last_ok_detection > self.ok_cooldown
            and current_time - self.manual_stop_time > self.manual_stop_cooldown
            and self.mode == "OFF"
            and not self.is_capturing
        ):
            print("✌️ [SUCCESS] Peace sign conditions met - Starting capture sequence")
            self.last_ok_detection = current_time
            self.mode = "ON"
            self.is_capturing = True
            self.countdown = 5
            self.last_countdown_update = current_time
            self.peace_sign_count = 0
            self.gesture_stability_count = 0
        elif self.gesture_stability_count >= self.required_peace_count:
            reasons = []
            if current_time - self.last_ok_detection <= self.ok_cooldown:
                reasons.append(
                    f"cooldown ({self.ok_cooldown - (current_time - self.last_ok_detection):.1f}s remaining)"
                )
            if current_time - self.manual_stop_time <= self.manual_stop_cooldown:
                reasons.append(
                    f"manual stop cooldown ({self.manual_stop_cooldown - (current_time - self.manual_stop_time):.1f}s remaining)"
                )
            if self.mode != "OFF":
                reasons.append(f"mode is {self.mode}")
            if self.is_capturing:
                reasons.append("already capturing")
            if reasons:
                print(f"✌️ [DEBUG] Peace sign detected but blocked by: {', '.join(reasons)}")

    def _handle_countdown_and_capture(self, current_time: float, frame) -> None:
        if self.countdown > 0:
            print(
                f"🔍 [DEBUG] Countdown: {self.countdown}, Mode: {self.mode}, Is_capturing: {self.is_capturing}"
            )
        # Đếm ngược
        if self.is_capturing and self.countdown > 0 and self.mode == "ON":
            if current_time - self.last_countdown_update >= self.countdown_interval:
                self.countdown -= 1
                self.last_countdown_update = current_time
                print(f"⏰ [DEBUG] Countdown decreased to: {self.countdown}")
                if self.countdown == 0:
                    print("📸 [DEBUG] Countdown reached 0, attempting to capture...")
                    capture_success = self.capture_image(frame)
                    if capture_success:
                        print(
                            f"✅ [DEBUG] Capture successful! Photos: {len(self.captured_photos)}/{self.max_photos}"
                        )
                        self.retry_captures_remaining = 0
                        self.capture_retry_deadline = 0.0
                        if len(self.captured_photos) < self.max_photos:
                            self.countdown = 5
                            self.last_countdown_update = current_time
                            print("🔄 [DEBUG] Starting next countdown cycle")
                        else:
                            print("🏁 [DEBUG] Max photos reached, resetting capture mode")
                            self._reset_capture_mode()
                    else:
                        print("❌ [DEBUG] Capture failed, scheduling retries...")
                        self.retry_captures_remaining = 2
                        self.capture_retry_deadline = current_time + 0.8
                        self.countdown = 0
        elif self.mode == "OFF" and self.countdown > 0:
            print("🛑 [DEBUG] Mode is OFF but countdown > 0, resetting...")
            self.countdown = 0
            self.is_capturing = False
        # Retry ngoài chu kỳ countdown
        if self.is_capturing and self.countdown == 0 and self.retry_captures_remaining > 0:
            if current_time <= self.capture_retry_deadline:
                try_frame = self.last_frame if self.last_frame is not None else frame
                print(
                    f"🔁 [DEBUG] Retry capture, attempts left: {self.retry_captures_remaining}"
                )
                success = self.capture_image(try_frame)
                self.retry_captures_remaining -= 1
                if success:
                    print("✅ [DEBUG] Retry capture successful!")
                    self.retry_captures_remaining = 0
                    self.capture_retry_deadline = 0.0
                    if len(self.captured_photos) < self.max_photos:
                        self.countdown = 5
                        self.last_countdown_update = current_time
                        print("🔄 [DEBUG] Starting next countdown cycle (after retry)")
                    else:
                        print(
                            "🏁 [DEBUG] Max photos reached after retry, resetting capture mode"
                        )
                        self._reset_capture_mode()
                else:
                    if self.retry_captures_remaining == 0:
                        print(
                            "🛑 [DEBUG] All retry attempts exhausted, resetting capture mode"
                        )
                        self._reset_capture_mode()
            else:
                print("⏳ [DEBUG] Retry window expired, resetting capture mode")
                self.retry_captures_remaining = 0
                self.capture_retry_deadline = 0.0
                self._reset_capture_mode()
    
    def _reset_capture_mode(self) -> None:
        self.is_capturing = False
        self.mode = "OFF"
        self.countdown = 0
        self.peace_sign_count = 0
        self.gesture_stability_count = 0


# Khởi tạo recognizer
recognizer = HandGestureRecognizer()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ New WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"❌ WebSocket disconnected. Remaining: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"Error sending message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Connection error during broadcast: {e}")
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    cap = None
    try:
        await manager.connect(websocket)
        # Khởi tạo camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            await manager.send_personal_message(
                json.dumps({"error": "Cannot open camera"}), websocket
            )
            return
        # Cấu hình camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("📹 Camera initialized for WebSocket connection")
        frame_count = 0
        while True:
            try:
                success, frame = cap.read()
                if not success:
                    print("❌ Failed to read frame from camera")
                    break
                result = recognizer.process_frame(frame)
                await manager.send_personal_message(json.dumps(result), websocket)
                await asyncio.sleep(0.033)  # ~30 FPS
                frame_count += 1
                if frame_count % 300 == 0:
                    print(f"📊 Processed {frame_count} frames")
            except WebSocketDisconnect:
                print("🔌 WebSocket client disconnected")
                break
            except Exception as e:
                print(f"❌ Error processing frame: {e}")
                break
    except Exception as e:
        print(f"❌ WebSocket endpoint error: {e}")
    finally:
        if cap is not None and cap.isOpened():
            cap.release()
            print("📹 Camera released")
        manager.disconnect(websocket)


@app.post("/toggle_mode")
async def toggle_mode():
    current_time = time.time()
    if recognizer.mode == "OFF" and not recognizer.is_capturing:
        recognizer.mode = "ON"
        recognizer.is_capturing = True
        recognizer.countdown = 5
        recognizer.last_countdown_update = current_time
    elif recognizer.mode == "ON" or recognizer.is_capturing:
        recognizer.mode = "OFF"
        recognizer.is_capturing = False
        recognizer.countdown = 0
    return {
        "mode": recognizer.mode,
        "is_capturing": recognizer.is_capturing,
        "countdown": recognizer.countdown,
    }


@app.post("/stop_capture")
async def stop_capture():
    current_time = time.time()
    recognizer.mode = "OFF"
    recognizer.is_capturing = False
    recognizer.countdown = 0
    recognizer.peace_sign_count = 0
    recognizer.gesture_stability_count = 0
    recognizer.last_gesture = "unknown"
    recognizer.gesture_history = []
    recognizer.peace_history = []
    recognizer.manual_stop_time = current_time
    recognizer.retry_captures_remaining = 0
    recognizer.capture_retry_deadline = 0.0
    print("🛑 Capture stopped by user request")
    return {
        "mode": recognizer.mode,
        "is_capturing": recognizer.is_capturing,
        "countdown": recognizer.countdown,
        "message": "Capture stopped successfully",
    }


@app.get("/status")
async def get_status():
    return {
        "mode": recognizer.mode,
        "zoom_level": recognizer.zoom_level,
        "is_capturing": recognizer.is_capturing,
        "countdown": recognizer.countdown,
        "photos_count": len(recognizer.captured_photos),
        "max_photos": recognizer.max_photos,
    }


@app.post("/reset")
async def reset_photos():
    recognizer.captured_photos = []
    recognizer.mode = "OFF"
    recognizer.is_capturing = False
    recognizer.countdown = 0
    recognizer.last_ok_detection = 0.0
    recognizer.last_countdown_update = 0.0
    recognizer.peace_sign_count = 0
    recognizer.last_gesture = "unknown"
    recognizer.gesture_stability_count = 0
    recognizer.last_reported_photos_count = 0
    recognizer.gesture_history = []
    recognizer.peace_history = []
    recognizer.retry_captures_remaining = 0
    recognizer.capture_retry_deadline = 0.0
    return {"message": "Reset successful"}


@app.get("/")
async def root():
    return {"message": "Photobooth AI Backend is running!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)