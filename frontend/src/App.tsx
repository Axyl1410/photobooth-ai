import { useEffect, useRef, useState } from "react";
import "./App.css";

interface Photo {
  id: string;
  dataUrl: string;
  timestamp: number;
}

interface WebSocketData {
  frame: string;
  gesture: string;
  zoom_level: number;
  mode: string;
  is_capturing: boolean;
  countdown: number;
  photos_count: number;
  photos_updated?: boolean;
  photos?: Photo[];
  peace_sign_count?: number; // ✅ Thêm Peace sign counter
  required_peace_count?: number; // ✅ Thêm số lần Peace cần thiết
  gesture_stability_count?: number; // ✅ Thêm gesture stability counter
  gesture_stability_required?: number; // ✅ Thêm số frame stability cần thiết
  // ✅ Retry capture status from backend
  retrying?: boolean;
  retry_attempts_remaining?: number;
}

type AppState = "capturing" | "selecting" | "composing" | "result";

function App() {
  const [appState, setAppState] = useState<AppState>("capturing");
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [selectedPhotos, setSelectedPhotos] = useState<string[]>([]);
  const [countdown, setCountdown] = useState(0);
  const [finalResult, setFinalResult] = useState<string | null>(null);
  const [appliedFilter, setAppliedFilter] = useState<string | null>(null);
  // composing flag not needed for rendering

  // New state for AI integration
  const [mode, setMode] = useState<"ON" | "OFF">("OFF");
  const [zoomLevel, setZoomLevel] = useState(1.0);
  const [currentGesture, setCurrentGesture] = useState("unknown");
  // wsData/peaceSignCount/requiredPeaceCount not needed with ref-based rendering
  const [gestureStabilityCount, setGestureStabilityCount] = useState(0); // ✅ Thêm gesture stability state
  const [gestureStabilityRequired, setGestureStabilityRequired] = useState(3); // ✅ Thêm required stability state
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryLeft, setRetryLeft] = useState(0);

  const imageRef = useRef<HTMLImageElement>(null);
  const pendingFrameRef = useRef<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [hasFrame, setHasFrame] = useState(false);
  const hasAutoSwitchedRef = useRef(false);
  const hasFrameRef = useRef(false);

  // Auto-cancel capture on page load/reload
  useEffect(() => {
    const handlePageLoad = async () => {
      try {
        // Tự động hủy capture khi load trang
        await fetch("http://localhost:8000/stop_capture", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
        });
        console.log("🔄 Auto-cancelled capture on page load");
      } catch {
        console.log("ℹ️ No active capture to cancel");
      }
    };

    handlePageLoad();
  }, []); // Chỉ chạy một lần khi component mount

  // Initialize WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket("ws://localhost:8000/ws");
        wsRef.current = ws;

        ws.onopen = () => {
          console.log("🔌 Connected to AI backend");
        };

        ws.onmessage = (event) => {
          try {
            const data: WebSocketData = JSON.parse(event.data);
            // Update frame with first-frame buffering
            if (data.frame) {
              if (!hasFrameRef.current) {
                pendingFrameRef.current = data.frame;
                hasFrameRef.current = true;
                setHasFrame(true);
              } else if (imageRef.current) {
                imageRef.current.src = data.frame;
              } else {
                pendingFrameRef.current = data.frame;
              }
            }

            // Only set state when values change to minimize renders
            setMode((prev) =>
              prev !== (data.mode as "ON" | "OFF")
                ? (data.mode as "ON" | "OFF")
                : prev
            );
            setZoomLevel((prev) =>
              prev !== data.zoom_level ? data.zoom_level : prev
            );
            setCurrentGesture((prev) =>
              prev !== data.gesture ? data.gesture : prev
            );
            setCountdown((prev) =>
              prev !== data.countdown ? data.countdown : prev
            );

            if (data.photos_updated && data.photos) {
              setPhotos(data.photos);

              // Auto proceed to selection after 6 photos - check right after photos update
              if (
                data.photos.length >= 6 &&
                appState === "capturing" &&
                !hasAutoSwitchedRef.current
              ) {
                hasAutoSwitchedRef.current = true;
                setTimeout(() => setAppState("selecting"), 2000);
              }
            }
            setGestureStabilityCount(data.gesture_stability_count || 0); // ✅ Cập nhật gesture stability
            setGestureStabilityRequired(data.gesture_stability_required || 3); // ✅ Cập nhật required stability
            setIsRetrying(Boolean(data.retrying));
            setRetryLeft(data.retry_attempts_remaining || 0);
          } catch (error) {
            console.error("Error parsing WebSocket data:", error);
          }
        };

        ws.onclose = () => {
          console.log("🔌 Disconnected from AI backend");
          // Reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (error) => {
          console.error("WebSocket error:", error);
        };
      } catch (error) {
        console.error("Error connecting to WebSocket:", error);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [appState]);

  // Apply the very first buffered frame once the <img> exists
  useEffect(() => {
    if (hasFrame && imageRef.current && pendingFrameRef.current) {
      imageRef.current.src = pendingFrameRef.current;
      pendingFrameRef.current = null;
    }
  }, [hasFrame]);

  const [isToggling, setIsToggling] = useState(false);

  const toggleMode = async () => {
    if (isToggling) return; // Prevent multiple clicks

    try {
      setIsToggling(true);
      const response = await fetch("http://localhost:8000/toggle_mode", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await response.json();
      setMode(data.mode);
      // capture flag used only to manage backend mode, no UI dependency
      setCountdown(data.countdown);
    } catch (error) {
      console.error("Error toggling mode:", error);
    } finally {
      setTimeout(() => setIsToggling(false), 1000);
    }
  };

  const resetPhotos = async () => {
    try {
      await fetch("http://localhost:8000/reset", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      setPhotos([]);
      setSelectedPhotos([]);
      hasAutoSwitchedRef.current = false;
      hasFrameRef.current = false;
      setMode("OFF");
      setCountdown(0);
      setGestureStabilityCount(0); // ✅ Reset gesture stability
      setHasFrame(false);
    } catch (error) {
      console.error("Error resetting photos:", error);
    }
  };

  const stopCapture = async () => {
    try {
      const response = await fetch("http://localhost:8000/stop_capture", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await response.json();
      setMode(data.mode);
      setCountdown(0);
    } catch (error) {
      console.error("Error stopping capture:", error);
    }
  };

  const selectPhoto = (photoId: string) => {
    setSelectedPhotos((prev) => {
      if (prev.includes(photoId)) {
        return prev.filter((id) => id !== photoId);
      } else if (prev.length < 3) {
        return [...prev, photoId];
      }
      return prev;
    });
  };

  const confirmSelection = () => {
    if (selectedPhotos.length === 3) {
      setAppState("composing");
      // Use setTimeout to ensure state update is processed before composePhotos
      setTimeout(() => {
        composePhotos();
      }, 0);
    }
  };

  const composePhotos = () => {
    const selectedPhotoData = selectedPhotos
      .map((id) => photos.find((photo) => photo.id === id))
      .filter(Boolean) as Photo[];

    // Load the frame image first
    const frameImg = new Image();
    frameImg.onload = () => {
      // Create composition canvas with frame dimensions
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Set canvas size to match frame image
      canvas.width = frameImg.width;
      canvas.height = frameImg.height;

      // Calculate photo dimensions based on frame
      const frameWidth = frameImg.width;
      const frameHeight = frameImg.height;
      const slotHeight = Math.floor(frameHeight / 3); // Divide frame into 3 equal vertical slots
      const slotWidth = Math.floor(frameWidth * 0.95); // Use 95% of frame width for photos to fill more space
      const slotX = Math.floor((frameWidth - slotWidth) / 2); // Center horizontally

      // Draw photos first (behind the frame)
      let photosLoaded = 0;
      const totalPhotos = selectedPhotoData.length;

      selectedPhotoData.forEach((photo, index) => {
        const img = new Image();
        img.onload = () => {
          const y = index * slotHeight;

          // Force photo to fill the entire slot width to eliminate side borders
          // This will crop top/bottom if needed but ensures no side borders
          const drawX = slotX;
          const drawY = y;
          const drawWidth = slotWidth;
          const drawHeight = slotHeight;

          // Draw photo to completely fill the slot (no aspect ratio preservation)
          // This ensures no black borders on the sides
          ctx.drawImage(img, drawX, drawY, drawWidth, drawHeight);

          photosLoaded++;

          // When all photos are loaded, draw the frame on top
          if (photosLoaded === totalPhotos) {
            // Draw the frame on top (this will only show non-transparent parts)
            ctx.drawImage(frameImg, 0, 0);

            // Convert to data URL and update state
            const resultDataUrl = canvas.toDataURL("image/jpeg", 0.9);
            setFinalResult(resultDataUrl);
            setAppState("result");
          }
        };
        img.src = photo.dataUrl;
      });
    };

    // Load the frame image
    frameImg.src = "/frame.png";
  };

  const applyFilter = (filterType: string) => {
    setAppliedFilter(filterType === "original" ? null : filterType);
  };

  const downloadResult = () => {
    if (!finalResult) return;

    const link = document.createElement("a");
    link.download = "photobooth-result.jpg";
    link.href = finalResult;
    link.click();
  };

  const takeNew = () => {
    resetPhotos();
    setAppState("capturing");
    setFinalResult(null);
    setAppliedFilter(null);
    // Reset camera stream state to force reload
    setHasFrame(false);
    hasFrameRef.current = false;
    hasAutoSwitchedRef.current = false;
    if (imageRef.current) {
      imageRef.current.src = "";
    }
    pendingFrameRef.current = null;
  };

  const renderCapturing = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 relative overflow-hidden">
      {/* Grid Background */}
      <div className="absolute inset-0 bg-[linear-gradient(rgba(59,130,246,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(59,130,246,0.1)_1px,transparent_1px)] bg-[size:50px_50px]"></div>

      <div className="relative z-10 container mx-auto px-4 py-4 lg:py-8">
        <div className="text-center mb-4 lg:mb-8">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-slate-800 mb-2">
            Photobooth with AI
          </h1>
          <p className="text-sm sm:text-base text-slate-600">
            <span className="hidden sm:inline">
              Hand gesture control - 👊 Zoom Out | ✋ Zoom In | ✌🏻 Peace Sign (2
              fingers) to Capture
            </span>
            <span className="sm:hidden">
              👊 Zoom Out | ✋ Zoom In | ✌🏻 Capture
            </span>
          </p>
        </div>

        <div className="flex flex-col lg:flex-row gap-4 lg:gap-8 max-w-7xl mx-auto">
          {/* Camera Section */}
          <div className="flex-1 mx-auto order-1 lg:order-1">
            <div
              className="relative bg-white rounded-xl lg:rounded-2xl shadow-2xl overflow-hidden"
              style={{ aspectRatio: "9/6" }}
            >
              {/* AI Camera Stream */}
              {hasFrame ? (
                <img
                  ref={imageRef}
                  alt="AI Camera Stream"
                  className="w-full h-full object-cover"
                />
              ) : (
                <div className="w-full h-full bg-gray-200 flex items-center justify-center">
                  <div className="text-gray-500 text-lg">
                    Connecting to AI Camera...
                  </div>
                </div>
              )}

              {/* Countdown Overlay */}
              {countdown > 0 && (
                <div className="absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center">
                  <div className="text-white text-[6rem] sm:text-[8rem] lg:text-[10rem] font-black countdown-number opacity-90">
                    {countdown}
                  </div>
                </div>
              )}

              {/* Status Overlay */}
              <div className="absolute top-2 left-2 lg:top-4 lg:left-4 bg-black bg-opacity-70 text-white px-2 lg:px-4 py-1 lg:py-2 rounded-lg status-overlay">
                <div className="text-xs lg:text-sm">
                  <div className="flex items-center gap-1 lg:gap-2 lg:block min-w-0">
                    <span className="flex-shrink-0">Mode:</span>
                    <span
                      className={`font-bold flex-shrink-0 ${
                        mode === "ON" ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {mode}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 lg:gap-2 lg:block min-w-0">
                    <span className="flex-shrink-0">Zoom:</span>
                    <span className="font-bold text-blue-400 flex-shrink-0">
                      {zoomLevel}x
                    </span>
                  </div>
                  <div className="flex items-center gap-1 lg:gap-2 lg:block min-w-0">
                    <span className="flex-shrink-0">Gesture:</span>
                    <span className="font-bold text-yellow-400 flex-shrink-0 truncate">
                      {currentGesture}
                    </span>
                  </div>
                  {/* ✅ Peace sign progress - responsive và không tràn */}
                  {currentGesture === "peace" && mode === "OFF" && (
                    <div className="mt-2 hidden lg:block max-w-full min-w-0">
                      <div className="text-xs text-gray-300 truncate">
                        Peace Sign Stability:
                      </div>
                      <div className="flex items-center gap-1 lg:gap-2 min-w-0 progress-container">
                        <div className="w-16 lg:w-20 bg-gray-600 rounded-full h-2 flex-shrink-0 progress-bar">
                          <div
                            className="bg-yellow-400 h-2 rounded-full transition-all duration-200"
                            style={{
                              width: `${
                                (gestureStabilityCount /
                                  gestureStabilityRequired) *
                                100
                              }%`,
                            }}
                          ></div>
                        </div>
                        <span className="text-xs font-bold text-yellow-400 flex-shrink-0">
                          {gestureStabilityCount}/{gestureStabilityRequired}
                        </span>
                      </div>
                      {gestureStabilityCount >= gestureStabilityRequired && (
                        <div className="text-xs text-green-400 mt-1 truncate max-w-full">
                          ✅ Ready!
                        </div>
                      )}
                    </div>
                  )}
                  {/* ✅ Retry indicator - compact and no overflow */}
                  {isRetrying && (
                    <div className="mt-1 hidden lg:block text-xs text-yellow-300 truncate max-w-full">
                      Retrying capture... ({retryLeft})
                    </div>
                  )}
                </div>
              </div>

              {/* Camera Frame */}
              <div className="absolute inset-0 border-4 border-blue-600 rounded-2xl pointer-events-none"></div>
            </div>

            {/* Control Buttons */}
            <div className="mt-4 lg:mt-6 space-y-2 lg:space-y-3">
              {/* Mode Toggle Button */}
              <button
                onClick={toggleMode}
                disabled={photos.length >= 6 || isToggling}
                className={`w-full font-bold py-3 lg:py-4 px-6 lg:px-8 rounded-xl text-lg lg:text-xl transition-colors duration-200 touch-manipulation ${
                  mode === "ON"
                    ? "bg-green-600 hover:bg-green-700 text-white"
                    : "bg-red-600 hover:bg-red-700 text-white"
                } ${
                  photos.length >= 6 || isToggling
                    ? "opacity-50 cursor-not-allowed"
                    : ""
                }`}
              >
                {photos.length >= 6 ? (
                  <>
                    <span className="hidden sm:inline">
                      Maximum Photos Reached
                    </span>
                    <span className="sm:hidden">Max Photos</span>
                  </>
                ) : isToggling ? (
                  "Switching..."
                ) : (
                  `MODE: ${mode}`
                )}
              </button>

              {/* Stop Capture Button - chỉ hiển thị khi đang capturing hoặc countdown */}
              {(mode === "ON" || countdown > 0) && (
                <button
                  onClick={stopCapture}
                  className="w-full font-bold py-3 lg:py-3 px-6 lg:px-6 rounded-xl text-base lg:text-lg bg-orange-600 hover:bg-orange-700 text-white transition-colors duration-200 touch-manipulation"
                >
                  🛑 Stop Capture
                </button>
              )}

              {/* Reset Photos Button */}
              {photos.length > 0 && (
                <button
                  onClick={resetPhotos}
                  className="w-full font-bold py-3 lg:py-3 px-6 lg:px-6 rounded-xl text-base lg:text-lg bg-gray-600 hover:bg-gray-700 text-white transition-colors duration-200 touch-manipulation"
                >
                  🗑️ <span className="hidden sm:inline">Clear All Photos</span>
                  <span className="sm:hidden">Clear</span>
                </button>
              )}
            </div>
          </div>

          {/* Thumbnails Section */}
          <div className="w-full lg:w-64 order-2 lg:order-2">
            <h3 className="text-base lg:text-lg font-semibold text-slate-700 mb-2 lg:mb-4">
              Photos ({photos.length}/6)
            </h3>
            <div className="flex lg:flex-col gap-2 lg:space-y-2 overflow-x-auto lg:overflow-x-visible pb-2 lg:pb-0">
              {photos.map((photo, index) => (
                <div key={photo.id} className="relative flex-shrink-0">
                  <img
                    src={photo.dataUrl}
                    alt={`Photo ${index + 1}`}
                    className="w-24 h-16 lg:w-full lg:h-36 object-cover rounded-lg shadow-md"
                    style={{ aspectRatio: "9/6" }}
                  />
                  <div className="absolute top-1 right-1 lg:top-2 lg:right-2 bg-blue-600 text-white text-xs px-1.5 lg:px-2 py-0.5 lg:py-1 rounded-full">
                    {index + 1}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSelecting = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 relative overflow-hidden">
      <div className="absolute inset-0 bg-[linear-gradient(rgba(59,130,246,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(59,130,246,0.1)_1px,transparent_1px)] bg-[size:50px_50px]"></div>

      <div className="relative z-10 container mx-auto px-4 py-4 lg:py-8">
        <div className="text-center mb-4 lg:mb-8">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-slate-800 mb-2">
            Select 3 Photos
          </h1>
          <p className="text-sm sm:text-base text-slate-600">
            Choose your favorite photos to compose
          </p>
        </div>

        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3 lg:gap-6 mb-6 lg:mb-8">
            {photos.map((photo, index) => (
              <div
                key={photo.id}
                onClick={() => selectPhoto(photo.id)}
                className={`relative cursor-pointer transform transition-all duration-200 hover:scale-105 touch-manipulation ${
                  selectedPhotos.includes(photo.id)
                    ? "ring-4 ring-blue-500 scale-105"
                    : "hover:shadow-lg"
                }`}
              >
                <img
                  src={photo.dataUrl}
                  alt={`Photo ${index + 1}`}
                  className="w-full h-32 sm:h-48 lg:h-64 object-cover rounded-lg lg:rounded-xl shadow-md"
                  style={{ aspectRatio: "9/6" }}
                />
                {selectedPhotos.includes(photo.id) && (
                  <div className="absolute top-2 right-2 lg:top-4 lg:right-4 bg-blue-600 text-white w-6 h-6 lg:w-8 lg:h-8 rounded-full flex items-center justify-center font-bold text-sm lg:text-base">
                    ✓
                  </div>
                )}
                <div className="absolute bottom-2 left-2 lg:bottom-4 lg:left-4 bg-black bg-opacity-50 text-white px-2 py-1 lg:px-3 lg:py-1 rounded-full text-xs lg:text-sm">
                  Photo {index + 1}
                </div>
              </div>
            ))}
          </div>

          <div className="text-center">
            <p className="text-sm sm:text-base text-slate-600 mb-4">
              Selected: {selectedPhotos.length}/3 photos
            </p>
            <button
              onClick={confirmSelection}
              disabled={selectedPhotos.length !== 3}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-bold py-3 lg:py-4 px-6 lg:px-8 rounded-xl text-lg lg:text-xl transition-colors duration-200 touch-manipulation"
            >
              Confirm Selection
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  const renderComposing = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 relative overflow-hidden">
      <div className="absolute inset-0 bg-[linear-gradient(rgba(59,130,246,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(59,130,246,0.1)_1px,transparent_1px)] bg-[size:50px_50px]"></div>

      <div className="relative z-10 container mx-auto px-4 py-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-slate-800 mb-2">
            Composing Your Photos
          </h1>
          <p className="text-slate-600 mb-8">
            Please wait while we create your photobooth result...
          </p>

          <div className="flex justify-center">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600"></div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderResult = () => (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 relative overflow-hidden">
      <div className="absolute inset-0 bg-[linear-gradient(rgba(59,130,246,0.1)_1px,transparent_1px),linear-gradient(90deg,rgba(59,130,246,0.1)_1px,transparent_1px)] bg-[size:50px_50px]"></div>

      <div className="relative z-10 container mx-auto px-4 py-4 lg:py-8">
        <div className="text-center mb-4 lg:mb-8">
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-bold text-slate-800 mb-2">
            Your Photobooth Result
          </h1>
          <p className="text-sm sm:text-base text-slate-600">
            Download or take new photos
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          {finalResult && (
            <div className="relative mb-6 lg:mb-8">
              <img
                src={finalResult}
                alt="Photobooth Result"
                className={`w-full max-w-sm sm:max-w-md lg:max-w-lg mx-auto rounded-xl lg:rounded-2xl shadow-2xl object-contain ${
                  appliedFilter === "white"
                    ? "brightness-110 contrast-105"
                    : appliedFilter === "pink"
                    ? "sepia-20 saturate-150 hue-rotate-320"
                    : appliedFilter === "black"
                    ? "brightness-70 contrast-120 saturate-80"
                    : appliedFilter === "yellow"
                    ? "sepia-40 saturate-150 hue-rotate-30 brightness-105"
                    : ""
                }`}
              />
            </div>
          )}

          {/* Filter Options */}
          <div className="flex justify-center gap-2 lg:gap-3 mb-6 lg:mb-8 flex-wrap">
            <button
              onClick={() => applyFilter("original")}
              className={`px-3 lg:px-5 py-2 rounded-lg font-semibold transition-colors touch-manipulation text-sm lg:text-base ${
                appliedFilter === "original" || appliedFilter === null
                  ? "bg-blue-200 text-blue-800"
                  : "bg-white text-gray-600 hover:bg-gray-100"
              }`}
            >
              Original
            </button>
            <button
              onClick={() => applyFilter("white")}
              className={`px-3 lg:px-5 py-2 rounded-lg font-semibold transition-colors touch-manipulation text-sm lg:text-base ${
                appliedFilter === "white"
                  ? "bg-gray-200 text-gray-800"
                  : "bg-white text-gray-600 hover:bg-gray-100"
              }`}
            >
              White
            </button>
            <button
              onClick={() => applyFilter("pink")}
              className={`px-3 lg:px-5 py-2 rounded-lg font-semibold transition-colors touch-manipulation text-sm lg:text-base ${
                appliedFilter === "pink"
                  ? "bg-pink-200 text-pink-800"
                  : "bg-white text-gray-600 hover:bg-gray-100"
              }`}
            >
              Pink
            </button>
            <button
              onClick={() => applyFilter("black")}
              className={`px-3 lg:px-5 py-2 rounded-lg font-semibold transition-colors touch-manipulation text-sm lg:text-base ${
                appliedFilter === "black"
                  ? "bg-gray-800 text-white"
                  : "bg-white text-gray-600 hover:bg-gray-100"
              }`}
            >
              Black
            </button>
            <button
              onClick={() => applyFilter("yellow")}
              className={`px-3 lg:px-5 py-2 rounded-lg font-semibold transition-colors touch-manipulation text-sm lg:text-base ${
                appliedFilter === "yellow"
                  ? "bg-yellow-200 text-yellow-800"
                  : "bg-white text-gray-600 hover:bg-gray-100"
              }`}
            >
              Yellow
            </button>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col sm:flex-row justify-center gap-4 lg:gap-6">
            <button
              onClick={downloadResult}
              className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 lg:py-4 px-6 lg:px-8 rounded-xl text-lg lg:text-xl transition-colors duration-200 touch-manipulation"
            >
              Download
            </button>
            <button
              onClick={takeNew}
              className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 lg:py-4 px-6 lg:px-8 rounded-xl text-lg lg:text-xl transition-colors duration-200 touch-manipulation"
            >
              Take New
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  switch (appState) {
    case "selecting":
      return renderSelecting();
    case "composing":
      return renderComposing();
    case "result":
      return renderResult();
    default:
      return renderCapturing();
  }
}

export default App;
