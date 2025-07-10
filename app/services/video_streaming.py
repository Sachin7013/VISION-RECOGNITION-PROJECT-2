import cv2
import os
from fastapi.responses import StreamingResponse
from app.services import object_detector, object_tracker

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def stream_video(filename: str):
    """
    Stream video frames with object detection
    """
    path = os.path.join(UPLOAD_DIR, filename)
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {path}")

    def generate_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Here you can do object detection (placeholder)
            # Example: draw a dummy rectangle
            cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")