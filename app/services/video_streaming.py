import cv2
import os
from fastapi.responses import StreamingResponse
from app.services import object_detector, object_tracker

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

frame_count = 0
DETECTION_INTERVAL = 5  # Detect every 5 frames
RESIZE_WIDTH = 640    


def stream_video(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise RuntimeError(f"Video file does not exist: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {path}")

    def generate_frames(cap):
        tracks = []  # Keep last tracks for interpolating between detections

        while True:
            success, frame = cap.read()
            if not success:
                break

            # Resize frame for faster processing
            scale_factor = RESIZE_WIDTH / frame.shape[1]
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * scale_factor)))

            global frame_count
            frame_count += 1

            # Detect every N frames
            if frame_count % DETECTION_INTERVAL == 0:
                detections = object_detector.detect_objects(frame)
                tracks = object_tracker.track_objects(frame, detections)

            # Draw tracks (even on intermediate frames)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                x1, y1, x2, y2 = map(int, track.to_ltrb())
                label = track.det_class
                track_id = track.track_id

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} #{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            # Yield frame for streaming
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return StreamingResponse(generate_frames(cap), media_type="multipart/x-mixed-replace; boundary=frame")