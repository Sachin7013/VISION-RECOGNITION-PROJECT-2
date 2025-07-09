from ultralytics import YOLO

model = YOLO("yolov9e.pt")

def detect_objects(frame):
    results = model(frame)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id]
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "label": label,
            "confidence": conf,
            "bbox": [x1, y1, x2 - x1, y2 - y1]
        })
    return detections
