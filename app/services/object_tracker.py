from deep_sort_realtime.deepsort_tracker import DeepSort

import math

tracker = DeepSort(max_age=30)

def track_objects(frame, detections):
    formatted = []
    for det in detections:
        x, y, w, h = det['bbox']
        formatted.append(([x, y, x+w, y+h], det['confidence'], det['label']))
    return tracker.update_tracks(formatted, frame=frame)

def calculate_angle(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))
