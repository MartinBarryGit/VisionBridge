import os

import cv2
import numpy as np
from ultralytics import YOLO

from config import parent_dir

alpha = 0.2           
iou_threshold = 0.5 
max_missed_frames = 30
model_path = os.path.join(parent_dir, "runs/detect/multi_dataset/weights/best.pt")

class Track:
    def __init__(self, box, score):
        self.box = box
        self.score = score
        self.missed = 0

    def update(self, new_box, new_score):
        self.box = alpha * new_box + (1 - alpha) * self.box
        self.score = alpha * new_score + (1 - alpha) * self.score
        self.missed = 0

    def mark_missed(self):
        self.missed += 1

    def is_active(self):
        return self.missed < max_missed_frames


def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2b - x1b) * (y2b - y1b)

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

tracks = []

def smooth_with_tracking(new_boxes, conf_threshold=0.3):
    global tracks
    updated_tracks = [False] * len(tracks)
    used_track_indices = set()

    for new_box_obj in new_boxes:
        box = new_box_obj.xyxy[0].cpu().numpy()
        score = float(new_box_obj.conf.cpu().numpy())

        if score < conf_threshold:
            continue

        matched = False
        for i, track in enumerate(tracks):
            if i in used_track_indices:
                continue
            if iou(box, track.box) > iou_threshold:
                track.update(box, score)
                updated_tracks[i] = True
                used_track_indices.add(i)
                matched = True
                break

        if not matched:
            tracks.append(Track(box, score))
            updated_tracks.append(True)

    for i, was_updated in enumerate(updated_tracks):
        if not was_updated:
            tracks[i].mark_missed()

    tracks = [t for t in tracks if t.is_active()]

    return [t.box for t in tracks], [t.score for t in tracks]

def main():
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Impossible d'ouvrir la webcam")
        return

    print("Appuie 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Impossible de lire une frame")
            break

        results = model(frame)
        boxes = results[0].boxes

        smoothed_boxes, smoothed_scores = smooth_with_tracking(boxes)

        annotated_frame = frame.copy()
        for box, score in zip(smoothed_boxes, smoothed_scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"door {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Live Detection (EMA + IoU + Tracking)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
