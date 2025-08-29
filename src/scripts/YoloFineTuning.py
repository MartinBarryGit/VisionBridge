from ultralytics import YOLO
import os
from config import data_dir

# Option 1: Use a pre-trained YOLOv8 model (recommended)
model = YOLO('yolov8n.pt')  # or 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
# This will automatically download the model if not present
model.train(data=f'{data_dir}/DoorDetect_yolo_training/dataset.yaml', epochs=5)
valid_results = model.val()
print(valid_results)