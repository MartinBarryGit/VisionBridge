import os

from ultralytics import YOLO

from config import parent_dir

model_path = os.path.join(
    parent_dir,
    "runs/detect/multi_dataset/weights/best.pt"
)
model = YOLO(model_path)
print(f"Modèle chargé depuis : {model_path}")

export_dir = os.path.join(
    parent_dir,
    "runs/detect/multi_dataset/weights"
)

tflite_float32 = model.export(
    format="tflite",
    imgsz=640,
    half=False,
    dynamic=False,
    nms=True
)
print(f"Modèle TFLite Float32 : {tflite_float32}")

tflite_float16 = model.export(
    format="tflite",
    imgsz=640,
    half=True,
    dynamic=False,
    nms=True
)
print(f"Modèle TFLite Float16 : {tflite_float16}")

tflite_model = YOLO(tflite_float32)  
results = tflite_model("pred.png")
results[0].show()
