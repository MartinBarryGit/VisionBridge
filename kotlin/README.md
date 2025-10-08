# Kotlin Door Detection - Quick Test

Simple Kotlin version matching the Python `door_detection_functions.py`.

## Quick Start

1. **Export model to ONNX** (if not done already):
```bash
cd ..
python src/scripts/export_model_to_onnx.py
cd kotlin
```

2. **Run the Kotlin version**:
```bash
chmod +x run.sh
./run.sh
```

Or manually:
```bash
gradle wrapper
./gradlew run
```

## What it does

- Loads `kotlin/assets/best.onnx` model
- Reads test image from `dataset/DoorDetect_yolo_training/images/val/train_000000.jpg`
- Runs YOLO detection with tracking (exactly like Python version)
- Shows annotated image in a window

## Files

- `utils/door_detection_functions.kt` - Main code (mirrors Python version)
- `build.gradle.kts` - Gradle build config
- `run.sh` - Quick run script

## Output

Similar to Python version:
```
Model loaded: kotlin/assets/best.onnx
Loading image: dataset/DoorDetect_yolo_training/images/val/train_000000.jpg
Running detection...
Door 0: Center at 320, Score: 0.85, Image center: 320
```

Shows window with green boxes around detected doors.

## Next Steps

- Add webcam support (like Python `cv2.VideoCapture(0)`)
- Add sound playback (like Python `SoundPlayer`)
- Port to Android once tested on desktop
