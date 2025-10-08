#!/bin/bash
# Simple build and run script - no Gradle, no Python

set -e

cd "$(dirname "$0")"

echo "========================================"
echo "VisionBridge - Kotlin Door Detection"
echo "========================================"
echo ""

# Check for kotlinc
if ! command -v kotlinc &> /dev/null; then
    echo "❌ kotlinc not found. Install with:"
    echo "   sudo snap install --classic kotlin"
    exit 1
fi

# Download ONNX Runtime JAR if needed
ONNX_JAR="onnxruntime-1.16.3.jar"
if [ ! -f "$ONNX_JAR" ]; then
    echo "📦 Downloading ONNX Runtime JAR..."
    wget -q --show-progress "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/1.16.3/$ONNX_JAR"
    echo "✅ Downloaded ONNX Runtime"
else
    echo "✅ ONNX Runtime JAR found"
fi

# Download OpenCV JAR if needed
OPENCV_JAR="opencv-4.7.0-0.jar"
if [ ! -f "$OPENCV_JAR" ]; then
    echo "📦 Downloading OpenCV JAR..."
    wget -q --show-progress "https://repo1.maven.org/maven2/org/openpnp/opencv/4.7.0-0/$OPENCV_JAR"
    echo "✅ Downloaded OpenCV"
else
    echo "✅ OpenCV JAR found"
fi

# Extract OpenCV native library
OPENCV_NATIVES_DIR="opencv-natives"
if [ ! -d "$OPENCV_NATIVES_DIR" ]; then
    echo "📦 Extracting OpenCV native libraries..."
    mkdir -p "$OPENCV_NATIVES_DIR"
    unzip -q "$OPENCV_JAR" "nu/pattern/opencv/linux/x86_64/*" -d "$OPENCV_NATIVES_DIR" 2>/dev/null || true
    echo "✅ Extracted OpenCV natives"
fi

# Find the native library path
NATIVE_LIB_PATH="$OPENCV_NATIVES_DIR/nu/pattern/opencv/linux/x86_64"
if [ ! -d "$NATIVE_LIB_PATH" ]; then
    echo "⚠️  OpenCV native library not found, trying to use system OpenCV..."
    NATIVE_LIB_PATH="/usr/lib/x86_64-linux-gnu"
fi

# Check model
if [ ! -f "assets/best.onnx" ]; then
    echo ""
    echo "❌ Model not found: assets/best.onnx"
    echo "   Please run: python src/scripts/export_model_to_onnx.py"
    exit 1
fi

# Compile
echo ""
echo "🔨 Compiling Kotlin code..."
kotlinc -cp "$ONNX_JAR:$OPENCV_JAR" utils/door_detection_functions.kt -include-runtime -d door_detection.jar

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful"
else
    echo "❌ Compilation failed"
    exit 1
fi

# Run
echo ""
echo "🚀 Running live detection..."
echo "   Press 'q' in the window to quit"
echo ""
java -cp "door_detection.jar:$ONNX_JAR:$OPENCV_JAR" -Djava.library.path="$NATIVE_LIB_PATH" Door_detection_functionsKt

echo ""
echo "✅ Done!"
