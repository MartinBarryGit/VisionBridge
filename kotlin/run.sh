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

# Detect OS and architecture
OS_NAME=$(uname -s)
ARCH=$(uname -m)

echo "🔍 Detected OS: $OS_NAME, Architecture: $ARCH"

# Set platform-specific paths
case "$OS_NAME" in
    "Darwin")
        if [ "$ARCH" = "arm64" ]; then
            OPENCV_PLATFORM_PATH="nu/pattern/opencv/osx/ARMv8"
            SYSTEM_LIB_PATH="/opt/homebrew/lib"
        else
            OPENCV_PLATFORM_PATH="nu/pattern/opencv/osx/x86_64"
            SYSTEM_LIB_PATH="/usr/local/lib"
        fi
        ;;
    "Linux")
        OPENCV_PLATFORM_PATH="nu/pattern/opencv/linux/x86_64"
        SYSTEM_LIB_PATH="/usr/lib/x86_64-linux-gnu"
        ;;
    *)
        echo "❌ Unsupported OS: $OS_NAME"
        exit 1
        ;;
esac

# Extract OpenCV native library
OPENCV_NATIVES_DIR="opencv-natives"
if [ ! -d "$OPENCV_NATIVES_DIR" ]; then
    echo "📦 Extracting OpenCV native libraries for $OS_NAME..."
    mkdir -p "$OPENCV_NATIVES_DIR"
    
    echo "� Extracting from: $OPENCV_PLATFORM_PATH/*"
    unzip -q "$OPENCV_JAR" "$OPENCV_PLATFORM_PATH/*" -d "$OPENCV_NATIVES_DIR"
    
    if [ "$(ls -A $OPENCV_NATIVES_DIR 2>/dev/null)" ]; then
        echo "✅ Extracted OpenCV natives"
    else
        echo "❌ Failed to extract native libraries"
        exit 1
    fi
fi

# Find the native library path
NATIVE_LIB_PATH="$OPENCV_NATIVES_DIR/$OPENCV_PLATFORM_PATH"

if [ ! -d "$NATIVE_LIB_PATH" ]; then
    echo "❌ Native library path not found: $NATIVE_LIB_PATH"
    exit 1
fi

echo "✅ Using native library path: $NATIVE_LIB_PATH"

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

# Add native access flag for newer Java versions and better library path handling
JAVA_OPTS="--enable-native-access=ALL-UNNAMED -Djava.library.path=$NATIVE_LIB_PATH"

# For macOS, also try common OpenCV installation paths
if [ "$OS_NAME" = "Darwin" ]; then
    if [ -d "/opt/homebrew/lib" ]; then
        JAVA_OPTS="$JAVA_OPTS:/opt/homebrew/lib"
    fi
    if [ -d "/usr/local/lib" ]; then
        JAVA_OPTS="$JAVA_OPTS:/usr/local/lib"
    fi
fi

echo "Using library path: $NATIVE_LIB_PATH"
java -cp "door_detection.jar:$ONNX_JAR:$OPENCV_JAR" $JAVA_OPTS Door_detection_functionsKt

echo ""
echo "✅ Done!"
