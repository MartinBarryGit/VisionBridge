#!/bin/bash

echo "=== VisionBridge Kotlin Setup and Run Script (No Gradle) ==="

# Check if kotlinc is available
if ! command -v kotlinc &> /dev/null; then
    echo "❌ kotlinc not found. Please install Kotlin compiler."
    echo "   On macOS: brew install kotlin"
    echo "   Or download from: https://kotlinlang.org/docs/command-line.html"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable is not set"
    echo "   Please set it with: export OPENAI_API_KEY='your-api-key-here'"
    echo "   You can get an API key from: https://platform.openai.com/api-keys"
    echo ""
fi

# Create lib directory for dependencies
mkdir -p lib

# Download required JAR files if they don't exist
echo "📦 Checking dependencies..."

# ONNX Runtime
if [ ! -f "lib/onnxruntime-1.16.3.jar" ]; then
    echo "Downloading ONNX Runtime..."
    curl -L -o lib/onnxruntime-1.16.3.jar "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/1.16.3/onnxruntime-1.16.3.jar"
fi

# OpenAI Java Client
if [ ! -f "lib/openai-java-0.12.0.jar" ]; then
    echo "Downloading OpenAI Java client..."
    curl -L -o lib/openai-java-0.12.0.jar "https://repo1.maven.org/maven2/com/openai/openai-java/0.12.0/openai-java-0.12.0.jar"
fi

# OkHttp (required by OpenAI client)
if [ ! -f "lib/okhttp-4.12.0.jar" ]; then
    echo "Downloading OkHttp..."
    curl -L -o lib/okhttp-4.12.0.jar "https://repo1.maven.org/maven2/com/squareup/okhttp3/okhttp/4.12.0/okhttp-4.12.0.jar"
fi

# Okio (required by OkHttp)
if [ ! -f "lib/okio-3.6.0.jar" ]; then
    echo "Downloading Okio..."
    curl -L -o lib/okio-3.6.0.jar "https://repo1.maven.org/maven2/com/squareup/okio/okio/3.6.0/okio-3.6.0.jar"
fi

# Kotlin stdlib
if [ ! -f "lib/kotlin-stdlib-1.9.20.jar" ]; then
    echo "Downloading Kotlin stdlib..."
    curl -L -o lib/kotlin-stdlib-1.9.20.jar "https://repo1.maven.org/maven2/org/jetbrains/kotlin/kotlin-stdlib/1.9.20/kotlin-stdlib-1.9.20.jar"
fi

# Build classpath
CLASSPATH="lib/*:opencv-4.7.0-0.jar:."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/
mkdir -p build

# Compile Kotlin files
echo "🔧 Compiling Kotlin files..."
kotlinc -cp "$CLASSPATH" utils/*.kt Main.kt -d build/

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 Running VisionBridge..."
    
    # Set library path for native libraries
    export DYLD_LIBRARY_PATH="opencv-natives/nu/pattern/opencv/osx/ARMv8:$DYLD_LIBRARY_PATH"
    
    # Run the application
    java -cp "$CLASSPATH:build" -Djava.library.path="opencv-natives/nu/pattern/opencv/osx/ARMv8" MainKt
else
    echo "❌ Compilation failed. Please check the error messages above."
    exit 1
fi
