#!/bin/bash

echo "=== VisionBridge Kotlin Simple Setup ==="

# Check if kotlinc is available
if ! command -v kotlinc &> /dev/null; then
    echo "❌ kotlinc not found. Please install Kotlin compiler."
    echo "   On macOS: brew install kotlin"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY environment variable is not set"
    echo "   Please set it with: export OPENAI_API_KEY='your-api-key-here'"
    echo "   You can get an API key from: https://platform.openai.com/api-keys"
    echo ""
    echo "   To set it for this session:"
    echo "   export OPENAI_API_KEY='sk-your-key-here'"
    echo ""
fi

# Use existing JARs in the directory
CLASSPATH="onnxruntime-1.16.3.jar:opencv-4.7.0-0.jar:."

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
