#!/bin/bash

echo "=== VisionBridge Kotlin Minimal Test ==="

# Check if kotlinc is available
if ! command -v kotlinc &> /dev/null; then
    echo "❌ kotlinc not found. Please install Kotlin compiler."
    echo "   On macOS: brew install kotlin"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OPENAI_API_KEY environment variable is not set"
    echo "   Please set it with: export OPENAI_API_KEY='your-api-key-here'"
    echo "   You can get an API key from: https://platform.openai.com/api-keys"
    echo ""
    read -p "Do you want to continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/
mkdir -p build

# Compile only the minimal files (no OpenCV dependencies)
echo "🔧 Compiling minimal Kotlin files..."
kotlinc utils/AI_door_descriptor.kt SimpleMain.kt -include-runtime -d build/minimal.jar

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo ""
    echo "🚀 Running VisionBridge Minimal Test..."
    
    # Run the application
    java -jar build/minimal.jar
else
    echo "❌ Compilation failed. Please check the error messages above."
    exit 1
fi
