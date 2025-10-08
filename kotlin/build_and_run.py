#!/usr/bin/env python3
"""
Run Kotlin code via Jython/JPype or just compile with kotlinc properly
This downloads dependencies and compiles correctly
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# Configuration
ONNX_JAR = "onnxruntime-1.16.3.jar"
ONNX_URL = f"https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/1.16.3/{ONNX_JAR}"
KOTLIN_FILE = "utils/door_detection_functions.kt"
OUTPUT_JAR = "door_detection.jar"

def check_kotlin():
    """Check if kotlinc is installed"""
    try:
        subprocess.run(["kotlinc", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ kotlinc not found. Install with:")
        print("   sudo snap install --classic kotlin")
        return False

def download_onnx_jar():
    """Download ONNX Runtime JAR if not present"""
    if os.path.exists(ONNX_JAR):
        print(f"✅ {ONNX_JAR} already exists")
        return True
    
    print(f"📦 Downloading {ONNX_JAR}...")
    try:
        urllib.request.urlretrieve(ONNX_URL, ONNX_JAR)
        print(f"✅ Downloaded {ONNX_JAR}")
        return True
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        return False

def compile_kotlin():
    """Compile Kotlin code with ONNX Runtime in classpath"""
    print(f"🔨 Compiling {KOTLIN_FILE}...")
    
    cmd = [
        "kotlinc",
        "-cp", ONNX_JAR,
        KOTLIN_FILE,
        "-include-runtime",
        "-d", OUTPUT_JAR
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Compilation failed:")
        print(result.stderr)
        return False
    
    print(f"✅ Compiled to {OUTPUT_JAR}")
    return True

def run_kotlin():
    """Run the compiled Kotlin code"""
    print("🚀 Running detection...")
    
    cmd = [
        "java",
        "-cp", f"{OUTPUT_JAR}:{ONNX_JAR}",
        "Door_detection_functionsKt"
    ]
    
    subprocess.run(cmd)

def main():
    print("=" * 60)
    print("VisionBridge - Kotlin Door Detection (Simple Build)")
    print("=" * 60)
    print()
    
    # Change to kotlin directory
    os.chdir(Path(__file__).parent)
    
    # Check prerequisites
    if not check_kotlin():
        sys.exit(1)
    
    # Check model
    if not os.path.exists("assets/best.onnx"):
        print("❌ Model not found: assets/best.onnx")
        print("   Run: python src/scripts/export_model_to_onnx.py")
        sys.exit(1)
    
    # Download dependencies
    if not download_onnx_jar():
        sys.exit(1)
    
    # Compile
    if not compile_kotlin():
        sys.exit(1)
    
    # Run
    run_kotlin()
    
    print()
    print("✅ Done!")

if __name__ == "__main__":
    main()
