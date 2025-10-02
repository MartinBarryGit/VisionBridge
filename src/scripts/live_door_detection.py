"""
Live Door Detection Script
Uses pre-trained YOLO model for real-time door detection via webcam
"""

import time

import cv2
import numpy as np
from ultralytics import YOLO


def main():
    # Load your pre-trained model
    model_path = "/Users/barry/Desktop/HES-SO/VisionBridge/src/runs/detect/multi_dataset/weights/best.pt"
    
    print(f"Loading model from: {model_path}")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully!")
        print(f"Model classes: {model.names}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Variables for FPS calculation
    fps_counter = 0
    start_time = time.time()
    fps = 0
    
    print("Starting live detection... Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run inference
        results = model(frame, conf=0.3, iou=0.5)  # Adjust confidence threshold as needed
        
        # Draw results on frame
        annotated_frame = results[0].plot(
            line_width=2,
            font_size=1,
            font='Arial.ttf',
            pil=False,
            labels=True,
            boxes=True,
            conf=True
        )
        
        # Calculate and display FPS
        fps_counter += 1
        if fps_counter % 10 == 0:  # Update FPS every 10 frames
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = end_time
        
        # Add FPS text to frame
        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add instructions
        cv2.putText(annotated_frame, "Press 'q' to quit", (10, annotated_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('Live Door Detection', annotated_frame)
        
        # Check for quit key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            # Save screenshot
            timestamp = int(time.time())
            filename = f"detection_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"Screenshot saved as: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Live detection stopped.")

if __name__ == "__main__":
    main()
