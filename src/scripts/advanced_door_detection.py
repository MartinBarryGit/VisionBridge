"""
Advanced Live Door Detection Script
Enhanced version with video file support, detection statistics, and more features
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import argparse
import os

class DoorDetector:
    def __init__(self, model_path, conf_threshold=0.3, iou_threshold=0.5):
        """Initialize the door detector"""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        print(f"Loading model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("Model loaded successfully!")
            print(f"Model classes: {self.model.names}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Statistics
        self.total_frames = 0
        self.frames_with_detections = 0
        self.detection_count = {"Door": 0, "Door handle": 0}
        
    def detect_in_frame(self, frame):
        """Run detection on a single frame"""
        results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)
        return results[0]
    
    def update_statistics(self, results):
        """Update detection statistics"""
        self.total_frames += 1
        
        if len(results.boxes) > 0:
            self.frames_with_detections += 1
            
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                if class_name in self.detection_count:
                    self.detection_count[class_name] += 1
    
    def draw_statistics(self, frame):
        """Draw statistics on the frame"""
        y_offset = 70
        line_height = 25
        
        # Background for statistics
        cv2.rectangle(frame, (10, 50), (350, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 50), (350, 180), (255, 255, 255), 2)
        
        # Statistics text
        stats = [
            f"Total Frames: {self.total_frames}",
            f"Frames with Detections: {self.frames_with_detections}",
            f"Detection Rate: {(self.frames_with_detections/max(1, self.total_frames)*100):.1f}%",
            f"Doors Detected: {self.detection_count.get('Door', 0)}",
            f"Handles Detected: {self.detection_count.get('Door handle', 0)}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (15, y_offset + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def live_detection(self, source=0, save_video=False, output_path="detection_output.mp4"):
        """Run live detection on webcam or video file"""
        
        # Initialize video capture
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
            print(f"Using webcam (device {source})")
        else:
            cap = cv2.VideoCapture(source)
            print(f"Processing video file: {source}")
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(source, str) else 0
        
        print(f"Video properties: {width}x{height} @ {fps}fps")
        if total_frames > 0:
            print(f"Total frames to process: {total_frames}")
        
        # Initialize video writer if saving
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")
        
        # Variables for FPS calculation
        fps_counter = 0
        start_time = time.time()
        current_fps = 0
        
        print("Starting detection... Press 'q' to quit, 's' to save screenshot")
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, str):
                    print("End of video file reached")
                else:
                    print("Error reading from webcam")
                break
            
            frame_number += 1
            
            # Run detection
            results = self.detect_in_frame(frame)
            
            # Update statistics
            self.update_statistics(results)
            
            # Draw results
            annotated_frame = results.plot(
                line_width=2,
                font_size=1,
                pil=False,
                labels=True,
                boxes=True,
                conf=True
            )
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 10 == 0:
                end_time = time.time()
                current_fps = 10 / (end_time - start_time)
                start_time = end_time
            
            # Draw FPS
            cv2.putText(annotated_frame, f'FPS: {current_fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw progress for video files
            if total_frames > 0:
                progress = (frame_number / total_frames) * 100
                cv2.putText(annotated_frame, f'Progress: {progress:.1f}%', (10, height - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw statistics
            self.draw_statistics(annotated_frame)
            
            # Add instructions
            cv2.putText(annotated_frame, "Press 'q' to quit, 's' for screenshot", 
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Save frame to video if enabled
            if writer:
                writer.write(annotated_frame)
            
            # Display frame
            cv2.imshow('Door Detection', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('s') or key == ord('S'):
                timestamp = int(time.time())
                filename = f"detection_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r') or key == ord('R'):
                # Reset statistics
                self.total_frames = 0
                self.frames_with_detections = 0
                self.detection_count = {"Door": 0, "Door handle": 0}
                print("Statistics reset")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        print(f"Total frames processed: {self.total_frames}")
        print(f"Frames with detections: {self.frames_with_detections}")
        print(f"Overall detection rate: {(self.frames_with_detections/max(1, self.total_frames)*100):.2f}%")
        print(f"Total doors detected: {self.detection_count.get('Door', 0)}")
        print(f"Total door handles detected: {self.detection_count.get('Door handle', 0)}")

def main():
    parser = argparse.ArgumentParser(description='Live Door Detection')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source: 0 for webcam, or path to video file')
    parser.add_argument('--conf', type=float, default=0.3, 
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--iou', type=float, default=0.5, 
                       help='IoU threshold for NMS (0.0-1.0)')
    parser.add_argument('--save', action='store_true', 
                       help='Save output video')
    parser.add_argument('--output', type=str, default='detection_output.mp4', 
                       help='Output video path')
    
    args = parser.parse_args()
    
    # Convert source to int if it's a digit (webcam)
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Model path
    model_path = "/Users/barry/Desktop/HES-SO/VisionBridge/src/scripts/runs/detect/multi_dataset/weights/best.pt"
    
    # Create detector
    try:
        detector = DoorDetector(model_path, args.conf, args.iou)
        detector.live_detection(source, args.save, args.output)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
