
from ultralytics import YOLO
import os
from config import parent_dir, data_dir
import cv2
# Load the fine-tuned model for prediction
model_path = os.path.join(parent_dir, "scripts/runs/detect/multi_dataset/weights", 'best.pt')
# model_path = "pretrained.pt"
# model_path = os.path.join(parent_dir, "scripts/runs/detect/door_detection/weights", 'best.pt')

finetuned_model = YOLO(model_path)

# Example prediction on test images
test_images_dir = os.path.join(data_dir, 'VIDD/images/val')
# test_images_dir = os.path.join(data_dir, 'DoorDetect_yolo_training/images/val')
# test_images_dir = os.path.join(data_dir, 'doors windows/images/val')
test_images_dir = os.path.join(data_dir, 'Doors_Merged/images/val')
if os.path.exists(test_images_dir):
    # Get first few images from the dataset for testing
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')][:30]
    
    for img_name in test_images:
        img_path = os.path.join(test_images_dir, img_name)
        print(f"\nMaking prediction on: {img_name}")
        
        # Run prediction
        results = finetuned_model(img_path)
        res = results[0].plot()
        # Display the image with predictions
        cv2.imshow(f'Predictions for {img_name}', res)
        cv2.waitKey(0)  # Wait for a key press to close the image
        ## save the image with predictions
        output_path = "pred.png"
        cv2.imwrite(output_path, res)
        print(f"Saved prediction image to: {output_path}")
        cv2.destroyAllWindows()