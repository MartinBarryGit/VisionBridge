
from ultralytics import YOLO
import os
from config import parent_dir, data_dir
import cv2
import numpy as np
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
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]
    import random
    random.shuffle(test_images)
    test_images = test_images[:50]  # Limit to first 50 images for quick testing

    for img_name in test_images:
        img_path = os.path.join(test_images_dir, img_name)
        print(f"\nMaking prediction on: {img_name}")
        ## load image and rotatte it with random angle
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error: Could not read image {img_path}")
            continue
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        angle = random.uniform(-40, 40)  # Random angle between -10 and
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        # Run prediction
        results = finetuned_model(image)
        res = results[0].plot()
        # Display the image with predictions
        cv2.imshow(f'Predictions for {img_name}', res)
        cv2.waitKey(0)  # Wait for a key press to close the image
        ## save the image with predictions
        output_path = "pred.png"
        cv2.imwrite(output_path, res)
        print(f"Saved prediction image to: {output_path}")
        cv2.destroyAllWindows()