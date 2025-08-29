"""
Dataset preparation script for YOLO fine-tuning
Takes existing YOLO-format dataset and splits it into train/val with proper structure
"""

import os
import shutil
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import glob
from config import data_dir

def create_yolo_directories(output_dir):
    """Create YOLO dataset directory structure"""
    dirs = [
        os.path.join(output_dir, 'images', 'train'),
        os.path.join(output_dir, 'images', 'val'),
        os.path.join(output_dir, 'labels', 'train'),
        os.path.join(output_dir, 'labels', 'val')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_class_names_from_labels(labels_dir):
    """Extract class names from existing label files"""
    class_ids = set()
    
    for label_file in glob.glob(os.path.join(labels_dir, '*.txt')):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_ids.add(int(parts[0]))
    
    # Create default class names based on your DoorDetect dataset
    # Adjust these based on your actual classes
    class_mapping = {
        0: "Door",
        1: "Door handle",
        2: "Cabinet door",
        3: "Refrigerator door"
    }
    
    # Return only classes that exist in the dataset
    existing_classes = {class_id: class_mapping.get(class_id, f"class_{class_id}") 
                       for class_id in sorted(class_ids)}
    
    return existing_classes

def create_yaml_config(output_dir, class_names):
    """Create YAML configuration file for YOLO training"""
    config = {
        'path': os.path.abspath(output_dir),
        'train': 'images/train',
        'val': 'images/val', 
        'nc': len(class_names),
        'names': class_names  # class_names is already a dict {0: "Door", 1: "Door handle"}
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return yaml_path

def get_image_label_pairs(images_dir, labels_dir):
    """Get matching image and label file pairs"""
    pairs = []
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    for img_path in image_files:
        # Get corresponding label file
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{img_name}.txt")
        
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
        else:
            print(f"Warning: No label file found for {img_path}")
    
    return pairs

def copy_files(pairs, output_dir, split):
    """Copy image and label files to the appropriate split directory"""
    for i, (img_path, label_path) in enumerate(pairs):
        # Create new filenames
        img_name = f"{split}_{i:06d}.jpg"
        label_name = f"{split}_{i:06d}.txt"
        
        # Copy image
        img_dest = os.path.join(output_dir, 'images', split, img_name)
        shutil.copy2(img_path, img_dest)
        
        # Copy label
        label_dest = os.path.join(output_dir, 'labels', split, label_name)
        shutil.copy2(label_path, label_dest)
        
        if (i + 1) % 10 == 0:
            print(f"Copied {i + 1} {split} files...")

def prepare_yolo_dataset(input_dataset_name="DoorDetect-Dataset", output_dir="door_handle_yolo_dataset", train_ratio=0.8):
    """
    Main function to prepare YOLO dataset from existing YOLO format dataset
    """
    # Input paths
    input_dataset_path = os.path.join(data_dir, input_dataset_name)
    images_dir = os.path.join(input_dataset_path, 'images')
    labels_dir = os.path.join(input_dataset_path, 'labels')
    
    # Check if input dataset exists
    if not os.path.exists(input_dataset_path):
        print(f"Error: Input dataset not found at {input_dataset_path}")
        return None, None
        
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found at {images_dir}")
        return None, None
        
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found at {labels_dir}")
        return None, None
    
    print(f"Loading dataset from: {input_dataset_path}")
    
    # Get image-label pairs
    pairs = get_image_label_pairs(images_dir, labels_dir)
    print(f"Found {len(pairs)} image-label pairs")
    
    if len(pairs) == 0:
        print("No valid image-label pairs found!")
        return None, None
    
    # Create output directory structure
    output_path = os.path.join(data_dir, output_dir)
    print(f"Creating dataset in: {output_path}")
    create_yolo_directories(output_path)
    
    # Get class names from existing labels
    class_names = get_class_names_from_labels(labels_dir)
    print(f"Detected classes: {class_names}")
    
    # Split into train and validation
    train_pairs, val_pairs = train_test_split(
        pairs, 
        train_size=train_ratio, 
        random_state=42
    )
    
    print(f"Train samples: {len(train_pairs)}")
    print(f"Validation samples: {len(val_pairs)}")
    
    # Copy training files
    print("Copying training files...")
    copy_files(train_pairs, output_path, 'train')
    
    # Copy validation files
    print("Copying validation files...")
    copy_files(val_pairs, output_path, 'val')
    
    # Create YAML configuration
    yaml_path = create_yaml_config(output_path, class_names)
    print(f"Created YAML config: {yaml_path}")
    
    print(f"\nDataset preparation complete!")
    print(f"Dataset location: {output_path}")
    print(f"Use this YAML file for training: {yaml_path}")
    
    return output_path, yaml_path

if __name__ == "__main__":
    # Configuration
    INPUT_DATASET = "DoorDetect-Dataset"  # Your existing dataset name
    OUTPUT_DIR = "DoorDetect_yolo_training"
    TRAIN_RATIO = 0.8
    
    print("Starting YOLO dataset preparation...")
    print(f"Input dataset: {INPUT_DATASET}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train/Val ratio: {TRAIN_RATIO:.1%}/{(1-TRAIN_RATIO):.1%}")
    
    dataset_path, yaml_path = prepare_yolo_dataset(
        input_dataset_name=INPUT_DATASET,
        output_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO
    )
    
    if dataset_path and yaml_path:
        print("\n" + "="*50)
        print("DATASET PREPARATION SUMMARY")
        print("="*50)
        print(f"Dataset path: {dataset_path}")
        print(f"YAML config: {yaml_path}")
        print("\nTo train YOLO model, use:")
        print(f"yolo train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640")
        print(f"# Or with more specific settings:")
        print(f"yolo train data={yaml_path} model=yolov8s.pt epochs=50 imgsz=640 batch=16 lr0=0.01")
    else:
        print("Dataset preparation failed. Please check the input dataset path.")
