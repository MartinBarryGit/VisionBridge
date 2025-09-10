# Multi-Dataset YOLO Training with Selective Loss

This repository contains several approaches to train YOLO on multiple datasets with selective loss calculation, specifically designed for your use case where you have:

1. **Dataset1** (DoorDetect_yolo_training): 4 classes (Door, Door handle, Cabinet door, Refrigerator door)
2. **Dataset2** (VIDD): 1 class (Door only)

The goal is to train on both datasets simultaneously, but only penalize the model for the classes that are actually annotated in each dataset.

## Available Training Approaches

### 1. Simple Merged Dataset Approach (Recommended for Quick Start)
**File**: `YoloFineTuning.py` (modified)

**How it works**:
- Physically merges both datasets into a single training dataset
- Images from VIDD get a "vidd_" prefix for identification
- The model learns all 4 classes, but VIDD images only have door annotations
- Natural bias towards door detection from VIDD dataset

**Pros**:
- Simple to implement and understand
- No custom loss functions required
- Uses standard YOLO training pipeline

**Cons**:
- Not truly selective - model still tries to predict other classes on VIDD images
- May learn false negatives for non-door classes

### 2. Multi-Dataset Trainer (Intermediate)
**File**: `MultiDatasetYoloTraining.py`

**How it works**:
- Custom trainer that tracks dataset sources
- Implements basic selective loss masking
- Merges datasets with source tracking

**Pros**:
- More control over training process
- Better dataset source tracking
- Foundation for advanced customizations

**Cons**:
- More complex implementation
- Still limited in true selective loss application

### 3. Selective Loss Implementation (Advanced)
**File**: `SelectiveLossYoloTraining.py`

**How it works**:
- Custom loss function that applies different class masks per dataset
- True selective loss calculation based on dataset source
- More sophisticated loss masking

**Pros**:
- Implements true selective loss masking
- More precise control over which classes contribute to loss
- Better theoretical foundation

**Cons**:
- Complex implementation
- Requires deep understanding of YOLO internals
- May need debugging for specific use cases

### 4. Advanced Selective Loss (Most Sophisticated)
**File**: `AdvancedSelectiveLossTraining.py`

**How it works**:
- Complete custom loss implementation with per-image class masking
- Sophisticated selective loss calculation
- Proper integration with YOLO's internal loss computation

**Pros**:
- Most accurate implementation of your requirement
- True selective loss masking per image
- Complete control over loss calculation

**Cons**:
- Most complex to understand and modify
- Requires expert knowledge of YOLO architecture
- May need adjustments for different YOLO versions

## Quick Start Guide

### Option 1: Simple Approach (Recommended)
```bash
cd src/scripts
python YoloFineTuning.py
```

This will:
1. Create a merged dataset combining both your datasets
2. Train YOLO on the merged dataset using standard training
3. Save the model as `yolov8_merged_dataset_finetuned.pt`

### Option 2: Advanced Selective Loss
```bash
cd src/scripts  
python AdvancedSelectiveLossTraining.py
```

This will:
1. Prepare datasets with proper source identification
2. Train with true selective loss masking
3. Save the model as `yolov8_selective_loss_model.pt`

## Dataset Structure After Merging

```
merged_dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg          # From DoorDetect (4 classes)
│   │   ├── image2.jpg          # From DoorDetect (4 classes)
│   │   ├── vidd_door1.jpg      # From VIDD (door only)
│   │   └── vidd_door2.jpg      # From VIDD (door only)
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt          # All 4 classes possible
│   │   ├── image2.txt          # All 4 classes possible  
│   │   ├── vidd_door1.txt      # Only class 0 (door)
│   │   └── vidd_door2.txt      # Only class 0 (door)
│   └── val/
│       └── ...
└── dataset.yaml
```

## How Selective Loss Works

### For Images from DoorDetect Dataset:
- **Classes**: 0=Door, 1=Door handle, 2=Cabinet door, 3=Refrigerator door
- **Loss Calculation**: Normal loss on all 4 classes
- **Penalty**: Applied for all incorrect predictions

### For Images from VIDD Dataset:
- **Classes**: Only 0=Door is annotated
- **Loss Calculation**: Only door class (0) contributes to loss
- **Penalty**: No penalty for predicting classes 1, 2, 3 (they're ignored)

This means the model can still predict door handles, cabinet doors, etc. on VIDD images, but it won't be penalized for these predictions since they're not annotated in the VIDD dataset.

## Configuration

### Training Parameters
The training configuration is defined in `yolo_training_config.yaml`:
- **Epochs**: 100 (adjustable)
- **Batch size**: 16 (adjustable based on your GPU memory)
- **Learning rate**: 0.01 with cosine decay
- **Data augmentation**: Enabled with reasonable parameters

### Model Selection
You can change the base model by modifying the model initialization:
```python
# Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
model = YOLO('yolov8n.pt')  # Nano (fastest)
model = YOLO('yolov8s.pt')  # Small
model = YOLO('yolov8m.pt')  # Medium  
model = YOLO('yolov8l.pt')  # Large
model = YOLO('yolov8x.pt')  # Extra Large (most accurate)
```

## Expected Results

After training, your model will:

1. **Detect all 4 classes**: Door, Door handle, Cabinet door, Refrigerator door
2. **Better door detection**: Improved from VIDD dataset exposure
3. **Balanced performance**: Good performance on all classes without bias against door handles/cabinet doors
4. **Robust generalization**: Works well on both indoor scenes (VIDD) and varied environments (DoorDetect)

## Monitoring Training

Training progress is saved in `runs/detect/[experiment_name]/`:
- **Weights**: Best and last model weights
- **Plots**: Loss curves, metrics, confusion matrix
- **Validation**: Validation results and sample predictions

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size in config
2. **No improvement**: Check learning rate, increase epochs
3. **Class imbalance**: Adjust loss weights in config
4. **Poor validation**: Check data quality, increase dataset size

### Performance Tips:

1. **Start with simple approach** (Option 1) to establish baseline
2. **Monitor class distribution** in merged dataset
3. **Use data augmentation** to increase effective dataset size
4. **Validate on both datasets** separately to check performance

## File Dependencies

Make sure you have:
- `config.py` with `data_dir` pointing to your dataset directory
- `yolo_training_config.yaml` with training parameters
- Both datasets in the expected directory structure
- Required Python packages: `ultralytics`, `torch`, `yaml`, `pathlib`

## Next Steps

1. **Start with the simple approach** to get a working baseline
2. **Evaluate results** on both datasets
3. **Try advanced selective loss** if you need more precise control
4. **Fine-tune hyperparameters** based on your specific requirements
