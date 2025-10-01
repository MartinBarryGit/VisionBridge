import glob
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from ultralytics import YOLO

from config import data_dir

# Configuration


DATASET = "Doors_Merged"
# Base model to fine-tune (Ultralytics hub weights or local .pt)
BASE_MODEL = os.environ.get("YOLO_BASE", "yolov8n.pt")
# If provided, use training overrides from this yaml (ultralytics training args)
TRAIN_CFG_PATH = Path(__file__).with_name("multi_dataset.yaml")





def write_merged_yaml(dst_root: Path) -> Path:
    cfg = {
        "path": str(dst_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "Door"},
    }
    yml = dst_root / "dataset.yaml"
    with open(yml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return yml


def load_train_overrides(cfg_path: Path) -> Dict:
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r") as f:
        overrides = yaml.safe_load(f) or {}
    # Remove model if present; we'll control it via BASE_MODEL
    overrides.pop("model", None)
    # data will be set dynamically
    overrides.pop("data", None)
    return overrides


def main():
    dst_root = Path(data_dir) / DATASET

    merged_yaml = write_merged_yaml(dst_root)
    print(f"Data YAML: {merged_yaml}")

    # Train YOLOv8
    print(f"Starting YOLOv8 fine-tuning from {BASE_MODEL}...")
    model = YOLO(BASE_MODEL)
    overrides = load_train_overrides(TRAIN_CFG_PATH)
    # Ensure a reasonable default if no cfg
    overrides.setdefault("epochs", 50)
    overrides.setdefault("batch", 16)
    overrides.setdefault("device", 0)
    # Always set data to our merged dataset
    overrides["data"] = str(merged_yaml)

    results = model.train(**overrides)
    print("Training complete.")
    try:
        print(f"Best weights: {model.trainer.best}")
    except Exception:
        print("Best weights path not available.")


if __name__ == "__main__":
    main()