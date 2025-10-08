import os
import glob
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from ultralytics import YOLO

from config import data_dir
import glob
# Configuration
# Datasets to merge (must be YOLO-format with images/train,val and labels/train,val)
DATASETS = glob.glob(os.path.join(data_dir, "*/"))
DATASETS = [path.split(os.path.sep)[-2] for path in DATASETS if "Merged" not in path and path.strip() != ""]
# Output merged dataset directory name (under dataset/)
OUTPUT_DATASET = "Doors_Merged"
# Whether to keep images that have no 'door' boxes after filtering (negative samples)
INCLUDE_NEGATIVES = True


def read_data_yaml(ds_root: Path) -> Dict:
    yml = ds_root / "dataset.yaml"
    if not yml.exists():
        raise FileNotFoundError(f"dataset.yaml not found in {ds_root}")
    with open(yml, "r") as f:
        return yaml.safe_load(f)


def find_door_class_id(names) -> Optional[int]:
    # names can be dict like {0: "Door", 1: "Door handle"} or a list ["Door", ...]
    if isinstance(names, dict):
        for k, v in names.items():
            try:
                idx = int(k)
            except Exception:
                idx = k
            if isinstance(v, str) and v.strip().lower() == "door":
                return int(idx)
    elif isinstance(names, list):
        for idx, v in enumerate(names):
            if isinstance(v, str) and v.strip().lower() == "door":
                return int(idx)
    return None


def ensure_dirs(root: Path):
    for p in [root / "images/train", root / "images/val", root / "labels/train", root / "labels/val"]:
        p.mkdir(parents=True, exist_ok=True)


def find_image_for_label(images_split_dir: Path, stem: str) -> Optional[Path]:
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    for ext in exts:
        p = images_split_dir / f"{stem}{ext}"
        if p.exists():
            return p
    # Fallback: glob by stem.*
    matches = list(images_split_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def filter_label_lines_to_door(label_path: Path, door_class_id: int) -> List[str]:
    filtered = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
            except Exception:
                continue
            if cls == door_class_id:
                # Remap class id -> 0 and keep first 4 bbox values
                # YOLO detection format: class cx cy w h
                vals = parts[1:5]
                if len(vals) < 4:
                    continue
                filtered.append("0 " + " ".join(vals[:4]) + "\n")
    return filtered


def merge_one_dataset(src_root: Path, dst_root: Path, include_negatives: bool, start_counters: Dict[str, int]) -> Dict[str, int]:
    data_cfg = read_data_yaml(src_root)
    names = data_cfg.get("names", {})
    door_id = find_door_class_id(names)
    if door_id is None:
        print(f"Skipping {src_root.name}: no 'door' class found in names={names}")
        return start_counters

    for split in ("train", "val"):
        labels_dir = src_root / "labels" / split
        images_dir = src_root / "images" / split
        if not labels_dir.exists() or not images_dir.exists():
            print(f"Warning: {src_root.name} missing split {split}")
            continue

        label_files = sorted(glob.glob(str(labels_dir / "*.txt")))
        for lf in label_files:
            lf_path = Path(lf)
            stem = lf_path.stem
            img_path = find_image_for_label(images_dir, stem)
            if img_path is None:
                # Try alternative: sometimes images might be in flat dir
                print(f"Warning: image for {lf_path} not found")
                continue

            door_lines = filter_label_lines_to_door(lf_path, door_id)
            if not door_lines and not include_negatives:
                continue

            # Create unique filename
            idx = start_counters.setdefault(split, 0)
            start_counters[split] = idx + 1
            img_ext = img_path.suffix.lower()
            new_img_name = f"{split}_{idx:06d}{img_ext}"
            new_lbl_name = f"{split}_{idx:06d}.txt"

            # Copy image
            dst_img = dst_root / "images" / split / new_img_name
            shutil.copy2(img_path, dst_img)

            # Write label (empty file for negatives)
            dst_lbl = dst_root / "labels" / split / new_lbl_name
            with open(dst_lbl, "w") as out:
                for line in door_lines:
                    out.write(line)

        print(f"Merged {src_root.name} {split}: total now {start_counters.get(split, 0)} files")

    return start_counters




def main():
    ds_roots = [Path(data_dir) / d for d in DATASETS]
    for p in ds_roots:
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")

    dst_root = Path(data_dir) / OUTPUT_DATASET
    ensure_dirs(dst_root)

    counters = {"train": 0, "val": 0}
    for src in ds_roots:
        counters = merge_one_dataset(src, dst_root, INCLUDE_NEGATIVES, counters)

    print(f"Merged dataset ready: {dst_root}")

if __name__ == "__main__":
    main()