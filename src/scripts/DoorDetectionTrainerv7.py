import os
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Optional
import torch
import yaml
from ultralytics import YOLO, settings

from src.config import data_dir

# Configuration


DATASET = "Doors_OpenImages"
# Base model to fine-tune (Ultralytics hub weights or local .pt)
# If provided, use training overrides from this yaml (ultralytics training args)
TRAIN_CFG_PATH = Path(__file__).with_name("multi_dataset.yaml")
DOOR_CLASS = os.environ.get("OI_DOOR_CLASS", "Door")
OPEN_IMAGES_SPLIT = os.environ.get("OI_SPLIT", "train")
MODELS_DIR =  "/home/martin.barry/projects/VisionBridge/largefiles/models"
RUNS_DIR = "/home/martin.barry/projects/VisionBridge/largefiles/runs"
OPEN_IMAGES_DIR = Path("/home/martin.barry/projects/VisionBridge/largefiles/door_open_images/")
POSITIVE_MAX = int(os.environ.get("OI_POSITIVE_MAX", "0"))  # 0 = all available
NEGATIVE_TARGET = int(os.environ.get("OI_NEGATIVE_TARGET", "0"))  # 0 = auto-ratio
NEGATIVE_RATIO = float(os.environ.get("OI_NEGATIVE_RATIO", "0.5"))
VAL_SPLIT = float(os.environ.get("OI_VAL_SPLIT", "0.2"))
SEED = int(os.environ.get("OI_SEED", "42"))


def _print_openimages_hint() -> None:
    print(
        textwrap.dedent(
            """
            This script downloads Open Images through FiftyOne dataset zoo.
            Docs: https://storage.googleapis.com/openimages/web/index.html
            You can control size with env vars:
              - OI_POSITIVE_MAX (default 0 = all positives with Door label)
              - OI_NEGATIVE_TARGET (default 0 = auto from OI_NEGATIVE_RATIO)
              - OI_NEGATIVE_RATIO (default 0.5)
              - OI_SPLIT (default train)
                            - OI_DATASET_DIR (default FiftyOne cache dir)
            """
        ).strip()
    )


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(fin.read())


def _normalize_box_xywh_to_yolo(bbox_xywh: List[float]) -> List[float]:
    x, y, w, h = bbox_xywh
    x_center = x + w / 2.0
    y_center = y + h / 2.0
    return [x_center, y_center, w, h]


def _write_label_file(label_path: Path, boxes: List[List[float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w") as f:
        for box in boxes:
            x_center, y_center, w, h = _normalize_box_xywh_to_yolo(box)
            f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


def _find_door_boxes(sample, door_class: str) -> List[List[float]]:
    possible_fields = ("ground_truth", "detections")
    for field in possible_fields:
        labels = sample[field]
        if labels is None:
            continue
        detections = getattr(labels, "detections", []) or []
        return [det.bounding_box for det in detections if det.label == door_class]
    return []


def _collect_open_images_samples(
    split: str,
    door_class: str,
    positive_max: int,
    seed: int,
    dataset_dir: Optional[Path] = None,
) -> Dict[str, List[Dict]]:
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise RuntimeError(
            "fiftyone is required to download Open Images. Install with `pip install fiftyone`."
        ) from exc
    print("Collecting Open Images samples with FiftyOne...")
    if dataset_dir is not None:
        print(f"Using custom dataset directory for FiftyOne: {dataset_dir}")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(dataset_dir)
        fo.config.dataset_zoo_dir = str(dataset_dir)

    positive_kwargs = {
        "split": split,
        "label_types": ["detections"],
        "classes": [door_class],
        "only_matching": True,
        "shuffle": True,
        "seed": seed,
    }
    if positive_max > 0:
        positive_kwargs["max_samples"] = positive_max
    # if dataset_dir is not None:
    #     positive_kwargs["dataset_dir"] = str(dataset_dir)

    target_dir = str(dataset_dir) if dataset_dir is not None else "FiftyOne default"
    print(
        f"Downloading positives from Open Images split='{split}' for class '{door_class}' to '{target_dir}'..."
    )
    positives_ds = foz.load_zoo_dataset("open-images-v7", **positive_kwargs)

    positives: List[Dict] = []
    positive_paths = set()
    for sample in positives_ds.iter_samples(progress=True):
        boxes = _find_door_boxes(sample, door_class)
        if not boxes:
            continue
        sample_path = str(Path(sample.filepath).resolve())
        positive_paths.add(sample_path)
        positives.append({"filepath": sample_path, "boxes": boxes})

    print("Downloading candidates for negatives (images without Door labels)...")
    negative_kwargs = {
        "split": split,
        "label_types": ["detections"],
        "classes": [door_class],
        "only_matching": False,
        "shuffle": True,
        "seed": seed,
    }
    # if dataset_dir is not None:
    #     negative_kwargs["dataset_dir"] = str(dataset_dir)

    negatives_candidates_ds = foz.load_zoo_dataset("open-images-v7", **negative_kwargs)

    negatives: List[Dict] = []
    for sample in negatives_candidates_ds.iter_samples(progress=True):
        sample_path = str(Path(sample.filepath).resolve())
        if sample_path in positive_paths:
            continue
        boxes = _find_door_boxes(sample, door_class)
        if boxes:
            continue
        negatives.append({"filepath": sample_path, "boxes": []})

    return {"positives": positives, "negatives": negatives}


def build_openimages_yolo_dataset(dst_root: Path) -> Path:
    random.seed(SEED)
    _print_openimages_hint()
    dataset_dir = Path(OPEN_IMAGES_DIR)
    print(f"Using dataset directory: {dataset_dir if dataset_dir else 'FiftyOne default'}")
    samples = _collect_open_images_samples(
        split=OPEN_IMAGES_SPLIT,
        door_class=DOOR_CLASS,
        positive_max=POSITIVE_MAX,
        seed=SEED,
        dataset_dir=dataset_dir,
    )
    positives = samples["positives"]
    negatives = samples["negatives"]

    if not positives:
        raise RuntimeError(
            "No positive Door samples found in Open Images. "
            "Check OI_DOOR_CLASS/OI_SPLIT settings."
        )

    if NEGATIVE_TARGET > 0:
        requested_negatives = NEGATIVE_TARGET
    else:
        requested_negatives = max(1, int(len(positives) * NEGATIVE_RATIO))

    negatives = negatives[: min(requested_negatives, len(negatives))]
    all_samples = positives + negatives
    random.shuffle(all_samples)

    val_count = max(1, int(len(all_samples) * VAL_SPLIT))
    train_samples = all_samples[val_count:]
    val_samples = all_samples[:val_count]

    for split_name, split_samples in (("train", train_samples), ("val", val_samples)):
        for idx, sample in enumerate(split_samples):
            src = Path(sample["filepath"])
            stem = f"{split_name}_{idx:07d}"
            img_dst = dst_root / "images" / split_name / f"{stem}{src.suffix.lower()}"
            label_dst = dst_root / "labels" / split_name / f"{stem}.txt"
            _copy_image(src, img_dst)
            _write_label_file(label_dst, sample["boxes"])

    print(
        f"Prepared dataset at {dst_root} | "
        f"positives={len(positives)} negatives={len(negatives)} "
        f"train={len(train_samples)} val={len(val_samples)}"
    )
    return write_merged_yaml(dst_root)





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

    # model is loaded explicitly in main()
    overrides.pop("model", None)
    # data will be set dynamically
    overrides.pop("data", None)
    return overrides


def main(dst_root: Optional[Path] = None) -> None:
    dst_root = Path(data_dir) / DATASET
    dst_root.mkdir(parents=True, exist_ok=True)

    merged_yaml = build_openimages_yolo_dataset(dst_root)
    print(f"Data YAML: {merged_yaml}")
    overrides = load_train_overrides(TRAIN_CFG_PATH)
    model_name = "yolov8m.pt"
    model_path = Path(model_name)
    if not model_path.is_absolute():
        model_path = Path(MODELS_DIR) / model_name
    Path(RUNS_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    settings.update({"weights_dir": MODELS_DIR})
    if not model_path.exists():
        model_path = Path(model_name)
    # Train YOLOv8
    print(f"Starting YOLOv8 fine-tuning from {model_path}...")
    model = YOLO(str(model_path))
    
    # Ensure a reasonable default if no cfg
    
    overrides.setdefault("device", "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # Always set data to our merged dataset
    overrides["data"] = str(merged_yaml)
    overrides["project"] = RUNS_DIR

    model.train(**overrides)
    print("Training complete.")
    try:
        print(f"Best weights: {model.trainer.best}")
    except Exception:
        print("Best weights path not available.")


if __name__ == "__main__":
    main()
