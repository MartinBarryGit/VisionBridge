import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

# Pre-filter during download to only get images with both Door and Door handle
# This significantly reduces download time and storage

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    classes=["Door"],  # Only download images with these classes
    max_samples=9999,  # Reduced since we're being more selective
    seed=51,
    shuffle=True,
)
## filter to images with doors AND handles/knobs
# First, get all images that have either doors or door handles
view_with_both_labels = dataset.filter_labels(
    "detections",
    F("label").is_in(["Door"]),
    only_matches=True,
)

# Then filter to only images that have BOTH labels
view = view_with_both_labels.match(
    F("detections.detections").map(F("label")).contains("Door") &
    F("detections.detections").map(F("label")).contains("Door handle")
)   
session = fo.launch_app(view)
session.wait()