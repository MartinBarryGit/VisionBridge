import pickle as pkl

from config import data_dir
import cv2
import numpy as np
pkl_file_path = f"{data_dir}/ADE20K_2021_17_01/index_ade20k.pkl"

print(f"Loading data from {pkl_file_path}...")
with open(pkl_file_path, "rb") as f:
    data = pkl.load(f)
print(data.keys())
index_door = data["objectnames"].index("door")
print(f"Index of 'door': {index_door}")
indices_opening  = [ data["objectnames"].index(obj) for obj in ["handle", "knob",] ]
print(f"Indices of 'handle' and 'knob': {indices_opening}")
print(data["objectPresence"])
full_door_indices = np.where((data["objectPresence"][index_door, :] == 1)
                         * ((data["objectIsPart"][indices_opening[0],:] == 1) + (data["objectIsPart"][indices_opening[1],:] == 1)))[0]
print(f"Number of images with 'door': {len(full_door_indices)} / {data['objectPresence'].shape[1]}")
for idx in full_door_indices:
    print(f" - {data['folder'][idx]}/{data['filename'][idx]}")
    cv2.imshow("img", cv2.imread(f"{data_dir}/{data['folder'][idx]}/{data['filename'][idx]}"))
    cv2.waitKey(0)
cv2.destroyAllWindows()