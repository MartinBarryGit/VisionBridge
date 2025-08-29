from config import data_dir
import os
import glob
import cv2
def label_color(label):
    colors ={ "door": (0, 255, 0), "handle": (255, 0, 0), "cabinet door": (0, 0, 255), "refrigerator door": (255, 255, 0)}
    return colors[label]
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    x_center, y_center, width, height = box
    x_center *= image.shape[1]
    y_center *= image.shape[0]
    width *= image.shape[1]
    height *= image.shape[0]
    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)
    p1, p2 = (x1, y1), (x2, y2)
    #
    cv2.rectangle(image, p1, p2, label_color(label), thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    label_color(label),
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return image
labels_list = ['door', 'handle', 'cabinet door', 'refrigerator door']
image_paths = glob.glob(os.path.join(data_dir, 'DoorDetect_yolo_training', 'images', "*", '*.jpg'))
for path in image_paths:
    image = cv2.imread(path)
    label_path = path.replace('images', 'labels').replace('.jpg', '.txt')
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            labels = f.readlines()
        for label in labels:
            parts = label.strip().split()
            class_id = parts[0]
            bounding_box = list(map(float, parts[1:5]))
            image = box_label(image, bounding_box, label=labels_list[int(class_id)])
    cv2.imshow("Annotated Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
