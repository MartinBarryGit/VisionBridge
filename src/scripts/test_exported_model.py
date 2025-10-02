import os

import cv2
import numpy as np
import tensorflow as tf

parent_dir = "/Users/moreno/sources/VisionBridge/src"
tflite_model_path = os.path.join(parent_dir, "runs/detect/multi_dataset/weights/best_float16.tflite")
image_path = "pred.png"
img_size = 640 
conf_threshold = 0.25
iou_threshold = 0.45

interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h0, w0 = img.shape[:2]

img_resized = cv2.resize(img_rgb, (img_size, img_size))
img_input = img_resized.astype(np.float32) / 255.0
img_input = np.expand_dims(img_input, axis=0) 

interpreter.set_tensor(input_details[0]['index'], img_input)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
output_data = np.squeeze(output_data)

def xywh2xyxy(box):
    x, y, w, h = box
    return np.array([x - w/2, y - h/2, x + w/2, y + h/2])

boxes = []
scores = []

for i in range(output_data.shape[1]): 
    conf = output_data[4, i]
    if conf > conf_threshold:
        box = output_data[:4, i]
        box_xyxy = xywh2xyxy(box)
        box_xyxy[[0,2]] *= w0 / img_size
        box_xyxy[[1,3]] *= h0 / img_size
        boxes.append(box_xyxy)
        scores.append(conf)

if len(boxes) > 0:
    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size=100, iou_threshold=iou_threshold)
    boxes = boxes[indices.numpy()]
    scores = scores[indices.numpy()]

for box, score in zip(boxes, scores):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
