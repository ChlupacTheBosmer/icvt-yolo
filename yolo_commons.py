from ultralytics.models.yolo import YOLO
import torch
import os
import numpy as np

def process_detection_results(result):

    # Define default values
    boxes = []
    confs = []
    classes = []

    # If any visitor detected
    if len(result.boxes) > 0:

        # Set variables
        detection = 1
        number_of_visitors = result.__len__()
        for i, box in enumerate(result.boxes):
            boxes.append(result.boxes.xywhn[i].cpu().numpy())
            confs.append(result.boxes.conf[i].cpu().numpy())
            classes.append(int(list(result.boxes.cls)[i]))

    else:
        detection = 0
        number_of_visitors = 0
    return detection, number_of_visitors, boxes, confs, classes

def save_label_file(label_path, boxes, visitor_category):

    with open(f"{label_path}.txt", 'w') as f:
        for arr in boxes:
            rounded_arr = np.round(arr, 5)
            line = f"{visitor_category} {' '.join(map(str, rounded_arr))}\n"
            f.write(line)