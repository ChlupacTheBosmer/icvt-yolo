from typing import Generator, Union, List, Tuple
import numpy as np


def process_detection_results(result: Generator, numpy_arrays: bool = True) -> Tuple[int, int, List[Union[np.ndarray, list]], List[Union[float, np.ndarray]], List[int]]:

    # Initialize default values
    detection = 0
    number_of_visitors = 0
    boxes = []
    confs = []
    classes = []

    if len(result.boxes) > 0:
        # Set detection variables
        detection = 1
        number_of_visitors = len(result.boxes)

        for i, box in enumerate(result.boxes):
            # Gather boxes and confidence scores
            box_data = box.xywhn.cpu().numpy()[0] if numpy_arrays else box.xywhn.cpu().tolist()[0]
            conf_data = box.conf.cpu().numpy()[0] if numpy_arrays else box.conf.cpu().tolist()[0]

            boxes.append(box_data)
            confs.append(conf_data)
            classes.append(int(list(box.cls)[i]))

    return detection, number_of_visitors, boxes, confs, classes

def save_label_file(label_path, boxes, visitor_category):

    with open(f"{label_path}.txt", 'w') as f:
        for arr in boxes:
            rounded_arr = np.round(arr, 5)
            line = f"{visitor_category} {' '.join(map(str, rounded_arr))}\n"
            f.write(line)
