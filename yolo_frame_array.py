#from ultralytics import YOLO
from ultralytics.models.yolo import YOLO
import torch
import os
from ..yolo.yolo_commons import process_detection_results
import numpy as np

def detect_visitors_in_frame_array(frame_numpy_array, metadata, model_path: str = os.path.join('resources', 'yolo', 'best.pt'), image_size: tuple = (640, 640),):

    print(f"(Y) - YOLO func initiated")
    # Load a pretrained YOLOv8n model
    model = YOLO(model_path)

    # Define variables
    frame_numbers = metadata['frame_numbers']
    visit_numbers = metadata['visit_numbers']
    roi_number = metadata['roi_number']
    device = 0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    #print(f"(Y) - YOLO func defined variables")
    #print(f"CUDA AVAILABILITY ----------- <{torch.cuda.device_count()}>")
    # FLatten the 4D array into a list of 3D arrays
    list_of_frames = np.split(frame_numpy_array, frame_numpy_array.shape[0], axis=0)
    # Removing singleton dimension
    list_of_frames = [np.squeeze(frame) for frame in list_of_frames]

    # Run inference on the source
    results = model(list_of_frames, stream=True, save_txt=False, save=False, device=device, imgsz=image_size)  # generator of Results objects

    for r, frame_number, visit_number in zip(results, frame_numbers, visit_numbers):

        print(f"(Y) - YOLO func result generated")

        detection, number_of_visitors, boxes, confs, classes = process_detection_results(r)


        yield frame_number, roi_number, visit_number, detection, number_of_visitors, boxes, confs, classes


