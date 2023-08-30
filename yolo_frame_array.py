from ultralytics import YOLO
import torch
import os
from yolo_commons import process_detection_results

def detect_visitors_in_frame_array(frame_numpy_array, metadata, model_path: str = os.path.join('resources', 'yolo', 'best.pt'), image_size: tuple = (640, 640),):

    # Load a pretrained YOLOv8n model
    model = YOLO(model_path)

    # Define variables
    frame_numbers = metadata['frame_numbers']
    visit_numbers = metadata['visit_numbers']
    roi_number = metadata['roi_number']
    object_detection_metadata = {}
    device = 0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None

    # Run inference on the source
    results = model(frame_numpy_array, stream=True, save_txt=False, save=False, device=device, imgsz=image_size)  # generator of Results objects

    for r, frame_number, visit_number in zip(results, frame_numbers, visit_numbers):

        detection, number_of_visitors, boxes, confs, classes = process_detection_results(r)

        object_detection_metadata = (frame_number, roi_number, visit_number, detection, number_of_visitors, boxes, confs, classes)

        yield object_detection_metadata

