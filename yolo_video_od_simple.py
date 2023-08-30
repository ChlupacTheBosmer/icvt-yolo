from ultralytics import YOLO
import torch
import os
from ..yolo.yolo_commons import process_detection_results

def detect_visitors_on_video(video_path, model_path: str = os.path.join('resources', 'yolo', 'best.pt'), image_size: tuple = (640, 640), frames_to_skip = 1):

    # Load a pretrained YOLOv8n model
    model = YOLO(model_path)

    # Define variables
    object_detection_metadata = {}
    frame_skip = frames_to_skip
    device = 0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None

    # Run inference on the source
    results = model(video_path, stream=True, save_txt=False, save=False, vid_stride=frames_to_skip, device=device, imgsz=image_size)  # generator of Results objects

    frame_number = -(frame_skip-1)
    for r in results:

        detection, number_of_visitors, boxes, confs, classes = process_detection_results(r)

        object_detection_metadata = [frame_number, detection, number_of_visitors, boxes, confs, classes]

        yield object_detection_metadata

if __name__ == '__main__':

    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    od_metadata = detect_visitors_on_video(r"D:\Dílna\Kutění\Python\Metacentrum\metacentrum\videos\GR2_L2_LavSto2_20220524_09_29.mp4", os.path.join('..', '..', 'resources', 'yolo', 'best.pt'))
    for e in od_metadata:
        print(e)