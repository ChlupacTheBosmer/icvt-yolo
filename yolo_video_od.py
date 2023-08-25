from ultralytics import YOLO
import os


def detect_visitors_on_video(video_path, roi_number, model_path: str = os.path.join('resources', 'yolo', 'best.pt')):

    # Load a pretrained YOLOv8n model
    model = YOLO(model_path)

    # Define variables
    source = 'out.mp4'
    roi_number = 1
    object_detection_metadata = {}


    # Run inference on the source
    # TODO: specify parameters
    results = model(video_path, stream=True, save_txt=True, save=True)  # generator of Results objects

    frame_number = 0
    for r in results:

        # Define default values
        detection = False
        number_of_visitors = 0
        boxes = []
        confs = []
        classes = []

        # Increase frame counter
        frame_number += 1

        # If any visitor detected
        if len(r.boxes) > 0:

            # Set variables
            detection = True
            number_of_visitors = r.__len__()
            print(f"detections: {r.__len__()}")
            for i, box in enumerate(r.boxes):
                boxes.append(r.boxes.xywhn[i].numpy().tolist())
                confs.append(r.boxes.conf[i].numpy().tolist())
                classes.append(int(list(r.boxes.cls)[i]))

                print(f"bbox: {r.boxes.xywhn[i].numpy().tolist()}")
                print(f"conf: {r.boxes.conf[i].numpy().tolist()}")
                print(f"class names:{int(list(r.boxes.cls)[i])}")
        else:
            detection = False
            number_of_visitors = 0

        object_detection_metadata[frame_number] = [detection, number_of_visitors, boxes, confs, classes]

    return object_detection_metadata

if __name__ == '__main__':
    od_metadata = detect_visitors_on_video('out.mp4', 0)
    print(od_metadata)