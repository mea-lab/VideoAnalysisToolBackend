# detectors/yolo_detectors.py

import os
import torch
import cv2
from ultralytics import YOLO

# Fallback for MPS on Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def create_yolo_detector(model_path="yolov8s.pt", device='cpu'):
    """
    Example function that creates a YOLO object 
    """
    # Choose device
    device = 'cuda' if torch.cuda.is_available() else device
    if torch.backends.mps.is_available():
        device = 'mps'

    return YOLO(model_path), device


def yolo_tracker(file_path, model_path="yolov8s.pt", device='cpu'):
    """
    Similar to your old YOLOTracker(...) code, but placed in 'yolo_detectors.py'.
    Runs YOLOv8 tracking and returns bounding boxes (every 10 frames).
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else device
    if torch.backends.mps.is_available():
        device = 'mps'

    # Load YOLO model
    model = YOLO(model_path)

    cap = cv2.VideoCapture(file_path)
    boundingBoxes = []
    frameNumber = 0
    data = []  # store the last set of detections between frames

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frameNumber % 10 == 0:
            # Run YOLOv8 tracking on this frame
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                verbose=False,
                device=device
            )
            data = []  # reset data for these frames

            if (len(results) > 0 and
                results[0].boxes is not None and
                results[0].boxes.id is not None):
                ind = results[0].boxes.id.cpu().numpy().astype(int)
                box = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for i in range(len(ind)):
                    temp = {
                        'id': int(ind[i]),
                        'x': int(box[i][0]),
                        'y': int(box[i][1]),
                        'width': int(box[i][2] - box[i][0]),
                        'height': int(box[i][3] - box[i][1]),
                        'Subject': False
                    }
                    data.append(temp)

            frameResults = {
                'frameNumber': frameNumber,
                'data': data
            }
            boundingBoxes.append(frameResults)
        else:
            # For frames in-between, we repeat the last known data 
            frameResults = {
                'frameNumber': frameNumber,
                'data': data
            }
            boundingBoxes.append(frameResults)

        frameNumber += 1

    # Prepare final output
    outputDictionary = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'boundingBoxes': boundingBoxes
    }
    cap.release()
    return outputDictionary
