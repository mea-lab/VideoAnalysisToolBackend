import os
import torch
import cv2
from ultralytics import YOLO

from .base_detector import BaseDetector

class YoloDetectors(BaseDetector):
    """
    Detector for bounding boxes using YOLOv8.
    """

    def __init__(self, model_path: str = "yolov8s.pt", device: str = 'cpu'):
        """
        Initialize the YOLO detector with optional model path and device.
        """
        self.model_path = model_path
        self.device = device
        # Initialize model and device
        self.model, self.device = self.get_detector

    @property
    def get_detector(self) -> tuple:
        """
        Returns a tuple of (YOLO model, device) ready for detection.
        """
        # Fallback for MPS on Mac
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        # Choose device
        device = 'cuda' if torch.cuda.is_available() else self.device
        if torch.backends.mps.is_available():
            device = 'mps'
        # Load YOLO model
        model = YOLO(self.model_path)
        return model, device

    def track(self, file_path: str) -> dict:
        """
        Runs YOLOv8 tracking on the given video file and returns bounding boxes (every 10 frames).
        """
        cap = cv2.VideoCapture(file_path)
        bounding_boxes = []
        frame_number = 0
        data = []  # store last detections

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_number % 10 == 0:
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    verbose=False,
                    device=self.device
                )
                data = []
                if (len(results) > 0
                        and results[0].boxes is not None
                        and results[0].boxes.id is not None):
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

                bounding_boxes.append({
                    'frameNumber': frame_number,
                    'data': data
                })
            else:
                bounding_boxes.append({
                    'frameNumber': frame_number,
                    'data': data
                })

            frame_number += 1

        output = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'boundingBoxes': bounding_boxes
        }
        cap.release()
        return output
