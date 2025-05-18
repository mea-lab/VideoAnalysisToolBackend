import os
import cv2
import math
import json
import uuid
import mediapipe as mp
import numpy as np
import traceback
from django.core.files.storage import FileSystemStorage

from .base_task import BaseTask
from app.analysis.detectors.mp_hand_detector import HandDetector
from app.analysis.signal_analyzers.peakfinder_signal_analyzer import PeakfinderSignalAnalyzer

class FingerTapLeftTask(BaseTask):
    """
    Left Finger Tap Task:
      - Calculates the signal as the Euclidean distance between the thumb tip and index finger tip.
      - Normalization factor is the maximum distance between the middle finger tip and wrist over all frames.
    This task always uses the left hand.
    """

# ------------------------------------------------------------------
# --- START: Abstract properties definitions
# ------------------------------------------------------------------
    LANDMARKS = {
        "WRIST": 0,
        "THUMB_CMC": 1,
        "THUMB_MCP": 2,
        "THUMB_IP": 3,
        "THUMB_TIP": 4,
        "INDEX_FINGER_MCP": 5,
        "INDEX_FINGER_PIP": 6,
        "INDEX_FINGER_DIP": 7,
        "INDEX_FINGER_TIP": 8,
        "MIDDLE_FINGER_MCP": 9,
        "MIDDLE_FINGER_PIP": 10,
        "MIDDLE_FINGER_DIP": 11,
        "MIDDLE_FINGER_TIP": 12,
        "RING_FINGER_MCP": 13,
        "RING_FINGER_PIP": 14,
        "RING_FINGER_DIP": 15,
        "RING_FINGER_TIP": 16,
        "PINKY_MCP": 17,
        "PINKY_PIP": 18,
        "PINKY_DIP": 19,
        "PINKY_TIP": 20
    }
    
    # Properties are set via prepare_video_parameters.
    original_bounding_box = None
    enlarged_bounding_box = None
    video = None
    fps = None
    start_time = None
    start_frame_idx = None
    end_time = None
    end_frame_idx = None
    file_path = None

# ------------------------------------------------------------------
# --- END: Abstract properties definitions
# ------------------------------------------------------------------





# -------------------------------------------------------------
# --- START: Abstract methods definitions
# -------------------------------------------------------------
    def api_response(self, request):
        """
        Entry point for processing the left finger tap task.
        """
        try:
            # 1) Process video and define all abstract class parameters
            self.prepare_video_parameters(request)

            # 2) Get detector
            detector = self.get_detector()

            # 3) Get analyzer
            signal_analyzer = self.get_signal_analyzer()

            # 4) Extract landmarks using the defined detector
            essential_landmarks, all_landmarks = self.extract_landmarks(detector)
            
            # 5) Calculate the signal using the land marks
            normalization_factor = self.calculate_normalization_factor(essential_landmarks)

            # 6) Calculate the  normalization factor using the land marks
            raw_signal = self.calculate_signal(essential_landmarks)
            
            # 7) Get output from the signal analyzer
            output = signal_analyzer.analyze(
                normalization_factor=normalization_factor,
                raw_signal=raw_signal,
                start_time=self.start_time,
                end_time=self.end_time
            )
            
            # 6) Structure output
            output["landMarks"] = essential_landmarks
            output["allLandMarks"] = all_landmarks
            output["normalization_factor"] = normalization_factor
            result = output

        except Exception as e:
            result = {'error': str(e)}
            traceback.print_exc()

        if self.video:
            self.video.release()
        if self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)

        return result
    
    def prepare_video_parameters(self, request):
        """
        Parses POST data, saves the video file, computes bounding boxes and frame indices.
        """
        APP_ROOT = os.path.dirname(os.path.abspath(__file__))
        try:
            json_data = json.loads(request.POST['json_data'])
            if 'video' not in request.FILES or len(request.FILES) == 0:
                raise Exception("'video' field missing or no files uploaded")
        except (KeyError, json.JSONDecodeError):
            raise Exception("Invalid or missing 'json_data' in POST data")

        file_name = f"{uuid.uuid4().hex[:15].upper()}.mp4"
        folder_path = os.path.join(APP_ROOT, '../video_uploads')
        file_path = os.path.join(folder_path, file_name)
        FileSystemStorage(folder_path).save(file_name, request.FILES['video'])

        video = cv2.VideoCapture(file_path)
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        original_bounding_box = json_data['boundingBox']
        start_time = json_data['start_time']
        end_time = json_data['end_time']
        fps = video.get(cv2.CAP_PROP_FPS)
        start_frame_idx = math.floor(fps * start_time)
        end_frame_idx = math.floor(fps * end_time)
        
        new_x = int(max(0, original_bounding_box['x'] - original_bounding_box['width'] * 0.125))
        new_y = int(max(0, original_bounding_box['y'] - original_bounding_box['height'] * 0.125))
        new_width = int(min(video_width - new_x, original_bounding_box['width'] * 1.25))
        new_height = int(min(video_height - new_y, original_bounding_box['height'] * 1.25))
        enlarged_bounding_box = {
            'x': new_x,
            'y': new_y,
            'width': new_width,
            'height': new_height
        }

        self.video = video
        self.file_path = file_path
        self.original_bounding_box = original_bounding_box
        self.enlarged_bounding_box = enlarged_bounding_box
        self.start_time = start_time
        self.end_time = end_time
        self.fps = fps
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx

        return {
            "video": video,
            "file_path": file_path,
            "original_bounding_box": original_bounding_box,
            "enlarged_bounding_box": enlarged_bounding_box,
            "start_time": start_time,
            "end_time": end_time,
            "start_frame_idx": start_frame_idx,
            "end_frame_idx": end_frame_idx
        }


    def get_detector(self) -> object:
        """
        Returns the mediapipe hand detector.
        """
        return HandDetector().get_detector()


    def get_signal_analyzer(self) -> object:
        """
        Returns the signal analyzer.
        """
        return PeakfinderSignalAnalyzer()


    def extract_landmarks(self, detector) -> tuple:
        """
        Processes video frames and extracts left hand landmarks.
        For each frame, retrieves thumb tip, index finger tip, middle finger tip, and wrist.
        """
        video = self.video
        start_frame_idx = self.start_frame_idx
        end_frame_idx = self.end_frame_idx
        fps = self.fps
        enlarged_bounding_box = self.enlarged_bounding_box
        essential_landmarks = []
        all_landmarks = []
        enlarged_coords = (
            enlarged_bounding_box['x'],
            enlarged_bounding_box['y'],
            enlarged_bounding_box['x'] + enlarged_bounding_box['width'],
            enlarged_bounding_box['y'] + enlarged_bounding_box['height']
        )
        
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        current_frame_idx = start_frame_idx

        while current_frame_idx < end_frame_idx:
            success, frame = video.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x1, y1, x2, y2 = enlarged_coords
            image_data = rgb_frame[y1:y2, x1:x2, :].astype(np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)
            timestamp = int(current_frame_idx / fps * 1000)
            detection_result = detector.detect_for_video(image, timestamp)
            
            # Look for the left hand
            hand_index = -1
            for idx, label in enumerate(detection_result.handedness):
                if label[0].category_name == "Left":
                    hand_index = idx
                    break

            if hand_index == -1 or not detection_result.hand_landmarks[hand_index]:
                essential_landmarks.append([])
                all_landmarks.append([])
            else:
                hand_landmarks = detection_result.hand_landmarks[hand_index]
                thumb = BaseTask.get_landmark_coords(hand_landmarks[self.LANDMARKS["THUMB_TIP"]], enlarged_coords)
                index_finger = BaseTask.get_landmark_coords(hand_landmarks[self.LANDMARKS["INDEX_FINGER_TIP"]], enlarged_coords)
                middle_finger = BaseTask.get_landmark_coords(hand_landmarks[self.LANDMARKS["MIDDLE_FINGER_TIP"]], enlarged_coords)
                wrist = BaseTask.get_landmark_coords(hand_landmarks[self.LANDMARKS["WRIST"]], enlarged_coords)
                essential = [thumb, index_finger, middle_finger, wrist]
                all_lms = BaseTask.get_all_landmarks_coord(hand_landmarks, enlarged_coords)
                essential_landmarks.append(essential)
                all_landmarks.append(all_lms)
            
            current_frame_idx += 1

        return essential_landmarks, all_landmarks


    def calculate_signal(self, essential_landmarks) -> list:
        """
        For each frame, computes the Euclidean distance between the thumb tip and index finger tip.
        Uses the previous valid value if landmarks are missing.
        """
        signal = []
        prev_dist = 0
        for frame_lms in essential_landmarks:
            if len(frame_lms) < 2:
                signal.append(prev_dist)
                continue
            thumb, index_finger = frame_lms[0], frame_lms[1]
            dist = math.dist(thumb, index_finger)
            prev_dist = dist
            signal.append(dist)
        return signal


    def calculate_normalization_factor(self, essential_landmarks) -> float:
        """
        Computes the maximum distance between the middle finger tip and wrist across frames.
        """
        distances = []
        for frame_lms in essential_landmarks:
            if len(frame_lms) < 4:
                continue
            middle_finger = frame_lms[2]
            wrist = frame_lms[3]
            d = math.dist(middle_finger, wrist)
            distances.append(d)
        return max(distances) if distances else 1.0
# -------------------------------------------------------------
# --- END: Abstract methods definitions
# -------------------------------------------------------------