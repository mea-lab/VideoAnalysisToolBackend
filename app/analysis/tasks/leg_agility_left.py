import math
import cv2
import mediapipe as mp
import numpy as np
import os, uuid, time, json, traceback
from django.core.files.storage import FileSystemStorage

from .base_task import BaseTask
from app.analysis.detectors.mediapipe_detectors import create_mediapipe_pose_heavy
from app.analysis.signal_processors.signal_processor import SignalAnalyzer

class LegAgilityLeftTask(BaseTask):
    """
    Leg Agility task for the left leg.
    Tracks shoulder midpoint, left knee, and hip midpoint.
    Signal is the vertical difference (shoulder y-coordinate minus knee y-coordinate).
    Normalization factor is the average distance between shoulder midpoint and hip midpoint.
    """
    LANDMARKS = {
        "NOSE": 0,
        "LEFT_EYE_INNER": 1,
        "LEFT_EYE": 2,
        "LEFT_EYE_OUTER": 3,
        "RIGHT_EYE_INNER": 4,
        "RIGHT_EYE": 5,
        "RIGHT_EYE_OUTER": 6,
        "LEFT_EAR": 7,
        "RIGHT_EAR": 8,
        "MOUTH_LEFT": 9,
        "MOUTH_RIGHT": 10,
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13,
        "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15,
        "RIGHT_WRIST": 16,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
        "LEFT_HEEL": 29,
        "RIGHT_HEEL": 30,
        "LEFT_FOOT_INDEX": 31,
        "RIGHT_FOOT_INDEX": 32
    }

    # Abstract properties required by BaseTask
    original_bounding_box = None
    enlarged_bounding_box = None
    video = None
    fps = None
    start_time = None
    start_frame_idx = None
    end_time = None
    end_frame_idx = None

    def api_response(self, request):
        try:
            # 1) Prepare video and set parameters from the request.
            self.prepare_video_parameters(request)
            # 2) Get the pose detector.
            detector = self.get_detector()
            # 3) Get signal analyzer.
            signal_analyzer = self.get_signal_analyzer()
            # 4) Extract landmarks across frames.
            essential_landmarks, all_landmarks = self.extract_landmarks(detector)
            # 5) Compute normalization factor.
            normalization_factor = self.calculate_normalization_factor(essential_landmarks)
            # 6) Compute the raw 1D signal (vertical difference between shoulder and knee).
            raw_signal = self.calculate_signal(essential_landmarks)
            # 7) Analyze the signal and structure the output.
            output = signal_analyzer.analyze(
                normalization_factor=normalization_factor,
                raw_signal=raw_signal,
                start_time=self.start_time,
                end_time=self.end_time
            )
            output["landMarks"] = essential_landmarks
            output["allLandMarks"] = all_landmarks
            output["normalization_factor"] = normalization_factor
            result = output
        except Exception as e:
            result = {'error': str(e)}
        if self.video:
            self.video.release()
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        return result

    def prepare_video_parameters(self, request):
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
        
        # Extract parameters from JSON.
        original_bounding_box = json_data['boundingBox']
        start_time = json_data['start_time']
        end_time = json_data['end_time']
        fps = video.get(cv2.CAP_PROP_FPS)
        start_frame_idx = math.floor(fps * start_time)
        end_frame_idx   = math.floor(fps * end_time)
        # Compute an enlarged bounding box.
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
        
        # Set class attributes.
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
    
    def get_detector(self):
        return create_mediapipe_pose_heavy()

    def get_signal_analyzer(self):
        return SignalAnalyzer()

    def extract_landmarks(self, detector):
        """
        Iterate through frames between start_frame_idx and end_frame_idx,
        crop each frame to the enlarged bounding box, run the pose detector,
        and extract the essential landmarks needed for the leg agility signal.
        """
        video = self.video
        start_frame_idx = self.start_frame_idx
        end_frame_idx = self.end_frame_idx
        fps = self.fps

        # Calculate coordinates for cropping.
        enlarged_bb = self.enlarged_bounding_box
        x1 = enlarged_bb['x']
        y1 = enlarged_bb['y']
        x2 = x1 + enlarged_bb['width']
        y2 = y1 + enlarged_bb['height']
        enlarged_coords = (x1, y1, x2, y2)

        essential_landmarks = []
        all_landmarks = []

        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        current_frame_idx = start_frame_idx

        while current_frame_idx < end_frame_idx:
            success, frame = video.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped_frame = rgb_frame[y1:y2, x1:x2, :].astype(np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_frame)

            timestamp = int((current_frame_idx / fps) * 1000)
            detection_result = detector.detect_for_video(image, timestamp)

            if not detection_result.pose_landmarks:
                essential_landmarks.append([])
                all_landmarks.append([])
            else:
                landmarks = detection_result.pose_landmarks[0]
                # Use the left knee since this is the left task.
                knee_idx = self.LANDMARKS["LEFT_KNEE"]
                knee = [
                    landmarks[knee_idx].x * (x2 - x1),
                    landmarks[knee_idx].y * (y2 - y1)
                ]
                # Average the left and right shoulder coordinates.
                left_shoulder = landmarks[self.LANDMARKS["LEFT_SHOULDER"]]
                right_shoulder = landmarks[self.LANDMARKS["RIGHT_SHOULDER"]]
                shoulder_mid = [
                    ((left_shoulder.x + right_shoulder.x) / 2) * (x2 - x1),
                    ((left_shoulder.y + right_shoulder.y) / 2) * (y2 - y1)
                ]
                # Average the left and right hip coordinates.
                left_hip = landmarks[self.LANDMARKS["LEFT_HIP"]]
                right_hip = landmarks[self.LANDMARKS["RIGHT_HIP"]]
                hip_mid = [
                    ((left_hip.x + right_hip.x) / 2) * (x2 - x1),
                    ((left_hip.y + right_hip.y) / 2) * (y2 - y1)
                ]
                essential = [shoulder_mid, knee, hip_mid]
                all_lms = BaseTask.get_all_landmarks_coord(landmarks, enlarged_coords)
                essential_landmarks.append(essential)
                all_landmarks.append(all_lms)
            current_frame_idx += 1

        return essential_landmarks, all_landmarks

    def calculate_signal(self, essential_landmarks):
        """
        For each frame, compute the difference (shoulder y - knee y).
        If no landmarks were detected in a frame, reuse the previous value.
        """
        signal = []
        prev_signal = 0
        for frame_lms in essential_landmarks:
            if len(frame_lms) < 3:
                signal.append(prev_signal)
                continue
            shoulder_mid, knee, _ = frame_lms
            diff = shoulder_mid[1] - knee[1]
            prev_signal = diff
            signal.append(diff)
        return signal

    def calculate_normalization_factor(self, essential_landmarks):
        """
        Compute the average distance between the shoulder midpoint and hip midpoint.
        """
        distances = []
        for frame_lms in essential_landmarks:
            if len(frame_lms) < 3:
                continue
            shoulder_mid, _, hip_mid = frame_lms
            d = math.dist(shoulder_mid, hip_mid)
            distances.append(d)
        return float(np.mean(distances)) if distances else 1.0
