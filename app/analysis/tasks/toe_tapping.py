# tasks/toe_tapping.py

import math
import numpy as np

import app.analysis.constants.mp_landmarks as MP_LANDMARKS

from ..utils.landmarks import (
    get_boundaries,
    get_all_landmarks_coord
)
from .base_task import BaseTask
import mediapipe as mp

class ToeTappingTask(BaseTask):
    """
    For 'toe tapping' tasks:
      - We track (shoulder midpoint, toe, hip midpoint) each frame.
      - The signal is (shoulder_mid_y - toe_y).
      - The normalizer is average distance shoulder->hip.
    """

    def __init__(self, task_name="Toe Tapping"):
        self.task_name = task_name
        self.is_left = ("left" in task_name.lower())

    def get_essential_landmarks(self, frame, frame_idx, bounding_box, detector, fps):
        [x1, y1, x2, y2] = get_boundaries(bounding_box)
        sub_frame = frame[y1:y2, x1:x2, :].astype(np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=sub_frame)

        detection_result = detector.detect_for_video(mp_image, int((frame_idx/fps)*1000))
        if not detection_result.pose_landmarks:
            return [], []

        landmarks = detection_result.pose_landmarks[0]

        knee_idx = MP_LANDMARKS.LEFT_KNEE if self.is_left else MP_LANDMARKS.RIGHT_KNEE
        toe_idx = MP_LANDMARKS.LEFT_FOOT_INDEX if self.is_left else MP_LANDMARKS.RIGHT_FOOT_INDEX

        # Shoulders
        left_shoulder = landmarks[MP_LANDMARKS.LEFT_SHOULDER]
        right_shoulder = landmarks[MP_LANDMARKS.RIGHT_SHOULDER]
        shoulder_mid = [
            ((left_shoulder.x + right_shoulder.x) / 2) * (x2 - x1),
            ((left_shoulder.y + right_shoulder.y) / 2) * (y2 - y1)
        ]

        # Hips
        left_hip = landmarks[MP_LANDMARKS.LEFT_HIP]
        right_hip = landmarks[MP_LANDMARKS.RIGHT_HIP]
        hip_mid = [
            ((left_hip.x + right_hip.x) / 2) * (x2 - x1),
            ((left_hip.y + right_hip.y) / 2) * (y2 - y1)
        ]

        toe_landmark = [
            landmarks[toe_idx].x * (x2 - x1),
            landmarks[toe_idx].y * (y2 - y1)
        ]
        # you might or might not use the knee
        knee_landmark = [
            landmarks[knee_idx].x * (x2 - x1),
            landmarks[knee_idx].y * (y2 - y1)
        ]

        essential = [shoulder_mid, toe_landmark, hip_mid]
        all_lms = get_all_landmarks_coord(landmarks, [x1, y1, x2, y2])
        return essential, all_lms

    def get_signal(self, display_landmarks):
        """
        (shoulder_mid_y - toe_y), shifted >= 0
        """
        signal = []
        for frame_lms in display_landmarks:
            if len(frame_lms) < 2:
                signal.append(0)
                continue
            shoulder, toe = frame_lms[:2]
            dist = shoulder[1] - toe[1]
            signal.append(dist)

        # shift so it's >= 0
        min_val = np.min(signal) if len(signal) else 0
        if min_val < 0:
            signal = list(np.array(signal) + abs(min_val))

        return signal

    def get_display_landmarks(self, all_essential_landmarks):
        """
        Average the shoulder midpoint across frames, then pair that with the toe for each frame.
        """
        shoulders = []
        for frame_lms in all_essential_landmarks:
            if len(frame_lms) >= 3:
                shoulders.append(frame_lms[0])

        if not shoulders:
            return [[] for _ in all_essential_landmarks]

        avg_shoulder = list(np.mean(np.array(shoulders), axis=0))

        display = []
        for frame_lms in all_essential_landmarks:
            if len(frame_lms) < 2:
                display.append([])
                continue
            toe = frame_lms[1]
            display.append([avg_shoulder, toe])
        return display

    def get_normalization_factor(self, all_essential_landmarks):
        """
        The average distance from shoulder_mid to hip_mid across frames.
        """
        distances = []
        for frame_lms in all_essential_landmarks:
            if len(frame_lms) < 3:
                continue
            shoulder_mid = frame_lms[0]
            hip_mid = frame_lms[2]
            dist = math.dist(shoulder_mid, hip_mid)
            distances.append(dist)

        return float(np.mean(distances)) if distances else 1.0
