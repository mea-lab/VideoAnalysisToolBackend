# tasks/hand_movement.py

import math

# If you keep your MP constants in app.analysis.constants:
import app.analysis.constants.mp_handlandmarks as MP_HAND_LANDMARKS

from ..utils.landmarks import (
    get_boundaries,
    get_hand_landmarks,
    get_landmark_coords,
    get_all_landmarks_coord
)
from .base_task import BaseTask

class HandMovementTask(BaseTask):
    """
    For 'hand movement' tasks:
      - We detect the full hand (index/middle/ring fingertips + wrist).
      - The signal is average fingertip-to-wrist distance.
      - The normalization factor is the max of (middle_finger,wrist) distance across frames.
    """

    def __init__(self, task_name="Hand Movement"):
        self.task_name = task_name
        self.is_left = ("left" in task_name.lower())

    def get_essential_landmarks(self, frame, frame_idx, bounding_box, detector, fps):
        """
        1) Crop frame to bounding box.
        2) Run MP hand detection.
        3) Extract index/middle/ring fingertips + wrist as essential landmarks.
        """
        # Reuse a helper that returns the “best” hand's landmarks (or empty if none).
        hand_landmarks = get_hand_landmarks(
            frame, frame_idx, bounding_box, detector, fps, is_left=self.is_left
        )
        if not hand_landmarks:
            return [], []

        bounds = get_boundaries(bounding_box)
        index_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], bounds)
        middle_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP], bounds)
        ring_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.RING_FINGER_TIP], bounds)
        wrist = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.WRIST], bounds)

        essential = [index_finger, middle_finger, ring_finger, wrist]
        all_lms = get_all_landmarks_coord(hand_landmarks, bounds)
        return essential, all_lms

    def get_signal(self, display_landmarks):
        """
        For each frame, we measure average distance of (index, middle, ring) to wrist.
        If missing landmarks, fallback to previous distance.
        """
        signal = []
        prev_dist = 0
        for frame_lms in display_landmarks:
            if len(frame_lms) < 4:
                # no detection => reuse prev
                signal.append(prev_dist)
                continue

            index_finger, middle_finger, ring_finger, wrist = frame_lms
            distance = (
                math.dist(index_finger, wrist) +
                math.dist(middle_finger, wrist) +
                math.dist(ring_finger, wrist)
            ) / 3.0

            prev_dist = distance
            signal.append(distance)

        return signal

    def get_display_landmarks(self, all_essential_landmarks):
        """
        We just display the same 4 points [index,middle,ring,wrist] as collected.
        No special averaging or skipping needed (unlike e.g. 'leg agility' might).
        """
        return all_essential_landmarks

    def get_normalization_factor(self, all_essential_landmarks):
        """
        The max distance from (middle_finger -> wrist) across all frames.
        This was your original approach for 'hand movement'.
        """
        distances = []
        for frame_lms in all_essential_landmarks:
            if len(frame_lms) < 4:
                continue
            # frame_lms = [index, middle, ring, wrist]
            middle_finger = frame_lms[1]
            wrist = frame_lms[3]
            d = math.dist(middle_finger, wrist)
            distances.append(d)

        return max(distances) if distances else 1.0
