# tasks/finger_tap.py

import math

import app.analysis.constants.mp_handlandmarks as MP_HAND_LANDMARKS
import numpy as np

from ..utils.landmarks import (
    get_boundaries,
    get_hand_landmarks,
    get_landmark_coords,
    get_all_landmarks_coord
)
from .base_task import BaseTask

class FingerTapTask(BaseTask):
    """
    For 'finger tap' tasks:
      - We typically track the distance between thumb tip and index finger tip.
      - Normalization factor might be the distance between middle finger and wrist.
    """

    def __init__(self, task_name="Finger Tap"):
        self.task_name = task_name
        self.is_left = ("left" in task_name.lower())

    def get_essential_landmarks(self, frame, frame_idx, bounding_box, detector, fps):
        """
        1) Detect the relevant hand.
        2) Extract thumb finger tip, index finger tip, middle finger tip, wrist.
        """
        hand_landmarks = get_hand_landmarks(
            bounding_box = bounding_box, detector=detector, current_frame_idx=frame_idx, current_frame=frame, fps=fps, is_left=self.is_left
        )
        if not hand_landmarks:
            return [], []

        bounds = get_boundaries(bounding_box)
        thumb_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.THUMB_TIP], bounds)
        index_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], bounds)
        middle_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP], bounds)
        wrist = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.WRIST], bounds)

        essential = [thumb_finger, index_finger, middle_finger, wrist]
        all_lms = get_all_landmarks_coord(hand_landmarks, bounds)
        return essential, all_lms

    def get_signal(self, display_landmarks):
        """
        The signal is simply the distance between (thumb tip, index tip).
        If landmarks missing, reuse last known distance.
        """
        signal = []
        prev_dist = 0
        for frame_lms in display_landmarks:
            if len(frame_lms) < 2:
                signal.append(prev_dist)
                continue
            thumb, index_finger = frame_lms[0], frame_lms[1]
            dist = math.dist(thumb, index_finger)
            prev_dist = dist
            signal.append(dist)
        return signal

    def get_display_landmarks(self, all_essential_landmarks):
        """
        We only need [thumb_finger, index_finger] for display. 
        (But we do have [thumb, index, middle, wrist] available.)
        """
        display_lms = []
        for frame_lms in all_essential_landmarks:
            if len(frame_lms) < 2:
                display_lms.append([])
            else:
                # first two are thumb, index
                display_lms.append([frame_lms[0], frame_lms[1]])
        return display_lms

    def get_normalization_factor(self, all_essential_landmarks):
        """
        Original approach used the distance from (middle_finger -> wrist).
        Return the maximum over all frames to scale the final signal.
        """
        distances = []
        for frame_lms in all_essential_landmarks:
            if len(frame_lms) < 4:
                continue
            # frame_lms = [thumb, index, middle, wrist]
            middle_finger = frame_lms[2]
            wrist = frame_lms[3]
            d = math.dist(middle_finger, wrist)
            distances.append(d)

        return max(distances) if distances else 1.0
