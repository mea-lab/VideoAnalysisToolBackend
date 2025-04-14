# detectors/mediapipe_detectors.py

import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def create_mediapipe_hand_detector():
    running_mode = vision.RunningMode
    current_dir = os.path.dirname(__file__)
    base_options = python.BaseOptions(
        model_asset_path=os.path.join(current_dir, '../models/hand_landmarker.task')
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=running_mode.VIDEO
    )
    return vision.HandLandmarker.create_from_options(options=options)

def create_mediapipe_pose_heavy():
    running_mode = vision.RunningMode
    current_dir = os.path.dirname(__file__)
    base_options = python.BaseOptions(
        model_asset_path=os.path.join(current_dir, '../models/pose_landmarker_heavy.task')
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=running_mode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options=options)

def create_mediapipe_pose_full():
    running_mode = vision.RunningMode
    current_dir = os.path.dirname(__file__)
    base_options = python.BaseOptions(
        model_asset_path=os.path.join(current_dir, '../models/pose_landmarker_full.task')
    )
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=running_mode.VIDEO
    )
    return vision.PoseLandmarker.create_from_options(options=options)
