# detectors/__init__.py

from .mediapipe_detectors import (
    create_mediapipe_hand_detector,
    create_mediapipe_pose_heavy,
    create_mediapipe_pose_full
)
from .yolo_detectors import create_yolo_detector

def get_detector(task_name):
    # logic from your old “detector.py”
    t = task_name.lower()
    if "hand movement" in t or "finger tap" in t:
        # typical hand
        return create_mediapipe_hand_detector(), False
    elif "leg agility" in t:
        return create_mediapipe_pose_heavy(), False
    elif "toe tapping" in t:
        return create_mediapipe_pose_full(), False
    else:
        # fallback or raise
        raise ValueError("No suitable detector for that task.")
