import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os


def get_detector(task):
    if "hand movement" in str.lower(task):
        return mp_hand(), False
    elif "leg agility" in str.lower(task):
        # return yolo_nas_pose(), False
        return mp_pose_heavy(), False
    elif "finger tap" in str.lower(task):
        return mp_hand(), False
    elif "toe tapping" in str.lower(task):
        return mp_pose_full(), False 
        # return test_pose(), False
    
   


# mediapipe pose detector heavy (largest model)
def mp_pose_heavy():
    VISION_RUNNING_MODE = mp.tasks.vision.RunningMode
    current_dir = os.path.dirname(__file__)
    base_options = python.BaseOptions(model_asset_path = os.path.join(current_dir, 'models/pose_landmarker_heavy.task'))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=VISION_RUNNING_MODE.VIDEO
    )

    return vision.PoseLandmarker.create_from_options(options=options)

# mediapipe pose detector full (mide-size model)
def mp_pose_full():
    VISION_RUNNING_MODE = mp.tasks.vision.RunningMode
    current_dir = os.path.dirname(__file__)
    base_options = python.BaseOptions(model_asset_path = os.path.join(current_dir, 'models/pose_landmarker_full.task'))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=VISION_RUNNING_MODE.VIDEO
    )

    return vision.PoseLandmarker.create_from_options(options=options)


def test_pose():
    return mp.solutions.pose.Pose()


# mediapipe hand detector
def mp_hand():
    VISION_RUNNING_MODE = mp.tasks.vision.RunningMode
    current_dir = os.path.dirname(__file__)
    base_options = python.BaseOptions(model_asset_path = os.path.join(current_dir, 'models/hand_landmarker.task'))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=VISION_RUNNING_MODE.VIDEO
    )

    return vision.HandLandmarker.create_from_options(options=options)


