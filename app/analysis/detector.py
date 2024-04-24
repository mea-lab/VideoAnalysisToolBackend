import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# yolo nas
import torch
from super_gradients.training import models


def get_detector(task):
    if "hand movement" in str.lower(task):
        return mp_hand(), True
    elif "leg agility" in str.lower(task):
        return yolo_nas_pose(), False
    elif "finger tap" in str.lower(task):
        return mp_hand(), True
    elif "toe tapping" in str.lower(task):
        return test_pose(), False


# mediapipe pose detector
def mp_pose():
    VISION_RUNNING_MODE = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='app/models/pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        running_mode=VISION_RUNNING_MODE.VIDEO
    )

    return vision.PoseLandmarker.create_from_options(options)


def test_pose():
    return mp.solutions.pose.Pose()


# mediapipe hand detector
def mp_hand():
    VISION_RUNNING_MODE = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='app/models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=VISION_RUNNING_MODE.VIDEO
    )

    return vision.HandLandmarker.create_from_options(options=options)


# yolo nas pose detector
def yolo_nas_pose():
    model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    return model
