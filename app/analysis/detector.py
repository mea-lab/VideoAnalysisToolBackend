import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# yolo nas
import torch
from super_gradients.training import models
import os
import urllib.request


def get_detector(task):
    if "hand movement" in str.lower(task):
        return mp_hand(), False
    elif "leg agility" in str.lower(task):
        return yolo_nas_pose(), False
    elif "finger tap" in str.lower(task):
        return mp_hand(), False
    elif "toe tapping" in str.lower(task):
        return test_pose(), False


# mediapipe pose detector
def mp_pose():
    VISION_RUNNING_MODE = mp.tasks.vision.RunningMode
    current_dir = os.path.dirname(__file__)
    base_options = python.BaseOptions(model_asset_path = os.path.join(current_dir, 'models/pose_landmarker_heavy.task'))
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
    current_dir = os.path.dirname(__file__)
    base_options = python.BaseOptions(model_asset_path = os.path.join(current_dir, 'models/hand_landmarker.task'))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        running_mode=VISION_RUNNING_MODE.VIDEO
    )

    return vision.HandLandmarker.create_from_options(options=options)


# yolo nas pose detector
def yolo_nas_pose():


    current_dir = os.path.dirname(__file__)
    modelpath = os.path.join(current_dir, 'models/yolo_nas_pose_l_coco_pose.pth')
    if not os.path.isfile(modelpath):
        urllib.request.urlretrieve("https://sg-hub-nv.s3.amazonaws.com/models/yolo_nas_pose_l_coco_pose.pth", modelpath)
    #load model from web --- NOT IN USE 
    # model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
    #load model from local --- IN USE
    model = models.get("yolo_nas_pose_l", pretrained_weights='coco_pose',checkpoint_path=modelpath)
    # model.load_state_dict(torch.load(modelpath))
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else device
    model.to(device)
    return model
