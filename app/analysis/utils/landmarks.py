# utils/landmarks.py

import cv2
import mediapipe as mp

import numpy as np

def get_boundaries(bounding_box):
    x1 = bounding_box['x']
    y1 = bounding_box['y']
    w = bounding_box['width']
    h = bounding_box['height']
    return [x1, y1, x1 + w, y1 + h]

def get_hand_index(detection_result, is_left):
    # Determine desired hand based on the input flag.
    direction = "Left" if is_left else "Right"

    handedness = detection_result.handedness

    # Iterate through detected hands to find the matching handedness.
    for idx in range(0, len(handedness)):
        if handedness[idx][0].category_name == direction:
            return idx

    # Return -1 if the desired hand is not found.
    return -1

def get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left, fps):
    # Convert current frame to RGB as expected by MediaPipe.
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    # Crop frame to the provided bounding box.
    [x1, y1, x2, y2] = get_boundaries(bounding_box)

    image_data = current_frame[y1:y2, x1:x2, :].astype(np.uint8)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)

    # Run hand detection on the cropped image.
    detection_result = detector.detect_for_video(image, int(current_frame_idx/fps*1000))
    current_frame_idx += 1

    # Get the index for the hand that matches the specified side.
    hand_index = get_hand_index(detection_result, is_left)

    if hand_index == -1:
        # If no hand is detected, return an empty list.
        return []
    # Return the detected hand landmarks.
    return detection_result.hand_landmarks[hand_index]

def get_landmark_coords(landmark, bounds):
    x1, y1, x2, y2 = bounds
    return [
        landmark.x * (x2 - x1),
        landmark.y * (y2 - y1)
    ]

def get_all_landmarks_coord(landmarks, bounds):
    x1, y1, x2, y2 = bounds
    coords = []
    for lm in landmarks:
        coords.append([
            lm.x * (x2 - x1),
            lm.y * (y2 - y1),
            lm.z
        ])
    return coords
