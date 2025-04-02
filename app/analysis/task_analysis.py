# import math
# import numpy as np
# import mediapipe as mp
# import cv2

# # Import custom constants for MediaPipe hand and body landmarks.
# import app.analysis.constants.mp_handlandmarks as MP_HAND_LANDMARKS
# import app.analysis.constants.mp_landmarks as MP_LANDMARKS
# # import app.analysis.constants.yolo_landmarks as YOLO_LANDMARKS

# from mediapipe.framework.formats import landmark_pb2

# # -----------------------------------------------------------------------------
# # Function: draw_opt
# # Description: Draws pose landmarks on an image using MediaPipe drawing utilities.
# # -----------------------------------------------------------------------------
# def draw_opt(rgb_image, detection_result):
#     # Get the list of pose landmarks from the detection result.
#     pose_landmarks_list = detection_result.pose_landmarks[0]
#     # Create a copy of the input image to draw on.
#     annotated_image = np.copy(rgb_image)
    
#     # Create a NormalizedLandmarkList protocol buffer.
#     pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     # Extend the protocol buffer with landmarks from the detection result.
#     pose_landmarks_proto.landmark.extend([
#         landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
#         for landmark in pose_landmarks_list
#     ])
    
#     # Draw the landmarks on the image using default pose connection styles.
#     mp.solutions.drawing_utils.draw_landmarks(
#         annotated_image,
#         pose_landmarks_proto,
#         mp.solutions.pose.POSE_CONNECTIONS,
#         mp.solutions.drawing_styles.get_default_pose_landmarks_style()
#     )
#     return annotated_image

# # -----------------------------------------------------------------------------
# # Function: draw_hand
# # Description: Draws hand landmarks on an image.
# # -----------------------------------------------------------------------------
# def draw_hand(rgb_image, hand_landmarks, bounds=None):
#     # Copy the original image to annotate.
#     annotated_image = np.copy(rgb_image)
#     pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()

#     try:
#         # Try adding hand landmarks directly if they are in normalized coordinates.
#         pose_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
#             for landmark in hand_landmarks
#         ])
#     except:
#         # If landmarks are not in normalized format, convert using bounding box boundaries.
#         [x1, y1, x2, y2] = bounds
#         pose_landmarks_proto.landmark.extend([
#             landmark_pb2.NormalizedLandmark(
#                 x=landmark[0] / (x2 - x1), 
#                 y=landmark[1] / (y2 - y1), 
#                 z=landmark[2]
#             ) for landmark in hand_landmarks
#         ])
#     # Draw the hand landmarks on the image.
#     mp.solutions.drawing_utils.draw_landmarks(
#         annotated_image,
#         pose_landmarks_proto
#     )
#     return annotated_image

# # -----------------------------------------------------------------------------
# # Function: get_essential_landmarks
# # Description: Returns task-specific landmarks based on the given task description.
# # -----------------------------------------------------------------------------
# def get_essential_landmarks(current_frame, current_frame_idx, task, bounding_box, detector, fps):
#     # Determine if the task involves the left side.
#     is_left = False
#     if "left" in str.lower(task):
#         is_left = True

#     # Depending on the task description, call the corresponding function.
#     if "hand movement" in str.lower(task):
#         return get_hand_movement_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector, fps)
#     elif "finger tap" in str.lower(task):
#         return get_finger_tap_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector, fps)
#     elif "leg agility" in str.lower(task):
#         return get_leg_agility_landmarks(bounding_box, detector, current_frame, current_frame_idx, is_left, fps)
#     elif "toe tapping" in str.lower(task):
#         return get_toe_tapping_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left, fps)

# # -----------------------------------------------------------------------------
# # Function: get_signal
# # Description: Computes a task-specific signal from display landmarks.
# # -----------------------------------------------------------------------------
# def get_signal(display_landmarks, task):
#     if "hand movement" in str.lower(task):
#         return get_hand_movement_signal(display_landmarks)
#     elif "finger tap" in str.lower(task):
#         return get_finger_tap_signal(display_landmarks)
#     elif "leg agility" in str.lower(task):
#         return get_leg_agility_signal(display_landmarks)
#     elif "toe tapping" in str.lower(task):
#         return get_toe_tapping_signal(display_landmarks)

# # -----------------------------------------------------------------------------
# # Function: get_normalization_factor
# # Description: Computes a normalization factor for the given task.
# # -----------------------------------------------------------------------------
# def get_normalization_factor(landmarks, task):
#     if "hand movement" in str.lower(task):
#         return get_hand_movement_nf(landmarks)
#     elif "finger tap" in str.lower(task):
#         return get_finger_tap_nf(landmarks)
#     elif "leg agility" in str.lower(task):
#         return get_leg_agility_nf(landmarks)
#     elif "toe tapping" in str.lower(task):
#         return get_toe_tapping_nf(landmarks)

# # -----------------------------------------------------------------------------
# # Function: get_display_landmarks
# # Description: Returns landmarks to be displayed based on the task.
# # -----------------------------------------------------------------------------
# def get_display_landmarks(landmarks, task):
#     if "hand movement" in str.lower(task):
#         return get_hand_movement_display_landmarks(landmarks)
#     elif "finger tap" in str.lower(task):
#         return get_finger_tap_display_landmarks(landmarks)
#     elif "leg agility" in str.lower(task):
#         return get_leg_agility_display_landmarks(landmarks)
#     elif "toe tapping" in str.lower(task):
#         return get_toe_tapping_display_landmarks(landmarks)

# # -----------------------------------------------------------------------------
# # Function: get_leg_agility_landmarks
# # Description: Extracts and computes landmarks for leg agility tasks using MediaPipe.
# # -----------------------------------------------------------------------------
# def get_leg_agility_landmarks(bounding_box, detector, current_frame, current_frame_idx, is_left, fps):
#     # Retrieve bounding box boundaries.
#     [x1, y1, x2, y2] = get_boundaries(bounding_box)

#     # Crop the frame to the region of interest defined by the bounding box.
#     Imagedata = current_frame[y1:y2, x1:x2, :].astype(np.uint8)
#     # Create a MediaPipe image from the ROI.
#     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=Imagedata)
#     # Perform pose detection on the cropped image; time is computed based on current frame index and fps.
#     detection_result = detector.detect_for_video(image, int((current_frame_idx/fps)*1000))
#     # Retrieve pose landmarks from the detection result.
#     landmarks = detection_result.pose_landmarks[0]

#     # Select knee index based on whether it's left or right.
#     knee_idx = MP_LANDMARKS.LEFT_KNEE if is_left else MP_LANDMARKS.RIGHT_KNEE
#     # Calculate knee coordinates relative to the original image using bounding box dimensions.
#     knee_landmark = [landmarks[knee_idx].x * (x2 - x1), landmarks[knee_idx].y * (y2 - y1)]

#     # Calculate shoulder midpoint from left and right shoulder landmarks.
#     left_shoulder = landmarks[MP_LANDMARKS.LEFT_SHOULDER]
#     right_shoulder = landmarks[MP_LANDMARKS.RIGHT_SHOULDER]
#     shoulder_midpoint = [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2]
#     shoulder_midpoint = [shoulder_midpoint[0] * (x2 - x1), shoulder_midpoint[1] * (y2 - y1)]

#     # Calculate hip midpoint from left and right hip landmarks.
#     left_hip = landmarks[MP_LANDMARKS.LEFT_HIP]
#     right_hip = landmarks[MP_LANDMARKS.RIGHT_HIP]
#     hip_midpoint = [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]
#     hip_midpoint = [hip_midpoint[0] * (x2 - x1), hip_midpoint[1] * (y2 - y1)]

#     # Return selected landmarks and all landmark coordinates.
#     return [shoulder_midpoint, knee_landmark, hip_midpoint], get_all_landmarks_coord(landmarks, get_boundaries(bounding_box))

# # -----------------------------------------------------------------------------
# # Function: get_leg_agility_signal
# # Description: Computes a signal for leg agility by comparing shoulder and knee positions.
# # -----------------------------------------------------------------------------
# def get_leg_agility_signal(landmarks_list):
#     signal = []
#     for landmarks in landmarks_list:
#         # Expect each landmarks list to contain shoulder midpoint and knee landmark.
#         [shoulder_midpoint, knee_landmark] = landmarks
#         # Compute vertical distance (difference in y-coordinates) between shoulder and knee.
#         distance = shoulder_midpoint[1] - knee_landmark[1]
#         signal.append(distance)

#     # Adjust the signal so that all values are non-negative.
#     signal = signal + np.abs(np.min(signal))
#     return list(signal)

# # -----------------------------------------------------------------------------
# # Function: get_leg_agility_nf
# # Description: Computes a normalization factor for leg agility using shoulder-hip distance.
# # -----------------------------------------------------------------------------
# def get_leg_agility_nf(landmarks_list):
#     values = []
#     for landmarks in landmarks_list:
#         [shoulder_midpoint, _, hip_midpoint] = landmarks
#         # Compute Euclidean distance between shoulder and hip midpoints.
#         distance = math.dist(shoulder_midpoint, hip_midpoint)
#         values.append(distance)
#     # Return the average distance as the normalization factor.
#     return np.mean(values)

# # -----------------------------------------------------------------------------
# # Function: get_leg_agility_display_landmarks
# # Description: Prepares landmarks for display by using an averaged shoulder midpoint.
# # -----------------------------------------------------------------------------
# def get_leg_agility_display_landmarks(landmarks_list):
#     shoulder_midpoint_position = [] 
#     # Collect shoulder midpoints from all frames.
#     for landmarks in landmarks_list:
#         [shoulder_midpoint, _, _] = landmarks
#         shoulder_midpoint_position.append(shoulder_midpoint)

#     # Compute the average shoulder midpoint across all frames.
#     shoulder_midpoint_position = np.array(shoulder_midpoint_position)
#     shoulder_midpoint_position = list(np.mean(shoulder_midpoint_position, axis=0))

#     display_landmarks = []
#     # For each frame, pair the averaged shoulder midpoint with the corresponding knee landmark.
#     for landmarks in landmarks_list:
#         [_, knee, _] = landmarks
#         display_landmarks.append([shoulder_midpoint_position, knee]) 
#     return display_landmarks

# # -----------------------------------------------------------------------------
# # Function: get_toe_tapping_landmarks
# # Description: Extracts landmarks for toe tapping tasks using MediaPipe.
# # -----------------------------------------------------------------------------
# def get_toe_tapping_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left, fps):
#     [x1, y1, x2, y2] = get_boundaries(bounding_box)

#     # Crop the frame based on the bounding box.
#     Imagedata = current_frame[y1:y2, x1:x2, :].astype(np.uint8)
#     # Create a MediaPipe image for the region of interest.
#     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=Imagedata)
#     # Run detection for the current frame time (in milliseconds).
#     detection_result = detector.detect_for_video(image, int((current_frame_idx/fps)*1000))
#     # Retrieve pose landmarks from the detection result.
#     landmarks = detection_result.pose_landmarks[0]

#     # Choose indices for knee and toe landmarks based on the side.
#     knee_idx = MP_LANDMARKS.LEFT_KNEE if is_left else MP_LANDMARKS.RIGHT_KNEE
#     toe_idx = MP_LANDMARKS.LEFT_FOOT_INDEX if is_left else MP_LANDMARKS.RIGHT_FOOT_INDEX

#     # Calculate shoulder midpoint.
#     left_shoulder = landmarks[MP_LANDMARKS.LEFT_SHOULDER]
#     right_shoulder = landmarks[MP_LANDMARKS.RIGHT_SHOULDER]
#     shoulder_midpoint = [(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2]
#     shoulder_midpoint = [shoulder_midpoint[0] * (x2 - x1), shoulder_midpoint[1] * (y2 - y1)]

#     # Calculate hip midpoint.
#     left_hip = landmarks[MP_LANDMARKS.LEFT_HIP]
#     right_hip = landmarks[MP_LANDMARKS.RIGHT_HIP]
#     hip_midpoint = [(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2]
#     hip_midpoint = [hip_midpoint[0] * (x2 - x1), hip_midpoint[1] * (y2 - y1)]

#     # Calculate toe and knee landmarks in absolute coordinates.
#     toe_landmark = [landmarks[toe_idx].x * (x2 - x1), landmarks[toe_idx].y * (y2 - y1)]
#     kee_landmark = [landmarks[knee_idx].x * (x2 - x1), landmarks[knee_idx].y * (y2 - y1)]

#     # Return selected landmarks along with all landmark coordinates.
#     return [shoulder_midpoint, toe_landmark, hip_midpoint], get_all_landmarks_coord(landmarks, get_boundaries(bounding_box))

# # -----------------------------------------------------------------------------
# # Function: get_toe_tapping_signal
# # Description: Computes the signal for toe tapping by comparing shoulder and toe positions.
# # -----------------------------------------------------------------------------
# def get_toe_tapping_signal(landmarks_list):
#     signal = []
#     # For each frame, calculate the vertical distance between the averaged shoulder and the toe.
#     for landmarks in landmarks_list:
#         [shoulder_midpoint_position, toe] = landmarks
#         distance = shoulder_midpoint_position[1] - toe[1]
#         signal.append(distance)
#     # Adjust the signal values to ensure non-negativity.
#     signal = signal + np.abs(np.min(signal))
#     return list(signal)  

# # -----------------------------------------------------------------------------
# # Function: get_toe_tapping_nf
# # Description: Computes a normalization factor for toe tapping using shoulder-hip distance.
# # -----------------------------------------------------------------------------
# def get_toe_tapping_nf(landmarks_list):
#     values = []
#     for landmarks in landmarks_list:
#         [shoulder_midpoint, _, hip_midpoint] = landmarks
#         distance = math.dist(shoulder_midpoint, hip_midpoint)
#         values.append(distance)
#     # Return the average distance as the normalization factor.
#     return np.mean(values)

# # -----------------------------------------------------------------------------
# # Function: get_toe_tapping_display_landmarks
# # Description: Prepares display landmarks for toe tapping tasks.
# # -----------------------------------------------------------------------------
# def get_toe_tapping_display_landmarks(landmarks_list):
#     shoulder_midpoint_position = [] 
#     # Compute average shoulder midpoint across frames.
#     for landmarks in landmarks_list:
#         [shoulder_midpoint, _, _] = landmarks
#         shoulder_midpoint_position.append(shoulder_midpoint)

#     shoulder_midpoint_position = np.array(shoulder_midpoint_position)
#     shoulder_midpoint_position = list(np.mean(shoulder_midpoint_position, axis=0))

#     display_landmarks = []
#     # For each frame, use the average shoulder midpoint paired with the toe landmark.
#     for landmarks in landmarks_list:
#         [_, toe, _] = landmarks
#         display_landmarks.append([shoulder_midpoint_position, toe]) 
#     return display_landmarks

# # -----------------------------------------------------------------------------
# # Function: get_hand_landmarks
# # Description: Retrieves hand landmarks from a given bounding box using the detector.
# # -----------------------------------------------------------------------------
# def get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left, fps):
#     # Convert current frame to RGB as expected by MediaPipe.
#     current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
#     # Crop frame to the provided bounding box.
#     [x1, y1, x2, y2] = get_boundaries(bounding_box)

#     image_data = current_frame[y1:y2, x1:x2, :].astype(np.uint8)
#     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)

#     # Run hand detection on the cropped image.
#     detection_result = detector.detect_for_video(image, int(current_frame_idx/fps*1000))
#     current_frame_idx += 1

#     # Get the index for the hand that matches the specified side.
#     hand_index = get_hand_index(detection_result, is_left)

#     if hand_index == -1:
#         # If no hand is detected, return an empty list.
#         return []
#     # Return the detected hand landmarks.
#     return detection_result.hand_landmarks[hand_index]

# # -----------------------------------------------------------------------------
# # Function: get_hand_movement_landmarks
# # Description: Extracts landmarks necessary for hand movement tasks.
# # -----------------------------------------------------------------------------
# def get_hand_movement_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector, fps):
#     # Retrieve hand landmarks from the specified region.
#     hand_landmarks = get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left, fps)
#     if not hand_landmarks:
#         return [],[]
#     bounds = get_boundaries(bounding_box)
#     # Get specific landmarks for the index finger, middle finger, ring finger, and wrist.
#     index_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], bounds)
#     middle_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP], bounds)
#     ring_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.RING_FINGER_TIP], bounds)
#     wrist = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.WRIST], bounds)

#     # Debug: Save output image for a specific frame index.
#     if current_frame_idx == 6667:
#         [x1, y1, x2, y2] = bounds
#         landmarks = []
#         landmarks.append(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP])
#         landmarks.append(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP])
#         landmarks.append(hand_landmarks[MP_HAND_LANDMARKS.RING_FINGER_TIP])
#         cv2.imwrite("outputs/" + str(current_frame_idx) + ".jpg", draw_hand(current_frame[y1:y2, x1:x2, :], landmarks))

#     # Return the selected landmarks and all landmark coordinates.
#     return [index_finger, middle_finger, ring_finger, wrist], get_all_landmarks_coord(hand_landmarks, bounds)

# # -----------------------------------------------------------------------------
# # Function: get_hand_movement_signal
# # Description: Computes a signal for hand movement based on distances between fingers and wrist.
# # -----------------------------------------------------------------------------
# def get_hand_movement_signal(landmarks_list):
#     signal = []
#     prevDist = 0
#     for landmarks in landmarks_list:
#         if len(landmarks) < 4:
#             # If not all required landmarks are present, use the previous distance.
#             signal.append(prevDist)
#             continue
#         [index_finger, middle_finger, ring_finger, wrist] = landmarks
#         # Compute average distance from the index, middle, and ring finger tips to the wrist.
#         distance = (math.dist(index_finger, wrist) + 
#                     math.dist(middle_finger, wrist) + 
#                     math.dist(ring_finger, wrist)) / 3
#         prevDist = distance
#         signal.append(distance)
#     return signal

# # -----------------------------------------------------------------------------
# # Function: get_hand_movement_nf
# # Description: Computes normalization factor for hand movement based on the middle finger and wrist.
# # -----------------------------------------------------------------------------
# def get_hand_movement_nf(landmarks_list):
#     values = []
#     for landmarks in landmarks_list:
#         if len(landmarks) < 4:
#             continue  # Skip if insufficient landmarks.
#         [_, middle_finger, _, wrist] = landmarks
#         distance = math.dist(middle_finger, wrist)
#         values.append(distance)
#     # Return the maximum distance as the normalization factor.
#     return np.max(values)

# # -----------------------------------------------------------------------------
# # Function: get_hand_movement_display_landmarks
# # Description: Prepares landmarks for display for hand movement tasks.
# # -----------------------------------------------------------------------------
# def get_hand_movement_display_landmarks(landmarks_list):
#     display_landmarks = []
#     for landmarks in landmarks_list:
#         if len(landmarks) < 4:
#             display_landmarks.append([])
#             continue  # Skip frames with insufficient landmarks.
#         [index_finger, middle_finger, ring_finger, wrist] = landmarks
#         display_landmarks.append([index_finger, middle_finger, ring_finger, wrist])
#     return display_landmarks

# # -----------------------------------------------------------------------------
# # Function: get_finger_tap_landmarks
# # Description: Extracts landmarks for the finger tapping task.
# # -----------------------------------------------------------------------------
# def get_finger_tap_landmarks(current_frame, current_frame_idx, bounding_box, is_left, detector, fps):
#     # Get hand landmarks using the specified bounding box.
#     hand_landmarks = get_hand_landmarks(bounding_box, detector, current_frame_idx, current_frame, is_left, fps)
#     if not hand_landmarks:
#         return [],[]

#     bounds = get_boundaries(bounding_box)
#     # Retrieve landmarks for thumb tip, index finger tip, middle finger tip, and wrist.
#     thumb_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.THUMB_TIP], bounds)
#     index_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], bounds)
#     middle_finger = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP], bounds)
#     wrist = get_landmark_coords(hand_landmarks[MP_HAND_LANDMARKS.WRIST], bounds)

#     # Debug: Save output image for a specific frame index.
#     if current_frame_idx == 1408:
#         [x1, y1, x2, y2] = bounds
#         landmarks = [hand_landmarks[MP_HAND_LANDMARKS.INDEX_FINGER_TIP], hand_landmarks[MP_HAND_LANDMARKS.THUMB_TIP]]
#         cv2.imwrite("outputs/" + str(current_frame_idx) + ".jpg", draw_hand(current_frame[y1:y2, x1:x2, :], landmarks))

#     # Return selected landmarks and all landmark coordinates.
#     return [thumb_finger, index_finger, middle_finger, wrist], get_all_landmarks_coord(hand_landmarks, bounds)

# # -----------------------------------------------------------------------------
# # Function: get_finger_tap_signal
# # Description: Computes the finger tapping signal by measuring the distance between the thumb and index finger.
# # -----------------------------------------------------------------------------
# def get_finger_tap_signal(landmarks_list):
#     signal = []
#     prev = 0
#     for landmarks in landmarks_list:
#         if not landmarks:
#             signal.append(prev)
#             continue
#         [thumb_finger, index_finger] = landmarks
#         # Compute Euclidean distance between thumb tip and index finger tip.
#         distance = math.dist(thumb_finger, index_finger)
#         prev = distance
#         signal.append(distance)
#     return signal

# # -----------------------------------------------------------------------------
# # Function: get_finger_tap_nf
# # Description: Computes normalization factor for finger tapping based on middle finger to wrist distance.
# # -----------------------------------------------------------------------------
# def get_finger_tap_nf(landmarks_list):
#     values = []
#     for landmarks in landmarks_list:
#         if not landmarks:
#             continue
#         [_, _, middle_finger, wrist] = landmarks
#         distance = math.dist(middle_finger, wrist)
#         values.append(distance)
#     return np.max(values)

# # -----------------------------------------------------------------------------
# # Function: get_finger_tap_display_landmarks
# # Description: Prepares display landmarks for the finger tapping task.
# # -----------------------------------------------------------------------------
# def get_finger_tap_display_landmarks(landmarks_list):
#     display_landmarks = []
#     for landmarks in landmarks_list:
#         if not landmarks:
#             display_landmarks.append([])
#             continue
#         [thumb_finger, index_finger, _, _] = landmarks
#         display_landmarks.append([thumb_finger, index_finger])
#     return display_landmarks

# # -----------------------------------------------------------------------------
# # Function: get_hand_index
# # Description: Determines the index of the hand (left or right) based on detection results.
# # -----------------------------------------------------------------------------
# def get_hand_index(detection_result, is_left):
#     # Determine desired hand based on the input flag.
#     direction = "Left" if is_left else "Right"

#     handedness = detection_result.handedness

#     # Iterate through detected hands to find the matching handedness.
#     for idx in range(0, len(handedness)):
#         if handedness[idx][0].category_name == direction:
#             return idx

#     # Return -1 if the desired hand is not found.
#     return -1

# # -----------------------------------------------------------------------------
# # Function: get_landmark_coords
# # Description: Converts normalized landmark coordinates to absolute pixel coordinates within the bounding box.
# # -----------------------------------------------------------------------------
# def get_landmark_coords(landmark, bounds):
#     [x1, y1, x2, y2] = bounds
#     return [landmark.x * (x2 - x1), landmark.y * (y2 - y1)]

# # -----------------------------------------------------------------------------
# # Function: get_all_landmarks_coord
# # Description: Converts all landmarks from normalized to absolute coordinates using the bounding box.
# # -----------------------------------------------------------------------------
# def get_all_landmarks_coord(landmark, bounds):
#     [x1, y1, x2, y2] = bounds
#     return [[item.x * (x2 - x1), item.y * (y2 - y1), item.z] for item in landmark]

# # -----------------------------------------------------------------------------
# # Function: get_all_landmarks_coord_YOLONAS
# # Description: Converts landmarks in YOLONAS format to a list of coordinates.
# # Note: Currently unused; provided for legacy compatibility.
# # -----------------------------------------------------------------------------
# def get_all_landmarks_coord_YOLONAS(landmark):
#     return [[item[0], item[1]] for item in landmark]

# # -----------------------------------------------------------------------------
# # Function: get_boundaries
# # Description: Computes the boundary coordinates (x1, y1, x2, y2) of the bounding box.
# # -----------------------------------------------------------------------------
# def get_boundaries(bounding_box):
#     x1 = int(bounding_box['x'])
#     y1 = int(bounding_box['y'])
#     x2 = x1 + int(bounding_box['width'])
#     y2 = y1 + int(bounding_box['height'])
#     return [x1, y1, x2, y2]
