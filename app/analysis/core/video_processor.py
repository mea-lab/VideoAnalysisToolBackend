# core/video_processor.py

import math
import cv2

from ..tasks import get_task_instance
from ..detectors import get_detector
from .signal_processor import SignalAnalyzer

class VideoProcessor:
    """
    Orchestrates the main pipeline:
      1) Expands bounding box
      2) Captures frames from the video between start/end times
      3) Uses a selected Task + Detector to get essential landmarks
      4) Delegates final signal analysis to build_analysis_output
    """

    def process_video(self, bounding_box, start_time, end_time, input_video, task_name):
        # 1) Open the video
        video = cv2.VideoCapture(input_video)

        # 2) Expand bounding box by 25%
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate new x and y coordinates
        new_x = int(max(0, bounding_box['x'] - bounding_box['width'] * 0.125))
        new_y = int(max(0, bounding_box['y'] - bounding_box['height'] * 0.125))
        
        # Calculate new width and height ensuring they don't exceed video boundaries
        new_width = int(min(video_width - new_x, bounding_box['width'] * 1.25))
        new_height = int(min(video_height - new_y, bounding_box['height'] * 1.25))
        
        # Update the bounding box with the new values
        bounding_box = {
            'x': new_x,
            'y': new_y,
            'width': new_width,
            'height': new_height
        }
        

        # 3) Determine start and end frames based on times
        fps = video.get(cv2.CAP_PROP_FPS)
        start_frame_idx = math.floor(fps * start_time)
        end_frame_idx = math.floor(fps * end_time)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

        # 4) Get the Task object & an appropriate Detector
        task = get_task_instance(task_name)
        detector, detector_update = get_detector(task_name)

        essential_landmarks = []
        all_landmarks = []

        current_frame_idx = start_frame_idx
        while current_frame_idx < end_frame_idx:
            success, frame = video.read()
            if not success:
                break

            # (If your approach requires re-initializing the detector each frame, do so)
            if detector_update:
                detector, _ = get_detector(task_name)

            # 5) Use the Task’s method to extract essential + all landmarks
            lms, all_lms = task.get_essential_landmarks(
                frame, current_frame_idx, bounding_box, detector, fps
            )

            # 6) If no landmarks this frame, fallback to previous frame’s data if possible
            if not lms:
                if essential_landmarks:
                    essential_landmarks.append(essential_landmarks[-1])
                    all_landmarks.append(all_landmarks[-1])
                else:
                    essential_landmarks.append([])
                    all_landmarks.append([])
            else:
                essential_landmarks.append(lms)
                all_landmarks.append(all_lms)

            current_frame_idx += 1

        video.release()

        display_landmarks = task.get_display_landmarks(essential_landmarks)
        normalization_factor = task.get_normalization_factor(essential_landmarks)

        analyzer = SignalAnalyzer()
        output = analyzer.analyze(
            task=task,
            display_landmarks=display_landmarks,
            normalization_factor=normalization_factor,
            fps=fps,
            start_time=start_time,
            end_time=end_time
        )

        # Attach raw landmarks to that final dictionary
        output["landMarks"] = display_landmarks
        output["allLandMarks"] = all_landmarks
        output["normalization_factor"] = normalization_factor
        return output