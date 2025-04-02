# core/video_processor.py

import math
import cv2

from ..tasks import get_task_instance
from ..detectors import get_detector
from ..utils.geometry import increase_bounding_box
from .analysis_output import build_analysis_output

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
        bounding_box = increase_bounding_box(bounding_box, video_width, video_height)

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

        # 7) Build final analysis output (signals, peaks, stats, etc.)
        return build_analysis_output(
            task, essential_landmarks, all_landmarks,
            fps, start_time, end_time
        )
