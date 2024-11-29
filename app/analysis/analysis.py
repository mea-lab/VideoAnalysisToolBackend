import math

import numpy as np
import cv2
from app.analysis.util import filter_signal, get_output, filter_signal_bandpass
from app.analysis.detector import get_detector
from app.analysis.task_analysis import get_essential_landmarks, get_signal, get_normalization_factor, \
    get_display_landmarks
import scipy.signal as signal


def increase_bounding_box(box, video_w, video_h):
    new_box = {}
    new_box['x'] = int(max(0, box['x'] - box['width'] * 0.125))
    new_box['y'] = int(max(0, box['y'] - box['height'] * 0.125))
    new_box['width'] = int(min(video_w - new_box['x'], box['width'] * 1.25))
    new_box['height'] = int(min(video_h - new_box['y'], box['height'] * 1.25))

    return new_box


def analysis(bounding_box, start_time, end_time, input_video, task_name):
    video = cv2.VideoCapture(input_video)

    # update bouding box to make it 25% larger.
    video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bounding_box = increase_bounding_box(bounding_box, video_width, video_height)

    fps = video.get(cv2.CAP_PROP_FPS)
    start_frame_idx = math.floor(fps * start_time)
    end_frame_idx = math.floor(fps * end_time)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    current_frame_idx = start_frame_idx

    essential_landmarks = []
    all_landmarks = []

    detector, detector_update = get_detector(task_name)

    while current_frame_idx < end_frame_idx:
        status, current_frame = video.read()

        if status is False:
            break

        if detector_update:
            detector, _ = get_detector(task_name)

        landmarks, allLandmarks = get_essential_landmarks(current_frame, current_frame_idx, task_name, bounding_box, detector, fps)

        # if frame doesn't have essential landmarks use previous landmarks 
        if not landmarks:
            try:
                essential_landmarks.append(essential_landmarks[-1])
                all_landmarks.append(all_landmarks[-1])
            except:
                essential_landmarks.append([])
                all_landmarks.append([])
                
            current_frame_idx += 1
            continue

        essential_landmarks.append(landmarks)
        all_landmarks.append(allLandmarks)
        current_frame_idx += 1



    # skip those landmarks which need not be displayed
    display_landmarks = get_display_landmarks(essential_landmarks, task_name)
    normalization_factor = get_normalization_factor(essential_landmarks, task_name)
    return get_analysis_output(task_name, display_landmarks, normalization_factor, fps, start_time, end_time, all_landmarks)


def get_analysis_output(task_name, display_landmarks, normalization_factor, fps, start_time, end_time, all_landmarks):
    task_signal = get_signal(display_landmarks, task_name)
    signal_of_interest = np.array(task_signal) / normalization_factor
    # signal_of_interest = filter_signal(signal_of_interest, cut_off_frequency=7.5)

    duration = end_time - start_time

    # upsample the signal to 60 FPS, the find peaks stage works best at 60 FPS
    fps = 60
    time_vector = np.linspace(0, duration, int(duration * fps))

    up_sample_signal = signal.resample(signal_of_interest, len(time_vector))

    up_sample_signal = up_sample_signal
    output = get_output(up_sample_signal, duration, start_time)


    output['landMarks'] = display_landmarks
    output['allLandMarks'] = all_landmarks
    output['normalization_factor'] = normalization_factor

    return output
   