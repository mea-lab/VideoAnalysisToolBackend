import numpy as np
import mediapipe as mp
import cv2
import math
import json
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# %%
import scipy.signal as signal
import scipy.interpolate as interpolate

from app.analysis.finderPeaksSignal import peakFinder


def json_serialize(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def filterSignal(rawSignal, fs=25, cutOffFrequency=5):
    b, a = signal.butter(2, cutOffFrequency, fs=fs, btype='low', analog=False)
    return signal.filtfilt(b, a, rawSignal)


def finger_tap(fps, bounding_box, start_time, end_time, input_video, is_left_leg):
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='app/models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2, running_mode=VisionRunningMode.VIDEO)

    detector = vision.HandLandmarker.create_from_options(options=options)
    # %%
    # detector = vision.HandLandmarker.create_from_options(options)
    video = cv2.VideoCapture(input_video)

    start_frame = round(fps * start_time)
    end_frame = round(fps * end_time)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frameCounter = start_frame

    knee_landmarks = []
    nose_landmarks = []
    landmarks_signal = []

    knee_landmark_pos = 8
    nose_landmark_pos = 4

    normalization_factor = 1

    if is_left_leg is True:
        knee_landmark_pos = 8

    while frameCounter < end_frame:
        status, frame = video.read()
        if status == False:
            break

        # detect landmarks
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # crop frame based on bounding box info
        x1 = bounding_box['x']
        y1 = bounding_box['y']
        x2 = x1 + bounding_box['width']
        y2 = y1 + bounding_box['height']
        Imagedata = frame[y1:y2, x1:x2, :].astype(np.uint8)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=Imagedata)
        detection_result = detector.detect_for_video(image, frameCounter)
        frameCounter = frameCounter + 1

        if is_left_leg:
            if detection_result.handedness[0][0].category_name == 'Left':
                index = 0
            else:
                index = 1
        else:
            if detection_result.handedness[0][0].category_name == 'Right':
                index = 0
            else:
                index = 1

        # index = 0 if is_left_leg and detection_result.handedness[0][0].category_name == 'Left' else
        # index = detection_result.handedness[0][0].category_name == 'Left' ? 0 : 1

        landmarks = detection_result.hand_landmarks[index]

        # landmarks = detection_result.hand_landmarks[0]

        # if (normalization_factor == 1):
        #     shoulder_mid = [((landmarks[11].x + landmarks[12].x) / 2) * (x2 - x1),
        #                     ((landmarks[11].y + landmarks[12].y) / 2) * (y2 - y1)]
        #     torso_mid = [((landmarks[23].x + landmarks[24].x) / 2) * (x2 - x1),
        #                  ((landmarks[23].y + landmarks[24].y) / 2) * (y2 - y1)]
        #     normalization_factor = math.dist(shoulder_mid, torso_mid)

        p = [landmarks[knee_landmark_pos].x * (x2 - x1), landmarks[knee_landmark_pos].y * (y2 - y1)]
        q = [landmarks[nose_landmark_pos].x * (x2 - x1), landmarks[nose_landmark_pos].y * (y2 - y1)]
        landmarks_signal.append([0, (math.dist(p, q) / normalization_factor)])
        # these are the coordinates of the landmark that you want to display in the video
        knee_landmarks.append([p, q])
        nose_landmarks.append(q)

        # landmarks_signal.append([landmarks[knee_landmark_pos].x - landmarks[nose_landmark_pos].x, landmarks[knee_landmark_pos].y - landmarks[nose_landmark_pos].y])
        # # these are the coordinates of the landmark that you want to display in the video
        # knee_landmarks.append([landmarks[knee_landmark_pos].x * (x2 - x1), landmarks[knee_landmark_pos].y * (y2 - y1)])
        # nose_landmarks.append([landmarks[nose_landmark_pos].x * (x2 - x1), landmarks[nose_landmark_pos].y * (y2 - y1)])

    # plt.imshow(frame[y1:y2,x1:x2,:])

    signalOfInterest = np.array(landmarks_signal)[:, 1]
    signalOfInterest = filterSignal(signalOfInterest, cutOffFrequency=7.5)

    # find peaks using custom algorithm
    currentFs = 1 / fps
    desiredFs = 1 / 60

    duration = end_time - start_time
    print(duration)

    timeVector = np.linspace(0, duration, int(duration / currentFs))

    newTimeVector = np.linspace(0, duration, int(duration / desiredFs))
    upsampleSignal = signal.resample(signalOfInterest, len(newTimeVector))

    distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(upsampleSignal, fs=60,
                                                                                         minDistance=3,
                                                                                         cutOffFrequency=7.5, prct=0.05)

    # plot
    # a plot like this should appear on the page.

    # green dots -> Peaks
    # red dots -> Left Valleys
    # blue dots -> Rigth Valleys
    # figure, ax = plt.subplots(figsize=(15, 5))
    # ax.plot(range(len(distance)), distance)
    # ax.set_ylim([np.min(distance), np.max(distance)])
    # ax.set_xlim([0, len(distance)])
    print(duration)
    print(frameCounter)
    line_time = []
    sizeOfDist = len(distance)
    for index, item in enumerate(distance):
        line_time.append((index / sizeOfDist) * duration + start_time)

    line_peaks = []
    line_peaks_time = []
    line_valleys_start = []
    line_valleys_start_time = []
    line_valleys_end = []
    line_valleys_end_time = []

    line_valleys = []
    line_valleys_time = []

    for index, item in enumerate(peaks):
        # ax.plot(item['openingValleyIndex'], distance[item['openingValleyIndex']], 'ro', alpha=0.75)
        # ax.plot(item['peakIndex'], distance[item['peakIndex']], 'go', alpha=0.75)
        # ax.plot(item['closingValleyIndex'], distance[item['closingValleyIndex']], 'bo', alpha=0.75)
        # line_valleys.append(prevValley+item['openingValleyIndex'])

        line_peaks.append(distance[item['peakIndex']])
        line_peaks_time.append((item['peakIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_start.append(distance[item['openingValleyIndex']])
        line_valleys_start_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_end.append(distance[item['closingValleyIndex']])
        line_valleys_end_time.append((item['closingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys.append(distance[item['openingValleyIndex']])
        line_valleys_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

    amplitude = []
    peakTime = []
    rmsVelocity = []
    maxOpeningSpeed = []
    maxClosingSpeed = []
    averageOpeningSpeed = []
    averageClosingSpeed = []

    for idx, peak in enumerate(peaks):
        # Height measures
        x1 = peak['openingValleyIndex']
        y1 = distance[peak['openingValleyIndex']]

        x2 = peak['closingValleyIndex']
        y2 = distance[peak['closingValleyIndex']]

        x = peak['peakIndex']
        y = distance[peak['peakIndex']]

        f = interpolate.interp1d(np.array([x1, x2]), np.array([y1, y2]))

        amplitude.append(y - f(x))

        # Opening Velocity
        rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']] ** 2)))

        averageOpeningSpeed.append((y - f(x)) / ((peak['peakIndex'] - peak['openingValleyIndex']) * (1 / 60)))
        averageClosingSpeed.append((y - f(x)) / ((peak['closingValleyIndex'] - peak['peakIndex']) * (1 / 60)))

        # timming
        peakTime.append(peak['peakIndex'] * (1 / 60))

    meanAmplitude = np.mean(amplitude)
    stdAmplitude = np.std(amplitude)

    meanRMSVelocity = np.mean(rmsVelocity)
    stdRMSVelocity = np.std(rmsVelocity)
    meanAverageOpeningSpeed = np.mean(averageOpeningSpeed)
    stdAverageOpeningSpeed = np.std(averageOpeningSpeed)
    meanAverageClosingSpeed = np.mean(averageClosingSpeed)
    stdAverageClosingSpeed = np.std(averageClosingSpeed)

    meanCycleDuration = np.mean(np.diff(peakTime))
    stdCycleDuration = np.std(np.diff(peakTime))
    rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    rate = len(peaks) / (peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']) / (1 / 60)

    earlyPeaks = peaks[:len(peaks) // 3]
    latePeaks = peaks[-len(peaks) // 3:]
    amplitudeDecay = np.mean(distance[:len(peaks) // 3]) / np.mean(distance[-len(peaks) // 3:])
    velocityDecay = np.sqrt(
        np.mean(velocity[earlyPeaks[0]['openingValleyIndex']:earlyPeaks[-1]['closingValleyIndex']] ** 2)) / np.sqrt(
        np.mean(velocity[latePeaks[0]['openingValleyIndex']:latePeaks[-1]['closingValleyIndex']] ** 2))
    rateDecay = (len(earlyPeaks) / (
            (earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) / (1 / 60))) / (
                        len(latePeaks) / (
                        (latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) / (1 / 60)))

    cvAmplitude = stdAmplitude / meanAmplitude
    cvCycleDuration = stdCycleDuration / meanCycleDuration
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed

    # results = np.array([meanAmplitude, stdAmplitude,
    #                     meanRMSVelocity, stdRMSVelocity, meanAverageOpeningSpeed, stdAverageOpeningSpeed,
    #                     meanAverageClosingSpeed, stdAverageClosingSpeed,
    #                     meanCycleDuration, stdCycleDuration, rangeCycleDuration, rate,
    #                     amplitudeDecay, velocityDecay, rateDecay,
    #                     cvAmplitude, cvCycleDuration, cvRMSVelocity, cvAverageOpeningSpeed, cvAverageClosingSpeed])

    jsonFinal = {
        "linePlot": {
            "data": distance,
            "time": line_time
        },
        "peaks": {
            "data": line_peaks,
            "time": line_peaks_time
        },
        "valleys": {
            "data": line_valleys,
            "time": line_valleys_time
        },
        "valleys_start": {
            "data": line_valleys_start,
            "time": line_valleys_start_time
        },
        "valleys_end": {
            "data": line_valleys_end,
            "time": line_valleys_end_time
        },
        "radar": {
            "A": [2.5, 3.8, 5.9, 4.2, 8.6, 4.9, 6.9, 9, 8.4, 7.3],
            "B": [3.8, 7.1, 7.4, 5.9, 4.2, 3.1, 6.5, 6.4, 3, 7],
            "labels": ["Mean Frequency", "Mean cycle amplitude", "CV cycle amplitude", "Mean cycle rms velocity",
                       "CV cycle rms velocity",
                       "Mean cycle duration", "CV cycle duration", "Range cycle duration", "Amplitude decay",
                       "Velocity decay"],
            "velocity": velocity
        },
        "radarTable": {
            "MeanAmplitude": meanAmplitude,
            "StdAmplitude": stdAmplitude,
            "MeanRMSVelocity": meanRMSVelocity,
            "StdRMSVelocity": stdRMSVelocity,
            "meanAverageOpeningSpeed": meanAverageOpeningSpeed,
            "stdAverageOpeningSpeed": stdAverageOpeningSpeed,
            "meanAverageClosingSpeed": meanAverageClosingSpeed,
            "stdAverageClosingSpeed": stdAverageClosingSpeed,
            "meanCycleDuration": meanCycleDuration,
            "stdCycleDuration": stdCycleDuration,
            "rangeCycleDuration": rangeCycleDuration,
            "rate": rate,
            "amplitudeDecay": amplitudeDecay,
            "velocityDecay": velocityDecay,
            "rateDecay": rateDecay,
            "cvAmplitude": cvAmplitude,
            "cvCycleDuration": cvCycleDuration,
            "cvRMSVelocity": cvRMSVelocity,
            "cvAverageOpeningSpeed": cvAverageOpeningSpeed,
            "cvAverageClosingSpeed": cvAverageClosingSpeed
        },
        "landMarks": knee_landmarks,
        "normalization_landmarks": nose_landmarks,
        "normalization_factor": normalization_factor

    }

    json_object = json.dumps(jsonFinal, default=json_serialize)

    # Writing to sample.json
    file_name = "finger_tap_left" if is_left_leg is True else "finger_tap_right"
    with open(file_name + ".json", "w") as outfile:
        outfile.write(json_object)

    return jsonFinal


def hand_analysis(fps, bounding_box, start_time, end_time, input_video, is_left_leg):
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = python.BaseOptions(model_asset_path='app/models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2, running_mode=VisionRunningMode.VIDEO)

    detector = vision.HandLandmarker.create_from_options(options=options)
    # %%
    # detector = vision.HandLandmarker.create_from_options(options)
    video = cv2.VideoCapture(input_video)

    start_frame = round(fps * start_time)
    end_frame = round(fps * end_time)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frameCounter = start_frame

    knee_landmarks = []
    nose_landmarks = []
    landmarks_signal = []

    knee_landmark_pos = 8
    nose_landmark_pos = 4

    normalization_factor = 1

    if is_left_leg is True:
        knee_landmark_pos = 8

    while frameCounter < end_frame:
        status, frame = video.read()
        if status == False:
            break

        # detect landmarks
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # crop frame based on bounding box info
        x1 = bounding_box['x']
        y1 = bounding_box['y']
        x2 = int(x1 + bounding_box['width'])
        y2 = int(y1 + bounding_box['height'])
        Imagedata = frame[y1:y2, x1:x2, :].astype(np.uint8)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=Imagedata)
        detection_result = detector.detect_for_video(image, frameCounter)
        frameCounter = frameCounter + 1

        if is_left_leg:
            if detection_result.handedness[0][0].category_name == 'Left':
                index = 0
            else:
                index = 1
        else:
            if detection_result.handedness[0][0].category_name == 'Right':
                index = 0
            else:
                index = 1

        # index = 0 if is_left_leg and detection_result.handedness[0][0].category_name == 'Left' else 1
        # index = detection_result.handedness[0][0].category_name == 'Left' ? 0 : 1

        landmarks = detection_result.hand_landmarks[index]

        # if (normalization_factor == 1):
        #     shoulder_mid = [((landmarks[11].x + landmarks[12].x) / 2) * (x2 - x1),
        #                     ((landmarks[11].y + landmarks[12].y) / 2) * (y2 - y1)]
        #     torso_mid = [((landmarks[23].x + landmarks[24].x) / 2) * (x2 - x1),
        #                  ((landmarks[23].y + landmarks[24].y) / 2) * (y2 - y1)]
        #     normalization_factor = math.dist(shoulder_mid, torso_mid)

        index = [landmarks[8].x * (x2 - x1), landmarks[8].y * (y2 - y1)]
        middle = [landmarks[12].x * (x2 - x1), landmarks[12].y * (y2 - y1)]
        ring = [landmarks[16].x * (x2 - x1), landmarks[16].y * (y2 - y1)]
        wrist = [landmarks[0].x * (x2 - x1), landmarks[0].y * (y2 - y1)]
        #
        # p = [landmarks[knee_landmark_pos].x * (x2 - x1), landmarks[knee_landmark_pos].y * (y2 - y1)]
        # q = [landmarks[nose_landmark_pos].x * (x2 - x1), landmarks[nose_landmark_pos].y * (y2 - y1)]

        distance = ((math.dist(index, wrist)) + (math.dist(middle, wrist)) + (math.dist(ring, wrist))) / 3

        # landmarks_signal.append([0, (math.dist(p, q) / normalization_factor)])
        landmarks_signal.append([0, distance])
        # these are the coordinates of the landmark that you want to display in the video
        knee_landmarks.append([index, middle, ring, wrist])
        # nose_landmarks.append()

        # landmarks_signal.append([landmarks[knee_landmark_pos].x - landmarks[nose_landmark_pos].x, landmarks[knee_landmark_pos].y - landmarks[nose_landmark_pos].y])
        # # these are the coordinates of the landmark that you want to display in the video
        # knee_landmarks.append([landmarks[knee_landmark_pos].x * (x2 - x1), landmarks[knee_landmark_pos].y * (y2 - y1)])
        # nose_landmarks.append([landmarks[nose_landmark_pos].x * (x2 - x1), landmarks[nose_landmark_pos].y * (y2 - y1)])

    # plt.imshow(frame[y1:y2,x1:x2,:])

    signalOfInterest = np.array(landmarks_signal)[:, 1]
    signalOfInterest = filterSignal(signalOfInterest, cutOffFrequency=7.5)

    # find peaks using custom algorithm
    currentFs = 1 / fps
    desiredFs = 1 / 60

    duration = end_time - start_time
    print(duration)

    timeVector = np.linspace(0, duration, int(duration / currentFs))

    newTimeVector = np.linspace(0, duration, int(duration / desiredFs))
    upsampleSignal = signal.resample(signalOfInterest, len(newTimeVector))

    distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(upsampleSignal, fs=60,
                                                                                         minDistance=3,
                                                                                         cutOffFrequency=7.5, prct=0.05)

    # plot
    # a plot like this should appear on the page.

    # green dots -> Peaks
    # red dots -> Left Valleys
    # blue dots -> Rigth Valleys
    # figure, ax = plt.subplots(figsize=(15, 5))
    # ax.plot(range(len(distance)), distance)
    # ax.set_ylim([np.min(distance), np.max(distance)])
    # ax.set_xlim([0, len(distance)])
    print(duration)
    print(frameCounter)
    line_time = []
    sizeOfDist = len(distance)
    for index, item in enumerate(distance):
        line_time.append((index / sizeOfDist) * duration + start_time)

    line_peaks = []
    line_peaks_time = []
    line_valleys_start = []
    line_valleys_start_time = []
    line_valleys_end = []
    line_valleys_end_time = []

    line_valleys = []
    line_valleys_time = []

    for index, item in enumerate(peaks):
        # ax.plot(item['openingValleyIndex'], distance[item['openingValleyIndex']], 'ro', alpha=0.75)
        # ax.plot(item['peakIndex'], distance[item['peakIndex']], 'go', alpha=0.75)
        # ax.plot(item['closingValleyIndex'], distance[item['closingValleyIndex']], 'bo', alpha=0.75)
        # line_valleys.append(prevValley+item['openingValleyIndex'])

        line_peaks.append(distance[item['peakIndex']])
        line_peaks_time.append((item['peakIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_start.append(distance[item['openingValleyIndex']])
        line_valleys_start_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys_end.append(distance[item['closingValleyIndex']])
        line_valleys_end_time.append((item['closingValleyIndex'] / sizeOfDist) * duration + start_time)

        line_valleys.append(distance[item['openingValleyIndex']])
        line_valleys_time.append((item['openingValleyIndex'] / sizeOfDist) * duration + start_time)

    amplitude = []
    peakTime = []
    rmsVelocity = []
    maxOpeningSpeed = []
    maxClosingSpeed = []
    averageOpeningSpeed = []
    averageClosingSpeed = []

    for idx, peak in enumerate(peaks):
        # Height measures
        x1 = peak['openingValleyIndex']
        y1 = distance[peak['openingValleyIndex']]

        x2 = peak['closingValleyIndex']
        y2 = distance[peak['closingValleyIndex']]

        x = peak['peakIndex']
        y = distance[peak['peakIndex']]

        f = interpolate.interp1d(np.array([x1, x2]), np.array([y1, y2]))

        amplitude.append(y - f(x))

        # Opening Velocity
        rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']] ** 2)))

        averageOpeningSpeed.append((y - f(x)) / ((peak['peakIndex'] - peak['openingValleyIndex']) * (1 / 60)))
        averageClosingSpeed.append((y - f(x)) / ((peak['closingValleyIndex'] - peak['peakIndex']) * (1 / 60)))

        # timming
        peakTime.append(peak['peakIndex'] * (1 / 60))

    meanAmplitude = np.mean(amplitude)
    stdAmplitude = np.std(amplitude)

    meanRMSVelocity = np.mean(rmsVelocity)
    stdRMSVelocity = np.std(rmsVelocity)
    meanAverageOpeningSpeed = np.mean(averageOpeningSpeed)
    stdAverageOpeningSpeed = np.std(averageOpeningSpeed)
    meanAverageClosingSpeed = np.mean(averageClosingSpeed)
    stdAverageClosingSpeed = np.std(averageClosingSpeed)

    meanCycleDuration = np.mean(np.diff(peakTime))
    stdCycleDuration = np.std(np.diff(peakTime))
    rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    rate = len(peaks) / (peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']) / (1 / 60)

    earlyPeaks = peaks[:len(peaks) // 3]
    latePeaks = peaks[-len(peaks) // 3:]
    amplitudeDecay = np.mean(distance[:len(peaks) // 3]) / np.mean(distance[-len(peaks) // 3:])
    velocityDecay = np.sqrt(
        np.mean(velocity[earlyPeaks[0]['openingValleyIndex']:earlyPeaks[-1]['closingValleyIndex']] ** 2)) / np.sqrt(
        np.mean(velocity[latePeaks[0]['openingValleyIndex']:latePeaks[-1]['closingValleyIndex']] ** 2))
    rateDecay = (len(earlyPeaks) / (
            (earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) / (1 / 60))) / (
                        len(latePeaks) / (
                        (latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) / (1 / 60)))

    cvAmplitude = stdAmplitude / meanAmplitude
    cvCycleDuration = stdCycleDuration / meanCycleDuration
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed

    # results = np.array([meanAmplitude, stdAmplitude,
    #                     meanRMSVelocity, stdRMSVelocity, meanAverageOpeningSpeed, stdAverageOpeningSpeed,
    #                     meanAverageClosingSpeed, stdAverageClosingSpeed,
    #                     meanCycleDuration, stdCycleDuration, rangeCycleDuration, rate,
    #                     amplitudeDecay, velocityDecay, rateDecay,
    #                     cvAmplitude, cvCycleDuration, cvRMSVelocity, cvAverageOpeningSpeed, cvAverageClosingSpeed])

    jsonFinal = {
        "linePlot": {
            "data": distance,
            "time": line_time
        },
        "peaks": {
            "data": line_peaks,
            "time": line_peaks_time
        },
        "valleys": {
            "data": line_valleys,
            "time": line_valleys_time
        },
        "valleys_start": {
            "data": line_valleys_start,
            "time": line_valleys_start_time
        },
        "valleys_end": {
            "data": line_valleys_end,
            "time": line_valleys_end_time
        },
        "radar": {
            "A": [2.5, 3.8, 5.9, 4.2, 8.6, 4.9, 6.9, 9, 8.4, 7.3],
            "B": [3.8, 7.1, 7.4, 5.9, 4.2, 3.1, 6.5, 6.4, 3, 7],
            "labels": ["Mean Frequency", "Mean cycle amplitude", "CV cycle amplitude", "Mean cycle rms velocity",
                       "CV cycle rms velocity",
                       "Mean cycle duration", "CV cycle duration", "Range cycle duration", "Amplitude decay",
                       "Velocity decay"],
            "velocity": velocity
        },
        "radarTable": {
            "MeanAmplitude": meanAmplitude,
            "StdAmplitude": stdAmplitude,
            "MeanRMSVelocity": meanRMSVelocity,
            "StdRMSVelocity": stdRMSVelocity,
            "meanAverageOpeningSpeed": meanAverageOpeningSpeed,
            "stdAverageOpeningSpeed": stdAverageOpeningSpeed,
            "meanAverageClosingSpeed": meanAverageClosingSpeed,
            "stdAverageClosingSpeed": stdAverageClosingSpeed,
            "meanCycleDuration": meanCycleDuration,
            "stdCycleDuration": stdCycleDuration,
            "rangeCycleDuration": rangeCycleDuration,
            "rate": rate,
            "amplitudeDecay": amplitudeDecay,
            "velocityDecay": velocityDecay,
            "rateDecay": rateDecay,
            "cvAmplitude": cvAmplitude,
            "cvCycleDuration": cvCycleDuration,
            "cvRMSVelocity": cvRMSVelocity,
            "cvAverageOpeningSpeed": cvAverageOpeningSpeed,
            "cvAverageClosingSpeed": cvAverageClosingSpeed
        },
        "landMarks": knee_landmarks,
        "normalization_landmarks": nose_landmarks,
        "normalization_factor": normalization_factor

    }

    json_object = json.dumps(jsonFinal, default=json_serialize)

    file_name = "hand_movement_left" if is_left_leg is True else "hand_movement_right"
    with open(file_name + ".json", "w") as outfile:
        outfile.write(json_object)

    return jsonFinal
