from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2
from django.core.files.storage import FileSystemStorage
import os
import uuid
from app.analysis.YOLOTracker import YOLOTracker
import time
import json
from app.leg_raise_2 import final_analysis, updatePeaksAndValleys, updateLandMarks


def home(req):
    return render(req, "index.html")
    # return HttpResponse("<h1>Hello world!</h1>")


def analyse_video(path=None):
    if path is None:
        return 0, 0

    try:
        data = cv2.VideoCapture(path)
    except Exception as e:
        print(f"Error in initialising cv2 with the video path : {e}")
        return 0, 0, 0

    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)

    if int(frames) == 0 or int(fps) == 0:
        return 0, 0, 0

    # calculate duration of the video
    seconds = round(frames / fps)

    return seconds, frames, fps


def analyse_video_frames(path=None):
    if path is None:
        return {}

    try:
        print("analysis started")
        start_time = time.time()
        ouputDict = YOLOTracker(path, 'yolov8n.pt', '')
        print("Analysis Done")
        print("--- %s seconds ---" % (time.time() - start_time))

        return ouputDict
    except Exception as e:
        print(f"Error in processing video : {e}")
        return {'error': str(e)}


def update_plot_data(json_data):
    try:
        print("updating plot started")
        start_time = time.time()
        outputDict = updatePeaksAndValleys(json_data)
        # ouputDict = YOLOTracker(path, 'yolov8n.pt', '')
        print("updating the plot is Done")
        print("--- %s seconds ---" % (time.time() - start_time))

        return outputDict
    except Exception as e:
        print(f"Error in processing update_plot_data : {e}")
        return {'error': str(e)}


def leg_analyse_video(json_data, path=None):
    if path is None:
        return {}

    try:
        print("analysis started")
        start_time = time.time()
        outputDict = final_analysis(json_data, path)
        # ouputDict = YOLOTracker(path, 'yolov8n.pt', '')
        print("Analysis Done")
        print("--- %s seconds ---" % (time.time() - start_time))

        return outputDict
    except Exception as e:
        print(f"Error in processing video : {e}")
        return {'error': str(e)}


def handle_upload(request):
    if len(request.FILES) == 0:
        raise Exception("No files are uploaded")

    if 'video' not in request.FILES:
        raise Exception("'video' field missing in form-data")

    video = request.FILES['video']
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))

    file_name = str(uuid.uuid4().hex[:15].upper()) + ".mp4"
    folder_path = os.path.join(APP_ROOT, 'uploads')

    file_path = os.path.join(folder_path, file_name)
    FileSystemStorage(folder_path).save(file_name, video)
    # video.save(path)
    print("video saved")

    val = analyse_video_frames(file_path)
    os.remove(file_path)

    return val


def handle_upload2(request):
    if len(request.FILES) == 0:
        raise Exception("No files are uploaded")

    if 'video' not in request.FILES:
        raise Exception("'video' field missing in form-data")

    video = request.FILES['video']
    try:
        json_data = json.loads(request.POST['json_data'])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON data")

    APP_ROOT = os.path.dirname(os.path.abspath(__file__))

    file_name = str(uuid.uuid4().hex[:15].upper()) + ".mp4"
    folder_path = os.path.join(APP_ROOT, 'uploads')

    file_path = os.path.join(folder_path, file_name)
    FileSystemStorage(folder_path).save(file_name, video)
    # video.save(path)
    print("video saved")

    val = leg_analyse_video(json_data, file_path)
    os.remove(file_path)

    return val


@api_view(['POST'])
def get_video_data(request):
    if request.method == 'POST':
        output = handle_upload(request)

        return Response(output)


@api_view(['POST'])
def leg_raise_task(request):
    if request.method == 'POST':
        output = handle_upload2(request)

        return Response(output)


@api_view(['POST'])
def updatePlotData(request):
    if request.method == 'POST':
        try:
            json_data = json.loads(request.POST['json_data'])
        except json.JSONDecodeError:
            raise Exception("Invalid JSON data")

        output = update_plot_data(json_data)

        return Response(output)


@api_view(['POST'])
def update_landmarks(request):
    if request.method == 'POST':
        try:
            json_data = json.loads(request.POST['json_data'])
        except json.JSONDecodeError:
            raise Exception("Invalid JSON data")

        output = updateLandMarks(json_data)

        return Response(output)
