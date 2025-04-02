# views.py (or wherever you define get_video_data)

from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
import os, uuid, time

# 1) Import your newly relocated function
from app.analysis.detectors.yolo_detectors import yolo_tracker

@api_view(['POST'])
def get_video_data(request):
    if len(request.FILES) == 0 or 'video' not in request.FILES:
        raise Exception("'video' field missing or no files uploaded")

    video = request.FILES['video']
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    file_name = str(uuid.uuid4().hex[:15].upper()) + ".mp4"
    folder_path = os.path.join(APP_ROOT, '../uploads')
    file_path = os.path.join(folder_path, file_name)
    FileSystemStorage(folder_path).save(file_name, video)

    try:
        print("analysis started")
        start_time = time.time()

        # 2) Build path to YOLO model
        current_dir = os.path.dirname(__file__)
        pathtomodel = os.path.join(current_dir, '../models/yolov8n.pt')

        # 3) Run YOLO-based tracker
        result = yolo_tracker(file_path, pathtomodel, device='')

        print("Analysis Done in %s seconds" % (time.time() - start_time))
    except Exception as e:
        result = {'error': str(e)}

    # Optionally remove file after
    os.remove(file_path)
    return Response(result)
