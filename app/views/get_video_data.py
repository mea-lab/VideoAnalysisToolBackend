from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
import os, uuid, time
from app.analysis.YOLOTracker import YOLOTracker

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
        current_dir = os.path.dirname(__file__)
        pathtomodel = os.path.join(current_dir, '../models/yolov8n.pt')
        result = YOLOTracker(file_path, pathtomodel, '')
        print("Analysis Done in %s seconds" % (time.time() - start_time))
    except Exception as e:
        result = {'error': str(e)}

    os.remove(file_path)
    return Response(result)
