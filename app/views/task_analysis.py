from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
import os, uuid, time, json, traceback
from app.analysis.analysis import analysis

def final_analysis(inputJson, inputVideo):
    data = inputJson

    boundingBox = data['boundingBox']
    fps = data['fps']
    start_time = data['start_time']
    end_time = data['end_time']
    task_name = data['task_name']
    
    return analysis(boundingBox, start_time, end_time, inputVideo, task_name)

@api_view(['POST'])
def task_analysis(request):
    if len(request.FILES) == 0 or 'video' not in request.FILES:
        raise Exception("'video' field missing or no files uploaded")

    try:
        json_data = json.loads(request.POST['json_data'])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON data")

    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    file_name = str(uuid.uuid4().hex[:15].upper()) + ".mp4"
    folder_path = os.path.join(APP_ROOT, '../uploads')
    file_path = os.path.join(folder_path, file_name)
    FileSystemStorage(folder_path).save(file_name, video := request.FILES['video'])

    try:
        print("analysis started")
        start_time = time.time()
        result = final_analysis(json_data, file_path)
        print("Analysis Done in %s seconds" % (time.time() - start_time))
    except Exception as e:
        print(f"Error in processing video: {e}")
        traceback.print_exc()
        result = {'error': str(e)}

    os.remove(file_path)
    return Response(result)
