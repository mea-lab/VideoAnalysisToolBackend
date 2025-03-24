from rest_framework.decorators import api_view
from rest_framework.response import Response
from app.analysis.analysis import get_analysis_output
import json

@api_view(['POST'])
def update_landmarks(request):
    try:
        json_data = json.loads(request.POST['json_data'])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON data")

    task_name = json_data['task_name']
    display_landmarks = json_data['landmarks']
    fps = json_data['fps']
    start_time =  json_data['start_time'] 
    end_time =  json_data['end_time']
    all_landmarks = json_data['allLandMarks']
    normalization_factor = json_data['normalization_factor']

    result = get_analysis_output(task_name, display_landmarks, normalization_factor, fps, start_time, end_time,all_landmarks)
    return Response(result)