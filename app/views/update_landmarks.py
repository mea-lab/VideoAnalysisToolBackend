from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import Http404
import json

# Import from your new structure
from app.analysis.tasks import get_task_instance
from app.analysis.core.signal_processor import SignalAnalyzer

@api_view(['POST'])
def update_landmarks(request):
    """
    Given a JSON payload containing:
      - 'task_name'
      - 'landmarks' (i.e. display_landmarks)
      - 'fps', 'start_time', 'end_time'
      - 'allLandMarks'
      - 'normalization_factor'
    Re-run the final step of analysis (peak finding, stats, etc.)
    using the new pipeline structure.
    """
    try:
        json_data = json.loads(request.POST['json_data'])
    except (KeyError, json.JSONDecodeError):
        raise Http404("Invalid or missing 'json_data' in POST body")

    # Extract fields
    task_name = json_data.get('task_name')
    display_landmarks = json_data.get('landmarks', [])
    fps = float(json_data.get('fps', 30))
    start_time = float(json_data.get('start_time', 0))
    end_time = float(json_data.get('end_time', 0))
    all_landmarks = json_data.get('allLandMarks', [])
    normalization_factor = float(json_data.get('normalization_factor', 1.0))

    # 1) Get the Task
    task = get_task_instance(task_name)

    # 2) Create a SignalAnalyzer
    analyzer = SignalAnalyzer()

    # 3) The 'analyzer.analyze' method will:
    #    - take the raw signal from task.get_signal(display_landmarks)
    #    - upsample, run peak finder, build stats
    analysis_output = analyzer.analyze(
        task=task,
        display_landmarks=display_landmarks,
        normalization_factor=normalization_factor,
        fps=fps,
        start_time=start_time,
        end_time=end_time
    )

    # 4) Attach the raw data if you want them in the final output
    analysis_output["landMarks"] = display_landmarks
    analysis_output["allLandMarks"] = all_landmarks
    analysis_output["normalization_factor"] = normalization_factor

    return Response(analysis_output)
