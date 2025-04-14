# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from django.http import Http404
# import json

# from app.analysis.tasks.hand_movement import HandMovementTask
# from app.analysis.tasks.finger_tap import FingerTapTask
# from app.analysis.tasks.leg_agility import LegAgilityTask
# from app.analysis.tasks.toe_tapping import ToeTappingTask

# # Import from your new structure
# from app.analysis.signal_processors.signal_processor import SignalAnalyzer

# def get_task_instance(task_name: str):
#     """
#     Factory function that returns the correct Task instance
#     based on the provided string.
#     """
#     name_lower = task_name.lower()
#     if "hand movement" in name_lower:
#         return HandMovementTask(task_name)
#     elif "finger tap" in name_lower:
#         return FingerTapTask(task_name)
#     elif "leg agility" in name_lower:
#         return LegAgilityTask(task_name)
#     elif "toe tapping" in name_lower:
#         return ToeTappingTask(task_name)
#     else:
#         raise ValueError(f"Unrecognized task name: {task_name}")

# @api_view(['POST'])
# def update_landmarks(request):
#     """
#     Given a JSON payload containing:
#       - 'task_name'
#       - 'landmarks' (i.e. display_landmarks)
#       - 'fps', 'start_time', 'end_time'
#       - 'allLandMarks'
#       - 'normalization_factor'
#     Re-run the final step of analysis (peak finding, stats, etc.)
#     using the new pipeline structure.
#     """
#     try:
#         json_data = json.loads(request.POST['json_data'])
#     except (KeyError, json.JSONDecodeError):
#         raise Http404("Invalid or missing 'json_data' in POST body")

#     # Extract fields
#     task_name = json_data.get('task_name')
#     display_landmarks = json_data.get('landmarks', [])
#     fps = float(json_data.get('fps', 30))
#     start_time = float(json_data.get('start_time', 0))
#     end_time = float(json_data.get('end_time', 0))
#     all_landmarks = json_data.get('allLandMarks', [])
#     normalization_factor = float(json_data.get('normalization_factor', 1.0))

#     # 1) Get the Task
#     task = get_task_instance(task_name)

#     # 2) Create a SignalAnalyzer
#     analyzer = SignalAnalyzer()

#     # 3) The 'analyzer.analyze' method will:
#     #    - take the raw signal from task.get_signal(display_landmarks)
#     #    - upsample, run peak finder, build stats
#     analysis_output = analyzer.analyze(
#         task=task,
#         display_landmarks=display_landmarks,
#         normalization_factor=normalization_factor,
#         fps=fps,
#         start_time=start_time,
#         end_time=end_time
#     )

#     # 4) Attach the raw data if you want them in the final output
#     analysis_output["landMarks"] = display_landmarks
#     analysis_output["allLandMarks"] = all_landmarks
#     analysis_output["normalization_factor"] = normalization_factor

#     return Response(analysis_output)
