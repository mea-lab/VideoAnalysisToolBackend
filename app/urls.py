from django.urls import path
from .views import get_video_data, task_analysis, updatePlotData, update_landmarks

urlpatterns = [
    path('video/', get_video_data),
    path('task_analysis/', task_analysis),
    path('update_plot/', updatePlotData),
    path('update_landmarks/', update_landmarks)
]
