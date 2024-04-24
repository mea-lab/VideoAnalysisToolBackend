from django.urls import path
from app.views import home, get_video_data, leg_raise_task, updatePlotData, update_landmarks

urlpatterns = [
    path('', home),
    path('video/', get_video_data),
    path('leg_raise/', leg_raise_task),
    path('update_plot/', updatePlotData),
    path('update_landmarks/', update_landmarks),
    path('toe_tap/',leg_raise_task)
]
