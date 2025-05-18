from django.urls import path
from app.views.get_video_data import get_video_data
from app.views.update_plot_data import updatePlotData
from app.views.update_landmarks import update_landmarks
from app.views.create_task_views import generate_task_urlpatterns


urlpatterns = [
    path('video/', get_video_data),
    path('update_plot/', updatePlotData),
    path('update_landmarks/', update_landmarks)
]

urlpatterns += generate_task_urlpatterns()