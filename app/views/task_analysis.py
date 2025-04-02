from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
import os, uuid, time, json, traceback

# Import the new VideoProcessor from your refactored pipeline
# Adjust the import path to match your project structure
from app.analysis.core.video_processor import VideoProcessor

def final_analysis(input_json, input_video):
    """
    Replaces the old `analysis(...)` call.
    Now uses the new `VideoProcessor().process_video(...)` approach.
    """
    # Extract required fields from input_json
    bounding_box = input_json['boundingBox']
    start_time   = input_json['start_time']
    end_time     = input_json['end_time']
    task_name    = input_json['task_name']
    
    # Create a VideoProcessor instance
    processor = VideoProcessor()
    # Run the pipeline
    result = processor.process_video(
        bounding_box, 
        start_time, 
        end_time, 
        input_video, 
        task_name
    )
    return result

@api_view(['POST'])
def task_analysis(request):
    """
    1) Validates the POST data (video file + JSON).
    2) Saves the uploaded video file temporarily.
    3) Calls `final_analysis(...)` to run the new pipeline.
    4) Returns the JSON result.
    """
    if 'video' not in request.FILES or len(request.FILES) == 0:
        raise Exception("'video' field missing or no files uploaded")

    try:
        json_data = json.loads(request.POST['json_data'])
    except (KeyError, json.JSONDecodeError):
        raise Exception("Invalid or missing 'json_data' in POST data")

    # Prepare to save the uploaded video
    APP_ROOT = os.path.dirname(os.path.abspath(__file__))
    file_name = f"{uuid.uuid4().hex[:15].upper()}.mp4"
    folder_path = os.path.join(APP_ROOT, '../uploads')
    file_path = os.path.join(folder_path, file_name)

    # Save the file
    FileSystemStorage(folder_path).save(file_name, video := request.FILES['video'])

    try:
        print("analysis started")
        start_t = time.time()

        # Call our new pipeline
        result = final_analysis(json_data, file_path)

        print("Analysis Done in %s seconds" % (time.time() - start_t))
    except Exception as e:
        print(f"Error in processing video: {e}")
        traceback.print_exc()
        result = {'error': str(e)}

    # Remove the video file after processing
    os.remove(file_path)
    return Response(result)
