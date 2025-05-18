import os
import math
import json
import uuid
import numpy as np
import traceback
from django.core.files.storage import FileSystemStorage
from rest_framework.response import Response
from rest_framework import status


from .base_task import BaseTask

class GaitTask(BaseTask):
    LANDMARKS = {
        "WRIST": 0,
        "THUMB_CMC": 1,
        "THUMB_MCP": 2,
        "THUMB_IP": 3,
        "THUMB_TIP": 4,
        "INDEX_FINGER_MCP": 5,
        "INDEX_FINGER_PIP": 6,
        "INDEX_FINGER_DIP": 7,
        "INDEX_FINGER_TIP": 8,
        "MIDDLE_FINGER_MCP": 9,
        "MIDDLE_FINGER_PIP": 10,
        "MIDDLE_FINGER_DIP": 11,
        "MIDDLE_FINGER_TIP": 12,
        "RING_FINGER_MCP": 13,
        "RING_FINGER_PIP": 14,
        "RING_FINGER_DIP": 15,
        "RING_FINGER_TIP": 16,
        "PINKY_MCP": 17,
        "PINKY_PIP": 18,
        "PINKY_DIP": 19,
        "PINKY_TIP": 20
    }
        
    # Properties are set via prepare_video_parameters.
    original_bounding_box = None
    enlarged_bounding_box = None
    video = None
    fps = None
    start_time = None
    start_frame_idx = None
    end_time = None
    end_frame_idx = None
    file_path = None
        

    def api_response(self, request):
        """
        Function that handles the api response for each task
        """
        try:
            # 1) Process video and define all abstract class parameters
            self.prepare_video_parameters(request)

            # 2) Get detector
            detector = self.get_detector()

            # 3) Get analyzer
            signal_analyzer = self.get_signal_analyzer()

            # 4) Extract landmarks using the defined detector
            essential_landmarks, all_landmarks = self.extract_landmarks(detector)
            
            # 5) Calculate the signal using the land marks
            normalization_factor = self.calculate_normalization_factor(essential_landmarks)

            # 6) Calculate the  normalization factor using the land marks
            raw_signal = self.calculate_signal(essential_landmarks)
            
            # 7) Get output from the signal analyzer
            output = signal_analyzer.analyze(
                normalization_factor=normalization_factor,
                raw_signal=raw_signal,
                start_time=self.start_time,
                end_time=self.end_time
            )
            
            # 6) Structure output
            output["landMarks"] = essential_landmarks
            output["allLandMarks"] = all_landmarks
            output["normalization_factor"] = normalization_factor
            result = output

        except Exception as e:
            result = {'error': str(e)}
            traceback.print_exc()
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if self.video:
            self.video.release()
        if self.file_path and os.path.exists(self.file_path):
            os.remove(self.file_path)

        return result

    
    def prepare_video_parameters(self, request):
        """
        Prepares video parameters from the HTTP request:
         - Parses JSON for bounding box and time codes.
         - Saves the uploaded video file.
         - Computes the expanded bounding box.
         - Determines FPS and start/end frame indices.
        Returns a dictionary of parameters. 
        MUST DEFINE ALL ABSTRACT PROPERTIES. 
        """
        return None

    def get_detector(self) -> object:
        """
        Getter for the detector used by the task.

        Returns an instance of the detector using the detectors classes
        """
        return None

    def get_signal_analyzer(self) -> object:
        """
        Getter for the signal analyzer used by the task

        Returns an instance of the signal analyze using the analyzer classes
        """
        return None


    
    def calculate_signal(self, essential_landmarks) -> list:
        """
        Given a set of display landmarks (one list per frame), return the raw 1D
        signal array.
        """
        return None


    def extract_landmarks(self, detector) -> tuple:
        """
        Process video frames between start_frame and end_frame and extract hand landmarks 
        for the left hand from each frame.
        
        Returns:
            tuple: (essential_landmarks, all_landmarks)
            - essential_landmarks: a list of lists where each inner list contains the key landmark coordinates for that frame.
            - all_landmarks: a list of lists containing all the landmark coordinates for that frame.
        """
        return None


    def calculate_normalization_factor(self, essential_landmarks) -> float:
        """
        Return a caluclated scalar factor used to normalize the raw 1D signal.
        """
        return None