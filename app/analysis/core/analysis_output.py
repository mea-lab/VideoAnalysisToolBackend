# core/analysis_output.py

from .signal_processor import SignalAnalyzer

def build_analysis_output(task, essential_landmarks, all_landmarks, fps,
                          start_time, end_time):
    """
    1) Convert essential_landmarks -> display_landmarks
    2) Get normalization factor
    3) Use SignalAnalyzer to get final time-series, peaks, stats, etc.
    4) Return a big dictionary
    """

    display_landmarks = task.get_display_landmarks(essential_landmarks)
    normalization_factor = task.get_normalization_factor(essential_landmarks)

    analyzer = SignalAnalyzer()
    output = analyzer.analyze(
        task=task,
        display_landmarks=display_landmarks,
        normalization_factor=normalization_factor,
        fps=fps,
        start_time=start_time,
        end_time=end_time
    )

    # Attach raw landmarks to that final dictionary
    output["landMarks"] = display_landmarks
    output["allLandMarks"] = all_landmarks
    output["normalization_factor"] = normalization_factor
    return output
