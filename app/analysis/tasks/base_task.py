# tasks/base_task.py

from abc import ABC, abstractmethod

class BaseTask(ABC):
    """
    Base class for all tasks (hand movement, finger tap, leg agility, toe tapping, etc.)
    Each concrete subclass must implement these abstract methods
    for retrieving & processing landmarks.
    """

    @abstractmethod
    def get_essential_landmarks(self, frame, frame_idx, bounding_box, detector, fps):
        """
        Extract the *minimal required* landmarks that define the movement
        we want to analyze (e.g. just the wrist + fingertip for finger taps).
        
        Parameters:
        -----------
        frame : np.ndarray
            The current video frame (e.g. from OpenCV).
        frame_idx : int
            The index of the current frame (relative to the full video).
        bounding_box : dict
            A dictionary with keys {x, y, width, height} for region-of-interest.
        detector : object
            An already-initialized detector (e.g. Mediapipe or YOLO).
        fps : float
            The video’s frames-per-second, used in time-based logic.

        Returns:
        --------
        (essential_landmarks, all_landmarks) : (list, list)
            essential_landmarks: The minimal set of key points (e.g. [index_tip, wrist]).
            all_landmarks: Possibly the full set of detected landmarks for debugging or reference.
        """
        pass

    @abstractmethod
    def get_signal(self, display_landmarks):
        """
        Given a “display” set of landmarks (one list per frame),
        return the raw 1D signal array we want to run peak-detection on.

        For example:
          - Hand movement: average finger-to-wrist distance
          - Finger tap: distance between thumb tip & index tip
          - Leg agility: difference between shoulder & knee positions
          - Toe tapping: difference between shoulder & foot positions

        Parameters:
        -----------
        display_landmarks : list of lists
            A per-frame list of (x, y) coordinates.

        Returns:
        --------
        signal : list of floats
            The 1D signal, one value per frame.
        """
        pass

    @abstractmethod
    def get_display_landmarks(self, all_essential_landmarks):
        """
        Possibly transform or reduce the essential landmarks for final display.

        For example, you might:
          - Average the shoulder position across frames to reduce jitter
          - Only keep [thumb, index finger] out of a larger set
          - Return all the essential landmarks if no special processing is needed

        Parameters:
        -----------
        all_essential_landmarks : list of lists
            Each sublist represents the essential landmarks for that frame.

        Returns:
        --------
        display_landmarks : list of lists
            The "cleaned" or "final" set of landmarks to be displayed or used by get_signal.
        """
        pass

    @abstractmethod
    def get_normalization_factor(self, all_essential_landmarks):
        """
        Return a scaling factor to normalize the raw 1D signal (if needed).

        Examples might be:
          - The maximum finger-to-wrist distance across all frames
          - The average distance between the shoulder & hip for leg tasks

        Parameters:
        -----------
        all_essential_landmarks : list of lists
            Each sublist is the essential landmarks for that frame.

        Returns:
        --------
        factor : float
            A scalar used to normalize the signal.
        """
        pass
