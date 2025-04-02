# tasks/__init__.py

from .hand_movement import HandMovementTask
from .finger_tap import FingerTapTask
from .leg_agility import LegAgilityTask
from .toe_tapping import ToeTappingTask

def get_task_instance(task_name: str):
    """
    Factory function that returns the correct Task instance
    based on the provided string.
    """
    name_lower = task_name.lower()
    if "hand movement" in name_lower:
        return HandMovementTask(task_name)
    elif "finger tap" in name_lower:
        return FingerTapTask(task_name)
    elif "leg agility" in name_lower:
        return LegAgilityTask(task_name)
    elif "toe tapping" in name_lower:
        return ToeTappingTask(task_name)
    else:
        raise ValueError(f"Unrecognized task name: {task_name}")
