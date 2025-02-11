import numpy as np
from typing import List
from dataclasses import dataclass
from project.motion import SimpleMotionModel

@dataclass
class Detection:
    """
    Data object to handle YOLO model outputs
    This is designed for yolo specifically 
    """
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    features: np.ndarray
    cls: int
    label: str



@dataclass
class Track:
    """
    track class which has an id, motion model, features, history, age, missed frames, class, and label
    used to store information about an object as it moves through the frames
    """
    id: int
    motion_model: SimpleMotionModel
    features: np.ndarray
    history: List[np.ndarray]  # List of past positions
    age: int
    missed_frames: int
    cls: int
    label: str 
