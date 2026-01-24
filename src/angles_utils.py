# src/angles_utils.py
# These are the Utility functions for computing joint angles from MediaPipe landmarks.

from __future__ import annotations
from dataclasses import dataclass #Lets define lightweight classes  and is usedfor storing data like points
import math #used for acos() and degree conversion
#Numpy used for vector operations and it also helps make angle formula easier and cleaner
import numpy as np

#Creating a small class to represent a 2D point
@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

#Converting 2D points to vectors which makes uisng math for the rest of the problems easier
def _to_np(p: Point2D) -> np.ndarray:
    return np.array([p.x, p.y], dtype=float)

#The main function
def angle_degrees(a: Point2D, b: Point2D, c: Point2D) -> float:
    """
    Computing the angle ABC (in degrees) using vector geometry.
    B is the vertex of the angle.

    Uses: arccos( (BA · BC) / (|BA| |BC|) ) (This is the standard vector angle formula)
    """
    # The System draws two lines and the angle between those two lines is the joint angle
    ba = _to_np(a) - _to_np(b)
    bc = _to_np(c) - _to_np(b)

    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
# This part here makes sure that he calculation is safe if something goes wrong the system would not crash but safely return angle not avalible
    if ba_norm == 0 or bc_norm == 0:
        return float("nan")

    cos_angle = float(np.dot(ba, bc) / (ba_norm * bc_norm))
    cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for numerical stability
# The angle that the computer calculates internally it converts it to degrees
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

#This makes it easier to use in angle claculations
def landmark_to_point2d(lm) -> Point2D:
    """
    Convert a MediaPipe landmark to a 2D point in normalized coordinates (0..1).
    """
    return Point2D(float(lm.x), float(lm.y))