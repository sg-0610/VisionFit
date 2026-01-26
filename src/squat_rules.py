# src/squat_rules.py
# Simple squat rules based on knee angle (Minimum Viabale product ruleset).

from dataclasses import dataclass

@dataclass(frozen=True)
class SquatFeedback:
    label: str
    detail: str


def evaluate_squat(knee_angle_deg: float) -> SquatFeedback:
    """
    Minimal  squat evaluation based on knee angle.
    Notes:
    - Higher knee angle means standing (near 180)
    - Lower knee angle means deeper squat
    """
    if knee_angle_deg != knee_angle_deg:  # Not a number check
        return SquatFeedback("No pose detected so, Could not compute knee angle.")

    # These are simple ranges these can be tuned later
    if knee_angle_deg > 160:
        return SquatFeedback("Too high", "Lower down (bend knees more).")
    if 110 <= knee_angle_deg <= 160:
        return SquatFeedback("Good range", "Nice depth. Keep chest up.")
    if 90 <= knee_angle_deg < 110:
        return SquatFeedback("Deep", "Good depth. Maintain control.")
    return SquatFeedback("Too deep", "Rise slightly to protect knees.")