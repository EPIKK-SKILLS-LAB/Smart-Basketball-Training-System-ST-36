import cv2
import numpy as np


LOWER_ORANGE = np.array([3, 80, 60], dtype=np.uint8)
UPPER_ORANGE = np.array([25, 255, 255], dtype=np.uint8)


def process_frame(frame: np.ndarray) -> dict:
    """Run basketball model processing on one decoded camera frame."""
    height, width = frame.shape[:2]

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    orange_mask = cv2.inRange(
        hsv,
        LOWER_ORANGE,
        UPPER_ORANGE,
    )

    orange_mask = cv2.morphologyEx(
        orange_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    contours, _ = cv2.findContours(
        orange_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    rim = None

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 150:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)

        if 8 <= radius <= min(width, height) * 0.25:
            candidate = {
                "x": round(float(x)),
                "y": round(float(y)),
                "radius": round(float(radius)),
                "area": round(float(area)),
            }

            if rim is None or candidate["area"] > rim["area"]:
                rim = candidate

    if rim is None:
        return {
            "detected": False,
            "confidence": 0.0,
            "rim": None,
            "feedback": "Searching for the rim...",
        }

    confidence = min(
        1.0,
        rim["area"] / max(1.0, width * height * 0.02),
    )

    return {
        "detected": confidence >= 0.25,
        "confidence": round(confidence, 2),
        "rim": rim,
        "feedback": (
            "Rim detected"
            if confidence >= 0.25
            else "Possible rim detected"
        ),
    }