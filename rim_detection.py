import cv2
import numpy as np
from config import RIM_X, RIM_Y, RIM_RADIUS


def get_rim_position(cap=None):
    """Attempt to detect the rim from a video capture. If detection fails,
    fall back to the fixed values from config.

    If a `cap` is provided, the function will read a single frame and then
    reset the video position to the start.
    """

    # fallback
    fallback = {
        "x": RIM_X,
        "y": RIM_Y,
        "radius": RIM_RADIUS
    }

    if cap is None:
        return fallback

    # Read one frame to analyze
    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, frame = cap.read()
    # reset to original position (usually 0)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    except Exception:
        pass

    if not ret or frame is None:
        return fallback

    h, w = frame.shape[:2]

    # focus search on right half of the image where hoop usually is
    roi = frame[0:h, int(w * 0.45):w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Hough Circle detection
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=120,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # choose the circle with largest radius (most likely rim)
        best = max(circles[0, :], key=lambda c: c[2])
        cx, cy, cr = best
        # convert roi coordinates back to full frame coordinates
        cx_full = int(cx + int(w * 0.45))
        cy_full = int(cy)

        return {"x": cx_full, "y": cy_full, "radius": int(cr)}

    return fallback