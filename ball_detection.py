import cv2
import numpy as np
from config import LOWER_ORANGE, UPPER_ORANGE

def detect_ball(frame):

    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower = np.array(LOWER_ORANGE)
    upper = np.array(UPPER_ORANGE)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:

        area = cv2.contourArea(cnt)

        if area < 300:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        if radius < 5:
            continue

        center = (int(x), int(y))

        return center, radius, mask

    return None, None, mask