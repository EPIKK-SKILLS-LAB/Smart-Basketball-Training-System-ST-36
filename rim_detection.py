import cv2
import numpy as np
import os
from config import FRAME_WIDTH, FRAME_HEIGHT, RIM_X, RIM_Y, RIM_RADIUS


def get_rim_position(cap=None, debug=True):
    """Attempt to detect the rim from a video capture.

    If detection fails, fall back to the fixed values from config.
    The ring detection uses the same resized frame size as the main loop.
    """

    fallback = {
        "x": RIM_X,
        "y": RIM_Y,
        "radius": RIM_RADIUS
    }

    if cap is None:
        return fallback

    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, frame = cap.read()
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    except Exception:
        pass

    if not ret or frame is None:
        print("[RIM] Failed to read frame from video")
        return fallback

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    h, w = frame.shape[:2]

    # Search in upper portion where rim typically is
    # Restrict to right side where rim is visible
    x0 = int(w * 0.4)  # Start from 40% of width
    x1 = w  # To right edge
    y0 = 0  # From top
    y1 = int(h * 0.6)  # To 60% of height

    roi = frame[y0:y1, x0:x1]
    
    #HSV to isolate bright/white areas (rim is typically bright)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # White/bright colors: low saturation, high value
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 100, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Convert to grayscale and apply Canny on white regions
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    
    # Only keep edges from bright/white areas
    edges = cv2.bitwise_and(edges, white_mask)

    if debug:
        os.makedirs("output", exist_ok=True)
        debug_roi = roi.copy()
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.imwrite("output/debug_raw_frame.png", frame)
        cv2.imwrite("output/debug_roi_extracted.png", debug_roi)
        cv2.imwrite("output/debug_white_mask.png", white_mask)
        cv2.imwrite("output/debug_edges.png", edges)

    # Try Hough circles with parameters tuned for rim
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=50,
        param1=100,  # Higher threshold - need stronger edges
        param2=20,   # Accumulator threshold
        minRadius=12,
        maxRadius=70,
    )

    best_candidate = None
    best_score = -float("inf")

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        print(f"[RIM] Found {len(circles)} candidates in ROI")

        for cx, cy, cr in circles:
            if cr < 12 or cr > 70:
                continue

            # Convert back to full frame coordinates
            global_x = cx + x0
            global_y = cy + y0
            
            # Score based on size (rim is typically 20-50 pixels)
            size_score = 100 if 18 < cr < 55 else 50
            
            score = size_score

            if score > best_score:
                best_score = score
                best_candidate = (global_x, global_y, cr)

        if best_candidate is not None and best_score > 0:
            cx_full, cy_full, cr_full = best_candidate
            print(f"[RIM] Detected at: ({cx_full}, {cy_full}), r={cr_full}, score={best_score:.1f}")
            if debug:
                debug_frame = frame.copy()
                cv2.circle(debug_frame, (cx_full, cy_full), int(cr_full), (0, 255, 0), 3)
                cv2.putText(debug_frame, f"Score: {best_score:.1f}", (cx_full-40, cy_full-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.imwrite("output/debug_detected_rim.png", debug_frame)
            return {"x": int(cx_full), "y": int(cy_full), "radius": int(cr_full)}
    else:
        print(f"[RIM] No circles found in ROI")