import cv2
import json
import os
import numpy as np

from config import *
from ball_detection import detect_ball
from rim_detection import get_rim_position
from trajectory import update_trajectory
from shot_analysis import (
    calculate_angle,
    predict_shot
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "resrc", "testSub1.mp4")

cap = cv2.VideoCapture(VIDEO_PATH)

# Try to detect the rim from the video first; fall back to config values inside the function
rim = get_rim_position(cap)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    ball_center, radius, mask = detect_ball(frame)

    trajectory = update_trajectory(
        ball_center,
        MAX_TRAJECTORY_POINTS
    )

    angle = calculate_angle(trajectory)

    shot_success = predict_shot(
        trajectory,
        rim
    )

    # DRAW BALL
    if ball_center is not None:

        cv2.circle(
            frame,
            ball_center,
            int(radius),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            "Ball",
            ball_center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    # DRAW RIM
    cv2.circle(
        frame,
        (rim["x"], rim["y"]),
        rim["radius"],
        (255,0,255),
        2
    )

    # DRAW TRAJECTORY
    for i in range(1, len(trajectory)):

        cv2.line(
            frame,
            trajectory[i-1],
            trajectory[i],
            (255,0,0),
            2
        )

    # SHOW ANGLE
    if angle is not None:

        cv2.putText(
            frame,
            f"Angle: {angle}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

    # SHOW SHOT STATUS
    status = "GOOD SHOT" if shot_success else "MISS"

    cv2.putText(
        frame,
        status,
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    # SAVE TRAJECTORY ARRAY
    output_data = {
        "trajectory": trajectory,
        "angle": angle,
        "shot_prediction": status
    }

    # Fit quadratic in time domain: x(t) and y(t) for a more stable fit
    arc_points = []
    poly_coeffs_x = None
    poly_coeffs_y = None

    if len(trajectory) >= 3:
        try:
            t = np.arange(len(trajectory), dtype=float)
            xs = np.array([p[0] for p in trajectory], dtype=float)
            ys = np.array([p[1] for p in trajectory], dtype=float)

            coeffs_x = np.polyfit(t, xs, 2)
            coeffs_y = np.polyfit(t, ys, 2)
            poly_coeffs_x = coeffs_x.tolist()
            poly_coeffs_y = coeffs_y.tolist()

            sample_t = np.linspace(0, len(trajectory) - 1, num=25)
            sample_x = np.polyval(coeffs_x, sample_t)
            sample_y = np.polyval(coeffs_y, sample_t)
            arc_points = [[int(x), int(y)] for x, y in zip(sample_x, sample_y)]
        except Exception:
            arc_points = []

    output_data["arc"] = arc_points
    output_data["poly_coeffs_x"] = poly_coeffs_x
    output_data["poly_coeffs_y"] = poly_coeffs_y

    os.makedirs("output", exist_ok=True)

    with open(
        "output/trajectory_data.json",
        "w"
    ) as f:

        json.dump(output_data, f)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()