import cv2
import json
import os

from config import *
from ball_detection import detect_ball
from rim_detection import get_rim_position
from trajectory import update_trajectory
from shot_analysis import (
    calculate_angle,
    predict_shot
)

cap = cv2.VideoCapture("C:\\Users\\HP ELITEBOOK\\Videos\\VID-20260530-WA0018.mp4")

rim = get_rim_position()

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
        ball_center,
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