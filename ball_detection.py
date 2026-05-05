import cv2
import numpy as np
import math

positions = []

def process_frame(frame):
    global positions

    frame = cv2.resize(frame, (640, 480))

    # Start with FULL FRAME (not ROI yet)
    roi = frame.copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # WIDE ORANGE RANGE (to guarantee detection)
    lower_orange = np.array([5, 80, 80])
    upper_orange = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # BASIC NOISE CLEANING
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball_detected = False
    center = None
    angle = None

    for cnt in contours:
        area = cv2.contourArea(cnt)

        #LOWER AREA THRESHOLD (so ball isn’t ignored)
        if area < 300:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        if radius < 5:
            continue

        center = (int(x), int(y))

        # Draw ball
        cv2.circle(roi, center, int(radius), (0, 255, 0), 2)
        cv2.putText(roi, "Ball", center,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        positions.append(center)
        ball_detected = True
        break

    # Limit trajectory memory
    if len(positions) > 20:
        positions.pop(0)

    # Draw trajectory
    for i in range(1, len(positions)):
        cv2.line(roi, positions[i-1], positions[i], (255, 0, 0), 2)

    # Calculate angle
    if len(positions) >= 2:
        x1, y1 = positions[-2]
        x2, y2 = positions[-1]

        dx = x2 - x1
        dy = y1 - y2

        if abs(dx) > 0:
            angle = math.degrees(math.atan2(dy, dx))

            if dy > 2:
                cv2.putText(roi, f"Angle: {int(angle)} deg", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                cv2.arrowedLine(roi, (x1, y1), (x2, y2), (0,255,255), 2)

    result = {
        "ball_detected": ball_detected,
        "position": center,
        "angle": angle,
        "trajectory": positions.copy()
    }

    return roi, mask, result


# ---------------------------
# MAIN LOOP
# ---------------------------
cap = cv2.VideoCapture(r"C:\Users\HP ELITEBOOK\Downloads\basketball2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, mask, result = process_frame(frame)

    cv2.imshow("Frame", processed_frame)
    cv2.imshow("Mask", mask)

    print(result)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()