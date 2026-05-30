import math

def calculate_angle(trajectory):

    if len(trajectory) < 2:
        return None

    x1, y1 = trajectory[-2]
    x2, y2 = trajectory[-1]

    dx = x2 - x1
    dy = y1 - y2

    if dx == 0:
        return 90

    angle = math.degrees(math.atan2(dy, dx))

    return round(angle, 2)


def predict_shot(ball_position, rim_position):

    if ball_position is None:
        return False

    bx, by = ball_position

    rx = rim_position["x"]
    ry = rim_position["y"]
    rr = rim_position["radius"]

    distance = math.sqrt((bx - rx)**2 + (by - ry)**2)

    return distance <= rr