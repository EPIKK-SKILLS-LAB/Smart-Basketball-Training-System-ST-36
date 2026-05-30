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


def predict_shot(trajectory, rim_position):

    if len(trajectory) < 5:
        return False

    bx, by = trajectory[-1]
    rx = rim_position["x"]
    ry = rim_position["y"]
    rr = rim_position["radius"]

    x1, y1 = trajectory[-5]
    x2, y2 = trajectory[-1]

    if x2 == x1 and y2 == y1:
        return False

    moving_down = (y2 - y1) > 0
    distance = math.sqrt((bx - rx) ** 2 + (by - ry) ** 2)
    close_to_rim = distance <= rr * 1.8

    line_distance = abs((y2 - y1) * rx - (x2 - x1) * ry + x2 * y1 - y2 * x1)
    line_distance /= math.hypot(y2 - y1, x2 - x1)
    aimed_at_rim = line_distance <= rr * 1.5

    return moving_down and close_to_rim and aimed_at_rim