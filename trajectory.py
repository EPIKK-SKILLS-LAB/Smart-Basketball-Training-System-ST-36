trajectory_points = []

def update_trajectory(center, max_points=50):

    global trajectory_points

    if center is not None:
        trajectory_points.append(center)

    if len(trajectory_points) > max_points:
        trajectory_points.pop(0)

    return trajectory_points