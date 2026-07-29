import numpy as np
from pybullet_planning.motion_planners.rrt_connect import rrt_connect, birrt


def distance_fn(q1, q2):
    return np.linalg.norm(np.array(q1) - np.array(q2))


def extend_fn(q1, q2, step_size=0.02):
    direction = np.array(q2) - np.array(q1)
    dist = np.linalg.norm(direction)
    if dist == 0:
        return []
    direction /= dist
    steps = int(np.floor(dist / step_size))
    return [
        (np.array(q1) + i * step_size * direction).tolist() for i in range(1, steps + 1)
    ]


def run_rrt_connect(start, goal, collision_fn, sample_fn, max_iterations=1000):
    """
    Run RRT-Connect algorithm to find a path from start to goal.
    start: List[float] - Start configuration.
    goal: List[float] - Goal configuration.
    collision_fn: Callable[[List[float]], bool] - Function to check for collisions.
    """
    path = rrt_connect(
        q1=start,
        q2=goal,
        sample_fn=sample_fn,
        distance_fn=distance_fn,
        extend_fn=extend_fn,
        collision_fn=collision_fn,
        max_iterations=max_iterations,
    )
    return path


def run_birrt(start, goal, collision_fn, sample_fn, max_iterations=1000):
    """
    Run Bidirectional RRT algorithm to find a path from start to goal.
    start: List[float] - Start configuration.
    goal: List[float] - Goal configuration.
    collision_fn: Callable[[List[float]], bool] - Function to check for collisions.
    """
    path = birrt(
        start=start,
        goal=goal,
        sample_fn=sample_fn,
        distance_fn=distance_fn,
        extend_fn=extend_fn,
        collision_fn=collision_fn,
        max_iterations=max_iterations,
    )
    return path
