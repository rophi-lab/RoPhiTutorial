"""RRT-Connect in joint space for collision-free path finding.

Grows two trees (from start and goal). Each iteration samples a free
configuration, extends one tree toward it by a fixed step, then tries to
connect the other tree to the new node. Returns a shortcutted polyline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


IsFreeFn = Callable[[np.ndarray], bool]
SegmentFreeFn = Callable[[np.ndarray, np.ndarray], bool]
SampleFn = Callable[[], Optional[np.ndarray]]


@dataclass
class _Tree:
    q: list[np.ndarray] = field(default_factory=list)
    parent: list[int] = field(default_factory=list)

    def add(self, q: np.ndarray, parent: int) -> int:
        self.q.append(np.asarray(q, dtype=np.float64).copy())
        self.parent.append(parent)
        return len(self.q) - 1

    def nearest(self, q: np.ndarray) -> int:
        diffs = np.asarray(self.q) - q[None, :]
        d2 = np.einsum("ij,ij->i", diffs, diffs)
        return int(np.argmin(d2))

    def path_to_root(self, idx: int) -> list[np.ndarray]:
        path = []
        while idx >= 0:
            path.append(self.q[idx])
            idx = self.parent[idx]
        return path  # leaf -> root


def _steer(q_from: np.ndarray, q_to: np.ndarray, step: float) -> np.ndarray:
    d = q_to - q_from
    n = float(np.linalg.norm(d))
    if n <= step or n < 1e-12:
        return q_to.copy()
    return q_from + (step / n) * d


def _shortcut(
    path: list[np.ndarray],
    segment_free: SegmentFreeFn,
    max_iters: int,
) -> list[np.ndarray]:
    """Greedy random shortcutting to remove zig-zags while staying free."""
    if len(path) <= 2:
        return path
    pts = [p.copy() for p in path]
    for _ in range(max_iters):
        if len(pts) <= 2:
            break
        i = int(np.random.randint(0, len(pts) - 1))
        j = int(np.random.randint(i + 1, len(pts)))
        if j <= i + 1:
            continue
        if segment_free(pts[i], pts[j]):
            pts = pts[: i + 1] + pts[j:]
    return pts


def rrt_connect(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    *,
    is_free: IsFreeFn,
    segment_free: SegmentFreeFn,
    sample_free: SampleFn,
    step_size: float = 0.25,
    max_iters: int = 2000,
    goal_bias: float = 0.05,
    shortcut_iters: int = 80,
) -> Optional[np.ndarray]:
    """Plan a collision-free joint-space path with RRT-Connect.

    Returns
    -------
    path : (K, n) or None
        Waypoints from start to goal (inclusive), or None on failure.
    """
    q_start = np.asarray(q_start, dtype=np.float64).reshape(-1)
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(-1)
    if not is_free(q_start) or not is_free(q_goal):
        return None
    if segment_free(q_start, q_goal):
        return np.vstack([q_start, q_goal])

    tree_start = _Tree()
    tree_goal = _Tree()
    tree_start.add(q_start, -1)
    tree_goal.add(q_goal, -1)

    # Which tree extends first this iteration (alternates after each attempt).
    grow_start_first = True

    def extend(tree: _Tree, q_target: np.ndarray) -> Optional[int]:
        i_near = tree.nearest(q_target)
        q_new = _steer(tree.q[i_near], q_target, step_size)
        if not is_free(q_new):
            return None
        if not segment_free(tree.q[i_near], q_new):
            return None
        return tree.add(q_new, i_near)

    def connect(tree: _Tree, q_target: np.ndarray) -> tuple[Optional[int], bool]:
        """Extend repeatedly toward q_target. Returns (last_idx, reached)."""
        last = None
        for _ in range(512):
            i_near = tree.nearest(q_target)
            q_new = _steer(tree.q[i_near], q_target, step_size)
            if not is_free(q_new) or not segment_free(tree.q[i_near], q_new):
                return last, False
            last = tree.add(q_new, i_near)
            if float(np.linalg.norm(q_new - q_target)) < 1e-8:
                return last, True
        return last, False

    def join(i_a: int, tree_a: _Tree, i_b: int, tree_b: _Tree, a_is_start: bool):
        path_a = tree_a.path_to_root(i_a)  # junction -> root_a
        path_b = tree_b.path_to_root(i_b)  # junction -> root_b
        path_a.reverse()
        path = path_a + path_b[1:]
        if not a_is_start:
            path.reverse()
        return path

    for _ in range(max_iters):
        if np.random.rand() < goal_bias:
            q_rand = q_goal if grow_start_first else q_start
        else:
            q_rand = sample_free()
            if q_rand is None:
                grow_start_first = not grow_start_first
                continue

        if grow_start_first:
            tree_a, tree_b = tree_start, tree_goal
            a_is_start = True
        else:
            tree_a, tree_b = tree_goal, tree_start
            a_is_start = False

        i_new = extend(tree_a, q_rand)
        if i_new is not None:
            i_conn, reached = connect(tree_b, tree_a.q[i_new])
            if reached and i_conn is not None:
                path = join(i_new, tree_a, i_conn, tree_b, a_is_start)
                if float(np.linalg.norm(path[0] - q_start)) > 1e-3:
                    path.reverse()
                path = _shortcut(path, segment_free, shortcut_iters)
                return np.vstack(path)

        grow_start_first = not grow_start_first

    return None
