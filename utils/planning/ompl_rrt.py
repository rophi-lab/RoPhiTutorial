"""RRT helpers.

Prefer :func:`utils.planning.rrt_connect.rrt_connect` for joint-space planning
used by ``MinimumJerkP2PControl``. This module kept as a placeholder for an
OMPL-backed variant.
"""

from utils.planning.rrt_connect import rrt_connect

__all__ = ["rrt_connect"]
