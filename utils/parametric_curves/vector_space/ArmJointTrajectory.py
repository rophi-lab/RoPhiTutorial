import numpy as np

from utils.parametric_curves.vector_space.MinimumJerkLinearTrajModels import (
    MinimumJerkLinearTrajModels,
)


class ArmJointTrajectory:
    """Multi-waypoint minimum-jerk trajectory through joint-space waypoints.

    Chains one :class:`MinimumJerkLinearTrajModels` segment per waypoint pair.
    Each segment's duration is chosen from a per-segment joint VELOCITY LIMIT
    (min-jerk then sets the acceleration profile), so making a segment faster is
    just a matter of raising its velocity limit. This is the knob used to stress
    the realtime grasp: slow -> fast sweeps of the same waypoint path.

    Parameters
    ----------
    waypoints : (K, dim) array-like
        K >= 2 joint configurations. waypoints[0] is the start (typically the
        arm config captured when stable_grasp begins).
    velocity_limits : (K-1, dim) or (K-1,) or (dim,) or scalar
        Per-segment, per-joint velocity limit [rad/s], > 0. A (dim,) or scalar
        value is broadcast to every segment; a (K-1,) value gives one scalar
        limit per segment (broadcast across joints).
    hold_at_end : bool
        If True (default), ``eval(t)`` past the total duration returns the last
        waypoint with zero velocity (hold). If False, it clamps identically
        (there is no other terminal behavior here -- looping/return is handled by
        the caller re-seeding the trajectory).

    Notes
    -----
    Velocity is continuous only in the min-jerk sense WITHIN a segment; at
    waypoints the segments are stitched at zero velocity (each min-jerk segment
    starts and ends at rest). That rest-to-rest stitching is intentional: it
    keeps every waypoint a well-defined, repeatable pose and avoids overshoot
    through the palm-flip.
    """

    def __init__(self, waypoints, velocity_limits, hold_at_end=True):
        wp = np.asarray(waypoints, dtype=np.float64)
        if wp.ndim != 2 or wp.shape[0] < 2:
            raise ValueError(
                f"waypoints must be (K>=2, dim); got {wp.shape}"
            )
        self.waypoints = wp
        self.K, self.dim = wp.shape
        self.n_segments = self.K - 1
        self.hold_at_end = bool(hold_at_end)

        vlims = self._normalize_velocity_limits(velocity_limits)

        self._segments = []
        self._seg_start_times = []  # cumulative start time of each segment
        t_acc = 0.0
        for i in range(self.n_segments):
            seg = MinimumJerkLinearTrajModels(
                wp[i], wp[i + 1], velocity_limit=vlims[i], clamp_time=True
            )
            self._segments.append(seg)
            self._seg_start_times.append(t_acc)
            t_acc += seg.get_T()
        self._seg_start_times = np.asarray(self._seg_start_times)
        self.total_time = float(t_acc)

    def _normalize_velocity_limits(self, velocity_limits):
        vl = np.asarray(velocity_limits, dtype=np.float64)
        if vl.ndim == 0:  # scalar -> all segments, all joints
            return np.full((self.n_segments, self.dim), float(vl))
        if vl.ndim == 1:
            if vl.shape[0] == self.dim:  # per-joint -> broadcast to segments
                return np.tile(vl, (self.n_segments, 1))
            if vl.shape[0] == self.n_segments:  # per-segment scalar
                return np.repeat(vl[:, None], self.dim, axis=1)
            raise ValueError(
                f"1D velocity_limits must be (dim={self.dim},) or "
                f"(n_segments={self.n_segments},); got {vl.shape}"
            )
        if vl.shape == (self.n_segments, self.dim):
            return vl
        raise ValueError(
            f"velocity_limits must be scalar, (dim,), (n_segments,), or "
            f"(n_segments, dim); got {vl.shape}"
        )

    def eval(self, t):
        """Return ``(q, qd)`` at time ``t`` (seconds since trajectory start).

        ``q`` and ``qd`` are ``(dim,)``. Before ``t=0`` returns the first
        waypoint at rest; after ``total_time`` holds the last waypoint at rest
        (when ``hold_at_end``).
        """
        t = float(t)
        if t <= 0.0:
            return self.waypoints[0].copy(), np.zeros(self.dim)
        if t >= self.total_time:
            return self.waypoints[-1].copy(), np.zeros(self.dim)

        # Locate the active segment (last seg whose start <= t).
        seg_idx = int(np.searchsorted(self._seg_start_times, t, side="right") - 1)
        seg_idx = max(0, min(seg_idx, self.n_segments - 1))
        local_t = t - self._seg_start_times[seg_idx]

        q, qd, _, _ = self._segments[seg_idx].forward(local_t)
        return q[0], qd[0]

    def is_finished(self, t):
        return float(t) >= self.total_time
