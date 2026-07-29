import numpy as np


class MinimumJerkLinearTrajModels:
    def __init__(self, q0, q1, T=None, velocity_limit=None, clamp_time=None):
        self.q0 = np.asarray(q0, dtype=np.float64).reshape(-1)  # (dim,)
        self.q1 = np.asarray(q1, dtype=np.float64).reshape(-1)  # (dim,)
        assert self.q0.shape == self.q1.shape, "q0 and q1 must have same shape"
        self.dim = self.q0.shape[0]
        self.clamp_time = clamp_time

        if velocity_limit is not None:
            vlim = np.asarray(velocity_limit, dtype=np.float64).reshape(-1)
            assert vlim.shape == (self.dim,), "velocity_limit must be (dim,)"
            if np.any(vlim <= 0):
                bad = np.where(vlim <= 0)[0]
                raise ValueError(
                    f"velocity_limit must be > 0 for all joints. Bad idx: {bad}"
                )
            self.T = float(self.choose_T_to_respect_velocity_limit(vlim))
        else:
            if T is None:
                raise ValueError("T must be provided if velocity_limit is not provided")
            self.T = float(T)
            if self.T <= 0:
                raise ValueError("T must be > 0")

        self.dq = self.q1 - self.q0  # (dim,)

    def get_T(self) -> float:
        return self.T

    @staticmethod
    def _poly(tau):
        """Return s, ds/dtau, d2s/dtau2, d3s/dtau3 for minimum-jerk polynomial."""
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        sd_tau = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
        sdd_tau = 60 * tau - 180 * tau**2 + 120 * tau**3
        sddd_tau = 60 - 360 * tau + 360 * tau**2
        return s, sd_tau, sdd_tau, sddd_tau

    def forward(self, t):
        """
        t : scalar, (N,), or (N,1)
        returns q, qd, qdd, qddd each (N, dim)
        """
        t = np.asarray(t, dtype=np.float64)

        # Make (N,1)
        if t.ndim == 0:
            t = t.reshape(1, 1)
        elif t.ndim == 1:
            t = t.reshape(-1, 1)
        elif t.ndim == 2 and t.shape[1] == 1:
            pass
        else:
            raise ValueError(f"t must be scalar, (N,), or (N,1). Got {t.shape}")

        if self.clamp_time:
            t = np.clip(t, 0.0, self.T)

        tau = t / self.T  # (N,1) in [0,1] if clamped

        s, sd_tau, sdd_tau, sddd_tau = self._poly(tau)

        # Convert tau-derivatives -> time derivatives
        invT = 1.0 / self.T
        sd = sd_tau * invT
        sdd = sdd_tau * invT**2
        sddd = sddd_tau * invT**3

        dq = self.dq[None, :]  # (1,dim)

        q = self.q0[None, :] + s * dq
        qd = sd * dq
        qdd = sdd * dq
        qddd = sddd * dq
        return q, qd, qdd, qddd

    def choose_T_to_respect_velocity_limit(self, velocity_limit):
        """
        velocity_limit : (dim,) > 0
        returns T : float
        """
        # Avoid division blow-ups already validated in __init__
        T_per_joint = (15.0 / 8.0) * np.abs(self.q1 - self.q0) / velocity_limit
        return np.max(T_per_joint)
