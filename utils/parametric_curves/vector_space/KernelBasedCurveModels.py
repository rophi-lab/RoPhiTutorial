import numpy as np


class KernelBasedCurveModels:
    def __init__(
        self,
        time_traj: np.ndarray,
        q_traj: np.ndarray,
        terminal_time: float,
        temparature=None,
    ):
        self._time_traj = time_traj  # (T, 1)
        self._q_traj = q_traj  # (T, dim)
        self._terminal_time = terminal_time  # float
        self._len_traj = time_traj.shape[0]

        if temparature is None:
            self._temparature = terminal_time / self._len_traj
        else:
            self._temparature = temparature

    def forward(self, t: np.ndarray):
        """
        t: (N, 1)
        returns:
          q  : (N, dim)
          qd : (N, dim)
        """
        sigma2 = self._temparature**2

        # (T, N)
        diff = self._time_traj - t.T
        k = np.exp(-(diff**2) / (2 * sigma2))

        Z = np.sum(k, axis=0, keepdims=True)  # (1, N)
        w = k / Z  # (T, N)

        # q(t)
        q = np.sum(w[:, :, None] * self._q_traj[:, None, :], axis=0)

        # dk/dt
        dk = k * (diff / sigma2)  # (T, N)

        # dw/dt
        dZ = np.sum(dk, axis=0, keepdims=True)  # (1, N)
        dw = dk / Z - w * (dZ / Z)  # (T, N)

        # qdot(t)
        qd = np.sum(dw[:, :, None] * self._q_traj[:, None, :], axis=0)

        # ddk/dt2
        ddk = k * (diff**2 / sigma2**2 - 1.0 / sigma2)  # (T, N)

        # d2w/dt2
        ddZ = np.sum(ddk, axis=0, keepdims=True)  # (1, N)
        ddw = (
            ddk / Z - 2.0 * dk * dZ / (Z**2) - w * (ddZ / Z) + 2.0 * w * (dZ**2 / Z**2)
        )  # (T, N)

        # qddot(t)
        qdd = np.sum(ddw[:, :, None] * self._q_traj[:, None, :], axis=0)

        return q, qd, qdd, None
