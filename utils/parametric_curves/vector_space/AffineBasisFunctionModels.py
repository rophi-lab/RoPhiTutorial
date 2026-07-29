import numpy as np

import sys

# THIS IS TO MAKE SURE THAT THE PYTHONPATH IS SET CORRECTLY FOR CASADI + PINOCCHIO
# sys.path.insert(0, "/opt/openrobots/lib/python3.10/site-packages")
# from pinocchio import casadi as cpin
import casadi as ca
import pinocchio as pin

from utils.parametric_curves.BaseCurves import BaseCurves
from utils.parametric_curves.vector_space.utils import get_basis_parameters


class AffineBasisFunctionModels(BaseCurves):
    def __init__(
        self,
        qi: np.ndarray = None,
        qf: np.ndarray = None,
        w: np.ndarray = None,
        basis: str = "Gaussian",
        num_basis: int = 30,
        curve_type: str = "affine_basis_function",
        terminal_time: float = 5,
        space_type: str = "vector_space",
        dim: int = 6,
        *args,
        **kwargs,
    ):
        """
        Base class for parametric curves.
        :param curve_type: Type of the curve (e.g., "vmp", "cubic", etc.)
        :param terminal_time: The time at which the curve ends.
        :param space_type: Type of space (e.g., "vector_space", "matrix_space", etc.)
        :param dim: Dimension of the space.
        """
        super().__init__(curve_type, terminal_time, space_type, dim, *args, **kwargs)

        self._basis = basis
        self._num_basis = num_basis

        if qi is not None:
            self._qi = qi
        else:
            self._qi = np.zeros((self._dim,))
        if qf is not None:
            self._qf = qf
        else:
            self._qf = np.zeros((self._dim,))
        if w is not None:
            self._w = w
        else:
            self._w = np.zeros((self._num_basis, self._dim))

        self._parameters = {"qi": self._qi, "qf": self._qf, "w": self._w}

        self._basis_params = get_basis_parameters(basis, num_basis)

        self.tf = self._terminal_time

    def _nominal(self, t: np.ndarray):
        """
        Returns the nominal path at time t.
        @param t: (n, 1) array of time values
        @return: (n, *dim) array of position values
        """
        num_times = len(t)
        if type(self._qf) == np.ndarray:
            q = self._qi.reshape(1, -1) * (
                1 - t / self._terminal_time
            ) + self._qf.reshape(1, -1) * (t / self._terminal_time)
        elif (type(self._qf) == ca.casadi.MX) or (type(self._qf) == ca.casadi.SX):
            term_ca = ca.repmat(ca.reshape(self._qf, 1, self._dim), num_times, 1) * (
                t / self._terminal_time
            )
            q = self._qi.reshape(1, -1) * (1 - t / self._terminal_time) + term_ca
        return q

    def _perturbation(self, t: np.ndarray):
        """
        Returns the perturbation at time t.
        @param t: (n, 1) array of time values
        @return: (n, *dim) array of position values
        """
        if self._basis == "Gaussian":
            t_mu = self._basis_params["mean"].reshape(1, -1)
            t_sigma = self._basis_params["std"]

            s = t / self._terminal_time
            phi = np.exp(-((s - t_mu) ** 2) / t_sigma**2)  # (n, num_basis)
            return phi @ self._w
        else:
            raise NotImplementedError(
                f"Basis function '{self._basis}' is not implemented."
            )

    def _nominal_velocity(self, t: np.ndarray):
        """
        Returns the nominal velocity at time t.
        @param t: (n, 1) array of time values
        @return: (n, *dim) array of velocity values
        """
        num_times = len(t)
        if type(self._qf) == np.ndarray:
            q_dot = (
                self._qf.reshape(1, -1) - self._qi.reshape(1, -1)
            ) / self._terminal_time
        elif (type(self._qf) == ca.casadi.MX) or (type(self._qf) == ca.casadi.SX):
            term_ca = ca.reshape(self._qf, 1, self._dim) / self._terminal_time
            q_dot = term_ca - self._qi.reshape(1, -1) / self._terminal_time

        return q_dot  # (1, *dim)

    def _perturbation_velocity(self, t: np.ndarray):
        """
        Returns the perturbation velocity at time t.
        @param t: (n, 1) array of time values
        @return: (n, *dim) array of velocity values
        """
        if self._basis == "Gaussian":
            t_mu = self._basis_params["mean"].reshape(1, -1)
            t_sigma = self._basis_params["std"]

            s = t / self._terminal_time
            phi = np.exp(-((s - t_mu) ** 2) / t_sigma**2)
            dphi_ds = -2 * (s - t_mu) / t_sigma**2 * phi
            dphi_dt = dphi_ds / self._terminal_time
            return dphi_dt @ self._w
        else:
            raise NotImplementedError(
                f"Basis function '{self._basis}' is not implemented."
            )

    def _perturbation_acceleration(self, t: np.ndarray):
        if self._basis == "Gaussian":
            t_mu = self._basis_params["mean"].reshape(1, -1)
            t_sigma = self._basis_params["std"]

            s = t / self._terminal_time
            phi = np.exp(-((s - t_mu) ** 2) / t_sigma**2)
            dphi_ds = -2 * (s - t_mu) / t_sigma**2 * phi
            d2phi_ds2 = -2 / t_sigma**2 * phi + (-2) * (s - t_mu) / t_sigma**2 * dphi_ds
            d2phi_dt2 = d2phi_ds2 / (self._terminal_time**2)
            return d2phi_dt2 @ self._w

    def _perturbation_jerk(self, t: np.ndarray):
        if self._basis == "Gaussian":
            t_mu = self._basis_params["mean"].reshape(1, -1)
            t_sigma = self._basis_params["std"]

            s = t / self._terminal_time
            phi = np.exp(-((s - t_mu) ** 2) / t_sigma**2)
            dphi_ds = -2 * (s - t_mu) / t_sigma**2 * phi
            d2phi_ds2 = -2 / t_sigma**2 * phi + (-2) * (s - t_mu) / t_sigma**2 * dphi_ds
            d3phi_ds3 = (
                -4 / t_sigma**2 * dphi_ds + (-2) * (s - t_mu) / t_sigma**2 * d2phi_ds2
            )
            d3phi_dt3 = d3phi_ds3 / (self._terminal_time**3)
            return d3phi_dt3 @ self._w

    def _position(self, t: np.ndarray):
        """
        Returns the position of the curve at time t.
        @param t: (n, 1) array of time values
        @return: (n, *dim) array of position values
        """
        q = self._nominal(t) + self._perturbation(t) * (t / self._terminal_time) * (
            1 - t / self._terminal_time
        )
        return q

    def get_dposition_dw(self, t: np.ndarray):
        """
        Returns the derivative of the position with respect to the weights at time t.
        @param t: (n, 1) array of time values
        @return: (n, dim, num_basis, dim) array of derivative of position with respect to weights values
        """
        t_mu = self._basis_params["mean"].reshape(1, -1)
        t_sigma = self._basis_params["std"]

        s = t / self._terminal_time
        phi = np.exp(-((s - t_mu) ** 2) / t_sigma**2)  # (n, num_basis)

        dq_dw = np.zeros((len(t), self._dim, self._num_basis, self._dim))
        temp = (
            t / self._terminal_time * (1 - t / self._terminal_time) * phi
        )  # (n, num_basis)

        # Kronecker delta over (dim, dim)
        delta = np.eye(self._dim)  # (dim, dim)

        # Broadcasting:
        # temp[:, None, :, None] -> (n, 1, num_basis, 1)
        # delta[None, :, None, :] -> (1, dim, 1, dim)
        # result -> (n, dim, num_basis, dim)
        dq_dw = temp[:, None, :, None] * delta[None, :, None, :]
        return dq_dw

    def velocity(self, t: np.ndarray):
        """
        Returns the velocity of the curve at time t.
        @param t: (n, 1) array of time values
        @return: (n, *dim) array of velocity values
        """
        q_dot = (
            self._nominal_velocity(t)
            + self._perturbation_velocity(t)
            * (t / self._terminal_time)
            * (1 - t / self._terminal_time)
            + self._perturbation(t)
            * (1 / self._terminal_time - 2 * t / (self._terminal_time**2))
        )
        return q_dot

    def acceleration(self, t: np.ndarray):
        """
        Returns the acceleration of the curve at time t.
        @param t: (n, 1) array of time values
        @return: (n, *dim) array of acceleration values
        """
        q_ddot = (
            self._perturbation_acceleration(t)
            * (t / self._terminal_time)
            * (1 - t / self._terminal_time)
            + 2
            * self._perturbation_velocity(t)
            * (1 / self._terminal_time - 2 * t / (self._terminal_time**2))
            + self._perturbation(t) * (-2 / self._terminal_time**2)
        )
        return q_ddot

    def jerk(self, t: np.ndarray):
        """
        Returns the jerk of the curve at time t.
        @param t: (n, 1) array of time values
        @return: (n, *dim) array of jerk values
        """
        q_dddot = (
            self._perturbation_jerk(t)
            * (t / self._terminal_time)
            * (1 - t / self._terminal_time)
            + 3
            * self._perturbation_acceleration(t)
            * (1 / self._terminal_time - 2 * t / (self._terminal_time**2))
            + 3 * self._perturbation_velocity(t) * (-2 / self._terminal_time**2)
        )
        return q_dddot

    def forward(self, t: np.ndarray):
        """
        Returns the forward kinematics of the curve at time t.
        @param t: (n, 1) array of time values
        @return: (n, *dim) array of position, velocity, acceleration, and jerk values
        """
        if self._basis == "Gaussian":
            t_mu = self._basis_params["mean"].reshape(1, -1)
            t_sigma = self._basis_params["std"]

            s = t / self._terminal_time
            phi = np.exp(-((s - t_mu) ** 2) / t_sigma**2)
            dphi_ds = -2 * (s - t_mu) / t_sigma**2 * phi
            d2phi_ds2 = -2 / t_sigma**2 * phi + (-2) * (s - t_mu) / t_sigma**2 * dphi_ds
            d3phi_ds3 = (
                -4 / t_sigma**2 * dphi_ds + (-2) * (s - t_mu) / t_sigma**2 * d2phi_ds2
            )
            dphi_dt = dphi_ds / self._terminal_time
            d2phi_dt2 = d2phi_ds2 / (self._terminal_time**2)
            d3phi_dt3 = d3phi_ds3 / (self._terminal_time**3)
            _perturbation = phi @ self._w
            _perturbation_velo = dphi_dt @ self._w
            _perturbation_accel = d2phi_dt2 @ self._w
            _perturbation_jerk = d3phi_dt3 @ self._w
        else:
            raise NotImplementedError(
                f"Basis function '{self._basis}' is not implemented."
            )

        q = self._nominal(t) + _perturbation * (t / self._terminal_time) * (
            1 - t / self._terminal_time
        )
        nominal_velocity = self._nominal_velocity(t)

        num_times = len(t)
        if type(nominal_velocity) == np.ndarray:
            nominal_velocity = np.tile(nominal_velocity, (num_times, 1))
        elif (type(nominal_velocity) == ca.casadi.MX) or (
            type(nominal_velocity) == ca.casadi.SX
        ):
            nominal_velocity = ca.repmat(
                ca.reshape(nominal_velocity, 1, self._dim), num_times, 1
            )
        q_dot = (
            nominal_velocity
            + _perturbation_velo
            * (t / self._terminal_time)
            * (1 - t / self._terminal_time)
            + _perturbation
            * (1 / self._terminal_time - 2 * t / (self._terminal_time**2))
        )
        q_ddot = (
            _perturbation_accel
            * (t / self._terminal_time)
            * (1 - t / self._terminal_time)
            + 2
            * _perturbation_velo
            * (1 / self._terminal_time - 2 * t / (self._terminal_time**2))
            + _perturbation * (-2 / self._terminal_time**2)
        )
        q_dddot = (
            _perturbation_jerk
            * (t / self._terminal_time)
            * (1 - t / self._terminal_time)
            + 3
            * _perturbation_accel
            * (1 / self._terminal_time - 2 * t / (self._terminal_time**2))
            + 3 * _perturbation_velo * (-2 / self._terminal_time**2)
        )
        return q, q_dot, q_ddot, q_dddot

    def fit(
        self,
        t: np.ndarray,
        q_traj: np.ndarray,
        lam: float = 1e-6,
        enforce_endpoint_zero_velocity: bool = False,
        beta: float = 1e3,
        *,
        alpha_acc: float = 0.0,
        t_acc: np.ndarray | None = None,
        a_max: float | None = None,
    ):
        """
        Least-squares fit for w with:
        - position matching at times t
        - optional soft endpoint zero-velocity constraints
        - optional soft acceleration regularization (component-wise)

        Parameters
        ----------
        t : (N,1) array
            Sample times in [0, T] for the position constraints.
        q_traj : (N,dim) array
            Target positions at those times.
        lam : float
            Ridge regularization on weights.
        enforce_endpoint_zero_velocity : bool
            If True, softly push q_dot(0)=q_dot(T)=0 with weight beta.
        beta : float
            Weight for endpoint velocity constraints.
        alpha_acc : float
            Weight for acceleration regularization (0 disables).
        t_acc : (M,1) array or None
            Times at which to penalize acceleration. If None, a dense grid is used.
        a_max : float or None
            Optional per-component soft bound scale (penalize accel/a_max).

        Returns
        -------
        w : (num_basis, dim) array
            Learned weight matrix.
        """
        T = self._terminal_time
        dim = self._dim
        s = t / T  # (N,1)

        # --- nominal position & velocity
        q_nom = self._nominal(t)  # (N,dim)
        qdot_nom_1x = self._nominal_velocity(np.array([[0.0]]))  # (1,dim)
        if isinstance(qdot_nom_1x, np.ndarray):
            qdot_nom_1x = qdot_nom_1x.reshape(1, dim)

        # --- Gaussian features at t
        mu = self._basis_params["mean"].reshape(1, -1)  # (1,B)
        sigma = self._basis_params["std"]  # scalar or (B,)
        B = self._num_basis

        # features for position constraints
        phi = np.exp(-((s - mu) ** 2) / (sigma**2))  # (N,B)

        # envelope e(s) = s(1-s)
        e = s * (1 - s)  # (N,1)

        # Avoid division by zero at endpoints in position constraints
        mask = (s.ravel() > 0.0) & (s.ravel() < 1.0)
        A = phi[mask]  # (M,B)
        R = (q_traj[mask] - q_nom[mask]) / e[mask]  # (M,dim)

        # --- optional: endpoint zero-velocity constraints (soft)
        if enforce_endpoint_zero_velocity:
            # q_dot(0) = qdot_nom + (phi0 @ w)/T = 0  -> (1/T)*phi0 @ w = -qdot_nom
            # q_dot(T) = qdot_nom - (phiT @ w)/T = 0  -> (-1/T)*phiT @ w = -qdot_nom
            phi0 = np.exp(-((0.0 - mu) ** 2) / (sigma**2)).reshape(-1)  # (B,)
            phiT = np.exp(-((1.0 - mu) ** 2) / (sigma**2)).reshape(-1)  # (B,)

            A_vel = np.vstack([(1.0 / T) * phi0, (-1.0 / T) * phiT])  # (2,B)
            b_vel = np.vstack([-qdot_nom_1x, -qdot_nom_1x])  # (2,dim)

            A = np.vstack([A, np.sqrt(beta) * A_vel])
            R = np.vstack([R, np.sqrt(beta) * b_vel])

        # --- optional: acceleration regularization (soft)
        # Build a linear map for q_ddot(t) = A_acc(t) @ w (component-wise)
        if alpha_acc > 0.0:
            if t_acc is None:
                # dense grid (covers endpoints too)
                t_acc = np.linspace(0.0, T, max(3 * len(t), 100)).reshape(-1, 1)
            s_acc = t_acc / T

            # features & time-derivatives at t_acc
            phi_acc = np.exp(-((s_acc - mu) ** 2) / (sigma**2))  # (M,B)
            dphi_ds = -2.0 * (s_acc - mu) / (sigma**2) * phi_acc  # (M,B)
            dphi_dt = dphi_ds / T  # (M,B)
            d2phi_ds2 = (-2.0 / (sigma**2)) * phi_acc + (
                -2.0 * (s_acc - mu) / (sigma**2)
            ) * dphi_ds
            d2phi_dt2 = d2phi_ds2 / (T**2)  # (M,B)

            # q_ddot = d2phi_dt2 * e + 2*dphi_dt*(1/T - 2t/T^2) + phi * (-2/T^2)
            e_acc = s_acc * (1.0 - s_acc)  # (M,1)
            coef_mid = (1.0 / T) - (2.0 * t_acc / (T**2))  # (M,1)

            A_acc = (
                d2phi_dt2 * e_acc + 2.0 * dphi_dt * coef_mid + phi_acc * (-2.0 / (T**2))
            )  # (M,B)

            # scale relative to a_max if provided (soft "bound")
            if a_max is not None and a_max > 0:
                A_acc = A_acc / float(a_max)

            # target accel ~ 0 (component-wise)
            zeros_acc = np.zeros((A_acc.shape[0], dim))

            A = np.vstack([A, np.sqrt(alpha_acc) * A_acc])
            R = np.vstack([R, zeros_acc])

        # --- ridge-regularized least squares solve
        ATA = A.T @ A + lam * np.eye(B)
        ATR = A.T @ R
        w = np.linalg.solve(ATA, ATR)  # (B,dim)

        self.set_w(w)
        return w

    def get_kinematic_trajectories(self, num_points: int = 100):
        """
        Returns the kinematic trajectories of the curve.
        :param num_points: Number of points to sample
        :return: Tuple of position, velocity, acceleration, and jerk trajectories
        """
        t = np.linspace(0, self._terminal_time, num_points)
        q, q_dot, q_ddot, q_dddot = self.forward(t.reshape(-1, 1))
        return q, q_dot, q_ddot, q_dddot

    def set_parameters(self, qi=None, qf=None, w=None):
        """
        Set the parameters of the curve.
        :param qi: Initial position
        :param qf: Final position
        :param w: Weights for the basis functions
        """
        if qi is not None:
            self._qi = qi
        if qf is not None:
            self._qf = qf
        if w is not None:
            self._w = w

        self._parameters = {"qi": self._qi, "qf": self._qf, "w": self._w}

    def set_qi(self, qi):
        """
        Set the initial position of the curve.
        :param qi: Initial position
        """
        self._qi = qi
        self._parameters["qi"] = self._qi

    def set_qf(self, qf):
        """
        Set the final position of the curve.
        :param qf: Final position
        """
        self._qf = qf
        self._parameters["qf"] = self._qf

    def set_w(self, w):
        """
        Set the weights for the basis functions.
        :param w: Weights for the basis functions
        """
        self._w = w
        self._parameters["w"] = self._w

    def get_qi(self):
        """
        Get the initial position of the curve.
        :return: Initial position
        """
        return self._qi

    def get_qf(self):
        """
        Get the final position of the curve.
        :return: Final position
        """
        return self._qf

    def get_w(self):
        """
        Get the weights for the basis functions.
        :return: Weights for the basis functions
        """
        return self._w
