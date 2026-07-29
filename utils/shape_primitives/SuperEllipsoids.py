from utils.shape_primitives.BaseShapePrimitives import BaseShapePrimitives
import numpy as np

from scipy.optimize import root_scalar
import trimesh


class SuperEllipsoids(BaseShapePrimitives):
    def __init__(
        self,
        p: np.ndarray,
        R: np.ndarray,
        a1: float,
        a2: float,
        a3: float,
        e1: float,
        e2: float,
        name: str = "super_ellipsoids",
    ):
        """
        Initialize the SuperEllipsoids object.
        :param p: A numpy array of shape (3,) representing the position of the center.
        :param R: A numpy array of shape (3, 3) representing the orientation of the ellipsoid.
        :param a1: Semi-axis length in the x-direction.
        :param a2: Semi-axis length in the y-direction.
        :param a3: Semi-axis length in the z-direction.
        :param e1: Exponent in the x-direction.
        :param e2: Exponent in the y-direction.
        :param name: Name of the shape primitive.
        """
        super().__init__(p, R, name)
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.e1 = e1
        self.e2 = e2

        assert self.a1 > 0, "Semi-axis a1 must be positive."
        assert self.a2 > 0, "Semi-axis a2 must be positive."
        assert self.a3 > 0, "Semi-axis a3 must be positive."
        assert self.e1 > 0, "Exponent e1 must be positive."
        assert self.e2 > 0, "Exponent e2 must be positive."

    def set_params(
        self,
        a1: float = None,
        a2: float = None,
        a3: float = None,
        e1: float = None,
        e2: float = None,
    ):
        """
        Set the parameters of the super ellipsoid.
        :param a1: Semi-axis length in the x-direction.
        :param a2: Semi-axis length in the y-direction.
        :param a3: Semi-axis length in the z-direction.
        :param e1: Exponent in the x-direction.
        :param e2: Exponent in the y-direction.
        """
        if a1 is not None:
            self.a1 = a1
        if a2 is not None:
            self.a2 = a2
        if a3 is not None:
            self.a3 = a3
        if e1 is not None:
            self.e1 = e1
        if e2 is not None:
            self.e2 = e2
        assert self.a1 > 0, "Semi-axis a1 must be positive."
        assert self.a2 > 0, "Semi-axis a2 must be positive."
        assert self.a3 > 0, "Semi-axis a3 must be positive."
        assert self.e1 > 0, "Exponent e1 must be positive."
        assert self.e2 > 0, "Exponent e2 must be positive."

    def get_dict_params(self) -> dict:
        """
        Get the parameters of the super ellipsoid as a dictionary.
        :return: A dictionary containing the parameters.
        """
        return {
            "a1": self.a1,
            "a2": self.a2,
            "a3": self.a3,
            "e1": self.e1,
            "e2": self.e2,
        }

    def get_vectorized_params(self) -> np.ndarray:
        """
        Get the parameters of the super ellipsoid as a vector.
        :return: A numpy array containing the parameters.
        """
        return np.array([self.a1, self.a2, self.a3, self.e1, self.e2])

    def get_approx_sign_distance(self, x: np.ndarray, output_all=False):
        """
        Compute the approximate signed distance from a point to the super ellipsoid.
        """
        x_tf = (
            x - self.p.reshape(1, 3)
        ) @ self.R  # Rotate the point to the ellipsoid's local frame
        x_tf_1 = x_tf[:, 0]
        x_tf_2 = x_tf[:, 1]
        x_tf_3 = x_tf[:, 2]

        x_tf_1_ = np.abs(x_tf_1 / self.a1) ** (2 / self.e2)
        x_tf_2_ = np.abs(x_tf_2 / self.a2) ** (2 / self.e2)
        x_tf_3_ = np.abs(x_tf_3 / self.a3) ** (2 / self.e1)

        f = (x_tf_1_ + x_tf_2_) ** (self.e2 / self.e1) + x_tf_3_
        d = np.linalg.norm(x_tf, axis=1) * np.abs(1 - f ** (-self.e1 / 2))
        d = np.sign(f - 1) * d  # Sign distance: positive outside, negative inside
        if output_all:
            return f, d, x_tf  # (N, ), (N, ), (N, 3)
        else:
            return d  # (N, )

    def get_grad_approx_sign_distance(
        self,
        x: np.ndarray,
    ):
        """
        Compute the gradient of the approximate signed distance from a point to the super ellipsoid.
        """
        eps = 1e-5
        n = len(x)
        x_perturb_1 = x + np.array([eps, 0, 0]).reshape(1, 3)
        x_perturb_2 = x + np.array([0, eps, 0]).reshape(1, 3)
        x_perturb_3 = x + np.array([0, 0, eps]).reshape(1, 3)
        ds = self.get_approx_sign_distance(
            np.vstack([x, x_perturb_1, x_perturb_2, x_perturb_3]),
            output_all=False,
        )
        d0 = ds[:n]
        d1 = ds[n : 2 * n]
        d2 = ds[2 * n : 3 * n]
        d3 = ds[3 * n :]
        grad = np.hstack(
            [
                (d1 - d0).reshape(-1, 1) / eps,
                (d2 - d0).reshape(-1, 1) / eps,
                (d3 - d0).reshape(-1, 1) / eps,
            ]
        )  # (N, 3)
        return grad

    def get_dist_and_grad(self, x: np.ndarray):
        """
        Compute the approximate signed distance and gradient from a point to the surface of the super ellipsoid.
        :param x: A numpy array of shape (N, 3) representing the points in the global frame.
        :return: A tuple of numpy arrays (d, grad) where d is the signed distance and grad is the gradient.
        """
        eps = 1e-5
        n = len(x)
        x_perturb_1 = x + np.array([eps, 0, 0]).reshape(1, 3)
        x_perturb_2 = x + np.array([0, eps, 0]).reshape(1, 3)
        x_perturb_3 = x + np.array([0, 0, eps]).reshape(1, 3)
        ds = self.get_approx_sign_distance(
            np.vstack([x, x_perturb_1, x_perturb_2, x_perturb_3]),
            output_all=False,
        )
        d0 = ds[:n]
        d1 = ds[n : 2 * n]
        d2 = ds[2 * n : 3 * n]
        d3 = ds[3 * n :]
        grad = np.hstack(
            [
                (d1 - d0).reshape(-1, 1) / eps,
                (d2 - d0).reshape(-1, 1) / eps,
                (d3 - d0).reshape(-1, 1) / eps,
            ]
        )  # (N, 3)
        return d0, grad

    def _get_polar_coordinates(self, surf_points_tf: np.ndarray):
        """
        Compute the polar coordinates (theta, phi) of the surface points in the ellipsoid's local frame.
        :param surf_points_tf: A numpy array of shape (N, 3) representing the surface points in the ellipsoid's local frame.
        :return: A tuple of numpy arrays (theta, phi) representing the polar coordinates.
        """
        x = surf_points_tf[:, 0]
        y = surf_points_tf[:, 1]
        z = surf_points_tf[:, 2]

        sin_e1_theta = z / self.a3
        val = np.abs(sin_e1_theta) ** (1 / self.e1)
        eps = 1e-10  # Small value to avoid division by zero
        theta = np.sign(sin_e1_theta) * np.arcsin(np.clip(val, -1 + eps, 1 - eps))

        cos_theta = np.cos(theta)
        cos_e1_theta = np.sign(cos_theta) * np.abs(cos_theta) ** self.e1
        sin_e2_phi = y / self.a2 / cos_e1_theta
        cos_e2_phi = x / self.a1 / cos_e1_theta

        sin_phi_abs = np.abs(sin_e2_phi) ** 1 / self.e2
        cos_phi_abs = np.abs(cos_e2_phi) ** 1 / self.e2
        sin_phi = np.sign(sin_e2_phi) * sin_phi_abs
        cos_phi = np.sign(cos_e2_phi) * cos_phi_abs
        phi = np.arctan2(sin_phi, cos_phi)
        return theta, phi

    # def _polar_to_cartesian(self, theta: np.ndarray, phi: np.ndarray):
    #     """
    #     Convert polar coordinates (theta, phi) to Cartesian coordinates in the ellipsoid's local frame.
    #     :param theta: A numpy array of shape (N,) representing the polar angle.
    #     :param phi: A numpy array of shape (N,) representing the azimuthal angle.
    #     :return: A numpy array of shape (N, 3) representing the Cartesian coordinates.
    #     """
    #     cos_theta = np.cos(theta).reshape(-1, 1)
    #     sin_theta = np.sin(theta).reshape(-1, 1)
    #     cos_phi = np.cos(phi).reshape(-1, 1)
    #     sin_phi = np.sin(phi).reshape(-1, 1)

    #     cos_e1_theta = np.sign(cos_theta) * np.abs(cos_theta) ** self.e1
    #     cos_e2_phi = np.sign(cos_phi) * np.abs(cos_phi) ** self.e2
    #     sin_e1_theta = np.sign(sin_theta) * np.abs(sin_theta) ** self.e1
    #     sin_e2_phi = np.sign(sin_phi) * np.abs(sin_phi) ** self.e2

    #     x_tf = self.a1 * cos_e1_theta * cos_e2_phi
    #     y_tf = self.a2 * cos_e1_theta * sin_e2_phi
    #     z_tf = self.a3 * sin_e1_theta
    #     return np.hstack([x_tf, y_tf, z_tf])

    # def sample_surface_points(self, num_points: int):
    #     """
    #     Sample points uniformly on the surface of the super ellipsoid.
    #     :param num_points: The number of points to sample.
    #     :return: A numpy array of shape (num_points, 3) representing the sampled surface points.
    #     """
    #     theta = np.random.uniform(-np.pi / 2, np.pi / 2, num_points)
    #     phi = np.random.uniform(-np.pi, np.pi, num_points)
    #     surf_points_tf = self._polar_to_cartesian(theta, phi)
    #     surf_points = surf_points_tf @ self.R.T + self.p.reshape(1, 3)
    #     return surf_points

    def get_surface_normal(self, surf_points: np.ndarray):
        """
        Compute the approximate normal vectors at the surface points of the super ellipsoid.
        :param surf_points: A numpy array of shape (N, 3) representing the surface points.
        :return: A numpy array of shape (N, 3) representing the normal vectors.
        """
        surf_points_tf = (surf_points - self.p.reshape(1, 3)) @ self.R
        theta, phi = self._get_polar_coordinates(surf_points_tf)

        cos_theta = np.cos(theta).reshape(-1, 1)
        sin_theta = np.sin(theta).reshape(-1, 1)
        cos_phi = np.cos(phi).reshape(-1, 1)
        sin_phi = np.sin(phi).reshape(-1, 1)

        eps = 1e-8

        cos_2_minus_e1_theta = np.sign(cos_theta) * (np.abs(cos_theta) + eps) ** (
            2 - self.e1
        )
        sin_2_minus_e1_theta = np.sign(sin_theta) * (np.abs(sin_theta) + eps) ** (
            2 - self.e1
        )
        cos_2_minus_e2_phi = np.sign(cos_phi) * (np.abs(cos_phi) + eps) ** (2 - self.e2)
        sin_2_minus_e2_phi = np.sign(sin_phi) * (np.abs(sin_phi) + eps) ** (2 - self.e2)

        normal_tf = np.hstack(
            [
                1 / self.a1 * cos_2_minus_e1_theta * cos_2_minus_e2_phi,
                1 / self.a2 * cos_2_minus_e1_theta * sin_2_minus_e2_phi,
                1 / self.a3 * sin_2_minus_e1_theta,
            ]
        )  # (N, 3)
        normal = normal_tf @ self.R.T  # Transform back to the global frame
        normal /= np.clip(
            np.linalg.norm(normal, axis=1, keepdims=True), a_min=1e-8, a_max=None
        )  # Normalize
        return normal

    def get_boundary_points_and_normals(
        self, x: np.ndarray, x_ref: np.ndarray, bracket=[0.0, 100.0]
    ):
        """
        Project each point in x onto the surface of the super ellipsoid along the ray from x_ref to x.
        Returns the boundary point and surface normal at the projection.
        """
        x_tf = (x - self.p.reshape(1, 3)) @ self.R
        x_ref_tf = (x_ref - self.p) @ self.R
        d = x_tf - x_ref_tf.reshape(1, 3)
        x_bd_tf = np.zeros_like(x_tf)

        for i in range(x.shape[0]):

            def level_set(k):
                pt = x_ref_tf + k * d[i]
                val = (
                    (np.abs(pt[0] / self.a1)) ** (2 / self.e2)
                    + (np.abs(pt[1] / self.a2)) ** (2 / self.e2)
                ) ** (self.e2 / self.e1) + (np.abs(pt[2] / self.a3)) ** (2 / self.e1)
                return val - 1

            res = root_scalar(level_set, bracket=bracket, method="brentq")
            k = res.root
            x_bd_tf[i] = x_ref_tf + k * d[i]

        # Transform back to global frame
        x_bd = x_bd_tf @ self.R.T + self.p.reshape(1, 3)
        normals = self.get_surface_normal(x_bd)
        return x_bd, normals

    def get_mesh(
        self,
        res_theta: int = 64,  # bands from south pole (-pi/2) to north pole (+pi/2)
        res_phi: int = 128,  # slices around z ([-pi, pi))
        with_normals: bool = False,
    ) -> trimesh.Trimesh:
        """
        Create a triangular surface mesh of the super-ellipsoid.

        Parameters
        ----------
        res_theta : int
            Number of latitude bands (>= 3).
        res_phi : int
            Number of longitude slices (>= 3).
        with_normals : bool
            If True, attach vertex normals computed from the analytic formula.

        Returns
        -------
        trimesh.Trimesh
            Watertight surface mesh in the WORLD frame.
        """
        assert res_theta >= 3 and res_phi >= 3

        # --- helpers: sign-preserving power ---
        def spow(x, e):
            return np.sign(x) * (np.abs(x) ** e)

        # We build rings excluding the poles, then add two pole vertices and fan triangles.
        eps = 1e-9
        theta_mid = np.linspace(
            -np.pi / 2 + eps, np.pi / 2 - eps, res_theta - 2
        )  # (T-2,)
        phi = np.linspace(-np.pi, np.pi, res_phi, endpoint=False)  # (P,)

        # Grid for the middle rings
        TH, PH = np.meshgrid(theta_mid, phi, indexing="ij")  # shapes: (T-2, P)

        # Parametric super-ellipsoid (local frame)
        cos_th = np.cos(TH)
        sin_th = np.sin(TH)
        cos_ph = np.cos(PH)
        sin_ph = np.sin(PH)

        cos_e1_th = spow(cos_th, self.e1)
        sin_e1_th = spow(sin_th, self.e1)
        cos_e2_ph = spow(cos_ph, self.e2)
        sin_e2_ph = spow(sin_ph, self.e2)

        x_mid = self.a1 * (cos_e1_th * cos_e2_ph)
        y_mid = self.a2 * (cos_e1_th * sin_e2_ph)
        z_mid = self.a3 * (sin_e1_th)

        # Flatten rings
        V_mid = np.stack([x_mid, y_mid, z_mid], axis=-1).reshape(-1, 3)  # ((T-2)*P, 3)

        # South and North poles (local frame)
        v_south = np.array([0.0, 0.0, -self.a3])  # theta = -pi/2
        v_north = np.array([0.0, 0.0, self.a3])  # theta = +pi/2

        # Concatenate vertices: [south, rings..., north]
        V_local = np.vstack([v_south[None, :], V_mid, v_north[None, :]])  # (N, 3)

        # Indices helpers
        Tm2 = res_theta - 2
        P = res_phi

        def ring_index(i, j):
            """Index into the 'mid rings' block flattened row-major (i in [0,Tm2-1], j in [0,P-1])."""
            return 1 + i * P + j  # +1 for south pole at index 0

        south_idx = 0
        north_idx = 1 + Tm2 * P

        faces = []

        # 1) Connect south pole to first ring (i=0)
        i = 0
        for j in range(P):
            jn = (j + 1) % P
            v1 = south_idx
            v2 = ring_index(i, j)
            v3 = ring_index(i, jn)
            faces.append([v1, v2, v3])

        # 2) Connect middle bands (between rings i and i+1)
        for i in range(Tm2 - 1):
            for j in range(P):
                jn = (j + 1) % P
                v00 = ring_index(i, j)
                v01 = ring_index(i, jn)
                v10 = ring_index(i + 1, j)
                v11 = ring_index(i + 1, jn)
                # two triangles per quad
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

        # 3) Connect last ring (i=Tm2-1) to north pole
        i = Tm2 - 1
        for j in range(P):
            jn = (j + 1) % P
            v1 = ring_index(i, j)
            v2 = north_idx
            v3 = ring_index(i, jn)
            faces.append([v1, v2, v3])

        faces = np.asarray(faces, dtype=np.int64)

        # --- transform to WORLD frame: x_world = R^T * x_local + p ---
        V_world = V_local @ self.R.T + self.p.reshape(1, 3)

        # Optionally compute analytic normals at the surface points (world frame)
        vertex_normals = None
        if with_normals:
            # Reuse your analytic normal function; it expects world-space surface points
            vertex_normals = self.get_surface_normal(V_world)

        # Generate random color for the mesh
        random_color = np.random.uniform(0.0, 1.0, 4)  # RGBA with random values
        random_color[3] = 1.0  # Set alpha to 1.0 for full opacity

        # add color
        mesh = trimesh.Trimesh(
            vertices=V_world,
            faces=faces,
            vertex_normals=vertex_normals,
            face_colors=random_color,  # Apply random color to all faces
            process=True,  # repair small issues, merge duplicates, set winding
        )
        return mesh


class SuperEllipses(BaseShapePrimitives):
    def __init__(
        self,
        p: np.ndarray,
        R: np.ndarray,
        a1: float,
        a2: float,
        e1: float,
        name: str = "super_ellipses",
    ):
        """
        Initialize the SuperEllipsoids object.
        :param p: A numpy array of shape (2,) representing the position of the center.
        :param R: A numpy array of shape (2, 2) representing the orientation of the ellipsoid.
        :param a1: Semi-axis length in the x-direction.
        :param a2: Semi-axis length in the y-direction.
        :param e1: Exponent in the x-direction.
        :param name: Name of the shape primitive.
        """
        super().__init__(p, R, name)
        self.a1 = a1
        self.a2 = a2
        self.e1 = e1

        assert self.a1 > 0, "Semi-axis a1 must be positive."
        assert self.a2 > 0, "Semi-axis a2 must be positive."
        assert self.e1 > 0, "Exponent e1 must be positive."

    def set_params(
        self,
        a1: float = None,
        a2: float = None,
        e1: float = None,
    ):
        """
        Set the parameters of the super ellipse.
        :param a1: Semi-axis length in the x-direction.
        :param a2: Semi-axis length in the y-direction.
        :param e1: Exponent in the x-direction.
        :param e2: Exponent in the y-direction.
        """
        if a1 is not None:
            self.a1 = a1
        if a2 is not None:
            self.a2 = a2
        if e1 is not None:
            self.e1 = e1
        assert self.a1 > 0, "Semi-axis a1 must be positive."
        assert self.a2 > 0, "Semi-axis a2 must be positive."
        assert self.e1 > 0, "Exponent e1 must be positive."

    def get_dict_params(self) -> dict:
        """
        Get the parameters of the super ellipse as a dictionary.
        :return: A dictionary containing the parameters.
        """
        return {
            "a1": self.a1,
            "a2": self.a2,
            "e1": self.e1,
        }

    def get_vectorized_params(self) -> np.ndarray:
        """
        Get the parameters of the super ellipse as a vector.
        :return: A numpy array containing the parameters.
        """
        return np.array([self.a1, self.a2, self.e1])

    def get_approx_sign_distance(self, x: np.ndarray, output_all=False):
        """
        Compute the approximate signed distance from a point to the super ellipse.
        """
        x_tf = (x - self.p.reshape(1, 2)) @ self.R
        x_tf_1 = x_tf[:, 0]
        x_tf_2 = x_tf[:, 1]
        x_tf_1_ = np.abs(x_tf_1 / self.a1) ** (2 / self.e1)
        x_tf_2_ = np.abs(x_tf_2 / self.a2) ** (2 / self.e1)

        f = x_tf_1_ + x_tf_2_
        d = np.linalg.norm(x_tf, axis=1) * np.abs(1 - f ** (-self.e1 / 2))
        d = np.sign(f - 1) * d
        if output_all:
            return f, d, x_tf
        else:
            return d

    def get_grad_approx_sign_distance(self, x: np.ndarray):
        """
        Compute the gradient of the approximate signed distance from a point to the super ellipse.
        """
        eps = 1e-5
        n = len(x)
        x_perturb_1 = x + np.array([eps, 0]).reshape(1, 2)
        x_perturb_2 = x + np.array([0, eps]).reshape(1, 2)
        ds = self.get_approx_sign_distance(
            np.vstack([x, x_perturb_1, x_perturb_2]),
            output_all=False,
        )
        d0 = ds[:n]
        d1 = ds[n : 2 * n]
        d2 = ds[2 * n :]
        grad = np.hstack(
            [
                (d1 - d0).reshape(-1, 1) / eps,
                (d2 - d0).reshape(-1, 1) / eps,
            ]
        )
        return grad

    def _get_polar_coordinates(self, surf_points_tf: np.ndarray):
        """
        Compute the polar coordinates (theta, phi) of the surface points in the ellipse's local frame.
        :param surf_points_tf: A numpy array of shape (N, 2) representing the surface points in the ellipse's local frame.
        :return: A tuple of numpy arrays (theta, phi) representing the polar coordinates.
        """
        x = surf_points_tf[:, 0]
        y = surf_points_tf[:, 1]

        # Inverse superellipse transformation
        x_norm = np.sign(x) * (np.abs(x / self.a1)) ** (1 / self.e1)
        y_norm = np.sign(y) * (np.abs(y / self.a2)) ** (1 / self.e1)

        theta = np.arctan2(y_norm, x_norm)  # true angular parameter
        return theta

    def _get_surface_normal(self, surf_points: np.ndarray):
        """
        Compute the approximate normal vectors at the surface points of the super ellipse.
        :param surf_points_tf: A numpy array of shape (N, 2) representing the surface points in the ellipse's local frame.
        :return: A numpy array of shape (N, 2) representing the normal vectors.
        """
        surf_points_tf = (surf_points - self.p.reshape(1, 2)) @ self.R
        x = surf_points_tf[:, 0]
        y = surf_points_tf[:, 1]

        n_x = (np.abs(x / self.a1) ** ((2 / self.e1) - 1)) * np.sign(x) / self.a1
        n_y = (np.abs(y / self.a2) ** ((2 / self.e1) - 1)) * np.sign(y) / self.a2

        normals = np.stack([n_x, n_y], axis=1)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)

        normals = normals @ self.R.T  # Transform back to the global frame
        return normals

    def get_boundary_points_and_normals(
        self, x: np.ndarray, x_ref: np.ndarray, bracket=[0.0, 10.0]
    ):
        x_tf = (x - self.p.reshape(1, 2)) @ self.R
        x_ref_tf = (x_ref - self.p) @ self.R
        d = x_tf - x_ref_tf.reshape(1, 2)
        x_bd_tf = np.zeros_like(x_tf)

        for i in range(x.shape[0]):

            def level_set(k):
                pt = x_ref_tf + k * d[i]
                val = (np.abs(pt[0] / self.a1)) ** (2 / self.e1) + (
                    np.abs(pt[1] / self.a2)
                ) ** (2 / self.e1)
                return val - 1

            res = root_scalar(level_set, bracket=bracket, method="brentq")
            k = res.root
            x_bd_tf[i] = x_ref_tf + k * d[i]

        # Back to global frame
        x_bd = x_bd_tf @ self.R.T + self.p.reshape(1, 2)

        # Compute normals
        normals = self._get_surface_normal(x_bd)
        return x_bd, normals
