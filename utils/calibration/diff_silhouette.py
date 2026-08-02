"""Differentiable gray / silhouette rendering with nvdiffrast."""

from __future__ import annotations

import numpy as np
import torch
import nvdiffrast.torch as dr

# OpenCV camera (z forward, y down) → OpenGL camera (z back, y up).
_GL_IN_CV = torch.tensor(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=torch.float32,
)


def projection_matrix_from_K(
    K: np.ndarray,
    height: int,
    width: int,
    znear: float = 0.01,
    zfar: float = 10.0,
) -> torch.Tensor:
    """Hartley–Zisserman K → OpenGL clip projection (FoundationPose y_down).

    Pair with a vertical flip of the raster (see DiffSilhouetteRenderer.render_world2cam);
    nvdiffrast's CUDA rasterizer matches OpenCV image coords only after that flip —
    same as FoundationPose.Utils.nvdiffrast_render.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    w, h = float(width), float(height)
    depth = float(zfar - znear)
    q = -(zfar + znear) / depth
    qn = -2.0 * (zfar * znear) / depth
    proj = np.array(
        [
            [2 * fx / w, 0.0, (-2 * cx + w) / w, 0.0],
            [0.0, 2 * fy / h, (2 * cy - h) / h, 0.0],
            [0.0, 0.0, q, qn],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return torch.as_tensor(proj, dtype=torch.float32)


def se3_from_wt(omega: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Axis-angle omega (3,) + translation t (3,) → 4x4 SE(3) on the same device."""
    from pytorch3d.transforms import so3_exp_map

    R = so3_exp_map(omega.reshape(1, 3))[0]
    T = torch.eye(4, device=omega.device, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def wt_from_se3(T: np.ndarray):
    """4x4 SE(3) → (omega, t) float32 torch CPU tensors."""
    from pytorch3d.transforms import so3_log_map

    T = np.asarray(T, dtype=np.float32)
    R = torch.as_tensor(T[:3, :3], dtype=torch.float32).unsqueeze(0)
    omega = so3_log_map(R)[0]
    t = torch.as_tensor(T[:3, 3], dtype=torch.float32)
    return omega, t


class DiffSilhouetteRenderer:
    """Render a constant-gray soft silhouette under OpenCV cam-in-base / world2cam."""

    def __init__(
        self,
        K: np.ndarray,
        height: int,
        width: int,
        gray: float = 0.7,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.H = int(height)
        self.W = int(width)
        self.gray = float(gray)
        self.K = np.asarray(K, dtype=np.float32)
        self.glctx = dr.RasterizeCudaContext()
        self.P = projection_matrix_from_K(self.K, self.H, self.W).to(self.device)
        self._gl_in_cv = _GL_IN_CV.to(self.device)

    def set_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        """Cache mesh in world / base frame (N,3) verts, (F,3) int faces."""
        self.pos = torch.as_tensor(vertices, dtype=torch.float32, device=self.device)
        self.faces = torch.as_tensor(faces, dtype=torch.int32, device=self.device)

    def render_world2cam(self, T_world2cam: torch.Tensor) -> torch.Tensor:
        """
        @param T_world2cam: (4,4) OpenCV world → camera, requires grad.
        @return: (H, W) gray image in [0, 1] with soft silhouette edges.
        """
        if T_world2cam.dim() == 2:
            T = T_world2cam.unsqueeze(0)
        else:
            T = T_world2cam
        T = T.to(dtype=torch.float32, device=self.device)
        # OpenCV → OpenGL cam, then project to clip.
        T_gl = self._gl_in_cv.unsqueeze(0) @ T
        mtx = self.P.unsqueeze(0) @ T_gl  # (1,4,4)

        ones = torch.ones((self.pos.shape[0], 1), device=self.device, dtype=torch.float32)
        pos_h = torch.cat([self.pos, ones], dim=-1)  # (V,4)
        pos_clip = torch.einsum("bij,vj->bvi", mtx, pos_h)

        rast, _ = dr.rasterize(self.glctx, pos_clip, self.faces, (self.H, self.W))
        feat = torch.full(
            (self.pos.shape[0], 1),
            self.gray,
            device=self.device,
            dtype=torch.float32,
        )
        color, _ = dr.interpolate(feat, rast, self.faces)
        color = dr.antialias(color, rast, pos_clip, self.faces)
        mask = (rast[..., -1:] > 0).float()
        img = (color * mask)[..., 0]  # (1,H,W)
        # Match OpenCV / MuJoCo RGB row order (FoundationPose flips Y after rast).
        img = torch.flip(img, dims=[1])
        return img[0]

    def render(self, T_cam2base: torch.Tensor) -> torch.Tensor:
        """
        Render with OpenCV cam-in-base (same convention as EasyHeC / eyeball_camera_pose):
        columns = right, down, forward; translation = camera eye in base/world.
        """
        if T_cam2base.dim() == 2:
            T_c2b = T_cam2base
        else:
            T_c2b = T_cam2base[0]
        # T_world2cam = inv(T_cam2base); keep grads through adjugate form.
        R = T_c2b[:3, :3]
        t = T_c2b[:3, 3]
        R_inv = R.transpose(0, 1)
        T_w2c = torch.eye(4, device=T_c2b.device, dtype=T_c2b.dtype)
        T_w2c[:3, :3] = R_inv
        T_w2c[:3, 3] = -R_inv @ t
        return self.render_world2cam(T_w2c)
