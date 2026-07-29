import copy

import numpy as np
import open3d as o3d
import torch


class PredefinedObj:
    def __init__(self, mesh_path: str, device: str | None = None):
        self._mesh_path = mesh_path
        self._mesh = o3d.io.read_triangle_mesh(mesh_path)

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = device

        # Uniform surface samples for a nearest-point distance proxy.
        self._surface_pc = np.asarray(
            self._mesh.sample_points_uniformly(number_of_points=1024).points
        )  # (N, 3)
        self._torch_surface_pc = torch.tensor(
            self._surface_pc, dtype=torch.float32, device=self._device
        )
        self._torch_rot = torch.eye(3, dtype=torch.float32, device=self._device)
        self._torch_trans = torch.zeros(3, dtype=torch.float32, device=self._device)

    def set_pose(self, trans: np.ndarray, rot: np.ndarray):
        self._torch_trans = torch.tensor(
            trans, dtype=torch.float32, device=self._device
        )
        self._torch_rot = torch.tensor(rot, dtype=torch.float32, device=self._device)

    def get_dist_and_grad(self, x):
        """
        x: (N, 3) world points → unsigned nearest surface distance + unit grad.
        """
        torch_x = torch.as_tensor(x, dtype=torch.float32, device=self._device)
        torch_surface_pc_tf = (
            self._torch_surface_pc @ self._torch_rot.T + self._torch_trans.view(1, 3)
        )
        dist_mat = torch.norm(
            (torch_surface_pc_tf.unsqueeze(0) - torch_x.unsqueeze(1)), dim=-1
        )
        min_dist, min_idx = torch.min(dist_mat, dim=1)
        nearest_surface_pc = torch_surface_pc_tf[min_idx]  # (N, 3)
        grad_dist = torch_x - nearest_surface_pc
        grad_dist = grad_dist / torch.clamp(
            torch.norm(grad_dist, dim=-1, keepdim=True), min=1e-9
        )
        return (min_dist.detach().cpu().numpy(), grad_dist.detach().cpu().numpy())

    def get_mesh(self):
        mesh = copy.deepcopy(self._mesh)
        mesh.rotate(self._torch_rot.cpu().numpy(), center=(0, 0, 0))
        mesh.translate(self._torch_trans.cpu().numpy())
        return mesh
