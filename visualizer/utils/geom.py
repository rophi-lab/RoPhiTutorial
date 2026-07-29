import numpy as np
import trimesh
import open3d as o3d


class Arrow:
    def __init__(
        self,
        start=np.zeros(3),
        direction=np.array([0, 0, 1]),
        length=0.03,
        radius=0.002,
        color=(255, 0, 0),
    ):
        self.arrow_trimesh = self._create_arrow(start, direction, length, radius, color)

    def _create_arrow(
        self, start, direction, length=0.03, radius=0.002, color=(255, 0, 0)
    ):
        direction = direction / np.linalg.norm(direction)
        mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=radius,
            cone_radius=radius * 1.5,
            cylinder_height=length * 0.7,
            cone_height=length * 0.3,
        )
        mesh_arrow.compute_vertex_normals()
        default_dir = np.array([0, 0, 1])
        target_dir = direction
        axis = np.cross(default_dir, target_dir)
        if np.linalg.norm(axis) < 1e-6:
            R = np.eye(3)
        else:
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(default_dir, target_dir), -1.0, 1.0))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

        mesh_arrow.rotate(R, center=(0, 0, 0))
        mesh_arrow.translate(start)

        vertices = np.asarray(mesh_arrow.vertices)
        faces = np.asarray(mesh_arrow.triangles)
        trimesh_mesh = trimesh.Trimesh(vertices, faces)

        # color
        trimesh_mesh.visual.face_colors = np.array(color)
        return trimesh_mesh
