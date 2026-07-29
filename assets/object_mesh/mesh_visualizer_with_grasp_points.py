import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import open3d as o3d
import numpy as np
import os
import yaml

try:
    import open3d.rendering as rendering
except ImportError:
    import open3d.visualization.rendering as rendering

from scipy.spatial.transform import Rotation

from assets.object_mesh import get_grasp_points, get_nominal_pose2bb

import trimesh

# --- Config ---
object_name = "green_bowl"
MESH_PATH = os.path.join(os.path.dirname(__file__), object_name, "textured_mesh.obj")
SPHERE_RADIUS = 0.01

# --- Load mesh ---
mesh = o3d.io.read_triangle_mesh(MESH_PATH)
mesh.compute_vertex_normals()
trimesh_mesh = trimesh.load(MESH_PATH)
cad_to_bb, _ = trimesh.bounds.oriented_bounds(trimesh_mesh)

grasp_points = get_grasp_points(object_name)
nominal_pose2detected_pose = np.linalg.inv(cad_to_bb) @ get_nominal_pose2bb(object_name)


# --- Create sphere geometries for grasp points ---
def create_grasp_spheres(points):
    spheres = []
    for p in points:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
        sphere.translate(p)
        sphere.compute_vertex_normals()
        spheres.append(sphere)
    return spheres


# --- Visualizer with callbacks ---
class GraspVisualizer:
    def __init__(self, mesh, grasp_points, nominal_pose2detected_pose):
        self.mesh = mesh
        grasp_points = grasp_points.copy()
        self.nominal_pose2detected_pose = nominal_pose2detected_pose.copy()

        self.Rot = self.nominal_pose2detected_pose[:3, :3]
        self.trans = self.nominal_pose2detected_pose[:3, 3]

        self.grasp_points = np.einsum(
            "ij,abj->abi", self.Rot, grasp_points
        ) + self.trans.reshape(1, 1, 3)

        self.spheres_left = create_grasp_spheres(self.grasp_points[:, 0, :])
        self.spheres_right = create_grasp_spheres(self.grasp_points[:, 1, :])

        # sample surface points on the mesh
        self.surface_points = self.mesh.sample_points_uniformly(number_of_points=1024)
        # visualize the surface points
        self.mesh.compute_vertex_normals()

        self.surface_points_spheres = create_grasp_spheres(
            np.asarray(self.surface_points.points)
        )

        # visualize frame (increase visibility)
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.frame.compute_vertex_normals()

        # transformed frame self.nominal_pose2detected_pose
        self.transformed_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1
        )
        self.transformed_frame.compute_vertex_normals()
        self.transformed_frame.rotate(self.Rot, center=(0, 0, 0))
        self.transformed_frame.translate(self.trans)

        # --- Add material with transparency ---
        self.mesh_material = rendering.MaterialRecord()
        self.mesh_material.base_color = [0.5, 0.5, 0.3, 0.1]
        self.mesh_material.shader = "defaultLitTransparency"

        # --- Add spheres with transparency ---
        self.spheres_material = rendering.MaterialRecord()
        self.spheres_material.base_color = [0.2, 0.2, 0.2, 0.5]
        self.spheres_material.shader = "defaultLitTransparency"

        self.left_grasp_points_material = rendering.MaterialRecord()
        self.left_grasp_points_material.base_color = [
            0.8,
            0.2,
            0.2,
            1,
        ]  # RGBA, alpha=0.5
        self.left_grasp_points_material.shader = "defaultLitTransparency"

        self.right_grasp_points_material = rendering.MaterialRecord()
        self.right_grasp_points_material.base_color = [
            0.2,
            0.2,
            0.8,
            1,
        ]  # RGBA, alpha=0.5
        self.right_grasp_points_material.shader = "defaultLitTransparency"

        # canonical_pose2pose

    def run(self):
        # Prepare mesh dict
        geometries = [
            {"name": "mesh", "geometry": self.mesh, "material": self.mesh_material}
        ]
        # Add grasp point spheres
        for i, sphere_l in enumerate(self.spheres_left):
            geometries.append(
                {
                    "name": f"grasp_sphere_l_{i}",
                    "geometry": sphere_l,
                    "material": self.left_grasp_points_material,
                }
            )
        for i, sphere_r in enumerate(self.spheres_right):
            geometries.append(
                {
                    "name": f"grasp_sphere_r_{i}",
                    "geometry": sphere_r,
                    "material": self.right_grasp_points_material,
                }
            )
        # Add surface point spheres
        for i, sphere in enumerate(self.surface_points_spheres):
            geometries.append(
                {
                    "name": f"surface_sphere_{i}",
                    "geometry": sphere,
                    "material": self.spheres_material,
                }
            )
        # geometries.append({"name": "frame", "geometry": self.frame})
        geometries.append(
            {"name": "transformed_frame", "geometry": self.transformed_frame}
        )
        o3d.visualization.draw(geometries)


if __name__ == "__main__":
    vis = GraspVisualizer(mesh, grasp_points, nominal_pose2detected_pose)
    vis.run()
