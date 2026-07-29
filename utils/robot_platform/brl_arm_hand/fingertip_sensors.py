import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_contact_point_sphere(theta, phi, r=0.01):
    n = np.array([0, 0, 1])
    R_theta = R.from_euler("x", theta, degrees=True).as_matrix()
    R_phi = R.from_euler("y", phi, degrees=True).as_matrix()
    R_cont = R_phi @ R_theta
    return r * (R_cont @ n)


def compute_contact_normal_ellipsoid(x, y, z, a, b, c):
    # x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
    # n = [2x/a^2, 2y/b^2, 2z/c^2]
    nl = np.array([2 * x / a**2, 2 * y / b**2, 2 * z / c**2])
    nl = nl / np.linalg.norm(nl)
    return nl


# ## Old functions from Sharmi
# def sensor_to_contact_frame(contact_data, theta, phi):

#     # for ellipsoid, this angle must be contact normal angle.
#     Fxyz = contact_data[:, 0:3]

#     # eventually to convert from sensor Fxyz to contact Fxyz
#     R_theta = (R.from_euler("x", theta, degrees=True).as_matrix()).squeeze()
#     R_phi = (R.from_euler("y", phi, degrees=True).as_matrix()).squeeze()
#     Fxyz_contact = R_phi.T @ R_theta.T @ Fxyz.T
#     return (Fxyz_contact.T).squeeze()


# def line_ellipsoid_intersection(R_cont, ellipse_params):
#     p0 = np.array([0, 0, 0])  # a point on the line
#     a = ellipse_params[0]
#     b = ellipse_params[1]
#     c = ellipse_params[2]
#     # Coefficients of the quadratic equation
#     A = (R_cont[0] ** 2) / a**2 + (R_cont[1] ** 2) / b**2 + (R_cont[2] ** 2) / c**2
#     B = 2 * (
#         (p0[0] * R_cont[0]) / a**2
#         + (p0[1] * R_cont[1]) / b**2
#         + (p0[2] * R_cont[2]) / c**2
#     )
#     C = (p0[0] ** 2) / a**2 + (p0[1] ** 2) / b**2 + (p0[2] ** 2) / c**2 - 1

#     discriminant = B**2 - 4 * A * C

#     # TODO: Better error handling if there is no intersection
#     if discriminant < 0:
#         return None  # No intersection
#     elif discriminant == 0:
#         t = -B / (2 * A)
#         p1 = p0 + t * R_cont
#         return np.array([p1])  # One intersection (tangent)
#     else:
#         sqrt_discriminant = np.sqrt(discriminant)
#         t1 = (-B + sqrt_discriminant) / (2 * A)
#         t2 = (-B - sqrt_discriminant) / (2 * A)
#         p1 = p0 + t1 * R_cont
#         p2 = p0 + t2 * R_cont

#         # return negative z point #TODO: Check this
#         if p1[2] > 0:
#             return p1
#         else:
#             return p2  # Two intersection points


# def sensor_to_contact_frame_ellipsoid(contact_data, theta_rad, phi_rad):
#     # for ellipsoid, calculate contact normal angles
#     # z is normal to surface
#     # should I put this here?
#     Fxyz = contact_data[:, 0:3]
#     nominal_contact = np.array([0.0, 0.0, 0.01])
#     ellipse_params = np.array([0.0105, 0.009, 0.00635])
#     R_theta = np.array(
#         [
#             [1, 0, 0],
#             [0, np.cos(theta_rad), -np.sin(theta_rad)],
#             [0, np.sin(theta_rad), np.cos(theta_rad)],
#         ]
#     )  # Rx by theta
#     R_phi = np.array(
#         [
#             [np.cos(phi_rad), 0, np.sin(phi_rad)],
#             [0, 1, 0],
#             [-np.sin(phi_rad), 0, np.cos(phi_rad)],
#         ]
#     )  # Ry by phi
#     R_cont = (R_phi @ R_theta @ nominal_contact).T
#     # calculate radius of vector (intersection between ellipsoid and line in direction of R_cont)
#     contact_vec = line_ellipsoid_intersection(R_cont, ellipse_params)
#     # calculate normal vector at contact point
#     contact_vec_normal = 2 * np.array(
#         [
#             contact_vec[0] / ellipse_params[0] ** 2,
#             contact_vec[1] / ellipse_params[1] ** 2,
#             contact_vec[2] / ellipse_params[2] ** 2,
#         ]
#     )  # find vector normal to surface at contact location
#     contact_vec_unit_normal = contact_vec_normal / np.linalg.norm(contact_vec_normal)
#     theta_rad_normal = np.arcsin(-contact_vec_unit_normal[1])
#     phi_rad_normal = np.arctan2(contact_vec_unit_normal[0], contact_vec_unit_normal[2])
#     R_theta_normal = np.array(
#         [
#             [1, 0, 0],
#             [0, np.cos(theta_rad_normal), -np.sin(theta_rad_normal)],
#             [0, np.sin(theta_rad_normal), np.cos(theta_rad_normal)],
#         ]
#     )  # Rx by theta
#     R_phi_normal = np.array(
#         [
#             [np.cos(phi_rad_normal), 0, np.sin(phi_rad_normal)],
#             [0, 1, 0],
#             [-np.sin(phi_rad_normal), 0, np.cos(phi_rad_normal)],
#         ]
#     )  # Ry by phi
#     R_cont_normal = R_phi_normal @ R_theta_normal
#     Fxyz_cont = R_cont_normal @ Fxyz.T

#     return (Fxyz_cont.T).squeeze()
