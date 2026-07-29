# https://mujoco.readthedocs.io/en/stable/XMLreference.html
# body / geom
# type: plane, hfield, sphere, capsule, ellipsoid, cylinder, box, mesh, sdf
# size:
# - plane: 3
# X half-size; Y half-size; spacing between square grid lines for rendering.
# If either the X or Y half-size is 0,
# the plane is rendered as infinite in the dimension(s) with 0 size.
# - hfield: 0
# The geom sizes are ignored and the height field sizes are used instead.
# - sphere: 1
# Radius of the sphere.
# - capsule: 1 or 2
# Radius of the capsule;
# half-length of the cylinder part when not using the fromto specification.
# - ellipsoid: 3
# X radius; Y radius; Z radius.
# - cylinder: 1 or 2
# Radius of the cylinder;
# half-length of the cylinder when not using the fromto specification.
# - box: 3
# X half-size; Y half-size; Z half-size.
# - mesh: 0
# The geom sizes are ignored and the mesh sizes are used instead.

import numpy as np
