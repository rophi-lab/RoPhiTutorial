import numpy as np


def read_obj_to_string(file_path):
    """
    Reads an OBJ file and returns its content as a string.

    Args:
        file_path (str): The path to the OBJ file.

    Returns:
        str: The content of the OBJ file as a string, or None if an error occurs.
    """
    try:
        with open(file_path, "r") as file:
            obj_string = file.read()
        return obj_string
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def convert_o3d_mesh_to_string(o3d_mesh):
    # Convert to numpy arrays
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    # Build .obj string
    lines = []

    # Vertices
    for v in vertices:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")

    # Faces (OBJ is 1-based indexing!)
    for f in faces:
        f1, f2, f3 = f + 1
        lines.append(f"f {f1} {f2} {f3}")

    # Join all lines into single string
    obj_str = "\n".join(lines)
    return obj_str


def get_obj_string_from_vertices_and_faces(vertices, faces):
    # Build .obj string
    lines = []

    # Vertices
    for v in vertices:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")

    # Faces (OBJ is 1-based indexing!)
    for f in faces:
        f1, f2, f3 = f + 1
        lines.append(f"f {f1} {f2} {f3}")

    # Join all lines into single string
    obj_str = "\n".join(lines)
    return obj_str
