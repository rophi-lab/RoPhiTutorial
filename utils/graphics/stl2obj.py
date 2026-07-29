import numpy as np
import argparse


from omegaconf import OmegaConf, DictConfig
import open3d as o3d


def convert_stl_to_obj(stl_filepath, obj_filepath):
    """Converts an STL file to OBJ format using Open3D.

    Args:
        stl_filepath (str): Path to the input STL file.
        obj_filepath (str): Path to save the output OBJ file.
    """
    mesh = o3d.io.read_triangle_mesh(stl_filepath)
    o3d.io.write_triangle_mesh(obj_filepath, mesh)


def main(args):
    # Example usage
    convert_stl_to_obj(args.input, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args, unknown = parser.parse_known_args()
    main(args)
