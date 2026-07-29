def add_viser_prim_geom(
    viser_server,
    name,
    type,
    size,
    color=(255, 0, 0),
    opacity=0.5,
):
    """
    Add a collision geometry to the scene.
    """
    if type == "box":
        return viser_server.scene.add_box(
            name, dimensions=size, color=color, opacity=opacity
        )
    elif type == "cylinder":
        return viser_server.scene.add_cylinder(
            name, radius=size[0], height=size[1], color=color, opacity=opacity
        )
    elif type == "sphere":
        return viser_server.scene.add_icosphere(
            name, radius=size[0], color=color, opacity=opacity
        )
    else:
        raise ValueError(f"Unknown geometry type: {type}")
