import mujoco as mj


def check_collision(mj_model, mj_data):
    """
    Check for collisions in a Mujoco simulation.
    @param[in] mj_model: Mujoco model
    @param[in] mj_data: Mujoco data
    """
    ncon = mj_data.ncon  # number of active contacts
    for i in range(ncon):
        contact = mj_data.contact[i]

        # These are the geom IDs of the contact pair
        geom1 = contact.geom1
        geom2 = contact.geom2

        # You can also get the body names if needed
        body1 = mj_model.geom_bodyid[geom1]
        body2 = mj_model.geom_bodyid[geom2]

        body1_name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_BODY, body1)
        body2_name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_BODY, body2)
        print(f"[{i+1}] contact between bodies: {body1_name}, {body2_name}")
    return ncon > 0
