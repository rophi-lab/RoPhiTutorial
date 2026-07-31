# Publishing collision geometry

Before a controller can say “stay away from the wall,” it needs a **description of shapes**. This tutorial is only about **publishing** those shapes from the environment over LCM. No avoidance yet — that comes in [05](05_collision_aware_grav_comp.md).

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/collision.yaml
```

You can run this env alone. Controllers and visualizers that need collision data will subscribe to the channels below (see [05](05_collision_aware_grav_comp.md)).

### Two channels, two frames

| Channel | Who uses it | Frame | Contents |
|---------|-------------|-------|----------|
| `sw_flexiv_arm_hand_robot_col_info` | Controller / viz | **Link-local** | Simple shapes attached to robot links (cylinders, spheres, …) |
| `sw_flexiv_arm_hand_static_col_info` | Controller / viz | **World** | Floor and walls |

Each named primitive is a small record:

- a **type** (`box`, `cylinder`, or `sphere`),
- **size** parameters,
- a 4×4 **offset** transform.

The env republishes on a timer (`col_info_update_dt`) so a controller that starts late still receives geometry.

### Why not reuse MuJoCo meshes?

MuJoCo contact meshes are for **physics**. Planning libraries such as **FCL** prefer a handful of **convex primitives** that are fast for distance queries. We deliberately keep a second, coarser geometric model for control.

---

## 2. Ideas

### Placing a robot shape in the world

A link moves with the robot. If a cylinder is defined in the link frame with offset $T_{\mathrm{offset}}$, its world pose at configuration $q$ is

$$
T^{\mathrm{world}}(q)
  = {}^{W}T_{\mathrm{link}}(q)\,T_{\mathrm{offset}}.
$$

Static walls skip the FK part: their offset is already expressed in the world.

### What “distance” will mean later

Given two placed shapes, a collision library returns:

- a **signed distance** $d$ (positive = separation, negative = penetration),
- two **nearest points**, one on each shape.

Controllers turn those into inequality constraints (“do not reduce $d$ too quickly”). This tutorial only ensures the shapes exist on the bus.

### Export details students trip on

- MuJoCo boxes often store **half**-extents; FCL boxes here use **full** side lengths.
- An infinite ground plane is exported as a **thin box** so distance queries stay well defined.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `env/flexiv_arm_5F_hand/FlexivArmHandColEnv.py` | Builds primitives and publishes `ColInfo` |
| Platform helper for arm primitives | Cylinders / spheres for links |
| `utils/collision/wall_geoms.py` | Floor / wall boxes |
| `ColInfoData` + `col_info.lcm` | Message schema |
| `configs/flexiv_arm_5F_hand/env/collision.yaml` | Channel names, republish period |

**Try this:** open the collision visualizer from [05](05_collision_aware_grav_comp.md) and confirm you see capsules/boxes on the arm and walls — that means these channels are alive.
