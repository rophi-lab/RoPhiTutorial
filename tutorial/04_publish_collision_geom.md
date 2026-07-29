# Publishing Collision Geometry from the Env

This note explains how the **collision env** models robot and environment
collision shapes for the *controller / visualizer* (FCL), and how it publishes
them over LCM. It is separate from MuJoCo’s own contact meshes (tutorial 02).

Run:

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/collision.yaml
```

Factory name: `flexiv_arm_5F_hand_col` → `FlexivArmHandColEnv`.

---

## Why a second collision model?

| Layer | Used by | Representation |
|-------|---------|----------------|
| MuJoCo `contact` geoms in MJCF | Physics contacts in sim | Detailed meshes / walls |
| **ColInfo** on LCM | Collision-aware controller + Viser overlay | Boxes / cylinders / spheres for **FCL** |

The controller never reads MuJoCo state. It only sees LCM. So the env must
**export** a compact primitive set that FCL understands. That export is what
this tutorial is about.

```
MJCF walls  ──extract_wall_col_geoms──►  static_col_info  ──LCM──►  controller / viz
platform URDF ──cylinders+sphere──────►  robot_col_info   ──LCM──►  controller / viz
```

---

## Payload: `ColInfoData`

Each geometry is one entry:

| Field | Meaning |
|-------|---------|
| `name` | ID (`link3`, `floor`, `wall_back`, …) |
| `type` | `"box"` \| `"cylinder"` \| `"sphere"` |
| `size` | Type-dependent (box: full extents $l,w,h$; cylinder: $r,h$; sphere: $r$) |
| `offset` | $4\times 4$ SE(3) pose of the primitive |

**Critical distinction — whose frame is `offset` in?**

| Stream | `offset` frame | Who moves it? |
|--------|----------------|---------------|
| **Static (env)** | **World** | Fixed; pose is absolute |
| **Robot** | **Link-local** | Consumer does $\mathrm{FK}(\mathrm{link})\,@\,\mathrm{offset}$ each tick |

Wire format: `lcm_type/lcm_type/col_info.lcm` → `ColInfoPublisher` /
`ColInfoSubscriber`.

---

## Environment (static) collision geoms

### Source of truth

Walls live in the scene MJCF (`hand_default.xml`):

```xml
<geom name="floor" type="plane" .../>
<geom name="wall_back" type="box" size="0.05 0.85 0.9" pos="-0.28 0 0.9" .../>
<geom name="wall_left" type="box" size="0.85 0.05 0.9" pos="0 -0.45 0.9" .../>
```

`FlexivArmHandColEnv` does **not** hard-code wall poses. At init it calls
`extract_wall_col_geoms(self.mj_model)` (`utils/collision/wall_geoms.py`), which
keeps naming rule: `floor` or names starting with `wall`.

### Export rules

| MJCF geom | Exported FCL primitive |
|-----------|------------------------|
| `box` | Box with **full** extents (`2 ×` MuJoCo half-sizes), world `pos`/`quat` |
| `plane` (`floor`) | Large thin **box** (FCL has no plane); top face at the plane height |

So: edit the XML walls once → MuJoCo contacts **and** the published static
ColInfo stay aligned.

---

## Robot collision geoms

### Not the MJCF contact meshes

Robot ColInfo comes from the **platform**, not from MuJoCo mesh geoms:

```python
# FlexivArmHandColEnv.__init__
for dict_col_info in self.platform.get_col_info_data():
    self.robot_col_info_data.add_col_geom(**dict_col_info)
```

`SimFlexivArm5FHand.get_col_info_data()` → `_build_arm_collision_primitives()`.

### How primitives are built

Using the Pinocchio URDF at neutral configuration:

1. For each arm link `link1`…`link7`, place a **cylinder** along the segment to
   the next frame (`link2`…`palm`), with hand-tuned radii
   `_ARM_LINK_RADII`. The last link is extended toward the wrist
   (`_LAST_LINK_EXTENSION`).
2. Add a **sphere** named `palm` with a fixed link-local offset.

Each entry stores:

- `name` = link / frame name (`link3`, `palm`, …) — must match what the
  controller uses for FK.
- `offset` = SE(3) **in that link’s frame** (cylinder axis + center).

The env publishes these offsets **once and re-sends them**; it does *not*
update them with live $q$. The controller / visualizer apply FK every control
tick:

$$
T_{\mathrm{world}} = {}^{W}T_{\mathrm{link}}(q)\,\,T_{\mathrm{offset}}.
$$

That keeps the ColInfo message small and rate-independent of joint streaming.

---

## How the env publishes

`FlexivArmHandColEnv._sync_data_from_sim` (after normal joint_meas publish):

1. Every `col_info_update_dt` seconds (default **0.1 s** in
   `configs/.../env/collision.yaml`):
2. Stamp both `ColInfoData` objects with sim time.
3. Push deep copies onto the pub queues:
   - **intr** → `robot_col_info_channel`
   - **extr** → `static_col_info_channel`

Pub manager (`flexiv_arm_hand_col_info`):

| Channel | Queue | Typical name |
|---------|-------|----------------|
| Robot | intr | `sw_flexiv_arm_hand_robot_col_info` |
| Static | extr | `sw_flexiv_arm_hand_static_col_info` |

Re-publishing static walls (even though they never move) lets a **late-joining**
controller / visualizer pick up geometry without a special handshake.

Default env (`default.yaml`) does **not** publish ColInfo — only the collision
env + `flexiv_arm_hand_col_info` pub manager do.

---

## Who consumes it

| Consumer | Use |
|----------|-----|
| `col_aware_grav_comp` | Build FCL objects; distance-check `col_pairs` |
| Collision Viser config | Overlay primitives; dashed distance lines |

Both must use the **same channel names** as the env YAML. Visualizer
“Show collision geometry” shows exactly what came off the wire.

---

## Mental model

```
            ┌─────────────────────────────────────┐
            │         FlexivArmHandColEnv         │
            │  (extends MujocoBaseEnv)            │
            ├─────────────────────────────────────┤
            │ init:                               │
            │   walls ← extract_wall_col_geoms()  │
            │   robot ← platform.get_col_info()   │
            │ every col_info_update_dt:           │
            │   publish static + robot ColInfo    │
            └──────────────┬──────────────────────┘
                           │ LCM
           ┌───────────────┴────────────────┐
           ▼                                ▼
   robot_col_info                    static_col_info
   (link-local cyl/sph)              (world boxes)
           │                                │
           └────────────┬───────────────────┘
                        ▼
              controller FK + FCL distance
              visualizer overlay
```

---

## Given a new robot and environment

The Flexiv cylinders / palm sphere and the `floor` / `wall_*` export are **this
robot’s design choices**, not a generic MuJoCo feature. For a new robot or cell
you must design your own collision geometry and wire it into the same ColInfo
pipeline:

1. **Environment.** Decide which obstacles matter (tables, shelves, cages). Put
   them in the MJCF (or a calibration file) as boxes / cylinders / spheres that
   FCL can use, and teach `extract_wall_col_geoms` (or a sibling helper) how to
   find and export them in **world** frame.
2. **Robot.** Choose a sparse set of primitives per link (capsules, spheres,
   boxes) that cover the dangerous volume without being so fat that the
   controller becomes overly conservative. Store **link-local** offsets;
   consumers will apply FK. Implement this in the platform’s
   `get_col_info_data()` (or equivalent).
3. **Names and pairs.** Geom names must match what the controller lists in
   `col_link_names`, `obs_names`, and `col_pairs` (and any self-collision pairs
   in the visualizer).
4. **Validate.** Turn on Viser “Show collision geometry” / distance lines and
   check that primitives hug the real shape and that distances make sense
   before trusting collision-aware control.

Good FCL geometry is a modeling task: too coarse → false collisions and timid
motion; too fine / too many pairs → slow planning and brittle numerics.

---

## Practical tips

1. **Move a wall in MJCF** → restart env; static ColInfo follows automatically.
2. **Change robot primitives** → edit `_build_arm_collision_primitives` (radii,
   palm sphere). Controllers that hard-code `col_link_names` / `col_pairs` must
   still match those names.
3. **MuJoCo mesh contact ≠ FCL capsule.** The arm can look clear in the viewer
   mesh but the published cylinder may still be close to a wall — use the Viser
   collision overlay to see the FCL shapes.
4. On hardware you still need *some* process to publish the same ColInfo
   channels (often this env with `vis_mode: vis_off`) so the controller’s FCL
   world matches the lab.

---

## Files

| File | Role |
|------|------|
| `env/flexiv_arm_5F_hand/FlexivArmHandColEnv.py` | Build + periodic publish |
| `utils/collision/wall_geoms.py` | MJCF walls → static boxes |
| `robot_platform/sim/.../SimFlexivArm5FHand.py` | Robot cylinders + palm sphere |
| `communication/.../FlexivArmHandColInfoPubManager.py` | LCM pub wiring |
| `configs/flexiv_arm_5F_hand/env/collision.yaml` | Channels + `col_info_update_dt` |
| `data_type/basic_types/ColInfoData.py` | In-process payload |
| `lcm_type/lcm_type/col_info.lcm` | On-wire schema |
