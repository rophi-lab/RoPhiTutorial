# Inverse Kinematics + Reach

Arm+hand: **sample a palm SE(3) goal and hand configuration**, **solve arm IK**, then **regulate to `q*` with joint-space impedance**. No path planner.

## Pipeline (keys)

| Key | Step | What you see in Viser |
|-----|------|------------------------|
| **`1`** | Sample palm pose `T_des` + hand joints `q_hand` | Purple **frame** at `T_des` + **floating palm + fingers** (no arm). |
| **`2`** | Solve IK for arm joints; check collisions | Floating hand clears; full-arm ghost at **`q*`**. Rejected if in collision. |
| **`3`** | Joint-space impedance about `q*` | Robot pulls toward the purple ghost (no yellow traj). |

Also: `g` grav-comp, `h` hold current, `p` pause, `q` quit.

Order is **1 → 2 → 3**. Pressing `1` clears later stages.

## Run

```bash
# terminal 1 — collision environment (walls)
python run_env.py --config configs/flexiv_arm_5F_hand/env/collision.yaml

# terminal 2
python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/ik_reach.yaml

# terminal 3
python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/ik_reach.yaml
```

Focus the controller terminal when pressing keys.

## Inverse kinematics (step 2)

We use **closed-loop IK (CLIK)** in Pinocchio on the `palm` frame. Hand joints stay fixed; only arm joints are free.

### Residual (body twist)

$$
\xi = \log\bigl(T(q)^{-1}\, T_{\mathrm{des}}\bigr) \in \mathfrak{se}(3)
$$

with $\log$ the Pinocchio `log6` map (position + rotation error as a 6-vector).

### Joint update

$$
\Delta q = J^{+}\,\xi,\qquad
J^{+} = J^{\mathsf T}\bigl(J J^{\mathsf T} + \lambda^{2} I\bigr)^{-1}
$$

$J$ is the **LOCAL** frame Jacobian (matches body `log6`). Iterate until $\|\xi\| < \varepsilon$ or the iteration limit. Then clamp joints to limits and check self/environment collisions with FCL.

## Sampling (step 1)

- **Translation:** uniform in `pose_pos_min` / `pose_pos_max`.
- **Rotation:** seed from current palm FK, then random SO(3) noise (`pose_rot_noise_rad`).
- **Hand:** uniform in joint limits (with margin) on `hand_sample_joint_idx`.

No IK or collision check until step `2`.

## Joint impedance (step 3)

No trajectory. Key `3` sets the hold / impedance setpoint to the IK solution `q*` and regulates with the plant law

$$
\tau = K_p(q^\star - q) - K_d\,\dot q + g(q),
$$

i.e. joint-space impedance with $K=K_p$, $D=K_d$ (gains from YAML `kp` / `kd`, same critical-damping recipe as [07](07_computed_torque_control.md) / [08](08_impedance_control.md)).

There is no collision avoidance while moving — if the straight pull hits a wall, resample (`1`) or solve a free IK goal (`2`).

## Config checklist

`configs/flexiv_arm_5F_hand/controllers/ik_reach.yaml`:

- `controller.name: ik_reach`
- `task_frame: palm`, `pose_pos_*`, `hand_sample_joint_idx`, `ik_*`
- `kp` / `kd` (joint impedance / hold gains)
- `pub_manager.joint_ctrl_channel` / `joint_target_channel` / `palm_pose_channel`

Visualizer `configs/visualizer/flexiv_arm_5F_hand/ik_reach.yaml` must list the same LCM channels (including `palm_pose_channel`).

## Relation to `ik_qp` / min-jerk P2P

- [`ik_qp.yaml`](../configs/flexiv_arm_5F_hand/controllers/ik_qp.yaml) — continuous QP SE(3) tracking with mink.
- [07](07_computed_torque_control.md) — collision-free RRT + min-jerk **path** then PD track.
- This tutorial — discrete pose sample → IK → **direct** joint impedance to `q*`.
