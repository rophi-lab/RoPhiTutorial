# Computed torque–style point-to-point motion

So far we either floated ([03](03_grav_comp.md)) or chased a live Cartesian target ([06](06_inverse_kinematics_quadratic_program.md)). Here we do classical **go-to-a-joint-goal** motion: sample a safe goal, **plan a path**, then **track** it with PD + gravity (and optional friction) feedforward.

“Computed torque” in the soft sense: we cancel known dynamics terms in $\tau_{\mathrm{ff}}$ and let PD clean up the rest.

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/collision.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/min_jerk_p2p.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/min_jerk_p2p.yaml
```

| Key | Meaning (suggested workflow) |
|-----|------------------------------|
| `1` | Sample a new **collision-free** joint goal |
| `2` | Build a smooth plan to that goal |
| `3` | **Track** the plan |
| `g` | Fall back to gravity compensation |
| `h` | Hold |
| `p`/`q` | Pause / quit |

### What you should observe

After `1`, a goal appears in Viser. After `2`, a joint-space path is drawn. After `3`, the arm follows it smoothly toward the goal. If sampling or planning fails, try `1` again (the goal might be hard to reach).

### LCM

| Channel | Direction | Role |
|---------|-----------|------|
| Joint meas + ColInfo | env → ctrl | State & geometry |
| `sw_flexiv_arm_hand_joint_ctrl` | ctrl → env | Tracking torques / setpoints |
| `sw_p2p_target_joint` | ctrl → viz | Sampled goal |
| `sw_p2p_joint_traj` | ctrl → viz | Planned trajectory |
| `sw_p2p_status` | ctrl → viz | Mode, errors, gains |
| `sw_p2p_gains_cmd` | viz → ctrl | Live gain tweaks from the GUI |

Control runs at 1 kHz; status overlays update more slowly.

---

## 2. Ideas

### Tracking law

Along a reference $(q_d(t),\dot q_d(t))$:

$$
\tau
  = K_p(q_d-q)
  + K_d(\dot q_d-\dot q)
  + \underbrace{g(q) + F_{\mathrm{jc}}\,\mathrm{sat}(\dot q_d/\phi)}_{\tau_{\mathrm{ff}}}.
$$

- PD pulls the error to zero.
- $g(q)$ removes the bulk of gravity.
- Optional Coulomb compensation helps overcome stiction when starting/stopping.

Gains are often initialized near **critical damping** using a rough mass $m_i$:

$$
k_{p,i} \approx \frac{k_{d,i}^{2}}{4m_i}.
$$

### Planning pipeline (story form)

1. **Sample** a joint vector that is far enough from walls and self-collisions (FCL).
2. **Connect** start to goal with RRT-Connect in joint space.
3. **Shortcut** useless zig-zags.
4. **Smooth** with a B-spline (or a densified polyline fallback).
5. **Time** the path with a rest-to-rest **minimum-jerk** profile so $\dot q_d$ starts and ends at zero.

The controller then only has to track; it does not rethink the whole path every millisecond.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/MinimumJerkP2PControl.py` | Keys `1`/`2`/`3`, planner calls, PD tracking |
| `utils/planning/` (RRT, B-spline, min-jerk helpers) | Path construction |
| `configs/.../min_jerk_p2p.yaml` | Gains, RRT parameters, clearance, timing |

**Student exercise:** raise `kd` with Viser while tracking and feel the motion get heavier; lower clearance and see planning fail more often.
