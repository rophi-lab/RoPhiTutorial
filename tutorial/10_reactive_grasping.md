# Reactive three-finger grasping (Flexiv + RH-5)

Tutorial port of the hierarchical reactive grasping pipeline from:

> **Y. Lee\*, T.-Y. Lin\*, A. Alexiev, S. Kim**,  
> *Hierarchical Reactive Grasping via Task-Space Velocity Fields and Joint-Space Quadratic Programming*,  
> ICRA 2026.  
> [arXiv:2509.01044](https://arxiv.org/abs/2509.01044) · [project page](https://reactivegrasp.github.io/)

This stack runs on **Flexiv Rizon + Robotis RH-5** in MuJoCo: collision-aware
fingertip path guidance → task-space velocity fields → joint-space IK QP, plus
a small tutorial FSM for squeeze / retract. Viser shows the object mesh,
antipodal candidates, fingertip cur→tgt, and the planned fingertip LVF paths.

It builds on:

- gravity / velocity damping and joint LCM ([03](03_grav_comp.md)),
- ColInfo + FCL ([04](04_publish_collision_geom.md), [05](05_collision_aware_grav_comp.md)),
- velocity-level IK QP ([06](06_inverse_kinematics_quadratic_program.md)).

---

## Run

Three terminals:

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/grasping.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/three_finger_reactive_grasp.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/grasping.yaml
```

| Key | Where | Action |
|-----|--------|--------|
| `o` | env (viewer or terminal) | Random XY (+ small yaw) **object respawn** — keeps sim + GUI running |
| `r` | env | Full env reset (also reseeds object; restarts viewer) |
| `g` / `d` | controller | Grav-comp / joint-space go toward `default_q` |
| `v` | controller | Start **reaching** (also clears IKQP fault latch) |
| `c` | controller | Force **closing** (squeeze → arm to default) |
| `p` / `q` | controller | Pause / quit |

Typical flow: wait for object pose → `v` (reach) → auto or `c` (close) →
`o` to move the bowl and try again → `d` / `g` if you need to settle.

---

## Architecture (two rates)

| Thread | Rate | Work |
|--------|------|------|
| Plan (`_plan_loop`) | `plan_freq` (50 Hz) | Object pose / FK, grasp selection, fingertip path + VFs, FCL, OSQP → $(\dot q_{\mathrm{des}},\tau_{\mathrm{ff}})$ |
| Control (`_update`) | `ctrl_freq` (1 kHz) | Track latest $\dot q_{\mathrm{des}}$ with $k_p{=}0$ + velocity damping + gravity FF |

Plans go through a **size-1 queue** (latest wins; drain-before-put). Each
thread uses its own Pinocchio `Data` (`_pin_data` vs `_pin_data_plan`).

Code: `controller/ThreeFingerReactiveGrasping.py`.

---

## 1. Why hierarchical?

Planning a long-horizon trajectory in full $n$-DoF joint space is expensive.
Lee et al. split the problem:

| Layer | Space | Role |
|-------|--------|------|
| **High** | Task space (fingertip $x_i$, $R_i$; hand joints) | Globally informed **velocity fields** |
| **Low** | Joint space $q\in\mathbb{R}^{27}$ | One-step **QP** that tracks those fields under collisions & limits |

Guidance stays cheap (small task dim); feasibility stays reactive (horizon
$H{=}$`plan_horizon`, full constraints).

---

## 2. Scene & perception (this tutorial)

| Piece | Role |
|-------|------|
| `FlexivArmHandGraspEnv` + `hand_grasping.xml` | Freejoint `grasp_object`, ColInfo, live SE(3) on `sw_grasp_object_pose` |
| `PredefinedObj` | Surface point-cloud distance + unit gradient (unsigned nearest surface) |
| Grasp candidates | `get_grasp_points_in_cad("green_bowl")` → world via live pose |
| MuJoCo collision | V-HACD convex parts for the bowl; FCL for arm↔walls/floor |

Spawn: random XY in `obj_spawn_xy_center` ± `obj_spawn_xy_half`, fixed $z$,
CAD→world orientation (+ small yaw). Press **`o`** to respawn without tearing
down the viewer.

---

## 3. High layer — task-space velocity fields

### 3.1 Grasp target

Antipodal pairs (and L/R flips) are transformed by the live object pose. The
active trio (thumb / index / middle) is chosen by a distance + alignment cost
(`_compute_target_grasp_data`). Width can widen for reaching and tighten for
closing (`squeeze_width_multiple`).

### 3.2 Fingertip linear fields (path-guided LVF)

Straight attractors clip concave bowls. Instead, each tip follows a short
clearance-aware path $c_i(s)$:

$$
\dot x_{i,\mathrm{des}}
  = v(x_i,x_i^*)\,\frac{c_i(s_1)-c_i(s_0)}{\|c_i(s_1)-c_i(s_0)\|}.
$$

**This repo’s path pipeline** (`_compute_fingertip_linear_velocity_des`):

1. **Heuristic init** — `heuristic_path_initialization_general` samples
   2-segment via paths that reduce violations of
   $d(x,\mathcal{O}) < \epsilon_{\mathrm{LVF}}$
   (`collision_margin_to_object_lvf`, `heuristic_via_length`).
2. **SQP polish** (paper TO loop) — `smooth_multi_finger_paths_sqp`
   (`num_path_sqp_iterations: 3`): OSQP minimizes discrete path curvature
   subject to linearized $d\ge\epsilon$, then Savitzky–Golay
   (`path_sqp_savgol_window`).
3. **Speed** — `speed_const_then_linear_clamped` (const far away, taper near
   goal, clamp so one plan step cannot overshoot).

Paths are stored in `_ft_paths` and published on `sw_grasp_ft_paths` for Viser.

### 3.3 Orientation & gripper fields

- **Orientation:** $\omega_{i,\mathrm{des}} \propto \hat n_i \times \hat n_i^*$
  (index / middle; tanh gates on angle / reaching error).
- **Gripper:** joint VF toward `q_nominal_gripper`, weighted up when far from
  the grasp (`weight_func_tanh` on reaching error).

---

## 4. Low layer — joint-space IK QP

Track weighted task plans $\{(w_k,J_k,\dot y_k^{\mathrm{des}})\}$:

$$
\begin{aligned}
\min_{\dot q}\quad
&\sum_k w_k\,\|J_k\dot q - \dot y_k^{\mathrm{des}}\|^2
 + \|\dot q\|_{W_{\mathrm{reg}}}^2
 + w_{\mathrm{rate}}\|\dot q-\dot q_{\mathrm{prev}}\|^2 \\
\text{s.t.}\quad
&\dot q_{\min}\le \dot q \le \dot q_{\max},\\
&q_{\min}\le q + \dot q\,H \le q_{\max},\\
&\text{FCL env rows (links↔walls/floor)},\\
&\text{object rows (phalanx/tip points↔bowl)}.
\end{aligned}
$$

Implemented in `_solve_IKQP` (OSQP):

- **Reg** — `ikqp_reg_arm` / `ikqp_reg_hand` (and optional `ikqp_qd_reg`
  vector); `ikqp_reg_qd_rate` for temporal smoothness.
- **Env** — FCL nearest points + point Jacobians
  (`_add_collision2env_constraints`).
- **Object** — `PredefinedObj.get_dist_and_grad`
  (`_add_collision2obstacles_constraints`), margin
  `collision_margin_to_object_qp`.
- **Escape cap** — collision lower bounds are clipped by
  `ikqp_max_escape_rate` so deep penetration cannot demand impossible
  velocities (which makes the QP primal-infeasible).

### IKQP fault latch

On primal infeasible / non-finite / over-limit solutions, OSQP’s
`res.x` can be a huge **infeasibility certificate** — never command it.
The controller:

1. commands $\dot q = 0$,
2. **latches** `_ikqp_fault` so later “solved” bangs at velocity limits
   cannot resume motion automatically,
3. clears the latch on **`v` / `d` / `g` / `c`**.

Without the latch, a single failed tick (~20 ms of zeros) is often followed by
an aggressive feasible recovery — that looks like a runaway.

---

## 5. FSM (tutorial wrapper)

| State | Behavior |
|-------|----------|
| **reaching** | Full hierarchy above. → **closing** when mid-fingertip error $<$ ~1 cm (`DIST_THR_REACHING2CLOSING_GRIPPER`). |
| **closing · squeeze** | Tip attractors + hand-close $\tau$ for `lift_delay_s`. |
| **closing · lift** | Arm joint VF → `default_q` (`lift_speed`); hand stays closed. Done when arm error $<$`lift_arm_eps`. |
| **lost grasp** | Mid error $>$ ~5 cm in closing → back to **reaching**. |

Closing / lift are tutorial conveniences; the paper contribution is the
**reaching** hierarchy.

---

## 6. Visualization

| Channel | Content |
|---------|---------|
| `sw_grasp_object_pose` | Live object SE(3) (env) |
| `sw_grasp_candidates` | Antipodal candidate points |
| `sw_grasp_fingertips` | Tip cur / tgt |
| `sw_grasp_ft_paths` | Heuristic+SQP fingertip polylines (reaching) |

Straight gray tip→tgt lines in Viser are **not** the LVF paths; the thick
colored polylines are `_ft_paths`.

---

## 7. Key files & knobs

| File | Role |
|------|------|
| `controller/ThreeFingerReactiveGrasping.py` | Fields, SQP paths, IKQP, FSM, fault latch |
| `utils/planning/heuristics.py` | Via-point path init |
| `utils/planning/path_sqp.py` | Multi-finger SQP path polish |
| `utils/shape_primitives/PredefinedObj.py` | Surface distance field |
| `env/flexiv_arm_5F_hand/FlexivArmHandGraspEnv.py` | Object pose + `o` respawn |
| `configs/flexiv_arm_5F_hand/controllers/three_finger_reactive_grasp.yaml` | Primary knobs |

Useful YAML:

- Paths: `num_lvf_points`, `num_path_sqp_iterations`, `path_sqp_savgol_window`,
  `heuristic_via_length`, `collision_margin_to_object_lvf`
- QP: `plan_horizon`, `collision_margin_to_object_qp`, `env_collision_margin`,
  `ikqp_reg_arm` / `ikqp_reg_hand` / `ikqp_reg_qd_rate`,
  `ikqp_max_escape_rate`, `joint_velocity_limit`
- LVF speed: `linear_velocity_field_v` / `_eps`
- Closing: `lift_delay_s`, `lift_speed`, `lift_arm_eps`,
  `squeeze_width_multiple`

Increase `ikqp_reg_arm` (e.g. $10$–$50$) for more conservative arm motion.

---

## 8. Takeaways

1. Guide fingertips with a **clearance-aware path** (heuristic + SQP), not a
   straight attractor into concavities.
2. Track in joint space with a **weighted QP** so collisions / limits are hard
   constraints.
3. Gate orientation / gripper fields with **tanh** priorities.
4. Keep planning slower than control; use a **latest-only** plan mailbox.
5. Treat IKQP infeasibility as a **fault**: zero velocity and latch until the
   operator resumes — do not trust OSQP’s certificate, and do not immediately
   accept a max-velocity “recovery” solve.

For proofs, experiments, and the original two-finger TO formulation, see the
paper and [reactivegrasp.github.io](https://reactivegrasp.github.io/).
