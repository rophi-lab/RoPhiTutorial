# Reactive three-finger grasping

This is the first “full story” grasping tutorial: the hand **chooses** where to touch an object, **plans** short fingertip paths that stay clear of the geometry, and **tracks** those wishes with a collision-aware joint velocity QP — then squeezes and lifts.

Paper: Lee et al., ICRA 2026 · [arXiv:2509.01044](https://arxiv.org/abs/2509.01044) · [reactivegrasp.github.io](https://reactivegrasp.github.io/)

If [06](06_inverse_kinematics_quadratic_program.md) was “one tip chases a keyboard target,” this is “three tips chase a living grasp on a moving object,” with more care about **which** grasp and **how** they approach.

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/grasping.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/three_finger_reactive_grasp.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/grasping.yaml
```

| Key | Where | Action |
|-----|--------|--------|
| `o` | env | Respawn the object at a new XY |
| `r` | env | Full reset |
| `v` | controller | Start **reaching** |
| `c` | controller | Force **closing** early |
| `g` / `d` | controller | Grav-comp / home to `default_q` |
| `p` / `q` | controller | Pause / quit |

### Suggested first run

Start all three processes → press `v` on the controller. Watch the fingertip paths and tip targets in Viser. When tips are close, the FSM should squeeze, then lift. Press `o` to try another object placement.

### What you should observe

- Candidate contact points appear on the object.
- Three tip targets settle (they should **not** flicker every frame — selection is sticky).
- Colored polylines show approach paths.
- The arm/hand moves smoothly toward the grasp; if the QP briefly fails, motion may pause for one plan cycle ($\dot q=0$) and then continue — it should **not** slam.

### Two clocks (keep this picture)

| Loop | Rate | Job |
|------|------|-----|
| **Control** | 1 kHz | Physics companion: track latest $\dot q_{\mathrm{des}}$, add gravity |
| **Plan** | 50 Hz | Pick grasp, build velocity fields, solve IKQP |

Planning is the “thinking”; control is the “reflex.” The plan mailbox keeps only the **latest** command (size 1).

### LCM (high level)

| Channel | Direction | Role |
|---------|-----------|------|
| Arm / hand `*_joint_meas` | env → ctrl / viz | State |
| `sw_flexiv_arm_hand_joint_ctrl` | ctrl → env | 27-DoF command |
| ColInfo (robot + static) | env → ctrl / viz | FCL shapes |
| `sw_grasp_object_pose` | env → ctrl / viz | Object CAD→world pose |
| `sw_grasp_candidates` / `sw_grasp_fingertips` / `sw_grasp_ft_paths` | ctrl → viz | Grasp overlays |

```
Env -- joints, ColInfo, T_obj --> Controller -- JointCtrl --> Env
                                      |
                                      +-- candidates, tips, paths --> Viser
```

---

## 2. Ideas

### Why a hierarchy?

Solving a long-horizon joint-space plan for 27 DoF every few milliseconds is heavy. Instead:

| Layer | Lives in | Intuition |
|-------|----------|-----------|
| **High** | Tip positions, tip orientations, hand joints | “Where should my fingers go?” → **velocity fields** |
| **Low** | Full $q\in\mathbb{R}^{27}$ | “Obey those wishes, but don’t hit walls” → **one-step IKQP** |

### Choosing a grasp (and sticking to it)

From the object pose $T_{\mathrm{obj}}$ and a CAD library of **antipodal** contact pairs (plus flipped pairs), pick a grasp. Re-picking the absolute best pair every plan tick can make tip targets **jump**. The controller therefore uses **sticky selection** with hysteresis: keep the current grasp unless another is clearly better for a while.

During reaching, contacts are widened (approach envelope). During closing, a `squeeze_width_multiple` brings them in. The “right” contact is split into **index** and **middle** along a direction orthogonal to the reach axis so three tips share the load.

### Fingertip linear velocity field (LVF)

For each tip $i$, build a clearance-aware polyline $c_i(s)$:

1. heuristic via-points,
2. multi-finger SQP that reduces path curvature while keeping distance to the object $\ge \epsilon_{\mathrm{LVF}}$,
3. optional Savitzky–Golay smoothing.

Then command a speed along the path:

$$
\dot x_{i,\mathrm{des}}
  = v(\|x_i-x_i^\star\|)\,
    \frac{c_i(s_1)-c_i(s_0)}{\|c_i(s_1)-c_i(s_0)\|}.
$$

Far from the goal tip → move faster; near → slow down. The path, not a straight line, keeps fingertips from diving into the mesh.

### Orientation and hand posture fields

$$
\omega_{i,\mathrm{des}} \propto \hat n_i \times \hat n_i^\star,
\qquad
\dot q_{h,\mathrm{des}} = \texttt{VF}(q_h,\,q_h^{\mathrm{nom}}).
$$

Weights use soft $\tanh$ gates: orientation and gripper closing matter more as the tips get close. That avoids “twisting early” while still far away.

### The IKQP (low-level tracker)

$$
\begin{aligned}
\min_{\dot q}\quad
&\sum_k w_k\|J_k\dot q-\dot x_k\|^2
 + \|\dot q\|_{W_{\mathrm{reg}}}^2
 + \lambda_{\mathrm{rate}}\|\dot q-\dot q_{\mathrm{prev}}\|^2 \\
\text{s.t.}\quad
&\text{velocity \& short-horizon joint limits},\\
&\text{FCL environment escape},\\
&\text{object clearance escape (rate-capped)}.
\end{aligned}
$$

**Regularization matters.** If arm regularization (`ikqp_reg_arm`) is too small, tip tasks dominate and the shoulder can race toward velocity limits. Values around $1.0$ (with a bit of rate regularization) keep reaching calm.

**Failure policy (important):** if OSQP is unsolved, inconsistent, or returns a non-finite / over-limit $\dot q$, command **$\dot q=0$ for that plan only** and try again next cycle. There is **no permanent fault latch**. Never send an infeasibility “certificate” vector to the robot.

### Finite-state machine

| State | Behavior |
|-------|----------|
| **reaching** | Full hierarchy. Auto-close when mid-tip error is $\lesssim 1\,\mathrm{cm}$. |
| **closing · squeeze** | Tip attractors + hand-close torque for `lift_delay_s`. |
| **closing · lift** | Arm velocity field toward `default_q`; hand stays closed. |
| **lost grasp** | If mid error grows past $\sim 5\,\mathrm{cm}$ while closing → go back to reaching. |

Tutorial [12](12_reactive_grasping_force_closure.md) keeps this reach, but replaces the squeeze with a live force-closure QP.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/ThreeFingerReactiveGrasping.py` | Plan thread, sticky grasp, LVF, IKQP, FSM |
| `env/flexiv_arm_5F_hand/FlexivArmHandGraspEnv.py` | Object pose, `o` respawn |
| `utils/planning/heuristics.py` / `path_sqp.py` | Path init + polish |
| `utils/shape_primitives/PredefinedObj.py` | Object distance field for clearance |
| `configs/.../three_finger_reactive_grasp.yaml` | Gains, LVF, IKQP, FSM thresholds |

Control loop: gravity (optional friction) + velocity damping on the latest plan. Plan and control use **separate** Pinocchio `Data` so threads do not race.

**If reaching looks aggressive:** raise `ikqp_reg_arm` (try $1$–$50$) or lower `linear_velocity_field_v` before changing the math.
