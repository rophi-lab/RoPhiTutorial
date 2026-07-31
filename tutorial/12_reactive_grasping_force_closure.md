# Reactive grasping with real-time force-closure closing

Tutorials [10](10_reactive_grasping.md) and [11](11_mujoco_grasp_eval.md) each solve half of a useful pipeline:

| Tutorial | Strength | Gap |
|----------|----------|-----|
| **10** | Reactive **reach** to a live object | Closing is a tutorial squeeze (hand PD / tip attractors), not force-aware |
| **11** | Real-time **force-closure** hold | Hand is already on the object; no reach |

This lesson **glues them together**: reach exactly like tutorial 10; once the FSM enters closing, squeeze and lift with the live distance-field + force-closure stack from tutorial 11 — and now the QP also **resists the object's weight**.

---

## 1. Run & LCM communication

Same env and visualizer as tutorial 10 — only the controller config changes. Warp is recommended for the GPU UDF.

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/grasping.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/three_finger_reactive_force_closure.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/grasping.yaml
```

| Key | Where | Action |
|-----|--------|--------|
| `v` | controller | Start **reaching** |
| `c` | controller | Jump straight to **closing** |
| `g` / `d` | controller | Grav-comp / home |
| `o` | env | Respawn object |
| `p` / `q` | controller | Pause / quit |

### Suggested first run

Start all three → `v`. During reach you should see the same tip paths as tutorial 10. After auto-close (or `c`), magenta / cyan contact arrows appear and the hand holds via $\tau$ while the arm later lifts. Watch the controller log for a one-shot line like `object physics from env: mass=...` — that means the QP knows the object's mass.

### What you should observe

Reaching should feel like tutorial 10 (calm, sticky targets). Closing should **not** fight itself: tip attractors are off by default so they do not cancel the force QP. With object mass enabled, contact forces should **push up against gravity** so the object is less likely to slip out during squeeze / lift. If the force OSQP fails, the robot freezes motion ($\dot q=0$) and keeps only robot gravity compensation until the QP recovers — no stale force or PD bang.

### Rates (three clocks)

| Loop | Rate | Job |
|------|------|-----|
| Sim / velocity control | 1 kHz | Track plan $\dot q$ + gravity (+ hand $\tau$ in closing) |
| Reach plan | 50 Hz | LVF + IKQP (same as tut 10) |
| Force QP | `force_ctrl_freq` (50 Hz) | UDF + force-closure (+ gravity wrench); hold last $\tau$ between solves |

### LCM

Everything from [10](10_reactive_grasping.md), plus:

| Channel | Content |
|---------|---------|
| `sw_grasp_object_physics` | NamedVec `physics_params`: $[m,\,r_{\mathrm{com}}^{\mathrm{CAD}},\,g^{W},\,\ldots]$ from MuJoCo |
| `sw_contact_force_arrows` | Magenta $[p\,|\,f]$ from the QP |
| `sw_contact_normal_arrows` | Cyan $[p\,|\,n]$ |

Object pose is CAD→world on `sw_grasp_object_pose` (no OBB conversion like the hand-only eval). Mass / COM / world gravity come from the env so the controller matches the simulated body.

```
                 plan 50 Hz                         ctrl 1 kHz
        ┌──────────────────────────┐       ┌──────────────────────────┐
reach → │ LVF + IKQP → (qd, τ≈0)   │──────►│ g(q) + vel damping       │
close → │ arm lift VF → (qd_arm,0) │──────►│ + τ_hand (force @ 50 Hz) │
        └──────────────────────────┘       └──────────────────────────┘
```

---

## 2. Ideas

### Reaching = tutorial 10

Sticky antipodal selection, clearance-aware tip paths, orientation / gripper fields, collision-aware IKQP. On OSQP failure: $\dot q=0$ **that plan only**. Non-closing `_update` calls the parent implementation so reach stays aligned with tutorial 10.

### Why closing must change the command mix

In tutorial 10, closing still thinks in **velocities** (tip attractors + hand close). Force-closure thinks in **contact forces → joint torques**. If both fight, the hand “goes crazy.” So in closing we:

1. set hand $\dot q_h=0$ (squeeze with torque only),
2. leave tip attractors **off** by default (`closing_tip_attractors: false`),
3. run the force stack at a modest rate (50 Hz) and clamp torques.

### Force regulation (tutorial 11 on the hand columns)

Each force tick, using the object UDF and hand surface samples:

$$
\tau_{\mathrm{hand}}
  = \tau_{\mathrm{QP}}
  + \tau_{\mathrm{attract}}
  + \tau_{\mathrm{approach}}.
$$

Contact detection, attraction, tip approach, and the lifted force-closure QP match [11](11_mujoco_grasp_eval.md). Jacobians come from the full 27-DoF model, but only **hand columns** ($7{:}27$) are used — contact wrenches must not yank the arm.

On force-closure OSQP failure (after one softened retry): **$\dot q=0$**, hand force terms cleared, and $\tau$ falls back to **robot** gravity compensation $g(q)$ only until the next successful solve.

### Object mass → gravity wrench (why the bowl stops dropping)

A pure force-closure cost drives the **net contact wrench** toward zero. That is a good squeeze in free fall, but a real object feels

$$
w_g
  = \begin{pmatrix} f_g \\ r_{\mathrm{com}} \times f_g \end{pmatrix},
\qquad
f_g = m\,g,
$$

expressed in the **CAD frame** (same frame as the contact map $G$ / `Jo`):

$$
f_g^{\mathrm{CAD}} = R_{\mathrm{CAD}\to W}^{\top}(m\,g^{W}),
\qquad
\tau_g^{\mathrm{CAD}} = r_{\mathrm{com}}^{\mathrm{CAD}} \times f_g^{\mathrm{CAD}}.
$$

The env publishes $(m,\,r_{\mathrm{com}}^{\mathrm{CAD}},\,g^{W})$ on `sw_grasp_object_physics`. The QP then minimizes something like

$$
\|G f + w_g\|^{2}
$$

(subject to friction cones, $f_n\ge 0$, torque limits, …), so contacts produce a net wrench $\approx -w_g$ — **holding the weight** instead of only “closing hard.”

To make that feasible, the total normal-force floor is also raised:

$$
\sum f_n
  \ge
  \max\bigl(f_{n,\min},\;
    \alpha\, m\|g\|\bigr),
$$

with `min_fn_gravity_scale` $=\alpha$ (default $1$).

| Knob (`grasp_squeeze_params_generator_config`) | Role |
|------------------------------------------------|------|
| `enable_gravity_wrench` | On/off for $w_g$ in the QP (default `true`) |
| `min_fn_gravity_scale` | Scale for the weight-based $\sum f_n$ floor |
| `object_mass` | Optional override if LCM physics is missing (`0` → use env) |
| `gravity` | World gravity vector used before / with physics |

### Lift while still holding

After `lift_delay_s`:

$$
\dot q_{\mathrm{arm}}
  =\texttt{VF}(q_{\mathrm{arm}},\,q_{\mathrm{arm}}^{\mathrm{nom}}),
\qquad
\dot q_h=0,
$$

while $\tau_{\mathrm{hand}}$ continues from **live** contacts (still including $w_g$). Lift finishes when $\|q_{\mathrm{arm}}-q_{\mathrm{arm}}^{\mathrm{nom}}\|$ is below `lift_arm_eps`.

### FSM (same labels, richer closing)

| State | Behavior |
|-------|----------|
| **reaching** | Tut 10 hierarchy → close near $\sim 1\,\mathrm{cm}$ mid-tip error |
| **closing · squeeze** | Force-closure (+ gravity wrench) @ `force_ctrl_freq` for `lift_delay_s` |
| **closing · lift** | Arm VF home; hand holds via $\tau_{\mathrm{QP}}$ against $w_g$ |
| **lost grasp** | Mid error $\gtrsim 5\,\mathrm{cm}$ → re-reach |

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/ThreeFingerReactiveForceClosureGrasping.py` | Subclass: inherit reach; closing force stack + $w_g$ |
| `controller/ThreeFingerReactiveGrasping.py` | Reaching + FSM shell |
| `env/flexiv_arm_5F_hand/FlexivArmHandGraspEnv.py` | Object pose + MuJoCo mass / COM / gravity publish |
| `utils/grasping/distance_field.py`, `force_gen.py` | UDF + OSQP (`gravity_wrench` argument) |
| `configs/.../three_finger_reactive_force_closure.yaml` | Reach knobs + `grasp_squeeze_params_generator_config` |
| `configs/.../env/grasping.yaml` | `object_physics_channel: sw_grasp_object_physics` |

**Startup order matters:** the UDF is built **before** the plan thread starts, so reaching never races an unfinished distance field. Closing uses a dedicated Pinocchio `_pin_data_force`, rate-limits the QP, and publishes contact arrows for Viser.

**If the object still drops:** confirm the physics log printed a positive mass; keep `enable_gravity_wrench: true`; try raising `min_fn_gravity_scale` (e.g. $1.2$–$1.5$) or `mu`. **If closing is unstable:** keep tip attractors off, leave `enable_achievable_force: false`, and do not crank `min_fn` far above what the finger torque limits can produce. Reaching issues are still tuned with `ikqp_reg_arm` as in tutorial 10.
