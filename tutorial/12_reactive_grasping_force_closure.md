# Reactive grasping with real-time force-closure closing

Tutorials [10](10_reactive_grasping.md) and [11](11_mujoco_grasp_eval.md) each solve half of a useful pipeline:

| Tutorial | Strength | Gap |
|----------|----------|-----|
| **10** | Reactive **reach** to a live object | Closing is a tutorial squeeze (hand PD / tip attractors), not force-aware |
| **11** | Real-time **force-closure** hold | Hand is already on the object; no reach |

This lesson **glues them together**: reach exactly like tutorial 10; once the FSM enters closing, squeeze and lift with the live distance-field + force-closure stack from tutorial 11.

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

Start all three → `v`. During reach you should see the same tip paths as tutorial 10. After auto-close (or `c`), magenta / cyan contact arrows appear and the hand holds via $\tau$ while the arm later lifts.

### What you should observe

Reaching should feel like tutorial 10 (calm, sticky targets). Closing should **not** fight itself: tip attractors are off by default so they do not cancel the force QP. If the QP briefly fails, the hand may soften toward `q_hand_close` instead of exploding.

### Rates (three clocks)

| Loop | Rate | Job |
|------|------|-----|
| Sim / velocity control | 1 kHz | Track plan $\dot q$ + gravity (+ hand $\tau$ in closing) |
| Reach plan | 50 Hz | LVF + IKQP (same as tut 10) |
| Force QP | `force_ctrl_freq` (50 Hz) | UDF + force-closure; hold last $\tau$ between solves |

### LCM

Everything from [10](10_reactive_grasping.md), plus contact overlays:

| Channel | Content |
|---------|---------|
| `sw_contact_force_arrows` | Magenta $[p\,|\,f]$ from the QP |
| `sw_contact_normal_arrows` | Cyan $[p\,|\,n]$ |

Object pose is CAD→world on `sw_grasp_object_pose` (no OBB conversion like the hand-only eval).

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
  + \tau_{\mathrm{approach}}
  \;(\;+\text{soft PD fallback if QP infeasible}\;).
$$

Contact detection, attraction, tip approach, and the lifted force-closure QP match [11](11_mujoco_grasp_eval.md). Jacobians come from the full 27-DoF model, but only **hand columns** ($7{:}27$) are used — contact wrenches must not yank the arm.

On OSQP failure: decay the previous $\tau_{\mathrm{QP}}$, add light PD toward `q_hand_close`, clamp limits, and optionally retry with softened constraints (e.g. disable achievable-force, lower $f_{n,\min}$).

### Lift while still holding

After `lift_delay_s`:

$$
\dot q_{\mathrm{arm}}
  =\texttt{VF}(q_{\mathrm{arm}},\,q_{\mathrm{arm}}^{\mathrm{nom}}),
\qquad
\dot q_h=0,
$$

while $\tau_{\mathrm{hand}}$ continues from **live** contacts. Lift finishes when $\|q_{\mathrm{arm}}-q_{\mathrm{arm}}^{\mathrm{nom}}\|$ is below `lift_arm_eps`.

### FSM (same labels, richer closing)

| State | Behavior |
|-------|----------|
| **reaching** | Tut 10 hierarchy → close near $\sim 1\,\mathrm{cm}$ mid-tip error |
| **closing · squeeze** | Force-closure @ `force_ctrl_freq` for `lift_delay_s` |
| **closing · lift** | Arm VF home; hand holds via $\tau_{\mathrm{QP}}$ |
| **lost grasp** | Mid error $\gtrsim 5\,\mathrm{cm}$ → re-reach |

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/ThreeFingerReactiveForceClosureGrasping.py` | Subclass: inherit reach; own closing force stack |
| `controller/ThreeFingerReactiveGrasping.py` | Reaching + FSM shell |
| `controller/MjEvalGraspControl.py` | Reference algorithms for UDF / QP |
| `utils/grasping/distance_field.py`, `force_gen.py` | Distance field + OSQP |
| `configs/.../three_finger_reactive_force_closure.yaml` | Reach knobs + `grasp_squeeze_params_generator_config` |

**Startup order matters:** the UDF is built **before** the plan thread starts, so reaching never races an unfinished distance field. Closing uses a dedicated Pinocchio `_pin_data_force`, rate-limits the QP, and publishes contact arrows for Viser.

**If closing is unstable:** keep tip attractors off, leave `enable_achievable_force: false`, and lower `min_fn` before raising force gains. Reaching issues are still tuned with `ikqp_reg_arm` as in tutorial 10.
