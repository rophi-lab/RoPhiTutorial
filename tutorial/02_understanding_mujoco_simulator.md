# Understanding the MuJoCo Simulator

This note walks through the Flexiv MJCF used by `run_env.py` — mainly
`assets/scene/flexiv_arm/hand_default.xml` (scene + actuators) and the included
`Rizon4_hand_robot.xml` (robot tree) — and how the physics timestep interacts
with the env loop.

---

## 1. Reading the `.xml`

An MJCF file is not just a mesh list. It defines the **kinematic tree**, the
**inertial** properties that make dynamics real, joint **dissipation**, contact
geometry, and actuators. MuJoCo compiles that into `mjModel` / `mjData` and
integrates the equations of motion each `mj_step`.

### Kinematic and inertial parameters

**Kinematics** live in the nested `<body>` / `<joint>` tree:

- Each `<body>` has a pose relative to its parent (`pos`, `quat`).
- Each `<joint>` has `axis`, `range`, and (often) `actuatorfrcrange`.
- Forward kinematics of the chain is fully determined by joint angles `q`.

Example (shoulder yaw) from `Rizon4_hand_robot.xml`:

```xml
<body name="link1" pos="0 0 0.155" quat="0 0 0 1">
  <inertial pos="0.0002 0.004 0.1451" ... mass="3.85"
            diaginertia="0.0308011 0.0307617 0.00763713"/>
  <joint name="joint1" axis="0 0 1" range="-2.8798 2.8798"
         actuatorfrcrange="-123 123" damping="1.0" frictionloss="0.5"/>
  ...
</body>
```

**Inertia** is *not* inferred from the visual mesh here. Each body carries an
explicit `<inertial>` (`mass`, CoM `pos` / `quat`, `diaginertia`). Collision
geoms use `density="0"` in the `contact` class so they do **not** double-count
mass — dynamics come only from `<inertial>`.

Wrong mass / CoM / inertia → wrong gravity torque, wrong acceleration under the
same command. Matching these to the real robot (or CAD) is the first step toward
sim-to-real.

### Damping and frictionloss

MuJoCo joints can dissipate energy in two complementary ways:

| Attribute | Model (approx.) | Feel |
|-----------|-----------------|------|
| `damping` | viscous: $\tau_d = -b\,\dot{q}$ | soft resistance proportional to speed |
| `frictionloss` | Coulomb-like dry friction (constant opposing motion) | “sticky” breakaway / slow crawl |

In this scene:

- Arm joints set `damping` / `frictionloss` per joint (proximal joints heavier:
  e.g. J1–J2 use `1.0` / `0.5`; distal use `0.5` / `0.2`).
- Finger joints use the `XM335` default class:
  `damping="0.01"` `frictionloss="0.02"` plus armature (below).

These are **plant** parameters. They sit on top of whatever the controller
sends. If you also put large `kd` in the PD command, you are damping twice
(controller + plant).

### Armature

`armature` adds reflected rotor inertia on the joint DoF. Conceptually it
increases the diagonal of the joint-space mass matrix:

$$
M(q)\,\ddot{q} \;\leftarrow\; \bigl(M_{\mathrm{links}}(q) + \mathrm{diag}(I_a)\bigr)\,\ddot{q}
$$

- Arm defaults: `armature="0"` in the `main` class (link inertia only).
- Finger `XM335` class: `armature="0.002418"` — small motors, non-negligible
  relative to light finger links.

Larger armature → slower acceleration for the same torque, smoother / less
jittery fingers, and a closer match to geared motors.

### How these add up to the physics equation

MuJoCo integrates the multibody equations (schematic joint-space form):

$$
\bigl(M(q) + \mathrm{diag}(I_a)\bigr)\,\ddot{q}
+ C(q,\dot{q})\,\dot{q}
+ g(q)
+ \tau_{\mathrm{damp}}(\dot{q})
+ \tau_{\mathrm{fric}}(\dot{q})
+ \tau_{\mathrm{contact}}(q,\dot{q})
= \tau_{\mathrm{act}}
$$

| Term | Comes from in the XML / sim |
|------|-----------------------------|
| $M(q)$ | body `<inertial>` + kinematic tree |
| $I_a$ | joint `armature` |
| $C,\,g$ | same inertias + gravity (`option` / world) |
| $\tau_{\mathrm{damp}}$ | joint `damping` |
| $\tau_{\mathrm{fric}}$ | joint `frictionloss` |
| $\tau_{\mathrm{contact}}$ | colliding `geom`s + contact solver params |
| $\tau_{\mathrm{act}}$ | `<actuator>` motors ← `mjData.ctrl` from the platform |

In this stack the platform writes

$$
\tau_{\mathrm{act}}
= K_p(q_{\mathrm{des}}-q) + K_d(\dot{q}_{\mathrm{des}}-\dot{q}) + \tau_{\mathrm{ff}}
$$

into `ctrl[]` each control tick (`SimFlexivArm5FHand`). With
`gravity_comp: false` on the plant, gravity must live in $\tau_{\mathrm{ff}}$
(as `GravCompControl` does via Pinocchio) or the arm falls.

### Contact model (short)

Contacts are soft constraints solved each step (not hard analytic impacts). Key
pieces in this XML:

- **Who collides:** `contype` / `conaffinity` bitmasks. The `contact` class uses
  `1/1`; `no_contact` uses `0/0`. Floor / walls in `hand_default.xml` use the
  default `1/1`, so they hit the robot.
- **Friction:** `friction="slide spin roll"` on geoms (e.g. walls `0.5 0.02 0.02`,
  robot contact class `1.0 0.02 0.0001`).
- **Compliance:** `solref` / `solimp` on the `contact` class set time-constant /
  impedance of the contact — softer contacts penetrate more but are stabler;
  stiffer contacts look “harder” but need a small enough timestep.
- **Exclusions:** `<contact><exclude .../></contact>` turns off pairs that
  intersect by construction (palm vs second finger phalanx) while leaving real
  self-collisions on.

Parent–child pairs are already filtered by MuJoCo (`filterparent`).

### Visual mesh vs collision mesh

Each link typically has **two** geoms sharing the same mesh file in this robot
XML, but different classes:

| Class | `contype` / `conaffinity` | `group` | Role |
|-------|---------------------------|---------|------|
| `no_contact` | `0` / `0` | `1` | Drawing only — what you see in the viewer |
| `contact` | `1` / `1` | `0` | Physics contacts — what the solver feels |

```xml
<geom class="no_contact" type="mesh" mesh="link3"/>
<geom class="contact"    type="mesh" mesh="link3"/>
```

Why split them?

- You can later swap the contact geom for a **convex hull / primitive** while
  keeping a detailed visual mesh (faster, more stable contacts).
- Viewer geom groups let you hide collision shapes (`group 0`) or visuals
  (`group 1`) independently.
- Inertia stays on `<inertial>` (`density="0"` on both), so mesh choice does not
  change mass.

**Important:** what you *see* is not always what *collides*. If the visual looks
clear of a wall but the arm still “hits” early, the contact mesh / wall pose is
the truth — same idea as the Viser collision overlay vs the pretty URDF.

---

## 2. Effect of the physics simulation step

### What `timestep` is

In `hand_default.xml`:

```xml
<option timestep="0.001" integrator="implicitfast"/>
```

That is the integration step $h = 0.001\,\mathrm{s}$ (1 kHz). Each
`mj.mj_step` advances simulated time by one `timestep`.

The env also sets this from config:

```python
# MujocoBaseEnv
self.mj_model.opt.timestep = self._sim_dt   # typically 1 / sim_freq
```

with `sim_freq: 1000` → `_sim_dt = 0.001` in `configs/.../env/default.yaml`.

### Control tick vs physics tick

One env control cycle does:

```python
for _ in range(self._sim_steps_per_control):
    ...
    mj.mj_step(...)
```

So simulated time advanced per control command is

$$
\Delta t_{\mathrm{ctrl}} = N_{\mathrm{steps}} \cdot h
$$

With the default `sim_steps_per_control: 1` and $h = 1\,\mathrm{ms}$, control and
physics are both 1 kHz — commands change every physics step.

If you set `sim_steps_per_control: 10` at the same $h$, each received command
is held for 10 ms of sim time (zero-order hold). That is cheaper if the
controller is slow, but the plant sees a coarser torque schedule.

### Why the step size matters

| Smaller $h$ | Larger $h$ |
|-------------|------------|
| More accurate contacts / stiff joints | Faster wall-clock (fewer steps / second of sim) |
| Better match to high-rate real control | More penetration, jitter, possible instability |
| More CPU for the same simulated second | Effective plant looks softer / laggy |

Contacts and high `solref` stiffness are the usual reason a step that “looks
fine” for free motion blows up when the hand hits an object: the contact
bandwidth and $h$ fight each other.

`integrator="implicitfast"` is a good default for stiff actuators / contacts at
1 kHz; switching integrator or raising $h$ without retuning `damping` /
contact `solref` often changes behavior a lot.

### Real-time vs sim-time

`env_mode: sim_real_time` paces the loop so one second of sim ≈ one second of
wall clock (sleeping if the CPU is ahead). `sim_sim_time` runs as fast as
possible — useful for batch rollouts, but LCM consumers and humans see a
speed-up.

Viewer updates (`view_freq: 30`) are **decoupled** from physics: you can
integrate at 1 kHz and only sync the GUI at 30 Hz.

### Practical checklist for this repo

1. Keep `timestep` / `sim_freq` at **1 ms / 1 kHz** unless you have a reason to
   change both together.
2. Prefer `sim_steps_per_control: 1` in real-time demos so publish rates line up
   with `*_joint_meas_freq`.
3. If contacts explode, first try slightly softer `solref` / more joint
   `damping` before jumping the timestep.
4. Retune `damping` / `frictionloss` / `armature` when comparing to hardware —
   they absorb a lot of sim-to-real gap that controllers alone cannot.

---

## Files to open next

| File | What to look at |
|------|-----------------|
| `assets/scene/flexiv_arm/hand_default.xml` | `option timestep`, walls, actuators, contact excludes |
| `assets/scene/flexiv_arm/Rizon4_hand_robot.xml` | defaults (`contact` / `no_contact` / `XM335`), inertias, joint damping |
| `configs/flexiv_arm_5F_hand/env/default.yaml` | `sim_freq`, `sim_steps_per_control`, `view_freq` |
| `env/MujocoBaseEnv.py` | `mj_step` loop, viewer sync |


## System identification in practice

1. **Kinematics and inertia** usually start from the manufacturer (URDF / datasheet / CAD). Those values are a good default, but they can be wrong or incomplete (payload, mounts, cable routing, missing motor inertia).
2. When the controller assumes a dynamics model (gravity compensation, inverse dynamics, collision-aware planning), **mismatched inertia shows up as residual gravity torque, tracking error, or unstable contact**. That is when you need system identification — estimate mass / CoM / inertia (and sometimes joint offsets) from motion or torque data.
3. **`damping` and `frictionloss` are almost never in the datasheet.** Identify them from free-motion / coast-down / low-speed tracking experiments; they often explain more of the sim-to-real gap than small inertia errors.

In this repo, plant dissipation lives in the MJCF; the controller’s `kd` / `arm_damping` are separate. Keep that split in mind when you identify and when you retune.

## Exercises

1. **Joint dissipation.** In `Rizon4_hand_robot.xml`, change arm `damping` and `frictionloss` (try zero vs 2× current values). With grav-comp running, compare how the arm settles, drifts, and responds to small disturbances.
2. **Physics step.** Change `sim_freq` / `timestep` (e.g. 1 kHz → 500 Hz or 2 kHz) and, separately, `sim_steps_per_control`. Watch free motion and contacts: larger $h$ often looks softer or jittery; mismatched control/physics rates change how “held” each torque command feels.
