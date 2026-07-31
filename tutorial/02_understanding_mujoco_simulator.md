# Understanding the MuJoCo simulator

This tutorial explains the **physics side** of the stack: what MuJoCo integrates, how your LCM command becomes torque, and why timing matters. You do not write new code here — you learn to *read* the plant that every controller talks to.

If [01](01_quick_start.md) was “who talks to whom,” this chapter is “what is being controlled.”

---

## 1. Run & LCM communication

Use the same three commands as the quick start:

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/default.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/grav_comp.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/default.yaml
```

No new channels. While it runs, keep this picture in mind:

- **Controller → env:** `sw_flexiv_arm_hand_joint_ctrl` (desired motion + feedforward torque + gains).
- **Env → controller:** joint measurements on the `*_joint_meas` channels.

### Timing knobs in the env YAML

| Knob | Intuition |
|------|-----------|
| `sim_freq: 1000` | Physics tries to take 1000 steps per simulated second ($h=1\,\mathrm{ms}$). |
| `sim_steps_per_control` | How many physics steps share **one** held motor command. |
| `env_mode: sim_real_time` | Do not run faster than wall-clock (good for interactive demos). |
| `env_mode: sim_sim_time` | Run as fast as the CPU allows (good for batch tests). |

---

## 2. Ideas (the plant model)

### Coordinates

The robot’s configuration is a vector of joint angles

$$
q \in \mathbb{R}^{n}
\qquad (n=27 \text{ for arm+hand}).
$$

Velocities are $\dot q$, accelerations $\ddot q$. MuJoCo advances $(q,\dot q)$ through time using the equations of motion below.

### Equations of motion (intuition first)

Newton’s law for a multi-joint robot is more than $F=ma$. Inertia depends on pose, joints couple through Coriolis forces, gravity pulls on every link, friction dissipates energy, and contacts push when you hit the table.

A standard form is:

$$
\bigl(M(q)+\mathrm{diag}(I_a)\bigr)\ddot q
  + C(q,\dot q)\dot q
  + g(q)
  + \tau_{\mathrm{damp}}
  + \tau_{\mathrm{fric}}
  + \tau_{\mathrm{contact}}
  = \tau_{\mathrm{act}}.
$$

| Symbol | Plain-language meaning |
|--------|-------------------------|
| $M(q)$ | Inertia matrix (“how hard it is to accelerate each joint”) |
| $I_a$ | Rotor / gearbox inertia (`armature` in MJCF) |
| $C\dot q$ | Velocity-dependent coupling (Coriolis / centrifugal) |
| $g(q)$ | Gravity torques (exactly what grav-comp cancels) |
| $\tau_{\mathrm{damp}}$ | Viscous friction $\approx -b\dot q$ |
| $\tau_{\mathrm{fric}}$ | Coulomb-like friction (`frictionloss`) |
| $\tau_{\mathrm{contact}}$ | Soft contact / constraint forces from MuJoCo |
| $\tau_{\mathrm{act}}$ | **What your controller asks the motors to produce** |

You do not need to derive this on day one. You *do* need to remember: **the controller never sets $\ddot q$ directly**; it sets $\tau_{\mathrm{act}}$ (through the actuator model), and physics decides the motion.

### How `JointCtrl` becomes $\tau_{\mathrm{act}}$

Inside the RoPhiTutorial platform, the actuator law is a familiar PD + feedforward form:

$$
\tau_{\mathrm{act}}
  = K_p\,(q_{\mathrm{des}}-q)
  + K_d\,(\dot q_{\mathrm{des}}-\dot q)
  + \tau_{\mathrm{ff}}.
$$

| Term | When students use it |
|------|----------------------|
| $K_p(q_{\mathrm{des}}-q)$ | Spring toward a joint pose (impedance, tracking, …) |
| $K_d(\dot q_{\mathrm{des}}-\dot q)$ | Damper / velocity tracking |
| $\tau_{\mathrm{ff}}$ | Model-based torque (gravity, Coriolis, contact forces, …) |

Special cases you will see often:

- **Pure gravity compensation:** $K_p=K_d=0$, $\tau_{\mathrm{ff}}\approx g(q)$.
- **Velocity field tracking:** $K_p=0$, $K_d>0$, $\dot q_{\mathrm{des}}$ from a planner, $\tau_{\mathrm{ff}}\approx g(q)$.

### Double gravity (a common bug)

If the **platform** also adds $g(q)$ internally (`gravity_comp: true`) *and* the controller sends $g(q)$ in $\tau_{\mathrm{ff}}$, the robot gets **$2g(q)$** and may leap upward. In these tutorials, leave platform gravity compensation **off** and let the controller own $g(q)$.

### Contacts vs planning collision models

MuJoCo contacts (meshes, `contype` / `conaffinity`, `solref` / `solimp`) decide **physical** bouncing and friction.

Separately, later tutorials publish **simple FCL shapes** for **planning** ([04](04_publish_collision_geom.md)). Those shapes are *not* required to match MuJoCo’s contact meshes. One is for physics; one is for “stay this far from the wall in the optimizer.”

---

## 3. How it is implemented

| Piece | Where to look | Why it matters |
|-------|---------------|----------------|
| Scene MJCF | `assets/scene/flexiv_arm/hand_default.xml` | Floor, lights, includes the robot |
| Robot MJCF | `assets/scene/flexiv_arm/Rizon4_hand_robot.xml` | Masses, damping, armature, actuators |
| Env base | `env/MujocoBaseEnv.py` | Step loop, real-time vs sim-time |
| Platform | `SimFlexivArm5FHand` (under env / assets) | Maps `JointCtrl` → MuJoCo `ctrl` |

When something “feels wrong” (too heavy, too slippery, double gravity), check MJCF and `platform.gravity_comp` before blaming the controller math.
