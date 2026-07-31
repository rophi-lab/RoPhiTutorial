# Gravity compensation

Your first real control law: make the arm feel **weightless** (almost), without telling it *where* to go.

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/default.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/grav_comp.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/default.yaml
```

| Key | Action |
|-----|--------|
| `p` / `q` | Pause / quit |

### What you should observe

If gravity compensation is working, gently pushing the arm in the MuJoCo viewer (or disturbing it) should feel easy: the arm **holds its pose against gravity** but does **not** spring back to a home configuration. Give it a small shove and it may drift slowly; that is expected.

### LCM (same as quick start)

| Channel | Direction | Role here |
|---------|-----------|-----------|
| `sw_flexiv_arm_joint_meas` | env → ctrl | Arm $q,\dot q$ |
| `sw_robotis_5F_hand_joint_meas` | env → ctrl | Hand $q,\dot q$ |
| `sw_flexiv_arm_hand_joint_ctrl` | ctrl → env | $\tau_{\mathrm{ff}}\approx g(q)$, with $K_p=K_d=0$ |

Keep `platform.gravity_comp: false` in the env YAML so gravity is not cancelled twice ([02](02_understanding_mujoco_simulator.md)).

---

## 2. Ideas (the algorithm)

### The problem in one sentence

Gravity produces joint torques $g(q)$ that depend on the configuration. If the motors send nothing, the arm collapses. If the motors send **exactly** $g(q)$, those gravity torques cancel and the arm “floats.”

### Feedforward gravity

We compute $g(q)$ from a URDF model (Pinocchio’s `computeGeneralizedGravity`) and send it as feedforward:

$$
\tau_{\mathrm{ff}} = g(q) - B_{\mathrm{arm}}\dot q_{\mathrm{arm}}.
$$

We set PD gains to zero and do not invent a goal pose:

$$
K_p = 0,\qquad K_d = 0,\qquad
q_{\mathrm{des}} = q,\qquad \dot q_{\mathrm{des}} = 0.
$$

So the actuator law from [02](02_understanding_mujoco_simulator.md) reduces to $\tau_{\mathrm{act}}=\tau_{\mathrm{ff}}$.

The small term $B_{\mathrm{arm}}\dot q_{\mathrm{arm}}$ is **extra damping on the arm only**. It bleeds kinetic energy so the arm does not swing forever. The hand typically gets pure $g(q)$.

### Why it settles (energy picture)

With matched models, the closed loop looks like

$$
M(q)\ddot q + C(q,\dot q)\dot q + B\dot q = 0.
$$

A standard energy argument shows that kinetic energy

$$
E = \tfrac12 \dot q^\top M(q)\dot q
$$

is nonincreasing when $B$ is positive definite on the moving joints:

$$
\dot E = -\dot q^\top B\dot q \le 0.
$$

So the robot coasts and eventually stops. Importantly, **every configuration with $\dot q=0$ is an equilibrium**. Gravity compensation does **not** create a preferred home pose — that is the job of impedance or tracking controllers later.

### Model mismatch (gentle warning)

If the URDF masses differ from MuJoCo, residual gravity remains and the arm slowly drifts. That is normal in simulation-to-reality transfer; students should treat perfect floating as a modeling assumption, not magic.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/GravCompControl.py` | Each control tick: read $q$, compute $g(q)$, subtract arm damping, publish `JointCtrl` |
| `configs/flexiv_arm_5F_hand/controllers/grav_comp.yaml` | `ctrl_freq`, URDF paths, `arm_damping`, joint↔channel map |

**Reading tip:** find where `kp` and `kd` are set to zero. That single choice is what distinguishes “float” from “servo to a pose.”

Next: [04](04_publish_collision_geom.md) adds geometry messages; [05](05_collision_aware_grav_comp.md) uses them so floating does not drive through walls.
