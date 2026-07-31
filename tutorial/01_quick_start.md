# Quick start

Welcome to RoPhiTutorial. This first lesson is not about fancy control theory — it is about **how the pieces talk to each other**. Once you can start the simulation and see the robot move under a simple controller, every later tutorial reuses the same pattern.

We use a **Flexiv Rizon** arm (7 joints) plus a **Robotis RH-5** hand (20 joints), simulated in MuJoCo, controlled in a separate Python process, and visualized in Viser.

---

## 1. Run & LCM communication

### What you will launch

Open **three terminals** in the repository root and run:

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/default.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/grav_comp.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/default.yaml
```

Start the **env** first (it owns the physics clock), then the controller, then the visualizer. You should see a MuJoCo window and a Viser page; the arm should look “floaty” under gravity compensation ([03](03_grav_comp.md)).

| Key | Where | Action |
|-----|--------|--------|
| `p` | env or controller terminal | Pause |
| `q` | env or controller terminal | Quit |

### Why three processes?

| Process | Everyday analogy | Job |
|---------|------------------|-----|
| **Env** | The physical world | Advance MuJoCo; publish “what the sensors see”; apply motor commands |
| **Controller** | The brain / real-time computer | Read sensors; compute torques / setpoints; publish commands |
| **Visualizer** | A camera + HUD | Draw the robot for you (does not affect physics) |

Keeping them separate means:

- physics can run at 1000 Hz while graphics runs at 30 Hz,
- a crash in the visualizer does not kill the controller,
- the same controller pattern can later talk to **real** hardware over the same message types.

### What is LCM?

**LCM** (Lightweight Communications and Marshalling) is a publish/subscribe bus. Think of named **channels** (like radio stations):

- one process **publishes** a message on a channel,
- any process that **subscribes** receives a copy.

You do not call the other process’s Python functions directly. You only exchange messages. That is the discipline of this codebase.

### Channels in this tutorial

| Channel name | Message type | Who → whom | Meaning in plain words |
|--------------|--------------|------------|------------------------|
| `sw_flexiv_arm_joint_meas` | `JointMeas` | env → controller & viz | Arm joint angles $q$ and velocities $\dot q$ (7 numbers each) |
| `sw_robotis_5F_hand_joint_meas` | `JointMeas` | env → controller & viz | Same for the hand (20 DoF) |
| `sw_flexiv_arm_hand_joint_ctrl` | `JointCtrl` | controller → env | What the motors should do next (see below) |
| `sim_clock` | time | env → controller | Simulation time so rates stay synchronized |

A `JointCtrl` message packages five vectors of length 27 (arm+hand):

$$
\bigl(
  q_{\mathrm{des}},\;
  \dot q_{\mathrm{des}},\;
  \tau_{\mathrm{ff}},\;
  K_p,\;
  K_d
\bigr).
$$

You will learn what each means in [02](02_understanding_mujoco_simulator.md). For now: the env turns that message into motor torques.

Later tutorials add more channels (collision shapes, object poses, arrows). The **pattern** never changes: env publishes world state; controller publishes commands; visualizer listens.

```
                 measurements
        ┌──────────────────────────► Controller
        │                                │
Env ────┤                                │  JointCtrl
        │                                ▼
        │◄───────────────────────────────┘
        │
        └────── measurements ──────────► Visualizer
```

---

## 2. Ideas (architecture)

### The control loop in words

Imagine repeating forever, about once every millisecond:

1. **Sense.** The env reads the current joint angles from MuJoCo and publishes them.
2. **Think.** The controller computes a new command from those angles (here: mostly “cancel gravity”).
3. **Act.** The env applies that command the next time it steps the physics.

This is a **feedback loop**: the command depends on the measurement, which depends on how the robot moved under the previous command.

### Why not one big Python script?

A single script that steps MuJoCo, runs OSQP, and draws OpenGL in one thread quickly becomes slow and brittle. Separating rates and responsibilities is standard on real robots (and in this course).

### Zero-order hold (gentle version)

The controller and the simulator may not wake up at the exact same instant. Between controller updates, the env **holds the last command constant**. That is called a zero-order hold. It is ordinary and expected; just remember that a slow controller means the robot “coasts” on an old command for longer.

---

## 3. How it is implemented

You do not need to read all of this on day one. Use it as a map.

| Piece | Path | What it does |
|-------|------|----------------|
| Env entry | `run_env.py` | Loads YAML, builds the MuJoCo env, starts LCM pubs/subs |
| Controller entry | `run_controller.py` | Loads YAML, builds a controller by `name`, starts the loop |
| Visualizer entry | `run_visualizer.py` | Loads YAML, opens Viser, subscribes to measurements |
| Default controller | `controller/GravCompControl.py` | Gravity compensation used in this quick start |
| Env config | `configs/flexiv_arm_5F_hand/env/default.yaml` | Scene file, `sim_freq`, channel names |
| Controller config | `configs/flexiv_arm_5F_hand/controllers/grav_comp.yaml` | `ctrl_freq`, URDF, which meas channels map to which joints |
| Viz config | `configs/visualizer/flexiv_arm_5F_hand/default.yaml` | URDF for drawing, same meas channels |

**Mental model:** YAML chooses *which* classes and *which* channel strings; Python classes implement *behavior*; LCM carries *numbers* between processes.

When you are comfortable starting and stopping this stack, continue to [02](02_understanding_mujoco_simulator.md) (what MuJoCo is doing) and [03](03_grav_comp.md) (your first real control law).
