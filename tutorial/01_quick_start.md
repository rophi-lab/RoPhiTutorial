# Quick Start

## Installation

```bash
conda create -n rophi python=3.12
conda activate rophi
pip install -r requirements.txt
```

## Run (three terminals)

From the repo root, start the sim environment, then a controller, then the visualizer:

```bash
# Terminal 1 — MuJoCo environment
python run_env.py --config configs/flexiv_arm_5F_hand/env/default.yaml

# Terminal 2 — gravity-compensation controller
python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/grav_comp.yaml

# Terminal 3 — Viser visualizer
python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/default.yaml
```

In the env / controller terminals: `p` pauses, `q` quits.

For the collision-aware stack, swap in:

- Env: `configs/flexiv_arm_5F_hand/env/collision.yaml`
- Controller: `configs/flexiv_arm_5F_hand/controllers/col_aware_grav_comp.yaml`
- Visualizer: `configs/visualizer/flexiv_arm_5F_hand/collision.yaml`

---

## Basics

This example is split into **three processes**. Each is started by a small `run_*.py` script that loads a YAML config, builds the object + LCM managers, then runs until you quit.

```
run_env.py  ──LCM──►  run_controller.py  ──LCM──►  run_env.py
    │                      ▲
    └──────── LCM ─────────┴──►  run_visualizer.py
```

### `run_env.py`

Owns the **MuJoCo simulation** (and later, the same interface on hardware).

> **NOTE — why this split exists.** Controllers and visualizers never talk to MuJoCo (or a robot SDK) directly; they only speak LCM. That is the point of the framework: **swap simulation for real hardware without rewriting control or viz.** Today you run `run_env.py`; later the same controller / visualizer configs will work against `run_hardware.py`, as long as the hardware bridge **publishes and subscribes to the exact same channels and message types** as the sim (same `joint_meas` / `joint_ctrl` names and payloads). If the wire format matches, sim and hardware are interchangeable.

What it does:

1. `get_env(cfg)` — builds the env from `cfg.name` (e.g. `default_mujoco`).
2. Creates an **env sub manager** (receives joint commands) and **env pub manager** (publishes joint measurements + clock).
3. Wires their queues into the env, starts the LCM threads, then `env.start()`.

Config knobs that matter for the default Flexiv demo:

| Key | Meaning |
|-----|---------|
| `scene_xml_path` | MuJoCo scene XML |
| `sim_freq` / `view_freq` | Physics and GUI rates |
| `platform` | Which robot platform (Flexiv arm + Robotis hand) |
| `sub_manager.ctrl_channel` | Where torque/PD commands arrive |
| `pub_manager.*_joint_meas_channel` | Where `q` / `qd` are published |

### `run_controller.py`

Owns the **control law**. It never touches MuJoCo directly — it only reads measurements and writes commands over LCM.

What it does:

1. `get_controller(cfg)` — e.g. `grav_comp` → pure Pinocchio gravity compensation.
2. Creates a **controller sub manager** (arm + hand meas + clock) and **controller pub manager** (joint ctrl).
3. Starts LCM threads, then `controller.start()` at `ctrl_freq`.

`GravCompControl` reads `q`, computes `τ_g(q)`, and publishes a `joint_ctrl` message with feedforward torque (plus light arm damping). Channel names in the controller YAML must match the env YAML exactly.

### `run_visualizer.py`

Owns the **Viser** browser UI. It is a pure subscriber: no commands, no clock.

What it does:

1. `get_vis_manager(cfg)` — for Flexiv, `FlexivArmHandVisManager`.
2. Subscribes to the arm and hand `joint_meas` channels.
3. Updates the URDF pose at `vis_freq` (typically 30 Hz).

The collision visualizer config adds overlays (collision primitives + distance lines) by also listening to `robot_col_info` / `static_col_info`.

> **Why use Viser if MuJoCo already renders?**
>
> MuJoCo's viewer exists only in simulation — on hardware there is no MuJoCo window.
> Viser listens to the same `joint_meas` LCM streams as the controller, so one visualizer
> works for both sim and real experiments, and it is where debug overlays live
> (collision geometry, distance lines, …) that are not in the MuJoCo scene viewer.

---

## Understanding communication (LCM)

### Why LCM?

[LCM](https://lcm-proj.github.io/) (Lightweight Communications and Marshalling) is a UDP multicast pub/sub bus. Env, controller, and visualizer each create their **own** `lcm.LCM()` instance and talk by **channel name**. They do not share memory or a process — the same pattern works for sim and hardware.

To inspect live traffic while the stack is running, use **`lcm-spy`** (ships with LCM):

```bash
lcm-spy
```

It lists active channels and lets you inspect message rates / fields — useful when debugging a silent loop (wrong channel name, nothing publishing, etc.).

Inside each process:

- A **subscriber thread** calls `lcm.handle_timeout(...)`, decodes messages, and pushes Python objects onto queues.
- A **publisher thread** (env / controller) drains queues and publishes.
- The **main thread** runs the sim / control / render loop and only touches queues.

That keeps realtime I/O off the control loop.

### Default Flexiv channels

| Channel | Payload | Direction |
|---------|---------|-----------|
| `sw_flexiv_arm_joint_meas` | arm `q`, `qd` (7) | Env → Controller, Visualizer |
| `sw_robotis_5F_hand_joint_meas` | hand `q`, `qd` (20) | Env → Controller, Visualizer |
| `sw_flexiv_arm_hand_joint_ctrl` | `q_des`, `qd_des`, `τ_ff`, `kp`, `kd` (27) | Controller → Env |
| `sim_clock` | simulation time | Env → Controller |

Closed loop:

1. Env steps MuJoCo and publishes joint measurements.
2. Controller reads measurements, computes torques, publishes `joint_ctrl`.
3. Env applies the command on the next tick.
4. Visualizer mirrors the published `q` in the browser.

### Collision extras

When you use the collision env / `col_aware_grav_comp` / collision visualizer, the env also publishes:

| Channel | Meaning |
|---------|---------|
| `sw_flexiv_arm_hand_robot_col_info` | Per-link collision primitives (link-local) |
| `sw_flexiv_arm_hand_static_col_info` | Floor / walls (world frame) |

The collision-aware controller uses these for FCL distance checks; the collision visualizer can overlay the same geometry and nearest-point lines.

### Matching configs

Channel strings are just names — if they disagree across YAMLs, nothing connects and nothing crashes loudly. When you change a channel in the env config, update the controller and visualizer configs to match.

## Real hardware

In simulation, MuJoCo can supply almost everything from one process: joint angles, camera RGB, ground-truth object poses, and more. So for sim studies, `run_env.py` can do a lot of jobs at once.

On real hardware those roles split across processes — for example `run_hardware.py` (robot bridge), `run_sensor.py` (cameras / force / etc.), `run_perception.py` (estimates that replace sim ground truth). Each process should **publish and subscribe over LCM with the same channel names and message payloads as the sim**. Controllers and visualizers then stay unchanged: they only care that the wire format matches, not whether the data came from MuJoCo or the lab.

For collision-aware control in a real experiment you still need to publish `robot_col_info` / `static_col_info` for the controller. You can keep running `run_env.py` for that role — just set `vis_mode: vis_off` so MuJoCo does not open a viewer.
