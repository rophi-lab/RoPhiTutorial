# Gravity Compensation Controller

This tutorial covers the pure gravity-compensation controller
(`controller/GravCompControl.py`, config
`configs/flexiv_arm_5F_hand/controllers/grav_comp.yaml`).

Goal: hold the Flexiv arm (+ hand) against gravity with **feedforward torque
only** — no PD tracking, no collision avoidance. It is the simplest closed
loop in the stack and a good sanity check that LCM, the plant, and the dynamics
model agree.

---

## Run it

Three terminals (from the repo root):

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/default.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/grav_comp.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/default.yaml
```

What you should see: the arm floats near its pose instead of collapsing under
gravity. You can gently push it in the MuJoCo viewer; it should move and then
settle with light damping.

---

## What the controller does

Each control tick (`ctrl_freq: 1000`):

1. Read the latest arm / hand `joint_meas` from LCM → $q$, $\dot{q}$.
2. Compute generalized gravity with Pinocchio:
   $$
   g(q) = \texttt{pin.computeGeneralizedGravity}(\mathrm{model}, \mathrm{data}, q)
   $$
3. Add a little viscous damping on the **arm only**:
   $$
   \tau_{\mathrm{ff}}
   = g(q) - b\,\dot{q}_{\mathrm{arm}}
   \qquad
   (b = \texttt{arm\_damping})
   $$
4. Publish a `JointCtrl` command with **zero gains**:
   $$
   K_p = 0,\quad K_d = 0,\quad
   q_{\mathrm{des}} = q,\quad
   \dot{q}_{\mathrm{des}} = 0,\quad
   \tau_{\mathrm{ff}} \text{ as above.}
   $$

Core loop (`GravCompControl._update`):

```python
tau_ff = pin.computeGeneralizedGravity(self._pin_model, self._pin_data, self._q)
tau_ff[:num_arm] -= arm_damping * self._qd[:num_arm]

cmd.set_data(t, q, zeros, tau_ff, zeros_kp, zeros_kd)
self.ctrl_pub_que.put(cmd)
```

So the plant is asked to apply $\tau_{\mathrm{ff}}$ only. There is no setpoint
holding beyond gravity + light damping.

### Closed loop

The plant (tutorial 02) obeys, schematically,

$$
M(q)\,\ddot{q} + C(q,\dot{q})\,\dot{q} + g_{\mathrm{plant}}(q)
+ \tau_{\mathrm{damp}} + \tau_{\mathrm{fric}} + \tau_{\mathrm{contact}}
= \tau_{\mathrm{act}}.
$$

With $K_p = K_d = 0$ and plant `gravity_comp: false`,

$$
\tau_{\mathrm{act}} = \tau_{\mathrm{ff}} = g_{\mathrm{ctrl}}(q) - B\,\dot{q},
$$

where $B = \mathrm{diag}(b,\ldots,b,0,\ldots,0)$ damps the arm DoFs only.
Substituting,

$$
M(q)\,\ddot{q} + C(q,\dot{q})\,\dot{q}
+ \bigl(g_{\mathrm{plant}}(q) - g_{\mathrm{ctrl}}(q)\bigr)
+ \tau_{\mathrm{damp}} + \tau_{\mathrm{fric}} + \tau_{\mathrm{contact}}
+ B\,\dot{q}
= 0.
$$

If the Pinocchio URDF matches the MuJoCo inertias, $g_{\mathrm{plant}} \approx g_{\mathrm{ctrl}}$ and gravity cancels. Away from contact the residual is mostly joint dissipation plus the controller’s $B\,\dot{q}$ — i.e. near free-float with light damping. Any mismatch $g_{\mathrm{plant}} - g_{\mathrm{ctrl}}$ shows up as a constant residual torque (drift up or down).

**Special case** — matched gravity ($g_{\mathrm{plant}}=g_{\mathrm{ctrl}}$) and no plant damping, friction, or contact:

$$
M(q)\,\ddot{q} + C(q,\dot{q})\,\dot{q} + B\,\dot{q} = 0.
$$

How this moves:

- **No preferred pose.** Gravity is cancelled and there is no $K_p$ spring, so every configuration is an equilibrium: if $\dot{q}=0$, then $\ddot{q}=0$. The arm does not “hold” a joint target; it only floats.
- **$C(q,\dot{q})\,\dot{q}$ does not dissipate energy.** Coriolis / centrifugal terms are energy-conserving (skew-symmetry of $\dot{M}-2C$). Alone they redistribute momentum along the chain but do not slow the robot down.
- **$B\,\dot{q}$ does dissipate.** Multiply the equation by $\dot{q}^\top$:
  $$
  \frac{\mathrm{d}}{\mathrm{d}t}\Bigl(\tfrac{1}{2}\dot{q}^\top M(q)\,\dot{q}\Bigr)
  = -\dot{q}^\top B\,\dot{q} \le 0.
  $$
  Kinetic energy strictly decreases wherever the damped joints are moving. The arm **coasts**, then **settles to rest** at whatever pose it reaches — like free-float in space with viscous joint drag.
- **If $B=0$ as well:** $M\ddot{q}+C\dot{q}=0$ conserves kinetic energy. Idealized motion never stops (modulo numerics); a push keeps the arm drifting forever.

So grav-comp + light $B$ feels like: push it, it moves, then gently stops — not like a stiff PD pose hold.

---

## Why the plant must not also add gravity

In `configs/flexiv_arm_5F_hand/env/default.yaml` the platform has:

```yaml
gravity_comp: false
```

The sim applies:

$$
\tau
= K_p(q_{\mathrm{des}}-q) + K_d(\dot{q}_{\mathrm{des}}-\dot{q}) + \tau_{\mathrm{ff}}
$$

With this controller, $K_p=K_d=0$, so $\tau = \tau_{\mathrm{ff}} \approx g(q)$.

If `gravity_comp: true`, the platform would **also** add MuJoCo’s
`qfrc_bias` gravity on the arm → **double gravity compensation** → the arm
pushes upward. That matches the real Flexiv setup in this project: onboard
grav-comp is off; the external controller owns $g(q)$.

> Until the first real `joint_ctrl` arrives, the platform still injects
> temporary arm gravity so the robot does not free-fall during startup. After
> that, only the controller’s $\tau_{\mathrm{ff}}$ counters gravity.

---

## LCM I/O

| Direction | Channel | Content |
|-----------|---------|---------|
| In | `sw_flexiv_arm_joint_meas` | arm $q$, $\dot{q}$ (7) |
| In | `sw_robotis_5F_hand_joint_meas` | hand $q$, $\dot{q}$ (20) |
| Out | `sw_flexiv_arm_hand_joint_ctrl` | 27-DoF command ($q_{\mathrm{des}}$, $\tau_{\mathrm{ff}}$, $K_p$, $K_d$) |

Same channels as in [01_quick_start](01_quick_start.md). Use `lcm-spy` to confirm
meas is flowing and ctrl is publishing at ~1 kHz.

---

## Config knobs

| Key | Role |
|-----|------|
| `urdf_path` | Pinocchio model for $g(q)$ — should match the robot (same as Viser URDF here) |
| `default_q` | Seed before the first measurement |
| `arm_damping` | Viscous $b$ on arm joints only (Nm/(rad/s)); default in YAML is `0.01` |
| `ctrl_freq` | Control rate (match env / real robot when possible) |
| `dict_joints` | Which meas channels map into which indices of the 27-vector |

**Hand:** gravity is still computed for all DoFs, but no extra `arm_damping` is
applied on fingers. Finger plant dissipation lives in the MJCF (`XM335`
`damping` / `frictionloss` / `armature`) — see
[02_understanding_mujoco_simulator](02_understanding_mujoco_simulator.md).

---

## Model mismatch (what breaks grav-comp)

Pure grav-comp is a **model-quality** test:

| Symptom | Likely cause |
|---------|----------------|
| Arm slowly sinks | $g(q)$ too small (mass / CoM low), or plant friction dominating |
| Arm drifts upward | $g(q)$ too large, or plant `gravity_comp: true` (double count) |
| Oscillates / jitter | `arm_damping` too low vs sensor noise; or wrong $\dot{q}$ |
| Works in sim, fails on hardware | URDF inertia ≠ real robot; need system ID (tutorial 02) |

Pinocchio’s $g(q)$ and MuJoCo’s gravity come from **different model files**
(URDF vs MJCF). They should be close; if not, fix the models before blaming the
controller.

---

## Relation to collision-aware grav-comp

`col_aware_grav_comp` builds on the same idea: when far from obstacles it
behaves like free float / grav-comp; near obstacles it adds repulsive motion
with nonzero $K_p$, $K_d$. Start here first — if pure grav-comp is wrong,
collision-aware will be wrong too.

---

## Exercises

1. **Kill feedforward.** Temporarily publish $\tau_{\mathrm{ff}}=0$ (or stop the
   controller). With `gravity_comp: false`, the arm should fall.
2. **Tune `arm_damping`.** Try `0`, `0.01`, `0.1`. Too small → drift / buzz;
   too large → sluggish “molasses” feel when you push the arm.
3. **Double gravity.** Set env `gravity_comp: true` while running this
   controller — the arm should float upward. Set it back to `false`.
4. **URDF sensitivity.** Scale a link mass in the Pinocchio URDF (or use a
   wrong `urdf_path`) and watch residual gravity appear as a constant push.

---

## Files

| File | Role |
|------|------|
| `controller/GravCompControl.py` | Control law |
| `configs/flexiv_arm_5F_hand/controllers/grav_comp.yaml` | Rates, URDF, channels, `arm_damping` |
| `robot_platform/sim/flexiv_arm_5F_hand/SimFlexivArm5FHand.py` | $\tau = K_p e + K_d \dot{e} + \tau_{\mathrm{ff}}$ (+ optional plant grav) |
| `configs/flexiv_arm_5F_hand/env/default.yaml` | `gravity_comp: false` |
| `run_controller.py` | Process entrypoint |
