# Computed Torque–Style P2P Tracking

`MinimumJerkP2PControl` (config
`configs/flexiv_arm_5F_hand/controllers/min_jerk_p2p.yaml`) samples a
**collision-free joint goal**, plans a **min-jerk** trajectory to it, then
tracks that trajectory with **joint-space PD + gravity / friction feedforward**.

This is the practical “computed torque–style” tracker used in this repo: the
plant applies

$$
\tau
=
K_p(q_{\mathrm{des}}-q)
+
K_d(\dot{q}_{\mathrm{des}}-\dot{q})
+
\tau_{\mathrm{ff}},
$$

with $\tau_{\mathrm{ff}}=g(q)+F_{\mathrm{jc}}\,\mathrm{sat}(\dot{q}_{\mathrm{des}}/\phi)$.
(Classical full computed torque also multiplies by $M(q)$ and tracks
$\ddot{q}_{\mathrm{des}}$; here we study PD+FF tracking of a planned $(q,\dot{q})$.)

It builds on:

- gravity / friction FF and the joint LCM plant ([03](03_grav_comp.md)),
- ColInfo from the collision env ([04](04_publish_collision_geom.md)),
- FCL clearance checks (same primitives as [05](05_collision_aware_grav_comp.md) /
  [06](06_inverse_kinematics_quadratic_program.md)).

**Study objects**

1. Collision-free min-jerk path planning (brief)
2. PD tracking + friction compensation
3. Tuning $K_p$, $K_d$; why critical damping
4. Parameter effects (Viser sliders + plot)

---

## Run

Three terminals (from the repo root):

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/collision.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/min_jerk_p2p.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/min_jerk_p2p.yaml
```

Scene: `hand_default.xml` (floor + walls). Goals / paths that violate
`safty_dist_thr` against published ColInfo are rejected.

**Keyboard (controller terminal):**

| Key | Action |
|-----|--------|
| `1` | sample a random **collision-free** arm goal |
| `2` | RRT-Connect → shortcut → B-spline → min-jerk time scale |
| `3` | PD-track the planned $(q(t),\dot{q}(t))$ |
| `g` | **grav-comp mode** ($K_p=K_d=0$, $\tau\approx g(q)$) |
| `h` | hold current pose with PD |
| `p` / `q` | pause / quit |

**Viser**

| UI | Meaning |
|----|---------|
| Purple ghost / EE sphere | sampled goal (`sw_p2p_target_joint`) |
| Yellow path | planned palm path (`sw_p2p_joint_traj`) |
| **P2P Tracking** | mode, effective $K_p$/$K_d$/$F_{\mathrm{jc}}$, uPlot $q$ vs $q_{\mathrm{des}}$ |
| **P2P Gains** | live `kp_scale`, `kd_scale`, `fjc_scale`, `friction_phi` |

Suggested workflow: `1` → `2` → `3`, watch the joint plot during tracking, then
nudge gain sliders and repeat `3` on a new plan.

---

## Architecture

| Stage | Where | Work |
|-------|--------|------|
| Sample / plan | background thread | FCL + RRT-Connect + B-spline + min-jerk tables |
| Control | 1 kHz `_update` | interpolate traj → PD setpoints + $\tau_{\mathrm{ff}}$ |
| Status / gains | LCM | `sw_p2p_status` (~50 Hz), `sw_p2p_gains_cmd` from Viser |

Planning is **not** in the 1 kHz loop. Control stamps trajectory start time on
the first tick after key `3` (after `_update_time`) so $t_{\mathrm{local}}$ does
not jump.

Modes:

| Mode | Key | Command |
|------|-----|---------|
| `hold` | `h` (default) | PD to $q_{\mathrm{hold}}$, $\dot{q}_{\mathrm{des}}=0$ |
| `track` | `3` | PD along planned traj |
| `grav` | `g` | $K_p=K_d=0$, $\tau_{\mathrm{ff}}=g(q)-b\dot{q}_{\mathrm{arm}}$ (+ friction on $\dot{q}$) |

---

## 1. Collision-free min-jerk planning (brief)

### Sample (`1`)

Randomize joints in `sample_joint_idx` (arm 0–6) inside joint limits with
margin `joint_limit_margin`. Accept only if env + self FCL distances stay
$\ge$ `safty_dist_thr` (`col_pairs`, `self_col_pairs`). Publish goal as
`joint_meas_t` on `sw_p2p_target_joint`.

### Plan (`2`)

1. **RRT-Connect** between current $q$ and goal (`rrt_step_size`,
   `rrt_max_iters`, `rrt_goal_bias`), then shortcut
   (`rrt_shortcut_iters`).
2. **B-spline** through waypoints (`bspline_samples`); if the curve cuts a
   corner into collision, fall back to a densified polyline.
3. **Min-jerk time scaling** (`path_to_min_jerk_trajectory`): rest-to-rest
   timing from path stretch and `joint_velocity_limit`, scaled by
   `traj_duration_scale` (or fixed `traj_duration`).

Result: discrete tables $t_k$, $q_k$, $\dot{q}_k$. Published (downsampled)
on `sw_p2p_joint_traj` for the yellow EE path.

### Track (`3`)

Linearly interpolate $q_{\mathrm{des}}(t)$, $\dot{q}_{\mathrm{des}}(t)$ from those
tables. At $t\ge T$, freeze at the goal and return to `hold`.

---

## 2. Computed torque–style PD + friction FF

### Plant law

MuJoCo / hardware (`SimFlexivArm5FHand`) applies exactly

$$
\tau
=
K_p(q_{\mathrm{des}}-q)
+
K_d(\dot{q}_{\mathrm{des}}-\dot{q})
+
\tau_{\mathrm{ff}}.
$$

Env `gravity_comp: false` — the controller owns gravity FF (same as
[03](03_grav_comp.md) / [06](06_inverse_kinematics_quadratic_program.md)).

### Feedforward

With `do_grav_comp` / `do_friction_comp`:

$$
\tau_{\mathrm{ff}}
=
g(q)
+
F_{\mathrm{jc}}\,\mathrm{sat}\!\left(
\tfrac{\dot{q}_{\mathrm{des}}}{\phi}
\right),
\qquad
\phi=\texttt{friction\_phi}.
$$

`$F_{\mathrm{jc}}$` is sized near MuJoCo `frictionloss` on the arm / XM335
fingers. During **grav** mode, Coulomb uses measured $\dot{q}$ (desired velocity
is zero). Viscous plant friction is **not** cancelled.

Live scales (`kp_scale`, `kd_scale`, `fjc_scale`) multiply base vectors from
YAML; Viser sliders publish them over `sw_p2p_gains_cmd`.

---

## 3. Tuning $K_p$, $K_d$; why critical damping

Per joint, treat the plant roughly as $m_i\ddot{e}_i + k_{d,i}\dot{e}_i +
k_{p,i}e_i \approx 0$ after gravity / Coulomb FF, with $m_i$ from
`mean_mass_matrix_diag` (MuJoCo `mj_fullM` at `default_q`, **including** finger
armature). Critically damped second-order poles when

$$
k_{p,i}
=
\frac{k_{d,i}^2}{4\,m_i}.
$$

That is how the default `kp` / `kd` lists were chosen:

| | Proximal arm | Distal arm | Fingers |
|--|--------------|------------|---------|
| $k_d$ | 20 | 10 / 3.5 | 0.05 |
| $m_i$ | $\sim$1–2 | $\sim$0.08 / 0.0025 | $\sim$0.0025 |
| $k_p$ | $\sim$48–73 | $\sim$250–1240 | $\sim$0.25 |

**Why critical damping**

- Underdamped ($k_p$ too large for fixed $k_d$): oscillatory overshoot on
  $q$ vs $q_{\mathrm{des}}$ (easy to see on the Viser plot).
- Overdamped ($k_p$ too small): sluggish lag behind the min-jerk profile.
- Critical: fastest non-oscillatory settle for this second-order model.

**How to tune in practice**

1. Pick $k_d$ for noise / torque smoothness (larger $k_d$ → more damping and
   typically larger critical $k_p$).
2. Set $k_p = k_d^2/(4m_i)$ from `mean_mass_matrix_diag`.
3. Scale both together with Viser `kp_scale` / `kd_scale` (keeps the damping
   ratio $\approx 1$ if you move them equally).
4. If you raise **only** `kp_scale`, expect underdamped ringing; **only**
   `kd_scale` → more overdamped lag.

Compare modes: press `g` to drop to pure grav-comp (no PD), then `h` or `3`
to restore tracking — isolates FF vs feedback.

---

## 4. Study the effect of parameters

Use the **same plan** (`1`→`2`) and re-track (`3`) while changing one knob.

| Knob | Expected effect on $q$ vs $q_{\mathrm{des}}$ |
|------|-----------------------------------------------|
| `kp_scale` ↑ alone | tighter / may ring (underdamped) |
| `kd_scale` ↑ alone | smoother, more lag (overdamped) |
| both scales ↑ equally | stiffer critically damped track |
| `fjc_scale` → 0 | more Coulomb lag / stick during velocity |
| `fjc_scale` ↑ | better velocity follow; too high → “dragging” along $\mathrm{sign}(\dot{q}_{\mathrm{des}})$ |
| `friction_phi` ↑ | softer Coulomb layer (slower saturation) |
| `traj_duration_scale` ↑ | slower min-jerk profile (easier to track) |
| `g` mode | free float under $g(q)$; plot $q_{\mathrm{des}}$ sticks to measured $q$ |

Watch **P2P Tracking** markdown: effective arm $K_p$, $K_d$, $F_{\mathrm{jc}}$,
and $\|q-q_{\mathrm{des}}\|_{\mathrm{arm}}$.

---

## Config map (high-signal keys)

| Key | Role |
|-----|------|
| `kp`, `kd`, `mean_mass_matrix_diag` | base PD (critical-damping recipe) |
| `kp_scale`, `kd_scale`, `fjc_scale` | live multipliers (Viser) |
| `do_grav_comp`, `do_friction_comp`, `Fjc`, `friction_phi` | FF |
| `arm_damping` | light viscous term in `grav` mode |
| `safty_dist_thr`, `col_pairs`, `self_col_pairs` | free-space checks |
| `rrt_*`, `bspline_samples`, `traj_dt`, `traj_duration_scale` | planning |
| `status_publish_stride` | Viser status rate |
| `gains_cmd_channel` / `status_channel` | `sw_p2p_gains_cmd` / `sw_p2p_status` |

---

## Exercises

1. **Critical damping.** With a fixed plan, set `kd_scale=1` and sweep
   `kp_scale` through $\{0.25, 1, 2\}$. Sketch under / critical / over on the
   joint plot.
2. **Friction.** Disable `do_friction_comp` in Viser, track, then re-enable
   with `fjc_scale=1`. Where does lag grow (start / mid / stop of the traj)?
3. **Grav vs hold.** At rest after a traj, press `g`, nudge the arm in MuJoCo,
   then `h`. Confirm `grav` floats and `hold` restores PD.
4. **Duration.** Plan with default `traj_duration_scale`, then raise it in
   YAML to `2.0` and replan. Does the same $K_p$,$K_d$ track more cleanly?
5. **Collision.** Enable collision geometry in Viser, sample until a goal
   near a wall, and confirm rejected samples / RRT detours keep clearance
   $\ge$ `safty_dist_thr`.

---

## Takeaways

1. Collision-free P2P = **sample free goal → geometric plan → min-jerk time
   law → PD+FF track**.
2. This stack’s “computed torque” is **PD + $g(q)$ + Coulomb FF** on the plant;
   planning supplies $(q_{\mathrm{des}},\dot{q}_{\mathrm{des}})$.
3. Critically damped $k_p=k_d^2/(4m)$ is a good first gain pair; scale $k_p$ and
   $k_d$ **together** when stiffening.
4. Viser $q$ vs $q_{\mathrm{des}}$ + gain/friction sliders are the right loop for
   studying tracking quality.
