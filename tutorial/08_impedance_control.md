# Joint-Space vs SE(3) Task-Space Impedance

`ImpedanceControl` (config
`configs/flexiv_arm_5F_hand/controllers/impedance.yaml`) regulates the arm
about a **fixed nominal pose** using either

1. **Joint-space impedance** — spring–damper on $q_{\mathrm{nom}}-q$, or
2. **Task-space SE(3) impedance** — geometric spring–damper on the palm frame
   $T\in\mathrm{SE}(3)$.

Students can toggle modes (`j` / `t` or the Viser dropdown), push the arm in
MuJoCo, and compare how the end-effector returns to the nominal frame.

Builds on gravity FF ([03](03_grav_comp.md)) and Pinocchio FK / Jacobians
([06](06_inverse_kinematics_quadratic_program.md)).

---

## Run

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/default.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/impedance.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/impedance.yaml
```

**Keyboard**

| Key | Action |
|-----|--------|
| `j` | joint-space impedance |
| `t` | SE(3) task impedance at `palm` |
| `p` / `q` | pause / quit |

**Viser (Impedance folder)**

| UI | Meaning |
|----|---------|
| Purple axes | nominal palm $T_{\mathrm{nom}}=\mathrm{FK}(q_{\mathrm{nom}})$ |
| Green axes | current palm $T(q)$ |
| Mode dropdown | joint / task (same as `j`/`t`) |
| Gain sliders | live scales for $K_q,D_q$ and $K_t,D_t$ |
| Markdown | $\|q-q_{\mathrm{nom}}\|$, $\|p-p_{\mathrm{nom}}\|$, $\|\xi\|$ |

Env tip: with default gravity + soft gains you can **drag the arm** in the
MuJoCo viewer and feel the restoring behavior.

---

## Nominal pose

YAML `default_q` is $q_{\mathrm{nom}}$. Task nominal is fixed at startup:

$$
T_{\mathrm{nom}} = \mathrm{FK}_{\mathrm{palm}}(q_{\mathrm{nom}}).
$$

Fingers stay soft-PD to $q_{\mathrm{nom}}$ (`kp_hand` / `kd_hand`) in both modes.
Arm impedance torques go through $\tau_{\mathrm{ff}}$ with plant $K_p=K_d=0$ on
the arm.

---

## 1. Joint-space impedance

$$
\tau
=
K_q(q_{\mathrm{nom}}-q)
-
D_q\,\dot{q}
+
g(q).
$$

$K_q$, $D_q$ are diagonal (config `kq`, `dq`), scaled by Viser `kq_scale` /
`dq_scale`. Defaults use the min-jerk critical-damping recipe
(`kq_i = dq_i^2 / (4 m_i)`) with the tuned `dq` in YAML; finger entries are 0
(held by `kp_hand` / `kd_hand`).

**Feel:** each joint is an independent spring. Pushing the palm along one
Cartesian direction generally **moves many joints**; restoring paths in
task space look “joint-like” / curved, and orientation is not regulated in
$\mathrm{SE}(3)$.

> **NOTE — damping-only “overdamping”.** On paper, raising $D_q$ alone while
> keeping $K_q$ fixed increases the damping ratio
> $\zeta = D_q / (2\sqrt{K_q m})$ and should **overdamp** the SISO joint model
> (sluggish return, no ringing). Try it with Viser `dq_scale` ↑ and
> `kq_scale` fixed.
>
> If the arm still **oscillates / chatters**, that is usually **not** the
> undamped second-order mode from the formula. Typical causes in this stack:
>
> 1. **$\dot{q}$ noise** — $\tau$ contains $-D_q\dot{q}$, so larger $D_q$
>    amplifies measurement noise into high-frequency torque chatter;
> 2. **delay** — 1 kHz discrete control + LCM with high $D$ can make the
>    delayed damping loop look unstable / limit-cycle;
> 3. **non-diagonal $M(q)$** — critical damping used a scalar
>    `mean_mass_matrix_diag`; coupled inertias and configuration dependence
>    break the single-DOF intuition;
> 4. **tiny distal $m$** — joints 6–7 still kick hard for modest $K_q$ when
>    $m$ is $\sim 10^{-2}$–$10^{-3}$.
>
> Safer: move **`kq_scale` and `dq_scale` together** (keep $\zeta\approx 1$),
> or lower both. Jitter from “more damping” is a feature of real closed loops,
> not a contradiction of the overdamping math.

---

## 2. SE(3) task-space impedance

Pinocchio **body** (LOCAL) quantities at `task_frame` (`palm`):

$$
\xi
=
\log\!\big(T^{-1}T_{\mathrm{nom}}\big)^{\vee}
\in\mathbb{R}^{6}
\quad\text{(layout $(\nu,\omega)$: linear then angular)},
$$

$$
V_b = J_b(q)\,\dot{q},
\qquad
F = K\,\xi - D\,V_b,
\qquad
\tau = J_b^{\top} F + g(q).
$$

Isotropic gains:

$$
K=\mathrm{diag}(k_t I_3,\,k_r I_3),
\qquad
D=\mathrm{diag}(d_t I_3,\,d_r I_3)
$$

(`kt_trans`, `kt_rot`, `dt_trans`, `dt_rot` × Viser scales).

**Feel:** the palm axes (green) are pulled toward the purple nominal frame as a
**pose spring**. Translations and rotations are co-located in the body wrench;
nullspace motions that leave $T$ unchanged are soft (not actively stiff).

---

## 3. What to compare

| Experiment | Joint | Task SE(3) |
|------------|-------|------------|
| Push palm in $+z$, release | multi-joint unwind; tip path may arc | tip returns nearly along the displacement |
| Twist palm about $z$ | may need several joints; ends can drift | orientation springs back about palm |
| Stiffen only `kt_trans_scale` | (no effect) | translation firmer, rotation same |
| Stiffen only `kq_scale` | all joints firmer | (scales unused in τ until you switch to joint) |

Watch $\|\xi\|$ and the two axis triads: in task mode they should re-align; in
joint mode they often stay **misaligned** while $\|q-q_{\mathrm{nom}}\|$ shrinks.

---

## Config map

| Key | Role |
|-----|------|
| `default_q`, `task_frame` | nominal $q$ and SE(3) frame |
| `kq`, `dq`, `kq_scale`, `dq_scale` | joint impedance |
| `kt_*`, `dt_*`, `*_scale` | SE(3) body impedance |
| `kp_hand`, `kd_hand` | soft finger hold |
| `do_grav_comp` | add $g(q)$ |
| `sw_imp_status` / `sw_imp_gains_cmd` | Viser |

---

## Exercises

1. **Same push, two modes.** Displace the palm by hand, release under `j`, then
   repeat under `t`. Sketch tip paths.
2. **Orientation.** With `t`, raise `kt_rot_scale` and twist the palm; then set
   it near 0 — does $\xi_\omega$ linger?
3. **Nullspace.** In task mode, can you wiggle a proximal joint without moving
   the green triad much? Compare to joint mode.
4. **Gain ratio.** Keep $k_t/d_t$ ~ critical / overdamped while sweeping
   `kt_trans_scale` and `dt_trans_scale` together vs separately.
5. **Damping only (joint).** Raise `dq_scale` alone. Does it look
   overdamped or does it chatter? Explain using the NOTE above.

---

## Takeaways

1. Joint impedance is simple and stable, but does **not** enforce Cartesian /
   orientation springs.
2. SE(3) impedance uses $\xi=\log(T^{-1}T_{\mathrm{nom}})^{\vee}$ and
   $J_b^{\top}F$ so the **palm frame** is the spring.
3. Comparing purple vs green triad + $\|\xi\|$ makes the difference visible;
   Viser gain sliders make stiffness experiments fast.
