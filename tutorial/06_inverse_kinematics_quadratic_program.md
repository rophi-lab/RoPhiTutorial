# Inverse Kinematics via Quadratic Program

`IKQPControl` (config `configs/flexiv_arm_5F_hand/controllers/ik_qp.yaml`)
moves a **task-space point** (index fingertip frame `index_tip`) toward a
keyboard-moved 3D target $p_{\mathrm{des}}$ by solving a velocity-level **IK
QP** with linearized FCL collision inequalities.

It builds on:

- gravity / friction feedforward and joint LCM ([03](03_grav_comp.md)),
- ColInfo geometry from the collision env ([04](04_publish_collision_geom.md)),
- the two-rate plan / control split used in collision-aware grav-comp
  ([05](05_collision_aware_grav_comp.md)).

---

## Run

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/collision.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/ik_qp.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/ik_qp.yaml
```

**Keyboard (controller terminal):**

| Key | Action |
|-----|--------|
| `w` / `s` | $p_{\mathrm{des}}$ ±$x$ |
| `a` / `d` | $p_{\mathrm{des}}$ ±$y$ |
| `r` / `f` | $p_{\mathrm{des}}$ ±$z$ |
| `[` / `]` | halve / double step size |
| `h` | home mode (joint-space go to `default_q`) |
| `p` / `q` | pause / quit |

**Viser:** green sphere = target $p_{\mathrm{des}}$, red sphere = current tip
$p$, cyan arrow = commanded tip velocity $J_p\dot{q}_{\mathrm{des}}$.

---

## Architecture (two rates)

| Thread | Rate | Work |
|--------|------|------|
| Plan (`_plan_loop`) | `plan_freq` (100 Hz) | FK, contracting field $v^\star$, FCL, OSQP → $\dot{q}_{\mathrm{des}}$ |
| Control (`_update`) | `ctrl_freq` (1 kHz) | track latest $\dot{q}_{\mathrm{des}}$, gravity / Coriolis / friction FF, publish pose |

FCL + OSQP are **not** run at 1 kHz. Doing that missed control ticks and made
$\dot{q}_{\mathrm{des}}$ update in bursts. Plans go through a size-1 queue
(latest wins); each thread has its own Pinocchio `Data`.

---

## 1. Task point and Jacobian

Config: `task_link: index_tip`, `task_point_local: [0,0,0]`.

$$
p(q) \;=\; p_{\mathrm{link}}(q) + R_{\mathrm{link}}(q)\,\Delta,
\qquad
\dot{p} \;=\; J_p(q)\,\dot{q}.
$$

$J_p\in\mathbb{R}^{3\times n}$ is the point Jacobian (`get_point_Jacobian` on
the link Jacobian in `LOCAL_WORLD_ALIGNED` form). With $\Delta=0$ on
`index_tip`, $J_p$ is just the linear block of that frame’s Jacobian.

---

## 2. Contracting Cartesian velocity field

Away from the target ($\|p_{\mathrm{des}}-p\| \ge \texttt{pos\_tol}$):

$$
v^\star
\;=\;
K_v\,(p_{\mathrm{des}}-p)
\;-\;
D_v\,\dot{p},
\qquad
\|v^\star\| \le v_{\max}.
$$

Config: `vel_gain` $=K_v$, `cart_damping` $=D_v$, `v_max`.
Inside the deadzone, $v^\star=0$ and the planner reports `at_goal`.

Modes:

- **`ik`** — track $p_{\mathrm{des}}$ via the QP below.
- **`home`** (`h`) — joint-space field
  $\dot{q}^\star = K_j(q_{\mathrm{home}}-q)-D_j\dot{q}$ (no IK QP).

---

## 3. IK QP (OSQP)

Decision variable $x=\dot{q}\in\mathbb{R}^{n}$. Cost (task tracking + small
damping):

$$
\min_x\;
\|J_p x - v^\star\|_2^2
+ \tfrac{\varepsilon}{2}\|x\|_2^2
\;=\;
\min_x\;
\tfrac12 x^\top P x + g^\top x,
$$

with (matching the code’s OSQP scaling)

$$
P \;=\; 2\,J_p^\top J_p + \varepsilon I,
\qquad
g \;=\; -2\,J_p^\top v^\star,
\qquad
\varepsilon=\texttt{qp\_reg}.
$$

### Joint velocity box

Over horizon $T=\texttt{plan\_horizon}$:

$$
l_i
=\max\!\Bigl(-\dot{q}_i^{\max},\;
\tfrac{q_i^{\mathrm{lo}}-q_i}{T}\Bigr),
\qquad
u_i
=\min\!\Bigl(\dot{q}_i^{\max},\;
\tfrac{q_i^{\mathrm{hi}}-q_i}{T}\Bigr).
$$

$q$ is clipped into $[q^{\mathrm{lo}},q^{\mathrm{hi}}]$ before forming these
rates so measurement overshoot cannot create $l>u$.

### Linearized collision inequalities

For each pair $(\ell,o)$ in `col_pairs`, FCL gives distance $d$ and nearest
points $p_\ell$, $p_o$. With point Jacobian $J_{\mathrm{pt}}$ at $p_\ell$,

$$
\hat{n}^\top J_{\mathrm{pt}}\,x
\;\ge\;
s,
\qquad
s
=\mathrm{clip}\!\left(
\tfrac{-d+d_{\mathrm{safe}}}{T},\;
-\infty,\;
s_{\max}
\right),
$$

where $d_{\mathrm{safe}}=\texttt{safty\_dist\_thr}$ and
$s_{\max}=\texttt{col\_sep\_rate\_max}$.

Interpretation: do not close the gap faster than reaching $d_{\mathrm{safe}}$
in time $T$; if already inside the margin, demand at most $s_{\max}$ of
opening rate (keeps the QP from asking for impossible escapes).

---

## 4. Infeasibility → stop

If OSQP cannot find any $x$ satisfying the **joint box and collision rows**,
the planner sets

$$
\dot{q}_{\mathrm{des}} \;=\; 0.
$$

There is **no** cascade that drops collision constraints or falls back to
unconstrained $J_p^{+}v^\star$. Prefer stopping over commanding a motion that
may violate $d>d_{\mathrm{safe}}$.

Typical cause: a large $v^\star$ (or a target behind a wall) that cannot be
realized without closing a near gap.

---

## 5. Control law (1 kHz)

Consume the latest plan, low-pass $\dot{q}_{\mathrm{des}}$ with `qd_filter`
$\alpha\in(0,1]$:

$$
\dot{q}_{\mathrm{filt}}
\leftarrow
\alpha\,\dot{q}_{\mathrm{des}}
+(1-\alpha)\,\dot{q}_{\mathrm{filt}}.
$$

| Situation | $q_{\mathrm{des}}$ | $K_p$ | $K_d$ |
|-----------|-------------------|-------|-------|
| At goal / hold | $q$ or $q_{\mathrm{home}}$ | from `default_kd` (critical damping) | `default_kd` |
| Tracking (`use_position_pd: false`) | $q$ | $0$ | `velocity_control_D_gain` |

Feedforward:

$$
\tau_{\mathrm{ff}}
\;=\;
g(q)
\;+\;
C(q,\dot{q})\,\dot{q}
\;+\;
F_{\mathrm{jc}}\,\mathrm{sat}\!\left(
\tfrac{\dot{q}_{\mathrm{des}}}{\phi}
\right),
$$

with Coulomb boundary layer $\phi=\texttt{friction\_phi}$ (viscous plant
friction is intentionally **not** cancelled).

Position-hold gains follow the delay-aware critical-damping design:

$$
K_p
\;=\;
\frac{K_d^{2}}{4\,\mathrm{diag}(\bar{M})},
\qquad
\bar{M}-\delta_t K_d > 0,
\quad
K_d-K_p\delta_t > 0
$$

(`est_time_delay` $=\delta_t$, `mean_mass_matrix_diag` $=\bar{M}$).

Plant applies

$$
\tau
\;=\;
K_p(q_{\mathrm{des}}-q)
+ K_d(\dot{q}_{\mathrm{des}}-\dot{q})
+ \tau_{\mathrm{ff}}.
$$

---

## 6. Publishing the target

Each control tick publishes `pose.pose3d_t` on `pose_channel`
(`sw_ik_target_pose`):

| Field | Meaning |
|-------|---------|
| `position` | $p_{\mathrm{des}}$ |
| `tip` | current task point $p$ |
| `velocity` | $J_p\dot{q}_{\mathrm{des}}$ (arrow in Viser) |

---

## Config map (high-signal keys)

| Key | Role |
|-----|------|
| `task_link` / `task_point_local` | tip frame |
| `vel_gain`, `cart_damping`, `v_max`, `pos_tol` | contracting field |
| `plan_freq`, `plan_horizon`, `qp_reg` | planner / QP |
| `safty_dist_thr`, `col_sep_rate_max`, `col_pairs` | collision |
| `velocity_control_D_gain`, `default_kd`, `mean_mass_matrix_diag` | tracking / hold |
| `Fjc`, `friction_phi` | Coulomb FF |
| `qd_filter` | control-side smoothing of $\dot{q}_{\mathrm{des}}$ |

---

## Exercises

1. **Deadzone.** Increase `pos_tol`. How does end-point chatter change near the
   green sphere?
2. **Horizon vs safety.** Halve `plan_horizon`. When does the QP print
   infeasible and freeze more often near walls?
3. **Field speed.** Raise `v_max` with collision viz on. Confirm that when the
   tip path would pierce a wall, $\dot{q}_{\mathrm{des}}$ goes to zero instead of
   sliding along the surface.
4. **Home.** Press `h`, then nudge with `w`. Confirm mode returns to `ik` and
   reseeds $p_{\mathrm{des}}$ at the current tip before applying the step.
5. **Tip frame.** Set `task_link: finger_r_link_2_index4` with
   `task_point_local: [0,0,0.02425]` and compare to `index_tip` (should match).

---

## Takeaways

1. Velocity-level IK + collision is naturally a **QP**: quadratic tracking cost,
   linear joint / distance inequalities.
2. Keep **planning slow** (FCL+OSQP) and **control fast** (PD + FF).
3. If the safe set and $v^\star$ conflict, **stop** — do not relax collision.
4. Friction / gravity FF and a dedicated velocity $K_d$ make $\dot{q}_{\mathrm{des}}$
   track on the plant; the QP alone does not apply torque.
