# Collision-Aware Gravity Compensation

`CollisionAwareGravCompControl` (config
`configs/flexiv_arm_5F_hand/controllers/col_aware_grav_comp.yaml`) extends pure
grav-comp ([03](03_grav_comp.md)): far from obstacles the arm free-floats on
$g(q)$; near obstacles a planner computes a repulsive $\dot{q}_{\mathrm{des}}$
and the control loop tracks it with PD, still with gravity feedforward.
Geometry is the ColInfo stream from the collision env
([04](04_publish_collision_geom.md)).

---

## Run

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/collision.yaml
python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/col_aware_grav_comp.yaml
python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/collision.yaml
```

Enable Viser **Show collision geometry** / **Show collision distances** to see
the same FCL pairs the planner uses.

---

## Architecture (two rates)

| Thread | Rate | Output |
|--------|------|--------|
| Plan (`_plan_loop`) | $f_{\mathrm{plan}}$ (`plan_freq`, 100 Hz) | state $\in\{\texttt{grav\_comp},\texttt{repulsive},\texttt{collision}\}$, $\dot{q}_{\mathrm{des}}$ |
| Control (`_update`) | $f_{\mathrm{ctrl}}$ (`ctrl_freq`, 1 kHz) | $\tau_{\mathrm{ff}}$, $K_p$, $K_d$, $q_{\mathrm{des}}$, $\dot{q}_{\mathrm{des}}$ on LCM |

Measurements are snapshot under a lock; plans go through a size-1 queue (latest
wins). Each thread has its own Pinocchio `Data`.

---

## 1. Placing FCL geometry

For each robot link name $\ell$ in `col_link_names`, ColInfo gives a primitive
with link-local offset $T_{\mathrm{off},\ell}\in\mathrm{SE}(3)$. With FK pose
${}^{W}T_{\ell}(q)$,

$$
T_{\ell}^{\mathrm{world}}(q)
= {}^{W}T_{\ell}(q)\,T_{\mathrm{off},\ell}.
$$

Static obstacles $o\in\texttt{obs\_names}$ already carry **world** offsets
$T_{o}^{\mathrm{world}}$ (walls / floor). Both become `fcl.CollisionObject`s.

Configured pairs $(\ell,o)\in\texttt{col\_pairs}$ are the only distances
queried.

---

## 2. Distance and margin

FCL returns signed distance $d_{\ell o}$ and nearest points
$p_{\ell},\,p_{o}\in\mathbb{R}^{3}$ (on the robot and obstacle surfaces).

Define the **safety margin** (config key `safty_dist_thr` $\equiv d_{\mathrm{safe}}$):

$$
m_{\ell o} \;=\; d_{\ell o} - d_{\mathrm{safe}}.
$$

Separation vector (robot → away from obstacle along the nearest-point line):

$$
\delta_{\ell o} \;=\; p_{\ell} - p_{o},
\qquad
\hat{n}_{\ell o} \;=\; \frac{\delta_{\ell o}}{\|\delta_{\ell o}\|}.
$$

(In code, $\|\delta\|$ is clipped below by $10^{-4}$.)

Let $d_{\mathrm{margin}}$ be `magin_thr` and $\alpha$ be `repulsive_coeff`. A
scalar weight turns the unit normal into a **Cartesian repulsive velocity
target** $v_{\ell o}^{\star}\in\mathbb{R}^{3}$:

$$
w_{\ell o}
=
\begin{cases}
0, & m_{\ell o} < 0 \quad\text{(penetration / past safety)}, \\[4pt]
\alpha\displaystyle\left(\frac{m_{\ell o}}{d_{\mathrm{margin}}}-1\right)^{2},
& 0 \le m_{\ell o} < d_{\mathrm{margin}}, \\[4pt]
0, & m_{\ell o} \ge d_{\mathrm{margin}}.
\end{cases}
$$

$$
v_{\ell o}^{\star} \;=\; w_{\ell o}\,\hat{n}_{\ell o}.
$$

So $w=0$ when far; as $m\to 0^{+}$, $w\to\alpha$ (strong push); if $m<0$ the
pair is marked unhealthy and the planner enters **collision** (stop) instead of
repulsing.

**State machine over all pairs:**

| State | Condition |
|-------|-----------|
| `collision` | $\exists\,(\ell,o):\; m_{\ell o}<0$ → $\dot{q}_{\mathrm{des}}=0$ |
| `grav_comp` | $\forall\,(\ell,o):\; m_{\ell o}\ge d_{\mathrm{margin}}$ → $\dot{q}_{\mathrm{des}}=0$ |
| `repulsive` | else (some pair in the band) → solve QP below |

---

## 3. Point Jacobian

Let $J_{\ell}(q)\in\mathbb{R}^{6\times n}$ be the spatial Jacobian of link
$\ell$, and $R_{\ell},\,p_{\ell}^{\mathrm{origin}}$ its orientation / origin.
The nearest point on the robot geom is offset from the link origin by
$\Delta = R_{\ell}^{\top}(p_{\ell}-p_{\ell}^{\mathrm{origin}})$ in the link
frame. The **point Jacobian** $J_{p}\in\mathbb{R}^{3\times n}$ maps joint
velocity to that point’s Cartesian velocity:

$$
\dot{p}_{\ell} \;=\; J_{p}(q)\,\dot{q}.
$$

(Implemented as `get_point_Jacobian` from the rotational/linear blocks of
$J_{\ell}$ and $\Delta$.)

---

## 4. Repulsive QP (OSQP) — in detail

This section only runs in state `repulsive` (at least one pair has
$0\le m_{\ell o}<d_{\mathrm{margin}}$). Decision variable:

$$
x \;=\; \dot{q}\;\in\;\mathbb{R}^{n}.
$$

### Why it is a quadratic program

A **QP** is any problem of the form

$$
\min_{x}\;
\tfrac12\,x^{\top} P\,x + g^{\top}x
\qquad
\text{subject to}\quad
l \le A x \le u,
$$

with $P\succeq 0$ (here $P\succ 0$ after regularization). Our planner is exactly
that, for two reasons:

1. **Cost.** We want several Cartesian velocities of nearest points to match
   targets $v^{\star}$. Each residual $\|J_p x - v^{\star}\|_{2}^{2}$ expands to
   $$
   x^{\top}(J_p^{\top}J_p)\,x
   - 2\,(J_p^{\top}v^{\star})^{\top}x
   + \|v^{\star}\|_{2}^{2},
   $$
   which is **quadratic in $x$** (constant dropped). Summing pairs and adding
   $\varepsilon\|x\|_{2}^{2}$ keeps a quadratic objective.
2. **Constraints.** Joint limits and “do not close distance too fast” are
   **linear inequalities** in $x=\dot{q}$. No nonlinear constraints → QP, not
   NLP. OSQP solves that class efficiently at `plan_freq`.

So “repulsive QP” means: *least-squares track of repulsive Cartesian
velocities, with linear safety/limit constraints, in joint-velocity space.*

### Desired Cartesian velocity $v^{\star}$

From §2, for each pair $(\ell,o)$:

$$
v_{\ell o}^{\star} \;=\; w_{\ell o}\,\hat{n}_{\ell o},
\qquad
w_{\ell o}
=
\begin{cases}
\alpha\bigl(m_{\ell o}/d_{\mathrm{margin}}-1\bigr)^{2},
& 0\le m_{\ell o}<d_{\mathrm{margin}},\\
0, & \text{otherwise}.
\end{cases}
$$

**When is $v_{\ell o}^{\star}=0$?**

| Case | $w$ | $v^{\star}$ | Meaning in the cost |
|------|-----|-------------|---------------------|
| Far: $m_{\ell o}\ge d_{\mathrm{margin}}$ | $0$ | $0$ | Prefer $\dot{p}_{\ell}\approx 0$ for that pair (do not thrash a far contact) |
| Inside band but weight formula… | $>0$ | $\neq 0$ | Prefer to move $p_{\ell}$ **away** along $\hat{n}$ |
| Colliding: $m_{\ell o}<0$ | $0$ | $0$ | QP is **not** run; state is `collision` instead |

Important: even when $v_{\ell o}^{\star}=0$, that pair **still appears in the
sum** (code loops over all `col_pairs`). With $v^{\star}=0$ the residual is
simply $\|J_p x\|_{2}^{2}$, i.e. a soft penalty on moving that nearest point.
Pairs that are actively close ($v^{\star}\neq 0$) pull $J_p x$ toward the
outward direction; far pairs pull those points toward stillness. The small
$\varepsilon\|x\|_{2}^{2}$ prefers the smallest joint motion that satisfies the
active targets.

If **every** $v^{\star}$ were zero, we would not be in `repulsive` (that is
`grav_comp`). In `repulsive`, **at least one** $v^{\star}\neq 0$; the others may
still be zero.

### Cost → OSQP matrices

Ideal unconstrained objective (sum over all pairs $\mathcal{P}=\texttt{col\_pairs}$):

$$
J(x)
=
\sum_{(\ell,o)\in\mathcal{P}}
\bigl\| J_{p,\ell o}\,x - v_{\ell o}^{\star}\bigr\|_{2}^{2}
+
\varepsilon\|x\|_{2}^{2}.
$$

Expand:

$$
J(x)
=
x^{\top}\!\left(
\sum_{\mathcal{P}} J_{p}^{\top}J_{p}
+ \varepsilon I
\right)\!x
-
2\left(
\sum_{\mathcal{P}} J_{p}^{\top}v^{\star}
\right)^{\!\top}\!x
+ \mathrm{const}.
$$

OSQP minimizes $\tfrac12 x^{\top}P x + g^{\top}x$. Matching coefficients:

$$
P
=
2\sum_{\mathcal{P}} J_{p,\ell o}^{\top} J_{p,\ell o}
+ 0.01\,I,
\qquad
g
=
-2\sum_{\mathcal{P}} J_{p,\ell o}^{\top} v_{\ell o}^{\star}.
$$

(Here $0.01\,I$ is what the code adds; then
$\tfrac12 x^{\top}(0.01\,I)x = 0.005\|x\|_{2}^{2}$, so $\varepsilon=0.005$.)

**Special case $v^{\star}=0$ for a subset $\mathcal{P}_0\subset\mathcal{P}$:**
those terms contribute $2J_p^{\top}J_p$ to $P$ and **nothing** to $g$. The
gradient of that piece is $2J_p^{\top}J_p\,x$, zero only if $J_p x=0$ (or $x$ in
$\ker J_p$). So far pairs act as “keep this surface point still” soft
constraints, while near pairs’ nonzero $v^{\star}$ shift $g$ and bias the
solution outward.

Unconstrained stationarity $Px+g=0$ is the normal equation of stacked
least squares — classic QP / linear least squares. Inequalities below make it a
**constrained** QP.

### Inequality constraints (still linear ⇒ still QP)

**1. Joint velocity and one-horizon travel limits**
($T=\texttt{plan\_horizon}$):

$$
\max\bigl(-\dot{q}_{\max},\; (q_{\min}-q)/T\bigr)
\;\le\;
x
\;\le\;
\min\bigl(\dot{q}_{\max},\; (q_{\max}-q)/T\bigr).
$$

Each bound is $l_i \le e_i^{\top} x \le u_i$ (rows of the identity).

**2. Distance-rate limits along the contact normal**

$$
a_{\ell o}^{\top}
=
\hat{n}_{\ell o}^{\top} J_{p,\ell o}
=
\frac{(p_{\ell}-p_{o})^{\top}}{d_{\ell o}}\,J_{p,\ell o},
$$

$$
\frac{-d_{\ell o}+d_{\mathrm{safe}}}{T}
\;\le\;
a_{\ell o}^{\top} x
\;\le\;
\frac{1000}{T}.
$$

Again linear in $x$. Lower bound $\approx 0$ near the safety surface forbids
closing the gap; upper bound is a loose “do not flee infinitely fast” cap.

Stack all rows into $l\le A x\le u$ and call OSQP. If bounds are inconsistent
($u<l$) or the solver fails → $\dot{q}_{\mathrm{des}}=0$.

### Picture

```
 near pair:  minimize ||J_p x - v*||²   with  v* = w n̂  (w>0)  → push away
 far pair:   minimize ||J_p x - 0 ||²                 → prefer still
 all:        + ε||x||² and linear limit / ḋ constraints
             → QP in x = q̇
```

---

## 5. Control law (1 kHz)

Let $\dot{q}_{\mathrm{des}}$ and state be the latest plan, $\Delta t=1/f_{\mathrm{ctrl}}$,
and $S=\mathrm{diag}(\texttt{kp\_multiplier})$.

$$
q_{\mathrm{des}}
=
q + S\,\dot{q}_{\mathrm{des}}\,\Delta t,
$$

$$
\tau_{\mathrm{ff}}
=
g(q) - b_{\mathrm{arm}}\,\dot{q}_{\mathrm{arm}}
\qquad (b_{\mathrm{arm}}=0.01\text{ on the first 7 DoFs}).
$$

Gains:

$$
(K_p,K_d)
=
\begin{cases}
(0,0), & \text{state}=\texttt{grav\_comp}, \\
(K_p^{\mathrm{cfg}},K_d^{\mathrm{cfg}}), & \text{else}.
\end{cases}
$$

The plant applies (tutorial 03):

$$
\tau
=
K_p(q_{\mathrm{des}}-q) + K_d(\dot{q}_{\mathrm{des}}-\dot{q}) + \tau_{\mathrm{ff}}.
$$

### Closed-loop reading by state

**`grav_comp`.** Same as [03](03_grav_comp.md): $K_p=K_d=0$,
$\tau=\tau_{\mathrm{ff}}\approx g(q)$, free float with light arm damping.

**`repulsive`.** PD tracks a one-step position increment along $\dot{q}_{\mathrm{des}}$
while gravity is cancelled. In Cartesian terms the planner asked for
$\dot{p}\approx v^{\star}$ (away from the wall); the QP maps that into joints
under velocity / distance-rate limits. Net effect: move clear of the obstacle,
then margins grow and the state returns to `grav_comp`.

**`collision`.** $\dot{q}_{\mathrm{des}}=0$, so $q_{\mathrm{des}}=q$, and PD +
gravity FF try to **freeze** the pose (brake), not push through contact.

---

## 6. Tuning (math → knobs)

| Symbol / role | Config key |
|---------------|------------|
| $d_{\mathrm{safe}}$ | `safty_dist_thr` |
| $d_{\mathrm{margin}}$ | `magin_thr` |
| $\alpha$ (peak $w$) | `repulsive_coeff` |
| $T$ | `plan_horizon` |
| $f_{\mathrm{plan}}$ | `plan_freq` |
| Pairs $(\ell,o)$ | `col_pairs` (+ `col_link_names`, `obs_names`) |
| $K_p,K_d$ when not floating | `kp`, `kd` |

Larger $\alpha$ / wider $d_{\mathrm{margin}}$ → earlier, stronger repulsion.
Larger $d_{\mathrm{safe}}$ → stops sooner. Wrong ColInfo or pairs → wrong
$d_{\ell o}$ and $J_p$ → the math is fine but the robot still hits the real
wall.

---

## Files

| File | Role |
|------|------|
| `controller/CollisionAwareGravCompControl.py` | FCL, QP, control |
| `configs/.../controllers/col_aware_grav_comp.yaml` | Thresholds, pairs, gains |
| `tutorial/03_grav_comp.md` | Pure free-float baseline |
| `tutorial/04_publish_collision_geom.md` | Where $T_{\mathrm{off}}$ and walls come from |
