# Inverse kinematics + reach

Earlier we chased a **moving** fingertip target with a velocity QP ([06](06_inverse_kinematics_quadratic_program.md)), and we tracked a **joint-space path** with PD ([07](07_computed_torque_control.md)). Here we do something in between:

1. Sample a desired **palm pose** (and a hand posture),
2. Solve **arm-only** inverse kinematics offline with a collision check,
3. Gently regulate the whole robot to that joint solution $q^\star$.

Think of it as: “decide where the palm should sit, find a joint pose that realizes it, then spring toward that pose.”

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/collision.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/ik_reach.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/ik_reach.yaml
```

| Key | Action |
|-----|--------|
| `1` | Sample a new target palm pose + hand joints |
| `2` | Solve IK; accept only if FCL says the pose is collision-free |
| `3` | Regulate to the accepted $q^\star$ |
| `g` / `h` | Gravity compensation / hold |
| `p` / `q` | Pause / quit |

### Suggested workflow

Press `1` until Viser shows a palm frame you like, then `2`. If IK fails or the solution collides, try `1` again. When you have a green (accepted) solution, press `3` and watch the arm pull toward it.

### What you should observe

Unlike [07](07_computed_torque_control.md), there is **no RRT path**. After `3`, the robot goes **straight in joint space** toward $q^\star$. That is simpler — and also why we only accept collision-free solutions: the path itself is not re-checked every step.

### LCM

| Channel | Direction | Role |
|---------|-----------|------|
| Arm / hand measurements | env → ctrl | Current $q,\dot q$ |
| Robot / static ColInfo | env → ctrl | FCL collision shapes |
| `sw_flexiv_arm_hand_joint_ctrl` | ctrl → env | Regulation or grav-comp |
| `sw_ik_reach_palm_pose` | ctrl → viz | Desired palm SE(3) for drawing |
| `sw_p2p_*` | ctrl ↔ viz | Status / gains (shared naming with the P2P visualizer) |

---

## 2. Ideas

### What “SE(3) palm target” means

A rigid pose is an element of $\mathrm{SE}(3)$: position plus orientation. Call the desired palm pose $T_{\mathrm{des}}$. Forward kinematics from the arm joints gives the current palm pose $T(q)$.

We measure the error **on the group**, not with naive Euler angles:

$$
\xi = \log\bigl(T(q)^{-1} T_{\mathrm{des}}\bigr)^{\vee}
   = \begin{pmatrix} \nu \\ \omega \end{pmatrix}.
$$

Here $\nu$ is a translation-like residual and $\omega$ a rotation-like residual in the palm’s body frame. When $\xi\approx 0$, the palm is where we want it.

### Closed-loop inverse kinematics (CLIK)

We only move the **arm** joints. Each iteration takes a damped least-squares step with the LOCAL (body) Jacobian $J$ that matches the `log6` chart:

$$
\Delta q = J^{+}\xi,
\qquad
J^{+} = J^{\top}(JJ^{\top}+\lambda^{2}I)^{-1}.
$$

- $\lambda$ (damping) keeps the step calm near singularities.
- Clamp joints to limits after each step.
- Stop when $\|\xi\|<\varepsilon_{\mathrm{IK}}$ or you hit the iteration budget.

The hand joints in $q^\star$ come from the sample (step `1`); they are not solved by CLIK.

### Collision acceptance

A mathematically perfect IK solution can still put a forearm through a wall. After CLIK converges, run FCL self- and environment-collision checks at $q^\star$. Reject and resample if anything overlaps.

### Regulation (the “go there” spring)

Once $q^\star$ is accepted:

$$
\tau = K_p(q^\star - q) - K_d\dot q + g(q).
$$

This is joint impedance toward a fixed goal — the same spirit as [08](08_impedance_control.md), but with a freshly computed setpoint instead of a fixed home.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/IKReachControl.py` | FSM: sample → CLIK → regulate |
| Pinocchio FK / Jacobian / `log6` | Pose residual and $J^{+}$ |
| `configs/.../ik_reach.yaml` | `task_frame`, `ik_tol`, `ik_damping`, pose sample box, gains, `col_pairs` |

**Learning tip:** if regulation looks too aggressive, lower $K_p$ (or raise $K_d$) from the GUI before blaming IK. If IK often fails near walls, shrink the pose sample box in the YAML.
