# Joint-space vs SE(3) impedance

Impedance control means: behave like a **spring–damper** relative to a nominal pose — compliant when pushed, restoring when released. This tutorial compares two ways to define that spring:

1. **Per joint** (easy to implement, configuration-dependent in Cartesian space),
2. **On the palm’s SE(3) pose** (geometric, natural for “hold this hand frame”).

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/default.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/impedance.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/impedance.yaml
```

| Key | Mode |
|-----|------|
| `j` | Joint-space impedance |
| `t` | Task-space (palm SE(3)) impedance |
| `p`/`q` | Pause / quit |

### What you should observe

Disturb the arm in MuJoCo. In both modes it should push back toward a nominal posture / palm pose. Switch `j` ↔ `t` and compare how the **hand frame** resists translation vs rotation. Viser can show errors and let you scale gains live.

### LCM

| Channel | Direction | Role |
|---------|-----------|------|
| Joint measurements | env → ctrl | $q,\dot q$ |
| `sw_flexiv_arm_hand_joint_ctrl` | ctrl → env | Usually pure $\tau_{\mathrm{ff}}$ (PD gains zero at the plant interface) |
| `sw_imp_status` | ctrl → viz | Errors / mode |
| `sw_imp_gains_cmd` | viz → ctrl | Gain scales from the GUI |

---

## 2. Ideas

### Shared setup

Let $q_{\mathrm{nom}}$ be a comfortable default posture and

$$
T_{\mathrm{nom}} = \mathrm{FK}_{\mathrm{palm}}(q_{\mathrm{nom}})
$$

the corresponding palm pose in $\mathrm{SE}(3)$ (position + orientation).

### Joint impedance

Treat each joint as its own spring–damper:

$$
\tau
  = K_q(q_{\mathrm{nom}}-q)
  - D_q\dot q
  + g(q).
$$

**Pros:** simple, stable if gains are modest.  
**Cons:** the Cartesian stiffness at the palm **changes with configuration** — the same $K_q$ does not mean “100 N/m in $x$.”

### SE(3) impedance (body frame)

Work with the palm pose $T(q)$ and the **body** (LOCAL) twist $V_b = J_b(q)\dot q$.

Measure pose error on the group with the matrix logarithm:

$$
\xi = \log\bigl(T(q)^{-1} T_{\mathrm{nom}}\bigr)^{\vee}
   = \begin{pmatrix} \nu \\ \omega \end{pmatrix}
\quad
(\nu\text{: translation-like},\;\omega\text{: rotation-like}).
$$

Choose diagonal stiffness / damping in that chart:

$$
K = \mathrm{diag}(k_t I_3,\, k_r I_3),
\qquad
D = \mathrm{diag}(d_t I_3,\, d_r I_3).
$$

Command a body wrench and map it to joints:

$$
F = K\xi - D V_b,
\qquad
\tau = J_b(q)^\top F + g(q).
$$

The hand may still use ordinary joint springs so fingers stay roughly open/closed as desired.

### A gentle warning about “just add damping”

In a perfect single-axis toy model, more $D$ only overdamps. On a real (or simulated) arm with **delay**, **noisy $\dot q$**, and **light wrists**, large $D$ can chatter. Increase $K$ and $D$ together, and prefer modest steps when tuning in Viser.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/ImpedanceControl.py` | Modes `j` / `t`; Pinocchio `log6` + LOCAL Jacobian |
| `configs/.../impedance.yaml` | `kq`/`dq`, `kt_*`/`dt_*`, hand gains, `task_frame` |

**Try:** set very soft $k_t$ in task mode and push the palm — it should yield in translation while still roughly holding orientation if $k_r$ stays larger.
