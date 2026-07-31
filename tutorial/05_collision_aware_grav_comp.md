# Collision-aware gravity compensation

Gravity compensation ([03](03_grav_comp.md)) lets the arm float — including into a wall. Here we add a **slow planner** that, when links get close to obstacles, gently pushes them away, while the **fast loop** still cancels gravity.

---

## 1. Run & LCM communication

```bash
python run_env.py --config configs/flexiv_arm_5F_hand/env/collision.yaml

python run_controller.py --config configs/flexiv_arm_5F_hand/controllers/col_aware_grav_comp.yaml

python run_visualizer.py --config configs/visualizer/flexiv_arm_5F_hand/collision.yaml
```

| Key | Action |
|-----|--------|
| `p` / `q` | Pause / quit |

### What you should observe

Far from walls, behavior feels like plain grav-comp. Move (or drift) toward a wall or floor: the arm should **back off** instead of sinking through the planning geometry. Viser can show collision shapes and distance cues.

### Two rates (important mental model)

| Loop | Typical rate | Responsibility |
|------|--------------|----------------|
| **Plan** | 100 Hz | Measure distances with FCL; solve a small QP for a desired joint velocity $\dot q_{\mathrm{des}}$ |
| **Control** | 1 kHz | Track that $\dot q_{\mathrm{des}}$ with damping + send $g(q)$ |

Planning is heavier (geometry + OSQP). Control must stay fast and boring.

### LCM

| Channel | Direction | Role |
|---------|-----------|------|
| Arm / hand `*_joint_meas` | env → ctrl | State |
| `sw_flexiv_arm_hand_robot_col_info` | env → ctrl / viz | Robot shapes ([04](04_publish_collision_geom.md)) |
| `sw_flexiv_arm_hand_static_col_info` | env → ctrl / viz | Walls / floor |
| `sw_flexiv_arm_hand_joint_ctrl` | ctrl → env | Command |

---

## 2. Ideas

### Step A — Where am I relative to the wall?

For each monitored link–obstacle pair, FCL returns a distance $d$ and nearest points. Define a **margin residual**

$$
m = d - d_{\mathrm{safe}}.
$$

- $m \gg 0$: comfortably far.
- $0 \le m < d_{\mathrm{margin}}$: inside a soft “warning band” → start repelling.
- $m < 0$: penetrating the safety margin → treat as collision / stop.

The outward unit direction $\hat n$ points from the obstacle toward the robot point.

### Step B — Decide a mode

| Situation | Mode | Intuition |
|-----------|------|-----------|
| Any $m<0$ | Stop | Do not command motion that digs deeper |
| All $m \ge d_{\mathrm{margin}}$ | Pure grav-comp | Nothing to avoid |
| Otherwise | Repulsive QP | Ask joints to move the threatened points along $\hat n$ |

### Step C — A soft repulsive “velocity field”

Inside the warning band, scale a desired Cartesian escape speed by how deep you are:

$$
w = \alpha\Bigl(\frac{m}{d_{\mathrm{margin}}}-1\Bigr)^{2},
\qquad
v^\star = w\,\hat n.
$$

Far edge of the band → $w\approx 0$. Near violation → larger $w$.

### Step D — Turn Cartesian wishes into joint rates (QP)

Each threatened point has a Jacobian $J$ mapping $\dot q$ to point velocity. We ask:

$$
\min_{\dot q}
  \sum \|J\dot q - v^\star\|^2
  + \varepsilon\|\dot q\|^2
$$

with joint speed limits and linearized “do not decrease distance too fast” constraints. The solution is the plan’s $\dot q_{\mathrm{des}}$.

### Step E — Fast tracking

Every millisecond:

$$
\tau_{\mathrm{ff}} \approx g(q) - b\dot q_{\mathrm{arm}},
$$

and PD / velocity damping tracks the latest $\dot q_{\mathrm{des}}$ from the plan mailbox.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `controller/CollisionAwareGravCompControl.py` | Plan thread (FCL + OSQP) + 1 kHz update |
| `FlexivArmHandColEnv` | Publishes ColInfo |
| `configs/.../col_aware_grav_comp.yaml` | `safty_dist_thr`, `magin_thr`, `repulsive_coeff`, `plan_freq`, `col_pairs` |

**Note:** some YAML keys are intentionally spelled `safty` / `magin` (historical). Match the file, do not “fix” them silently.

**Tuning intuition:** larger `repulsive_coeff` → stronger escape; larger `ik`-style regularization (if exposed) → milder joint motion; larger `safty_dist_thr` → earlier reaction.
