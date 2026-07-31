# MuJoCo mesh grasp evaluation (RH-5)

This tutorial asks a different question from [10](10_reactive_grasping.md). There we **reached** for a grasp. Here the hand is already at a precomputed multi-contact pose, and we ask:

> Can we **hold** this grasp under gravity and disturbances, using only geometry (a distance field) and a real-time force-closure QP?

It is a hand-only MuJoCo eval (Robotis RH-5), ported from ReactiveGrasp’s offline mesh script into the usual env / controller / visualizer pattern.

---

## 1. Run & LCM communication

GPU unsigned distance fields need Warp (`pip install warp` with CUDA). CPU mesh fallback exists but is slower.

```bash
python run_env.py --config configs/robotis_5f/env/grasp_eval.yaml

python run_controller.py --config configs/robotis_5f/controllers/grasp_eval.yaml

python run_visualizer.py --config configs/visualizer/robotis_5f/grasp_eval.yaml
```

| Key | Where | Action |
|-----|--------|--------|
| `r` | env | Reset object + current grasp |
| `n` | env | Next grasp index, then reset |
| `p` | env | Apply an SE(3) **perturbation** wrench for a short burst |
| `g` | env | Toggle world gravity |
| `v` | controller | Toggle reactive force control |
| `g` | controller | Toggle hand gravity compensation |
| `p` / `q` | controller | Pause / quit |

### Suggested first run

Start all three → press `v` on the controller. Contact arrows should appear; the hand should squeeze and hold. Try env `p` to nudge the object. Use `n` to walk through stored grasps.

### What you should observe

Magenta force arrows and cyan normals on the mesh; the object should stay in the fingers if the grasp is strong. Weak grasps may slip when you perturb or enable gravity.

### Rates

| Process | Rate | Work |
|---------|------|------|
| Env | 1 kHz | MuJoCo; apply $\tau$; optional contact overlay |
| Controller | 200 Hz | Distance field → contacts → QP → $\tau_{\mathrm{ff}}$ |
| Viser | 60 Hz | Mesh + arrows |

### LCM

| Channel | Content |
|---------|---------|
| `sim_hand_joint_meas` | Hand $q,\dot q,\tau\in\mathbb{R}^{20}$ |
| `sim_obj_pose_bb2world` | OBB→world pose (+ physics params) |
| `sim_joint_ctrl` | Pure feedforward $\tau$ ($K_p=K_d=0$ at the interface) |
| `sim_contact_forces_in_world` | `"i"`: $[p\,|\,f]$; `"n_i"`: $[p\,|\,n]$ |
| `sim_net_force_arrow` | Optional net force at the CAD origin |

CAD pose recovery:

$$
T_{\mathrm{CAD}\to W}
  = T_{\mathrm{BB}\to W}\,T_{\mathrm{CAD}\to\mathrm{BB}}.
$$

---

## 2. Ideas

### Placing a stored grasp

The palm is fixed in the MJCF. Grasp assets store poses relative to the palm (`SE3_grasp_pose_in_palm_frame`). For grasp index $i$, the env places the object’s freejoint so that CAD realizes that grasp, and sets the hand to $q_{\mathrm{grasp}}$. You start **already closed** on the object — no reaching phase.

### Distance field → virtual contacts

Build an **unsigned distance field** (UDF) of the object mesh (Warp grid on GPU, or CPU mesh queries). Sample points $p^\ell$ on the hand collision surfaces. For each sample, query distance $d$ and outward normal $\hat n_{\mathrm{out}}$. Squeeze normals point into the object: $\hat n=-\hat n_{\mathrm{out}}$.

Bucket the samples:

$$
\mathcal{C}=\{i:d_i<d_{\mathrm{contact}}\},
\qquad
\mathcal{A}=\{i:d_{\mathrm{contact}}\le d_i<d_{\mathrm{attract}}\}.
$$

- $\mathcal{C}$: “touching” — enter the force QP.
- $\mathcal{A}$: “nearby” — attract gently toward the surface.

A contact point on the object can be reconstructed as $c=p-d\,\hat n_{\mathrm{out}}$.

### Attraction and tip approach (before / beside the QP)

Nearby samples pull with a spring-like force along the squeeze normal:

$$
f_i = k_a d_i\hat n_i,
\qquad
\tau_{\mathrm{attract}}=\sum_{i\in\mathcal{A}} J_i^\top f_i.
$$

Fingers that are still far get a **tip approach** velocity field toward the nearest surface, mapped through a small regularized IK and an EMA, then applied as a masked damper $\tau_{\mathrm{approach}}$. Intuition: close the air gap so the QP has contacts to work with.

### Force-closure QP (the hold)

At contacts, build frames $\{n,t_1,t_2\}$ and a grasp map $G$ so the net wrench is $w=Gf$. Solve roughly:

$$
\min_f \;\|w\|^2
\quad(+\;\text{optional robustness weights})
$$

subject to non-negative normal forces, a friction pyramid, $\sum f_n \ge f_{n,\min}$, and optional torque / achievable-force limits.

Map the solved contact forces to joint torque:

$$
\tau_{\mathrm{QP}}=\sum_{i\in\mathcal{C}} J_i^\top f_i^{W}.
$$

If OSQP fails, keep (latch) the previous $\tau_{\mathrm{QP}}$ so the hand does not suddenly go limp. Total command:

$$
\tau
  = \tau_{\mathrm{QP}}
  + \tau_{\mathrm{attract}}
  + \tau_{\mathrm{approach}}
  + \mathbf{1}_{\mathrm{grav}}\,g(q).
$$

### Perturbation test

Env key `p` applies a short, critically damped body wrench from a random SE(3) offset. It is a quick robustness probe: good grasps recover; fragile ones slip.

---

## 3. How it is implemented

| Piece | Role |
|-------|------|
| `env/robotis_5f/MjMeshesGraspEvalEnv.py` | Physics, grasp placement, perturbation |
| `controller/MjEvalGraspControl.py` | UDF contacts, attraction, force QP |
| `utils/grasping/distance_field.py` | UDF / mesh factory |
| `utils/grasping/force_gen.py` | Contact frames + OSQP |
| `utils/grasping/link_surface_sampler.py` | Hand surface samples |
| `configs/robotis_5f/controllers/grasp_eval.yaml` | Thresholds and QP knobs |

`control_type` can select real-time feedforward force (default), impedance-style variants, or offline frozen-contact modes. Tutorial [12](12_reactive_grasping_force_closure.md) reuses this closing stack on the full arm+hand after a reactive reach.
