# FoundationPose in MuJoCo (sim RGB-D)

Track a textured mesh object in a **robot-free** MuJoCo scene: the env publishes RGB-D over LCM, FoundationPose estimates 6D pose, and Viser overlays the estimate against ground truth.

---

## 1. Prerequisites

Use the `rophi` conda env (or equivalent) with MuJoCo, LCM, torch, and CUDA.

### LCM multicast (Linux)

```bash
sudo ifconfig lo multicast
sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev lo
```

### nvdiffrast (CUDA)

`nvcc` and PyTorch must report the **same** CUDA major.minor (the build checks this):

```bash
nvcc --version                    # e.g. release 12.8
python -c "import torch; print(torch.version.cuda)"  # must match, e.g. 12.8
```

If they differ (common error: `detected CUDA version 12.8` vs `PyTorch 13.0`), either reinstall PyTorch for your toolkit **or** point `CUDA_HOME` at a toolkit that matches PyTorch.

**Option A — keep system CUDA 12.8, force-reinstall PyTorch cu128**  
(`pip install ... cu128` alone is a no-op if the same version is already installed as cu130):

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda)"  # expect ...+cu128 12.8
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

**Option B — keep PyTorch cu130, use a CUDA 13** `nvcc`  
Install the CUDA 13 toolkit (or `nvidia-cuda-nvcc` wheel) and set `CUDA_HOME` to that toolkit so `nvcc --version` reports 13.x.

Then build nvdiffrast:

```bash
pip install setuptools wheel ninja
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```



### Third-party stacks (not vendored)

FoundationPose and SAM2 are imported as `third_party.FoundationPose` / `sam2`.

**SAM2 (required for** `build_sam2_camera_predictor`**)** — use the real-time fork, not stock Meta SAM2.

Use `--no-build-isolation` so the CUDA extension builds against **your** env’s PyTorch (isolated builds often pull a mismatched CUDA torch and fail with `12.8` vs `13.0`):

```bash
# From the repo root; CUDA_HOME must match torch.version.cuda
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
python -c "import torch; print(torch.__version__, torch.version.cuda)"  # expect ...+cu128 12.8

mkdir -p third_party && cd third_party
git clone https://github.com/Gy920/segment-anything-2-real-time.git sam2
cd sam2
pip install -e . --no-build-isolation
python -c "from sam2.build_sam import build_sam2_camera_predictor; print('sam2 OK')"
cd ../..
```

Checkpoints for the tutorial config (`sam2.1_hiera_tiny`) are pulled from HuggingFace on first run (`facebook/sam2.1-hiera-tiny`). You can also download them into `third_party/sam2/checkpoints/` via that repo’s `checkpoints/download_ckpts.sh`.

**FoundationPose + pytorch3d:**

```bash
mkdir -p third_party && cd third_party
git clone https://github.com/NVlabs/FoundationPose.git
cd ../..

# pytorch3d — FoundationPose only needs `pytorch3d.transforms` (no CUDA ops).
# Prefer a CPU-only build (minutes, not hours). If a prior build hung:
#   pkill -f 'nvcc|pip install.*pytorch3d'
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
FORCE_CUDA=0 MAX_JOBS=8 pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
python -c "from pytorch3d.transforms import so3_exp_map; print('pytorch3d OK')"

# Remaining FoundationPose Python deps (includes `transformations`, open3d, trimesh, …)
pip install -r third_party/FoundationPose/requirements.txt

# Build the native `mycpp` extension (required — without it you get
# `AttributeError: module 'mycpp' has no attribute 'cluster_poses'`).
# CMake deps (Eigen, Boost, pybind11). Prefer system g++ (build_all_conda.sh
# uses /usr/bin/g++ when present) so the .so matches the libstdc++ that
# Torch/OpenCV already load; conda GCC 14+ can fail with CXXABI_1.3.15.
conda install -y -c conda-forge eigen boost-cpp pybind11 ninja
cd third_party/FoundationPose && bash build_all_conda.sh && cd ../..
python -c "from third_party.FoundationPose.Utils import mycpp; assert hasattr(mycpp,'cluster_poses'); print('mycpp OK', mycpp.__file__)"

# Network weights (required): download from
#   https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing
# and place under third_party/FoundationPose/weights/
# Need folders:
#   weights/2023-10-28-18-33-37/   # pose refiner
#   weights/2024-01-11-20-02-45/   # score model  (config.yml + ckpts)
mkdir -p third_party/FoundationPose/weights
# After download, you should have e.g.:
#   third_party/FoundationPose/weights/2024-01-11-20-02-45/config.yml
```

If you later need pytorch3d CUDA ops, rebuild for **your GPU arch only** (much faster than all arches), e.g. `TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 MAX_JOBS=8 pip install --no-build-isolation --force-reinstall "git+https://github.com/facebookresearch/pytorch3d.git"`.

Also required: `transformers` / HuggingFace access for the SAM2 checkpoint (downloaded on first run).

---



## 2. Run (three terminals)

```bash
# Terminal 1 — MuJoCo table + mustard bottle + RGB-D
python run_env.py --config configs/perception_tutorial/env/table_mesh_cam.yaml

# Terminal 2 — FoundationPose + SAM2 tracking
python run_perception.py --config configs/perception_tutorial/foundation_pose_sim.yaml

# Terminal 3 — Viser overlay (estimate + GT ghost)
python run_visualizer.py --config configs/perception_tutorial/visualizer/obj_pose.yaml
```

Open Viser at [http://localhost:8080](http://localhost:8080).

### Interaction


| Key / action     | Where                    | Effect                                          |
| ---------------- | ------------------------ | ----------------------------------------------- |
| Click on mustard | perception OpenCV window | First-frame SAM2 mask → FoundationPose register |
| `s`              | env                      | Toggle slow yaw spin (makes tracking obvious)   |
| `r`              | env                      | Reset object pose                               |
| `p` / `q`        | env / perception / vis   | Pause / quit                                    |




### Suggested first run

1. Start env → confirm mustard is on the table and the MuJoCo viewer shows the camera marker.
2. Start perception → wait until camera intrinsics/extrinsics initialize, then **click the bottle**.
3. Start visualizer → opaque mesh = FoundationPose estimate; translucent ghost = sim GT.
4. Press `s` in the env to spin the object and watch the estimate follow.

---



## 3. LCM channels


| Channel                 | Type            | Producer → consumer                                     |
| ----------------------- | --------------- | ------------------------------------------------------- |
| `upper_camera`          | `rgbd_t`        | Env → FoundationPose                                    |
| `upper_camera_info`     | `camera_info_t` | Env → FoundationPose (`K`, world→cam, `depth_factor=1`) |
| `fp_obj_pose`           | NamedVec 12     | FoundationPose → Viser                                  |
| `sim_obj_pose_bb2world` | NamedVec 12     | Env (GT) → Viser ghost                                  |


Pose encoding (oriented bounding box → world):

$$
\mathbf{v} = [t_x, t_y, t_z, R_{00},\ldots,R_{22}]
$$

CAD pose for mesh rendering:

$$
T_{\mathrm{CAD}\to W}
  = T_{\mathrm{BB}\to W}T_{\mathrm{CAD}\to\mathrm{BB}}.
$$

---



## 4. What you should observe

- After the click, the OpenCV debug view draws a posed box / axes on the bottle.
- In Viser, the opaque mesh should sit on the translucent GT ghost (small lag is normal while tracking).
- With spin (`s`), both meshes rotate together; if the estimate drifts, SAM2 auto-reset may re-register.



### Rates


| Process    | Typical rate                 | Work                             |
| ---------- | ---------------------------- | -------------------------------- |
| Env        | 500 Hz physics, ~15 Hz RGB-D | MuJoCo step + camera render      |
| Perception | camera-limited               | SAM2 mask + FoundationPose track |
| Viser      | 30 Hz                        | Mesh / axes update               |


---



## 5. Common failures


| Symptom                                                                              | Likely cause                                                         |
| ------------------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| `Couldn't create LCM`                                                                | Multicast route missing (see above)                                  |
| `No module named 'third_party.FoundationPose'` / `nvdiffrast` / `sam2` / `pytorch3d` / `transformations` | Third-party / FP requirements incomplete |
| `weights/.../config.yml` not found | Download FoundationPose weights into `third_party/FoundationPose/weights/` |
| Perception never leaves “waiting for camera”                                         | Env not running, or channel ≠ `upper_camera`                         |
| Click does nothing                                                                   | Click the OpenCV window focused on the bottle; ensure depth is valid |
| Viser mesh at origin forever                                                         | No messages on `fp_obj_pose` yet (click / register first)            |
| Texture missing / black object                                                       | Missing `material_0.png` next to `textured_mesh.obj`                 |


---



## 6. Code map


| Piece             | Path                                                               |
| ----------------- | ------------------------------------------------------------------ |
| Scene XML         | `assets/scene/perception_tutorial/table_mesh_cam.xml`              |
| Env               | `env/perception_tutorial/TableMeshCamEnv.py`                       |
| Cam LCM pub       | `communication/lcm/publisher/pub_manager/env/CamOnlyPubManager.py` |
| Perception config | `configs/perception_tutorial/foundation_pose_sim.yaml`             |
| Viser manager     | `visualizer/vis_manager/perception_tutorial/ObjPoseVisManager.py`  |


