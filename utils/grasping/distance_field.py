"""Distance field factory for grasp contact detection.

Default: Warp ``UDFGrid`` on ``cuda:0`` (GPU precompute of the UDF).
Fallback: CPU ``MeshDistanceField`` (trimesh proximity) if Warp is missing
or the requested device is unavailable.
"""

from __future__ import annotations

from typing import Any


def build_distance_field(
    mesh,
    resolution: int = 128,
    padding: float = 0.02,
    device: str = "cuda:0",
    backend: str = "auto",
):
    """Build a distance field for ``detect_contacts``.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    resolution, padding : UDF grid parameters (ignored by CPU backend)
    device : preferred Warp device, e.g. ``"cuda:0"`` or ``"cpu"``
    backend : ``"auto"`` | ``"udf"`` | ``"mesh"``
        ``auto`` / ``udf`` prefer GPU UDF; ``mesh`` forces CPU trimesh.

    Returns
    -------
    field : object with ``.query(points, with_sign=False)``
    backend_name : ``"udf"`` or ``"mesh"``
    """
    backend = str(backend).lower()
    if backend not in ("auto", "udf", "mesh"):
        raise ValueError(f"backend must be auto|udf|mesh, got {backend!r}")

    if backend == "mesh":
        from utils.grasping.mesh_contact_detector import MeshDistanceField

        print("[distance_field] Using CPU MeshDistanceField (backend=mesh).")
        return MeshDistanceField(mesh, resolution=resolution, padding=padding), "mesh"

    # Prefer Warp UDF on GPU.
    try:
        from utils.grasping.udf_contact_detector import UDFGrid

        field = UDFGrid(
            mesh, resolution=int(resolution), padding=float(padding), device=device
        )
        # UDFGrid itself falls back to cpu if cuda:0 is missing; report which.
        used = getattr(field, "_device", device)
        print(
            f"[distance_field] Using Warp UDFGrid "
            f"(resolution={resolution}, device={used})."
        )
        return field, "udf"
    except Exception as e:
        if backend == "udf":
            raise RuntimeError(
                f"UDF/Warp distance field required (backend=udf) but failed: {e}"
            ) from e
        from utils.grasping.mesh_contact_detector import MeshDistanceField

        print(
            f"[distance_field] Warp UDF unavailable ({e}); "
            "falling back to CPU MeshDistanceField."
        )
        return MeshDistanceField(mesh, resolution=resolution, padding=padding), "mesh"


def detect_contacts(*args: Any, **kwargs: Any):
    """Dispatch to the UDF-compatible ``detect_contacts`` implementation.

    Works with both ``UDFGrid`` and ``MeshDistanceField`` (same ``.query`` API).
    Prefers the Warp module's implementation; falls back to the mesh module.
    """
    try:
        from utils.grasping.udf_contact_detector import detect_contacts as _fn
    except Exception:
        from utils.grasping.mesh_contact_detector import detect_contacts as _fn
    return _fn(*args, **kwargs)
