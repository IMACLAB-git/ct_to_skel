"""Patient surfaces built on the SKEL topology (the principled alternative to posing marching-cubes meshes).

Design
------
* The displayed patient skin is a subdivided SKEL skin (SKEL vertices + midpoint subdivision) whose vertices are
  moved onto the CT body surface.  It therefore carries SKEL's own skinning weights (interpolated on the subdivision)
  and poses exactly like SKEL does - no nearest-neighbour weight transfer, no tearing at part boundaries.  Regions the
  CT does not cover (or estimated limbs) simply stay parametric.
* An anatomical gate discards CT structures that cannot belong to the patient: anything farther than a margin from the
  fitted body model (table, calibration phantoms, cables, wires) - independent of the scanner or protocol.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .labelmap import SKEL_PARTS


# ---------------------------------------------------------------------- subdivision with weights
def subdivide_with_weights(V: np.ndarray, F: np.ndarray, W: np.ndarray, levels: int = 1):
    """Midpoint-subdivide a triangle mesh ``levels`` times; per-vertex attributes ``W`` (n, k) are averaged on the new
    edge midpoints.  Returns (V2, F2, W2)."""
    V = np.asarray(V, dtype=np.float64); F = np.asarray(F, dtype=np.int64); W = np.asarray(W, dtype=np.float32)
    for _ in range(levels):
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        inv = inv.reshape(3, -1)                                   # (3, nF): edge ids for (01, 12, 20)
        mid = 0.5 * (V[uniq[:, 0]] + V[uniq[:, 1]])
        Wm = 0.5 * (W[uniq[:, 0]] + W[uniq[:, 1]])
        base = len(V)
        m01, m12, m20 = inv[0] + base, inv[1] + base, inv[2] + base
        a, b, c = F[:, 0], F[:, 1], F[:, 2]
        F = np.concatenate([np.stack([a, m01, m20], 1), np.stack([m01, b, m12], 1),
                            np.stack([m20, m12, c], 1), np.stack([m01, m12, m20], 1)])
        V = np.concatenate([V, mid]); W = np.concatenate([W, Wm])
    return V, F, W


def dense_skin_weights(model) -> np.ndarray:
    w = model.skin_weights
    w = w.to_dense() if w.is_sparse else w
    return w.cpu().numpy().astype(np.float32)


def top_k(W: np.ndarray, k: int = 4):
    idx = np.argsort(-W, axis=1)[:, :k]
    val = np.take_along_axis(W, idx, axis=1)
    val = val / np.maximum(val.sum(1, keepdims=True), 1e-9)
    return idx.astype(np.int16), val.astype(np.float32)


# ---------------------------------------------------------------------- anatomical gate
def gate_mesh_by_distance(mesh: trimesh.Trimesh, ref_pts: np.ndarray, max_mm: float, min_component_frac: float = 0.002):
    """Drop mesh vertices farther than ``max_mm`` from the reference point cloud (fitted SKEL skin or skeleton) and the
    faces touching them; crumbs smaller than ``min_component_frac`` of the faces are dropped too."""
    from .meshing import remove_vertices
    if mesh is None or len(mesh.faces) == 0:
        return mesh
    d, _ = cKDTree(np.asarray(ref_pts)).query(np.asarray(mesh.vertices))
    far = d > max_mm
    if not far.any():
        return mesh
    return remove_vertices(mesh, far, min_component_frac=min_component_frac)


def gate_components(mesh: trimesh.Trimesh, ref_pts: np.ndarray, max_mm: float) -> tuple[trimesh.Trimesh, int]:
    """Keep the connected components of a bone mesh whose median vertex lies within ``max_mm`` of the fitted
    skeleton (islands of a label, phantom inserts, calcifications far from bone are dropped).  Returns (mesh, n_dropped)."""
    if mesh is None or len(mesh.faces) == 0:
        return mesh, 0
    tree = cKDTree(np.asarray(ref_pts))
    comps = mesh.split(only_watertight=False)
    keep = []
    for c in comps:
        d, _ = tree.query(np.asarray(c.vertices)[:: max(len(c.vertices) // 3000, 1)])
        if np.median(d) <= max_mm:
            keep.append(c)
    if len(keep) == len(comps):
        return mesh, 0
    if not keep:
        return trimesh.Trimesh(), len(comps)
    return (trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]), len(comps) - len(keep)


def gate_pieces(pieces: list, ref_pts: np.ndarray, max_mm: float, skin_mesh: trimesh.Trimesh | None = None,
                skin_margin_mm: float = 15.0) -> tuple[list, list]:
    """Gate every unlabelled bone piece component-wise.  A component is anatomy when it lies within ``max_mm`` of the
    fitted skeleton AND (if ``skin_mesh`` is given) inside the fitted skin envelope (signed distance <= margin): a
    forearm the first fit misplaced by a few centimetres is still inside the arm, a phantom between the legs is not.
    Returns (kept, dropped)."""
    tree = cKDTree(np.asarray(ref_pts))
    kept, dropped = [], []
    for m in pieces:
        comps = m.split(only_watertight=False)
        keep = []
        for c in comps:
            v = np.asarray(c.vertices)[:: max(len(c.vertices) // 2000, 1)]
            ok = np.median(tree.query(v)[0]) <= max_mm
            if ok and skin_mesh is not None:
                sd = trimesh.proximity.signed_distance(skin_mesh, v)       # positive inside for trimesh
                ok = np.median(sd) >= -skin_margin_mm
            if ok:
                keep.append(c)
        if keep:
            kept.append(trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0])
        else:
            dropped.append(m)
    return kept, dropped


# ---------------------------------------------------------------------- patient skin
def refine_to_surface(V: np.ndarray, F: np.ndarray, ct_pts: np.ndarray, constrain: np.ndarray, max_dist_mm: float = 40.0,
                      iters: int = 8, smooth_iters: int = 20, step: float = 0.8, data_weight: float = 0.6,
                      ct_normals: np.ndarray | None = None, min_normal_dot: float = 0.0):
    """Move constrained vertices onto the CT surface with a graph-smoothed displacement field (non-rigid ICP in the
    spirit of Amberg et al. 2007): correspondences are the nearest CT surface samples, accepted only when the surface
    normals agree (``min_normal_dot``) and the distance is below a coarse-to-fine schedule (2x ``max_dist_mm`` down to
    ``max_dist_mm``).  This keeps a hand lying on the thigh from snapping onto the thigh surface.  Unconstrained
    vertices follow their neighbours harmonically.  KD-tree based (fast on 100k+ vertices)."""
    from .refine import _adjacency
    V = np.asarray(V, dtype=np.float64).copy()
    n = len(V)
    A = _adjacency(n, F)
    tree = cKDTree(np.asarray(ct_pts))
    stats = {}
    for it in range(iters):
        d_max = max_dist_mm * (2.0 - it / max(iters - 1, 1))          # 2x -> 1x max_dist over the iterations
        dist, nn = tree.query(V)
        disp = ct_pts[nn] - V
        ok = constrain & (dist < d_max)
        if ct_normals is not None:
            vn = trimesh.Trimesh(V, F, process=False).vertex_normals
            agree = np.einsum("ij,ij->i", vn, ct_normals[nn]) > min_normal_dot
            ok &= agree
        target = np.where(ok[:, None], disp, 0.0)
        D = target.copy()
        for _ in range(smooth_iters):
            nb = A @ D
            D = np.where(ok[:, None], data_weight * target + (1 - data_weight) * nb, nb)
        V = V + step * D
        stats = {"iter": it + 1, "constrained": int(ok.sum()), "mean_dist_before_mm": float(dist[ok].mean()) if ok.any() else float("nan")}
    dist, _ = tree.query(V)
    ok = constrain & (dist < max_dist_mm)
    stats["mean_dist_after_mm"] = float(dist[ok].mean()) if ok.any() else float("nan")
    stats["p95_dist_after_mm"] = float(np.percentile(dist[ok], 95)) if ok.any() else float("nan")
    return V, stats


def build_patient_skin(model, skin_verts_mm: np.ndarray, ct_skin: trimesh.Trimesh, est_ids: list[int],
                       y_range_mm: tuple[float, float] | None, levels: int = 2, margin_mm: float = 15.0):
    """Subdivided SKEL skin refined onto the CT skin.  Returns (mesh, W_dense (n,24), stats).

    Vertices of estimated parts and vertices outside the CT axial range stay parametric."""
    F0 = model.skin_f.cpu().numpy()
    W0 = dense_skin_weights(model)
    V, F, W = subdivide_with_weights(skin_verts_mm, F0, W0, levels)
    part = W.argmax(1)
    constrain = ~np.isin(part, np.asarray(est_ids, dtype=int)) if len(est_ids) else np.ones(len(V), dtype=bool)
    if y_range_mm is not None:
        lo, hi = y_range_mm
        constrain &= (V[:, 1] > lo + margin_mm) & (V[:, 1] < hi - margin_mm)
    # dense CT surface samples (vertices are ~3 mm apart on the marching-cubes skin: fine as targets)
    ct_pts = np.asarray(ct_skin.vertices, dtype=np.float64)
    V2, stats = refine_to_surface(V, F, ct_pts, constrain, ct_normals=np.asarray(ct_skin.vertex_normals))
    stats["vertices"] = int(len(V2)); stats["levels"] = int(levels)
    mesh = trimesh.Trimesh(V2, F, process=False)
    return mesh, W, stats


# ---------------------------------------------------------------------- CT skin for the extremities
EXTREMITY_PARTS = ("hand_r", "hand_l", "talus_r", "calcn_r", "toes_r", "talus_l", "calcn_l", "toes_l")


def merge_extremity_skin(patient_mesh: trimesh.Trimesh, W: np.ndarray, ct_skin: trimesh.Trimesh,
                         bone_pts: np.ndarray, bone_parts: np.ndarray, est_ids: list[int], min_faces: int = 300):
    """Replace the hands / feet of the SKEL-topology patient skin by the patient's own CT skin.

    The SKEL hand and foot are coarse blobs that cannot follow spread fingers or a plantar-flexed foot; the CT skin of
    these extremities is cut out (CT skin vertices whose nearest CT bone belongs to a hand / foot part) and attached
    rigidly to that part, while the SKEL skin loses its hand / foot faces.  Estimated (not scanned) extremities keep
    the SKEL skin.  Returns (mesh, W_dense) with the CT extremity vertices appended."""
    ext_ids = [SKEL_PARTS.index(n) for n in EXTREMITY_PARTS if SKEL_PARTS.index(n) not in set(est_ids)]
    if not ext_ids or ct_skin is None or len(ct_skin.faces) == 0 or len(bone_pts) == 0:
        return patient_mesh, W, {"added_faces": 0}
    part_of_ct = bone_parts[cKDTree(np.asarray(bone_pts)).query(np.asarray(ct_skin.vertices))[1]]
    f = np.asarray(ct_skin.faces)
    ext_face = np.isin(part_of_ct[f], ext_ids).all(axis=1)
    if not ext_face.any():
        return patient_mesh, W, {"added_faces": 0}
    ext = trimesh.Trimesh(np.asarray(ct_skin.vertices), f[ext_face], process=False)
    ext.remove_unreferenced_vertices()
    comps = [c for c in ext.split(only_watertight=False) if len(c.faces) >= min_faces]
    if not comps:
        return patient_mesh, W, {"added_faces": 0}
    ext = trimesh.util.concatenate(comps) if len(comps) > 1 else comps[0]
    ext_parts = bone_parts[cKDTree(np.asarray(bone_pts)).query(np.asarray(ext.vertices))[1]]
    # SKEL skin: drop the hand / foot faces (any corner dominated by an extremity part)
    top = W.argmax(1)
    pf = np.asarray(patient_mesh.faces)
    keep = ~np.isin(top[pf], ext_ids).any(axis=1)
    base = trimesh.Trimesh(np.asarray(patient_mesh.vertices), pf[keep], process=False)
    keep_v = np.zeros(len(patient_mesh.vertices), dtype=bool); keep_v[np.unique(pf[keep])] = True
    base.remove_unreferenced_vertices()
    W_base = W[keep_v]
    W_ext = np.zeros((len(ext.vertices), W.shape[1]), dtype=np.float32)
    W_ext[np.arange(len(ext.vertices)), ext_parts] = 1.0
    merged = trimesh.util.concatenate([base, ext])
    W_all = np.concatenate([W_base, W_ext])
    return merged, W_all, {"added_faces": int(len(ext.faces)), "removed_skel_faces": int((~keep).sum()),
                           "parts": sorted({SKEL_PARTS[i] for i in np.unique(ext_parts)})}
