"""Re-posing the patient model with the SKEL kinematics.

After registration every patient bone (CT surface or fitted template) is attached to one SKEL
part.  For a new pose q' the SKEL forward kinematics gives, per part j, a world transform
G'_j; the rigid motion that carries the fitted-pose geometry to the new pose is

    M_j = G'_j · G_j^-1          (G_j = [joints_ori_j | joints_j] in the fitted pose)

Bones move rigidly with M_j; the skin moves by linear blend skinning of the M_j with the
SKEL skin weights.  ``poses.json`` stores, per pose, the transforms for 11 interpolation
steps between the fitted pose and the target pose so the viewer can animate without FK.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .labelmap import SKEL_PARTS

try:
    from skel.kin_skel import pose_param_names as POSE_NAMES
except Exception:  # pragma: no cover
    POSE_NAMES = [f"q{i}" for i in range(46)]

N_STEPS = 11


def part_frames(model, poses: np.ndarray, betas: np.ndarray, trans: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """(24, 4, 4) world frames of the SKEL parts in mm for the given parameters."""
    from .fit import skel_forward

    dev = next(model.buffers()).device
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)[None]
    with torch.no_grad():
        out = skel_forward(model, t(poses), t(betas), t(trans), skelmesh=False, scale=torch.tensor([scale], device=dev))
    R = out.joints_ori[0].cpu().numpy()
    J = out.joints[0].cpu().numpy() * 1000.0
    G = np.tile(np.eye(4), (len(J), 1, 1))
    G[:, :3, :3] = R
    G[:, :3, 3] = J
    return G


def relative_transforms(G_fit: np.ndarray, G_new: np.ndarray) -> np.ndarray:
    return np.einsum("jab,jbc->jac", G_new, np.linalg.inv(G_fit))


# ---------------------------------------------------------------------- presets
def _deg(**kw):
    return {k: math.radians(v) for k, v in kw.items()}


# the arm presets also straighten the elbow/forearm/wrist and remove the fitted shoulder twist, so the whole
# arm hangs (or rises) from the shoulder regardless of the CT pose
_ARM_NEUTRAL = {k: 0.0 for side in "rl" for k in (f"shoulder_{side}_y", f"shoulder_{side}_z", f"elbow_flexion_{side}",
                                                   f"pro_sup_{side}", f"wrist_flexion_{side}", f"wrist_deviation_{side}")}

PRESETS = {
    "arms_down": _deg(shoulder_r_x=90, shoulder_l_x=-90, **_ARM_NEUTRAL),
    "arms_up": _deg(shoulder_r_x=-90, shoulder_l_x=90, **_ARM_NEUTRAL),
    "right_knee_90": _deg(knee_angle_r=90),
    "left_knee_90": _deg(knee_angle_l=90),
    "hips_45": _deg(hip_flexion_r=45, hip_flexion_l=45),
    "sitting": _deg(hip_flexion_r=90, hip_flexion_l=90, knee_angle_r=90, knee_angle_l=90),
    "trunk_flexion_20": _deg(lumbar_extension=-20),
    "head_turn_30": _deg(head_twist=30),
}


def apply_pose_spec(q_fit: np.ndarray, spec: dict[str, float], absolute: bool = True) -> np.ndarray:
    """Return a pose vector with the DOFs in ``spec`` (radians) set (absolute) or added (relative)."""
    q = np.asarray(q_fit, dtype=np.float64).copy()
    for name, val in spec.items():
        if name not in POSE_NAMES:
            raise KeyError(f"unknown SKEL pose parameter {name!r}; see skel.kin_skel.pose_param_names")
        i = POSE_NAMES.index(name)
        q[i] = val if absolute else q[i] + val
    return q


def parse_set(arg: str) -> dict[str, float]:
    """'hip_flexion_r=45,knee_angle_r=90' (degrees) -> radians dict."""
    out = {}
    for item in arg.split(","):
        if not item.strip():
            continue
        k, v = item.split("=")
        out[k.strip()] = math.radians(float(v))
    return out


# ---------------------------------------------------------------------- poses.json
def skin_weight_table(model, top: int = 4) -> tuple[np.ndarray, np.ndarray]:
    w = model.skin_weights
    w = w.to_dense() if w.is_sparse else w
    w = w.cpu().numpy()
    idx = np.argsort(-w, axis=1)[:, :top]
    val = np.take_along_axis(w, idx, axis=1)
    val = val / np.maximum(val.sum(1, keepdims=True), 1e-9)
    return idx.astype(np.int16), val.astype(np.float32)


# ---------------------------------------------------------------------- scapulohumeral rhythm
# SKEL has no posture library: the scapula DOFs (abduction, elevation, upward rotation) are free parameters.  A CT
# with the arms raised therefore leaves the scapulae elevated / upward-rotated, which looks wrong once the arms are
# lowered.  This rule moves the scapula with the arm elevation (classic ~2:1 scapulohumeral rhythm, scaled to the
# SKEL limits) while keeping the CT-fitted scapula near the CT pose.
_SCAP = {"r": ("scapula_abduction_r", "scapula_elevation_r", "scapula_upward_rot_r"),
         "l": ("scapula_abduction_l", "scapula_elevation_l", "scapula_upward_rot_l")}


def arm_elevation_deg(G: np.ndarray) -> dict[str, float]:
    """Elevation of each humerus from the part frames: 0 = hanging along the trunk, 180 = straight up."""
    out = {}
    up = G[SKEL_PARTS.index("thorax"), :3, 1]
    for side in "rl":
        d = G[SKEL_PARTS.index(f"ulna_{side}"), :3, 3] - G[SKEL_PARTS.index(f"humerus_{side}"), :3, 3]
        d = d / max(np.linalg.norm(d), 1e-9)
        out[side] = float(np.degrees(np.arccos(np.clip(np.dot(d, -up), -1.0, 1.0))))
    return out


def scapula_for_elevation(side: str, elev_deg: float) -> np.ndarray:
    """Scapula DOFs (rad) for an arm elevation: at rest below 30 deg, fully elevated/upward-rotated at 180 deg."""
    from skel.kin_skel import pose_limits
    frac = float(np.clip((elev_deg - 30.0) / 150.0, 0.0, 1.0))
    abd, ele, rot = _SCAP[side]
    lo_e, hi_e = pose_limits[ele]
    rest_e, raised_e = (lo_e, hi_e) if abs(lo_e) < abs(hi_e) else (hi_e, lo_e)      # rest = least elevated end
    lo_r, hi_r = pose_limits[rot]
    return np.array([0.0, rest_e + (raised_e - rest_e) * frac, max(lo_r, hi_r) * frac])


def apply_scapula_rhythm(model, fit: dict, q: np.ndarray, fade_deg: float = 60.0) -> np.ndarray:
    """Return ``q`` with the scapula DOFs following the arm elevation.

    Near the CT pose the fitted scapula is kept (the CT scapula bone is where the CT shows it); the further the arm
    moves away from the CT elevation, the more the scapula follows the rhythm rule alone.
    """
    from skel.kin_skel import pose_limits
    q_fit = np.asarray(fit["poses"], dtype=np.float64)
    q = np.asarray(q, dtype=np.float64).copy()
    e_fit = arm_elevation_deg(part_frames(model, q_fit, fit["betas"], fit["trans"], fit.get("scale", 1.0)))
    e_now = arm_elevation_deg(part_frames(model, q, fit["betas"], fit["trans"], fit.get("scale", 1.0)))
    for side in "rl":
        idx = [POSE_NAMES.index(n) for n in _SCAP[side]]
        w = float(np.clip(1.0 - abs(e_now[side] - e_fit[side]) / fade_deg, 0.0, 1.0))
        val = scapula_for_elevation(side, e_now[side]) + (q_fit[idx] - scapula_for_elevation(side, e_fit[side])) * w
        for k, n in zip(idx, _SCAP[side]):
            lo, hi = sorted(pose_limits[n])
            q[k] = float(np.clip(val[_SCAP[side].index(n)], lo, hi))
    return q


def skin_verts_mm(model, poses, betas, trans, scale: float = 1.0) -> np.ndarray:
    from .fit import skel_forward
    dev = next(model.buffers()).device
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)[None]
    with torch.no_grad():
        out = skel_forward(model, t(poses), t(betas), t(trans), skelmesh=False, scale=torch.tensor([scale], device=dev))
    return out.skin_verts[0].cpu().numpy() * 1000.0


def arms_down_clearance(model, fit: dict, ct_skin, q: np.ndarray, min_deg: float = 5.0, max_deg: float = 45.0,
                        step_deg: float = 5.0, max_inside: float = 0.03, ct_bones: dict | None = None,
                        scapula_auto: bool = True, symmetric: bool = True) -> tuple[np.ndarray, dict]:
    """Abduct each hanging arm just enough that its skin clears the patient's (CT) trunk skin.

    The estimated arm hangs from the parametric shoulder; a wide trunk (obese patient, non-rigidly refined CT skin)
    can swallow it.  Starting at ``min_deg`` the shoulder abduction grows until at most ``max_inside`` of the arm skin
    vertices more than 12 cm below the shoulder lie deeper than 8 mm inside the CT skin (sign of the CT surface
    normal at the closest point; the deltoid region is excluded because it always overlaps the CT shoulder).  With
    ``ct_bones`` the test uses the arm *bones* instead of the skin (an obese trunk legitimately swallows the upper-arm
    skin) and additionally requires at most 3 % of the humerus vertices within 10 mm of the CT thorax bones.
    The scapula rhythm is applied inside the search (the rest scapula moves the shoulder medially)."""
    from scipy.spatial import cKDTree
    from trimesh.proximity import closest_point
    from .fit import skel_forward
    from .skel_wrapper import bone_part_labels, skin_part_labels
    labels = skin_part_labels(model)
    blabels = bone_part_labels(model) if ct_bones else None
    thorax = ct_bones.get("thorax") if ct_bones else None
    ttree = cKDTree(np.asarray(thorax.vertices)) if thorax is not None and len(thorax.vertices) else None
    dev = next(model.buffers()).device
    tt = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)[None]
    q = np.asarray(q, dtype=np.float64).copy()
    info = {}
    for side, sign in (("r", 1.0), ("l", -1.0)):
        ix = POSE_NAMES.index(f"shoulder_{side}_x")
        arm = np.isin(labels, [SKEL_PARTS.index(n) for n in (f"humerus_{side}", f"ulna_{side}", f"radius_{side}")])
        armb = np.isin(blabels, [SKEL_PARTS.index(n) for n in (f"humerus_{side}", f"ulna_{side}", f"radius_{side}")]) if blabels is not None else None
        chosen = min_deg
        if not arm.any():
            continue
        for d in np.arange(min_deg, max_deg + 1e-6, step_deg):
            q[ix] = sign * math.radians(90.0 - d)
            q_try = apply_scapula_rhythm(model, fit, q) if scapula_auto else q
            with torch.no_grad():
                o = skel_forward(model, tt(q_try), tt(fit["betas"]), tt(fit["trans"]), skelmesh=ttree is not None,
                                 scale=torch.tensor([fit.get("scale", 1.0)], device=dev))
            V = o.skin_verts[0].cpu().numpy() * 1000.0
            sh_y = float(o.joints[0, SKEL_PARTS.index(f"humerus_{side}"), 1]) * 1000.0
            chosen = d
            ok_bone = True
            if ttree is not None:
                H = o.skel_verts[0].cpu().numpy()[blabels == SKEL_PARTS.index(f"humerus_{side}")] * 1000.0
                H = H[:: max(len(H) // 800, 1)]
                ok_bone = len(H) == 0 or (ttree.query(H)[0] < 10.0).mean() <= 0.03
            # the arm *bones* (not the skin: an obese trunk legitimately swallows the upper-arm skin) must not lie
            # deeper than 8 mm inside the CT skin below the shoulder region
            if ttree is not None and blabels is not None:
                A = o.skel_verts[0].cpu().numpy()[armb] * 1000.0
                A = A[:: max(len(A) // 1000, 1)]
            else:
                A = V[arm][:: max(int(arm.sum()) // 1000, 1)]
            A = A[(A[:, 1] < sh_y - 120.0) & (A[:, 1] > ct_skin.bounds[0][1] + 20.0)]
            if len(A) < 20:
                if ok_bone:
                    break
                continue
            closest, dist, tid = closest_point(ct_skin, A)
            deep = (np.einsum("ij,ij->i", ct_skin.face_normals[tid], A - closest) < 0) & (dist > 8.0)
            if deep.mean() <= max_inside and ok_bone:
                break
        q[ix] = sign * math.radians(90.0 - chosen)
        info[side] = float(chosen)
    if symmetric and len(info) == 2:                       # asymmetric hanging arms read as an error: use the larger angle
        d = max(info.values())
        for side, sign in (("r", 1.0), ("l", -1.0)):
            q[POSE_NAMES.index(f"shoulder_{side}_x")] = sign * math.radians(90.0 - d)
            info[side] = float(d)
    return q, info


def preset_pose(model, fit: dict, name: str, ct_skin=None, ct_bones: dict | None = None) -> np.ndarray:
    """Pose vector of a preset; ``arms_down`` additionally abducts the arms until they clear the CT trunk and ribs."""
    q = apply_pose_spec(fit["poses"], PRESETS[name])
    if name == "arms_down" and ct_skin is not None and len(ct_skin.faces):
        q, _ = arms_down_clearance(model, fit, ct_skin, q, ct_bones=ct_bones)
    return q


def build_pose_entry(model, fit: dict, name: str, q_target: np.ndarray, n_steps: int = N_STEPS,
                     scapula_auto: bool = True) -> dict:
    q_fit = np.asarray(fit["poses"], dtype=np.float64)
    G_fit = part_frames(model, q_fit, fit["betas"], fit["trans"], fit.get("scale", 1.0))
    frames = []
    for k in range(n_steps):
        t = k / (n_steps - 1)
        q = q_fit + t * (q_target - q_fit)
        if scapula_auto:
            q = apply_scapula_rhythm(model, fit, q)
        G = part_frames(model, q, fit["betas"], fit["trans"], fit.get("scale", 1.0))
        frames.append(relative_transforms(G_fit, G).reshape(24, 16).round(6).tolist())
    q_final = apply_scapula_rhythm(model, fit, q_target) if scapula_auto else np.asarray(q_target, dtype=np.float64)
    return {"name": name, "q": [float(v) for v in q_final], "frames": frames}


def write_poses(out_dir: str | Path, model, fit: dict, poses: dict[str, np.ndarray],
                ct_skin_corner_weights: tuple[np.ndarray, np.ndarray] | None = None,
                bone_entries: list[dict] | None = None, skel_verts_mm: np.ndarray | None = None) -> Path:
    """Write poses.json (+ optional binary CT-skin corner weights) into the output directory."""
    out = Path(out_dir)
    idx, val = skin_weight_table(model)
    data = {
        "parts": list(SKEL_PARTS),
        "pose_names": list(POSE_NAMES),
        "n_steps": N_STEPS,
        "skin": {"faces": model.skin_f.cpu().numpy().astype(int).tolist(),
                 "w_idx": idx.tolist(), "w": val.round(4).tolist()},
        "poses": [build_pose_entry(model, fit, n, q) for n, q in poses.items()],
    }
    if ct_skin_corner_weights is not None:
        cidx, cval = ct_skin_corner_weights
        (out / "ct_skin_weights.bin").write_bytes(cidx.astype(np.uint8).tobytes() + cval.astype(np.float32).tobytes())
        data["ct_skin_weights"] = {"file": "ct_skin_weights.bin", "n": int(len(cidx)), "top": int(cidx.shape[1])}
    if bone_entries is not None and skel_verts_mm is not None:
        from .skel_wrapper import bone_part_labels
        bidx, bval = skel_weight_table(model)
        data["bone_weights"] = write_bone_weights(out, bone_entries, skel_verts_mm, bidx, bval, part_labels=bone_part_labels(model))
    (out / "poses.json").write_text(json.dumps(data))
    return out / "poses.json"


def skel_weight_table(model, top: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Top-``top`` skeleton skinning weights per SKEL skeleton vertex (spine vertices blend two joints)."""
    w = model.skel_weights
    w = w.to_dense() if w.is_sparse else w
    w = w.cpu().numpy()
    idx = np.argsort(-w, axis=1)[:, :top]
    val = np.take_along_axis(w, idx, axis=1)
    val = val / np.maximum(val.sum(1, keepdims=True), 1e-9)
    return idx.astype(np.int16), val.astype(np.float32)


def bone_corner_weights(mesh, skel_verts_mm: np.ndarray, idx: np.ndarray, val: np.ndarray,
                        allowed: np.ndarray | None = None):
    """Per-STL-corner skeleton weights for a bone mesh: nearest fitted SKEL skeleton vertex's weights.

    ``allowed`` restricts the search to the SKEL skeleton vertices of the bone's own part: ribs under the scapula or
    the acromion next to the humeral head must never be bound to the neighbouring bone."""
    from scipy.spatial import cKDTree
    V = np.asarray(skel_verts_mm)
    sub = np.where(allowed)[0] if allowed is not None and allowed.any() else np.arange(len(V))
    _, nn = cKDTree(V[sub]).query(np.asarray(mesh.vertices))
    nn = sub[nn]
    faces = np.asarray(mesh.faces).reshape(-1)
    return idx[nn][faces], val[nn][faces]


def part_vertex_mask(part_labels: np.ndarray | None, part: str) -> np.ndarray | None:
    """Bool mask of the SKEL skeleton vertices that belong to ``part`` (None if labels are unavailable)."""
    if part_labels is None or part not in SKEL_PARTS:
        return None
    return np.asarray(part_labels) == SKEL_PARTS.index(part)


def write_bone_weights(out_dir, entries: list[dict], skel_verts_mm: np.ndarray, idx: np.ndarray, val: np.ndarray,
                       part_labels: np.ndarray | None = None) -> dict:
    """Write bone_weights.bin (uint8 idx + float32 w per corner) for every bone-like part; returns the index dict."""
    import trimesh
    out = Path(out_dir)
    parts, chunks, offset = {}, [], 0
    top = idx.shape[1]
    for e in entries:
        if e["kind"] not in ("bone", "bone_tpl") or not e.get("part"):
            continue
        m = trimesh.load(out / e["file"])
        ci, cv = bone_corner_weights(m, skel_verts_mm, idx, val, allowed=part_vertex_mask(part_labels, e["part"]))
        n = len(ci)
        parts[e["id"]] = {"offset": offset, "n": n}
        chunks.append(ci.astype(np.uint8).tobytes() + cv.astype(np.float32).tobytes())
        offset += n
    (out / "bone_weights.bin").write_bytes(b"".join(chunks))
    return {"file": "bone_weights.bin", "top": int(top), "parts": parts}




def ct_skin_corner_weights(ct_skin_mesh, skel_skin_verts_mm: np.ndarray, idx: np.ndarray, val: np.ndarray,
                           allowed: np.ndarray | None = None):
    """Per-STL-corner skin weights for the CT skin: nearest SKEL skin vertex's weights.

    ``allowed`` (bool per SKEL skin vertex) restricts the search, e.g. to non-estimated parts so that CT skin
    next to a raised arm is never bound to the (estimated) arm and dragged along when the arm is lowered."""
    vidx, vval = ct_skin_vertex_weights(ct_skin_mesh, skel_skin_verts_mm, idx, val, allowed)
    faces = np.asarray(ct_skin_mesh.faces).reshape(-1)          # corner order = STL triangle order
    return vidx[faces], vval[faces]


def ct_skin_vertex_weights(ct_skin_mesh, skel_skin_verts_mm: np.ndarray, idx: np.ndarray, val: np.ndarray,
                           allowed: np.ndarray | None = None, smooth_iters: int = 30, top: int = 4):
    """Per-vertex (idx, val) skin weights of the CT skin: nearest SKEL skin vertex's weights, then diffused over the
    CT mesh graph so that part boundaries (e.g. head | scapula) blend over ~1 cm instead of tearing when posed."""
    from scipy.spatial import cKDTree
    import scipy.sparse as sp
    V = np.asarray(skel_skin_verts_mm)
    sub = np.where(allowed)[0] if allowed is not None else np.arange(len(V))
    P = np.asarray(ct_skin_mesh.vertices)
    _, nn = cKDTree(V[sub]).query(P)
    nn = sub[nn]
    n, nj = len(P), int(idx.max()) + 1
    W = np.zeros((n, max(nj, 24)), dtype=np.float32)
    np.put_along_axis(W, idx[nn].astype(int), val[nn].astype(np.float32), axis=1)
    if smooth_iters > 0 and len(ct_skin_mesh.faces):
        e = np.asarray(ct_skin_mesh.edges_unique)
        A = sp.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
        deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1.0)
        Dinv = sp.diags(1.0 / deg)
        for _ in range(smooth_iters):
            W = 0.5 * W + 0.5 * (Dinv @ (A @ W))
    order = np.argsort(-W, axis=1)[:, :top]
    vval = np.take_along_axis(W, order, axis=1)
    vval = vval / np.maximum(vval.sum(1, keepdims=True), 1e-9)
    return order.astype(np.int16), vval.astype(np.float32)
