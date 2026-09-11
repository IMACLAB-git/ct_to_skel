"""CT-driven refinement of the fitted SKEL model (SKEL used as a reference, CT is the truth).

1. :func:`align_bones` — rigid (+ mild uniform scale) ICP of every SKEL bone part onto the
   corresponding CT bone (TotalSegmentator labels mapped to SKEL parts).  Gives bones that sit on
   the CT bones and, from them, accurate joint centres to re-fit the parametric model.
2. :func:`refine_skin` — non-rigid, smoothness-regularised displacement of the SKEL skin vertices
   onto the CT body surface (harmonic extension where the CT gives no constraint).

All coordinates are SKEL frame millimetres.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy import sparse
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------- rigid / similarity ICP
def kabsch(A: np.ndarray, B: np.ndarray, allow_scale: bool = True,
           scale_bounds: tuple[float, float] = (0.85, 1.15)) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (s, R, t) minimising ||s R a + t - b||."""
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = 1.0
    if allow_scale:
        var_a = (A0 ** 2).sum()
        s = float(np.clip((S * np.diag(D)).sum() / max(var_a, 1e-12), *scale_bounds))
    t = cb - s * (R @ ca)
    return s, R, t


def icp(src: np.ndarray, dst: np.ndarray, allow_scale: bool = True, iters: int = 80,
        reject_mm: tuple[float, float] = (40.0, 8.0), min_pairs: int = 30, trim: float = 0.8) -> tuple[np.ndarray, dict]:
    """Point-to-point ICP of ``src`` onto ``dst`` with a shrinking pair-rejection radius.

    Returns a 4x4 similarity transform and statistics.  Pairs farther than the current
    rejection radius are ignored, which makes the alignment tolerant to partially
    scanned bones (the CT side may be cut by the field of view).
    """
    tree = cKDTree(dst)
    T = np.eye(4)
    stats = {"pairs": 0, "residual_mm": float("nan"), "scale": 1.0, "iters": 0}
    cur = src
    for k in range(iters):
        # rejection radius shrinks over the first 60 % of the iterations; additionally only the
        # best ``trim`` fraction of pairs is used (trimmed ICP, robust to partial overlap)
        thr = reject_mm[0] + (reject_mm[1] - reject_mm[0]) * min(k / max(iters * 0.6, 1), 1.0)
        d, idx = tree.query(cur)
        keep = d < min(thr, np.percentile(d, trim * 100))
        if keep.sum() < min_pairs:
            break
        s, R, t = kabsch(src[keep], dst[idx[keep]], allow_scale)
        T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
        new = src @ (s * R).T + t
        moved = np.abs(new - cur).max()
        cur = new
        stats.update(pairs=int(keep.sum()), residual_mm=float(d[keep].mean()), scale=float(s), iters=k + 1)
        if moved < 1e-3:
            break
    d, _ = tree.query(cur)
    stats["final_mean_mm"] = float(d[d < reject_mm[1] * 2].mean()) if (d < reject_mm[1] * 2).any() else float("nan")
    stats["final_inlier_frac"] = float((d < reject_mm[1] * 2).mean())
    return T, stats


def apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------- bones
# SKEL kinematic tree: PARENT[i] is the parent part of part i (pelvis is the root)
PARENT = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 14, 15, 16, 17, 12, 19, 20, 21, 22]


@dataclass
class BoneAlignment:
    transforms: dict = field(default_factory=dict)     # part name -> 4x4
    stats: dict = field(default_factory=dict)          # part name -> icp stats
    verts: np.ndarray | None = None                    # transformed skeleton vertices
    joints: np.ndarray | None = None                   # transformed joints (24, 3)
    joint_weight: np.ndarray | None = None             # 1 for aligned parts, 0 otherwise


# parts that are aligned together as one rigid unit when only unlabelled CT bone is available
_RIGID_GROUPS = {"ulna_r": ("ulna_r", "radius_r"), "ulna_l": ("ulna_l", "radius_l"),
                 "tibia_r": ("tibia_r",), "tibia_l": ("tibia_l",)}


def align_bones(skel_verts: np.ndarray, labels: np.ndarray, part_names: list[str], joints: np.ndarray,
                ct_parts: dict[str, trimesh.Trimesh], bbox: np.ndarray | None = None,
                max_src: int = 15000, min_ct_verts: int = 200, allow_scale: bool = False,
                seed: int = 0, unlabeled: list | None = None) -> BoneAlignment:
    """ICP every SKEL bone part onto the CT bone mesh of the same part.

    ``skel_verts`` (Nk,3) with per-vertex ``labels`` (part ids), ``joints`` (24,3), ``ct_parts``
    mapping part name -> CT mesh; ``bbox`` (2,3) limits the SKEL points used to the CT coverage.
    Parts without a CT counterpart keep their parametric placement (identity transform).
    """
    rng = np.random.default_rng(seed)
    out = BoneAlignment(verts=skel_verts.copy(), joints=joints.copy(), joint_weight=np.zeros(len(joints)))
    for pid, name in enumerate(part_names):
        sel = np.where(labels == pid)[0]
        ct = ct_parts.get(name)
        if ct is None or len(ct.vertices) < min_ct_verts or len(sel) < 50:
            out.transforms[name] = np.eye(4)
            continue
        src = skel_verts[sel]
        if bbox is not None:
            inside = ((src >= bbox[0] - 10) & (src <= bbox[1] + 10)).all(1)
            if inside.sum() < 50:
                out.transforms[name] = np.eye(4)
                continue
            src_fit = src[inside]
        else:
            src_fit = src
        if len(src_fit) > max_src:
            src_fit = src_fit[rng.choice(len(src_fit), max_src, replace=False)]
        dst = np.asarray(ct.vertices)
        if len(dst) > 200000:
            dst = dst[rng.choice(len(dst), 200000, replace=False)]
        T, st = icp(src_fit, dst, allow_scale=allow_scale)
        out.transforms[name] = T
        out.stats[name] = st
        out.verts[sel] = apply_T(T, src)
        out.joints[pid] = apply_T(T, joints[pid:pid + 1])[0]
        out.joint_weight[pid] = 1.0
    # parts without a CT counterpart (outside the scan) follow their nearest aligned ancestor rigidly,
    # so that e.g. the forearm stays attached to the ICP-moved humerus and the shank to the femur
    aligned = {n for n, w in zip(part_names, out.joint_weight) if w >= 1.0}
    for pid, name in enumerate(part_names):
        if name in aligned:
            continue
        anc = PARENT[pid]
        while anc >= 0 and part_names[anc] not in aligned:
            anc = PARENT[anc]
        if anc < 0:
            continue
        T = out.transforms[part_names[anc]]
        direct_child = PARENT[pid] == anc
        # keep only the rigid part of the ancestor's similarity transform (do not scale the child)
        R = T[:3, :3] / np.cbrt(max(np.linalg.det(T[:3, :3]), 1e-9))
        Tr = np.eye(4); Tr[:3, :3] = R; Tr[:3, 3] = T[:3, 3] + (T[:3, :3] - R) @ joints[anc]
        out.transforms[name] = Tr
        sel = np.where(labels == pid)[0]
        if len(sel):
            out.verts[sel] = apply_T(Tr, skel_verts[sel])
        out.joints[pid] = apply_T(Tr, joints[pid:pid + 1])[0]
        if direct_child:
            # the child's joint (knee, elbow, ...) carried along by the parent's ICP encodes where the CT bone points:
            # a soft target so that the re-fit reproduces the scanned bone's orientation
            out.joint_weight[pid] = 0.5
        out.stats.setdefault(name, {})["follows"] = part_names[anc]
    # unlabelled CT bone (e.g. forearms when the appendicular labels are missing): align the unsupported
    # SKEL parts that lie near it, as rigid groups (ulna+radius), then carry their children along
    if unlabeled:
        pts = np.vstack([np.asarray(m.vertices) for m in unlabeled if len(m.vertices)])
        tree = cKDTree(pts)
        for lead, group in _RIGID_GROUPS.items():
            if lead not in part_names or lead in aligned:
                continue
            gids = [part_names.index(g) for g in group if g in part_names]
            sel = np.concatenate([np.where(labels == g)[0] for g in gids])
            if len(sel) < 50:
                continue
            src = out.verts[sel]                                     # already carried along by the parent
            # coarse initialisation: rotate the group about its proximal joint so that its long axis
            # follows the principal axis of the unlabelled bone that starts near that joint
            lead_id = part_names.index(lead)
            e = out.joints[lead_id]
            d0, _ = tree.query(src)
            already_near = (d0 < 40.0).mean() >= 0.5        # e.g. second pass after the re-fit: keep the pose
            cand = [np.asarray(m.vertices) for m in unlabeled if len(m.vertices) and
                    np.linalg.norm(np.asarray(m.vertices) - e, axis=1).min() < 130.0]
            if cand and not already_near:
                cpts = np.vstack(cand)
                ax = np.linalg.svd(cpts - cpts.mean(0), full_matrices=False)[2][0]
                if np.dot(cpts.mean(0) - e, ax) < 0:
                    ax = -ax                                           # point away from the joint
                far = out.joints[_descendants(lead_id)[-1]] if _descendants(lead_id) else src.mean(0)
                a = far - e
                a = a / max(np.linalg.norm(a), 1e-9)
                v = np.cross(a, ax); c = float(np.dot(a, ax)); sn = np.linalg.norm(v)
                if sn > 1e-6:
                    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    R0 = np.eye(3) + K + K @ K * ((1 - c) / (sn ** 2))
                    T0 = np.eye(4); T0[:3, :3] = R0; T0[:3, 3] = e - R0 @ e
                    ids0 = set(gids) | {cid for g in gids for cid in _descendants(g)}
                    for g in ids0:
                        gsel = np.where(labels == g)[0]
                        out.verts[gsel] = apply_T(T0, out.verts[gsel])
                        out.joints[g] = apply_T(T0, out.joints[g:g + 1])[0]
                        out.transforms[part_names[g]] = T0 @ out.transforms.get(part_names[g], np.eye(4))
                    src = out.verts[sel]
                    out.stats.setdefault(lead, {})["axis_prealigned_deg"] = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
            d, _ = tree.query(src)
            near = d < 80.0
            out.stats.setdefault(lead, {})["unlabeled_near_fraction"] = float(near.mean())
            if near.mean() < 0.15:
                continue
            probe = src[near][:: max(len(src[near]) // 2000, 1)]
            dst_idx = np.unique(np.concatenate([np.asarray(ix, dtype=int) for ix in tree.query_ball_point(probe, 80.0)]))
            dst = pts[dst_idx]
            out.stats[lead]["unlabeled_dst_points"] = int(len(dst))
            if len(dst) < 100:
                continue
            T, st = icp(src[near], dst, allow_scale=False, reject_mm=(80.0, 8.0), iters=120)
            # the CT may hold only a short stretch of the forearm (field of view), so judge the fit by the
            # points that found a partner: mean inlier distance and inlier fraction, not the trimmed residual
            if st.get("final_inlier_frac", 0) < 0.35 or st.get("final_mean_mm", 99) > 8.0 or st.get("residual_mm", 99) > 25.0:
                out.stats.setdefault(lead, {})["unlabeled_icp_rejected"] = st
                if "axis_prealigned_deg" in out.stats.get(lead, {}):
                    # keep the principal-axis alignment as a weak direction target (source "unlabeled_axis")
                    for g in gids:
                        out.joint_weight[g] = max(out.joint_weight[g], 0.5)
                        out.stats.setdefault(part_names[g], {})["source"] = "unlabeled_axis"
                        for cid in _descendants(g):
                            if part_names[cid] not in aligned and cid not in gids:
                                out.joint_weight[cid] = max(out.joint_weight[cid], 0.3)
                continue
            moved = set()
            for g in gids:
                gsel = np.where(labels == g)[0]
                out.verts[gsel] = apply_T(T, out.verts[gsel])
                out.joints[g] = apply_T(T, out.joints[g:g + 1])[0]
                out.transforms[part_names[g]] = T @ out.transforms.get(part_names[g], np.eye(4))
                out.joint_weight[g] = 1.0
                out.stats.setdefault(part_names[g], {}).update(residual_mm=st["residual_mm"], pairs=st["pairs"], source="unlabeled")
                moved.add(g)
            for g in gids:
                for cid in _descendants(g):
                    if cid in moved or part_names[cid] in aligned:
                        continue
                    csel = np.where(labels == cid)[0]
                    out.verts[csel] = apply_T(T, out.verts[csel])
                    out.joints[cid] = apply_T(T, out.joints[cid:cid + 1])[0]
                    out.transforms[part_names[cid]] = T @ out.transforms.get(part_names[cid], np.eye(4))
                    out.joint_weight[cid] = max(out.joint_weight[cid], 0.5)   # wrist etc.: soft target for the re-fit
                    moved.add(cid)
    return out


# ---------------------------------------------------------------------- skin
def _adjacency(n: int, faces: np.ndarray) -> sparse.csr_matrix:
    i = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    j = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    A = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1.0
    return sparse.diags(1.0 / deg) @ A          # row-normalised neighbour averaging


def refine_skin(verts: np.ndarray, faces: np.ndarray, ct_skin: trimesh.Trimesh,
                constrain: np.ndarray | None = None, max_dist_mm: float = 60.0, iters: int = 6,
                smooth_iters: int = 30, step: float = 0.8, data_weight: float = 0.6) -> tuple[np.ndarray, dict]:
    """Move SKEL skin vertices onto the CT body surface with a smooth displacement field.

    Each iteration finds the closest CT surface point per vertex, keeps displacements only for
    vertices allowed by ``constrain`` and closer than ``max_dist_mm`` (so that limbs missing in
    the CT are not dragged onto the torso), then smooths the displacement field over the mesh
    graph; unconstrained vertices receive the harmonic extension of their neighbours' motion.
    """
    V = np.asarray(verts, dtype=np.float64).copy()
    n = len(V)
    W = _adjacency(n, faces)
    constrain = np.ones(n, dtype=bool) if constrain is None else constrain.astype(bool)
    stats = {}
    for it in range(iters):
        closest, dist, _ = ct_skin.nearest.on_surface(V)
        disp = closest - V
        ok = constrain & (dist < max_dist_mm)
        target = np.where(ok[:, None], disp, 0.0)
        D = target.copy()
        for _ in range(smooth_iters):
            nb = W @ D
            D = np.where(ok[:, None], data_weight * target + (1 - data_weight) * nb, nb)
        V = V + step * D
        stats = {"iter": it + 1, "constrained": int(ok.sum()), "mean_dist_before_mm": float(dist[ok].mean()) if ok.any() else float("nan")}
    closest, dist, _ = ct_skin.nearest.on_surface(V)
    ok = constrain & (dist < max_dist_mm)
    stats["mean_dist_after_mm"] = float(dist[ok].mean()) if ok.any() else float("nan")
    stats["p95_dist_after_mm"] = float(np.percentile(dist[ok], 95)) if ok.any() else float("nan")
    return V, stats


# ---------------------------------------------------------------------- template bones -> patient shape
COMPACT_PARTS = ("pelvis", "femur_r", "femur_l", "tibia_r", "tibia_l", "humerus_r", "humerus_l",
                 "ulna_r", "ulna_l", "radius_r", "radius_l", "head", "lumbar_body")


# ---------------------------------------------------------------------- seams between CT bones and their continuation
def _descendants(pid: int) -> list[int]:
    out, frontier = [], [pid]
    while frontier:
        cur = frontier.pop()
        kids = [i for i, par in enumerate(PARENT) if par == cur]
        out += kids; frontier += kids
    return out


def _section_centroid(mesh: trimesh.Trimesh, y: float):
    """Length-weighted centroid (x, z) of the mesh's cross-section at height ``y`` (None if empty)."""
    segs = trimesh.intersections.mesh_plane(mesh, [0.0, 1.0, 0.0], [0.0, y, 0.0])
    if segs is None or len(segs) == 0:
        return None
    mid = segs.mean(axis=1)[:, [0, 2]]
    L = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    if L.sum() < 1e-6:
        return None
    return (mid * L[:, None]).sum(0) / L.sum()


def align_seams(skel_parts: dict[str, trimesh.Trimesh], ct_parts: dict[str, trimesh.Trimesh], part_names: list[str],
                band_mm: float = 10.0, taper_mm: float = 30.0, min_pts: int = 20,
                skin: trimesh.Trimesh | None = None, skin_part: np.ndarray | None = None) -> dict:
    """Translate the unscanned continuation of a bone (and its child bones) so that its cross-section
    centroid matches the CT bone's cross-section at the scan border.  Modifies ``skel_parts`` in place.

    The per-bone ICP can leave the shaft a few degrees / ~1 cm off along a long bone; visually the
    continuation then looks detached from the CT bone.  Matching the centroids at the border makes the
    seam continuous; the shape difference of the template remains.
    """
    offsets = {}
    part_offsets = {}                      # part name -> cumulative translation applied (full, untapered)
    for name, sk in skel_parts.items():
        ct = ct_parts.get(name)
        if ct is None or len(ct.faces) == 0 or name not in part_names:
            continue
        pid = part_names.index(name)
        V = np.asarray(sk.vertices)
        for side in ("below", "above"):
            if side == "below":
                edge = ct.bounds[0][1]
                if sk.bounds[0][1] > edge - 20:
                    continue
                # cross-sections at the border: the continuation must meet the CT cross-section exactly there
                c_ct = _section_centroid(ct, edge + band_mm / 2)
                c_sk = _section_centroid(sk, edge + band_mm / 2)
                w = np.clip((edge + band_mm + taper_mm - V[:, 1]) / taper_mm, 0.0, 1.0)   # 1 through the border band, fading to 0 further inside the CT
            else:
                edge = ct.bounds[1][1]
                if sk.bounds[1][1] < edge + 20:
                    continue
                c_ct = _section_centroid(ct, edge - band_mm / 2)
                c_sk = _section_centroid(sk, edge - band_mm / 2)
                w = np.clip((V[:, 1] - (edge - band_mm - taper_mm)) / taper_mm, 0.0, 1.0)
            if c_ct is None or c_sk is None:
                continue
            off = np.array([c_ct[0] - c_sk[0], 0.0, c_ct[1] - c_sk[1]])
            V = V + w[:, None] * off
            sk.vertices = V
            desc = [cid for cid in _descendants(pid) if part_names[cid] not in ct_parts]
            part_offsets[name] = part_offsets.get(name, np.zeros(3)) + off
            for cid in desc:
                cname = part_names[cid]
                part_offsets[cname] = part_offsets.get(cname, np.zeros(3)) + off
                if cname in skel_parts:
                    skel_parts[cname].vertices = np.asarray(skel_parts[cname].vertices) + off
            if skin is not None and skin_part is not None:
                # the estimated skin follows its bones: same taper for this part, full offset for the unscanned children
                S = np.asarray(skin.vertices)
                ws = np.zeros(len(S))
                ws[skin_part == pid] = (np.clip((edge + band_mm + taper_mm - S[:, 1]) / taper_mm, 0.0, 1.0)
                                        if side == "below" else np.clip((S[:, 1] - (edge - band_mm - taper_mm)) / taper_mm, 0.0, 1.0))[skin_part == pid]
                ws[np.isin(skin_part, desc)] = 1.0
                skin.vertices = S + ws[:, None] * off
            offsets[f"{name}:{side}"] = off.round(2).tolist()
    align_seams.last_part_offsets = part_offsets
    return offsets


# ---------------------------------------------------------------------- shaft direction + final placement
def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / max(np.linalg.norm(a), 1e-9); b = b / max(np.linalg.norm(b), 1e-9)
    v = np.cross(a, b); c = float(np.dot(a, b)); sn = np.linalg.norm(v)
    if sn < 1e-8:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (sn ** 2))


def align_shafts(verts: np.ndarray, labels: np.ndarray, part_names: list[str], joints: np.ndarray,
                 transforms: dict, ct_parts: dict, parts=("femur_r", "femur_l", "humerus_r", "humerus_l"),
                 span_mm: float = 45.0, step_mm: float = 8.0, max_deg: float = 35.0) -> dict:
    """Rotate a long bone (and its unscanned children) about its proximal joint so that its shaft
    direction, measured by cross-section centroids next to the scan border, matches the CT shaft.
    Modifies ``verts``, ``joints`` and ``transforms`` in place; returns per-part statistics."""
    stats = {}
    for name in parts:
        if name not in part_names or name not in ct_parts or len(ct_parts[name].faces) == 0:
            continue
        pid = part_names.index(name)
        sel = np.where(labels == pid)[0]
        if len(sel) < 50:
            continue
        ct = ct_parts[name]
        lo, hi = ct.bounds[0][1], ct.bounds[1][1]
        part_lo, part_hi = verts[sel][:, 1].min(), verts[sel][:, 1].max()
        if part_lo < lo - 20:            # continuation below the scan
            ys = lo + 6 + np.arange(0, span_mm, step_mm)
        elif part_hi > hi + 20:          # continuation above
            ys = hi - 6 - np.arange(0, span_mm, step_mm)
        else:
            continue
        ys = ys[(ys > lo + 3) & (ys < hi - 3)]
        if len(ys) < 3:
            continue
        # SKEL part as a mesh for sectioning: use a point-cloud slab centroid instead (robust, no faces needed)
        pv = verts[sel]
        c_ct, c_sk = [], []
        for y in ys:
            cc = _section_centroid(ct, float(y))
            slab = pv[np.abs(pv[:, 1] - y) < step_mm / 2]
            if cc is None or len(slab) < 5:
                continue
            c_ct.append([cc[0], y, cc[1]]); c_sk.append([slab[:, 0].mean(), y, slab[:, 2].mean()])
        if len(c_ct) < 3:
            continue
        c_ct, c_sk = np.array(c_ct), np.array(c_sk)
        d_ct = np.linalg.svd(c_ct - c_ct.mean(0), full_matrices=False)[2][0]
        d_sk = np.linalg.svd(c_sk - c_sk.mean(0), full_matrices=False)[2][0]
        if np.dot(d_ct, c_ct[-1] - c_ct[0]) < 0: d_ct = -d_ct
        if np.dot(d_sk, c_sk[-1] - c_sk[0]) < 0: d_sk = -d_sk
        ang = float(np.degrees(np.arccos(np.clip(np.dot(d_ct, d_sk), -1, 1))))
        if ang > max_deg or ang < 0.5:
            stats[name] = {"shaft_angle_deg": round(ang, 2), "applied": False}
            continue
        R = _rot_between(d_sk, d_ct)
        pivot = joints[pid]
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pivot - R @ pivot
        ids = [pid] + [cid for cid in _descendants(pid) if part_names[cid] not in ct_parts]
        for g in ids:
            gsel = np.where(labels == g)[0]
            verts[gsel] = apply_T(T, verts[gsel])
            joints[g] = apply_T(T, joints[g:g + 1])[0]
            transforms[part_names[g]] = T @ transforms.get(part_names[g], np.eye(4))
        stats[name] = {"shaft_angle_deg": round(ang, 2), "applied": True}
    return stats


def _part_mesh(verts: np.ndarray, faces: np.ndarray, labels: np.ndarray, pid: int) -> trimesh.Trimesh | None:
    """Mesh of one part whose vertex array is exactly ``verts[labels == pid]`` (order preserved)."""
    sel = np.where(labels == pid)[0]
    if len(sel) == 0:
        return None
    remap = -np.ones(len(verts), dtype=np.int64); remap[sel] = np.arange(len(sel))
    f = faces[(labels[faces] == pid).all(1)]
    return trimesh.Trimesh(verts[sel], remap[f], process=False)


def skin_follow_bones(skin_param: np.ndarray, w_idx: np.ndarray, w_val: np.ndarray, transforms: dict,
                      part_names: list[str]) -> np.ndarray:
    """Linear blend skinning of the parametric skin with the final per-part transforms."""
    M = np.stack([transforms.get(n, np.eye(4)) for n in part_names])          # (24, 4, 4)
    Vh = np.concatenate([skin_param, np.ones((len(skin_param), 1))], axis=1)
    out = np.zeros_like(skin_param)
    for k in range(w_idx.shape[1]):
        j = w_idx[:, k].astype(int)
        out += w_val[:, k][:, None] * np.einsum("vab,vb->va", M[j][:, :3, :], Vh)
    return out


def finalize_placement(al: BoneAlignment, labels: np.ndarray, faces: np.ndarray, part_names: list[str], ct_parts: dict,
                       skin_param: np.ndarray, w_idx: np.ndarray, w_val: np.ndarray, skin_part: np.ndarray) -> dict:
    """After the bone ICP: shaft direction, seams, then move the skin with its bones.

    Returns dict(verts, joints, transforms, skin, shaft, seams).  All in mm.
    """
    verts, joints = al.verts.copy(), al.joints.copy()
    transforms = {n: al.transforms.get(n, np.eye(4)).copy() for n in part_names}
    shaft = align_shafts(verts, labels, part_names, joints, transforms, ct_parts)
    # seams on the part meshes (tapered) and as full translations for the transforms
    parts = {}
    for pid, n in enumerate(part_names):
        m = _part_mesh(verts, faces, labels, pid)
        if m is not None and len(m.faces):
            parts[n] = m
    seams = align_seams(parts, ct_parts, part_names)
    offs = getattr(align_seams, "last_part_offsets", {})
    for n, m in parts.items():
        sel = np.where(labels == part_names.index(n))[0]
        verts[sel] = np.asarray(m.vertices)
    for n, off in offs.items():
        T = np.eye(4); T[:3, 3] = off
        transforms[n] = T @ transforms[n]
        joints[part_names.index(n)] = joints[part_names.index(n)] + off
    skin = skin_follow_bones(skin_param, w_idx, w_val, transforms, part_names)
    return {"verts": verts, "joints": joints, "transforms": transforms, "skin": skin, "shaft": shaft, "seams": seams}
