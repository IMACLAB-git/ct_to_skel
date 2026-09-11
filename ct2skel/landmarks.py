"""Anatomical joint targets for SKEL derived from TotalSegmentator bone masks.

The SKEL joint centres are OpenSim-style anatomical joints (note: the spine joints
sit at the *top* of each segment: lumbar_body ~ L1, thorax ~ T1, head ~ C1).  The
rules below are heuristics on the segmented bones (LPS mm); each target carries a weight
in [0, 1] expressing how well the heuristic matches the SKEL definition.
Joints that cannot be derived (structure missing or truncated by the scan
border) get weight 0.

All positions are returned in the CT world frame (LPS mm).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .dicom_io import Volume
from .labelmap import SKEL_PARTS, UNLATERALISED

PART_INDEX = {n: i for i, n in enumerate(SKEL_PARTS)}


@dataclass
class JointTargets:
    positions: np.ndarray = field(default_factory=lambda: np.zeros((24, 3)))   # LPS mm
    weights: np.ndarray = field(default_factory=lambda: np.zeros(24))
    notes: dict = field(default_factory=dict)

    def set(self, name: str, pos, weight: float, note: str = ""):
        i = PART_INDEX[name]
        self.positions[i] = np.asarray(pos, dtype=np.float64)
        self.weights[i] = float(weight)
        self.notes[name] = note

    def as_dict(self) -> dict:
        return {n: {"pos_lps_mm": self.positions[i].tolist(), "weight": float(self.weights[i]),
                    "note": self.notes.get(n, "")}
                for i, n in enumerate(SKEL_PARTS) if self.weights[i] > 0}


# ---------------------------------------------------------------------- mask utilities
def _world_coords(mask: np.ndarray, vol: Volume) -> np.ndarray:
    zyx = np.argwhere(mask)
    return vol.zyx_to_world(zyx)


def centroid(mask: np.ndarray, vol: Volume) -> np.ndarray:
    return _world_coords(mask, vol).mean(axis=0)


def touches_border(mask: np.ndarray, axis: int, side: str, margin: int = 1) -> bool:
    """True if the mask reaches within ``margin`` voxels of the volume border along ``axis``."""
    idx = np.where(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
    if len(idx) == 0:
        return True
    n = mask.shape[axis]
    return idx.min() <= margin if side == "low" else idx.max() >= n - 1 - margin


def _z_extent_truncated(mask: np.ndarray, vol: Volume, top: bool) -> bool:
    # axial axis is array axis 0 ; whether index 0 is superior depends on direction[2,2]
    z_sign = np.sign(vol.direction[2, 2]) or 1.0
    side_top = "high" if z_sign > 0 else "low"
    side = side_top if top else ("low" if side_top == "high" else "high")
    return touches_border(mask, 0, side)


def end_region(mask: np.ndarray, vol: Volume, length_mm: float, top: bool) -> np.ndarray:
    """Voxel world coords within ``length_mm`` of the superior (top) or inferior end of the mask."""
    pts = _world_coords(mask, vol)
    z = pts[:, 2]
    if top:
        sel = z >= z.max() - length_mm
    else:
        sel = z <= z.min() + length_mm
    return pts[sel]


def split_by_side(mask: np.ndarray, vol: Volume, midline_x: float) -> dict[str, np.ndarray]:
    """Split an un-lateralised mask into {'l': mask, 'r': mask} by connected-component centroid."""
    lab, num = ndi.label(mask)
    out = {"l": np.zeros_like(mask), "r": np.zeros_like(mask)}
    for i in range(1, num + 1):
        comp = lab == i
        cx = centroid(comp, vol)[0]
        side = "l" if cx > midline_x else "r"      # LPS +x = patient left
        out[side] |= comp
    return out


# ---------------------------------------------------------------------- joint rules
def derive_joint_targets(masks: dict[str, np.ndarray], vol: Volume,
                         midline_x: float | None = None) -> JointTargets:
    jt = JointTargets()
    if midline_x is None:
        ref = [n for n in ("sacrum", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_T12", "sternum")
               if n in masks]
        midline_x = centroid(masks[ref[0]], vol)[0] if ref else 0.0

    def has(n):
        return n in masks and masks[n].any()

    # ---- hips / knees from femur
    for side, ts in (("r", "femur_right"), ("l", "femur_left")):
        if not has(ts):
            continue
        m = masks[ts]
        if not _z_extent_truncated(m, vol, top=True):
            head = end_region(m, vol, 30.0, top=True)
            # femoral head is the medial portion of the proximal femur
            medial = head[:, 0] < np.median(head[:, 0]) if side == "l" else head[:, 0] > np.median(head[:, 0])
            if medial.sum() > 10:
                head = head[medial]
            jt.set(f"femur_{side}", head.mean(axis=0), 1.0, "femoral head centre (proximal 30 mm, medial half)")
        if not _z_extent_truncated(m, vol, top=False):
            knee = end_region(m, vol, 20.0, top=False)
            jt.set(f"tibia_{side}", knee.mean(axis=0), 0.8, "knee centre (distal femur 20 mm)")

    # ---- shoulders / elbows from humerus.  The humeral head is the end closest to the
    # shoulder girdle (scapula / clavicle of the same side); with the arms raised above the
    # head (common in chest/abdomen CT) the head is the *inferior* end, so do not assume.
    for side, ts, ref_names in (("r", "humerus_right", ("scapula_right", "clavicula_right")),
                                ("l", "humerus_left", ("scapula_left", "clavicula_left"))):
        if not has(ts):
            continue
        m = masks[ts]
        ref = next((centroid(masks[n], vol) for n in ref_names if has(n)), None)
        top_ok, bot_ok = not _z_extent_truncated(m, vol, top=True), not _z_extent_truncated(m, vol, top=False)
        top_c = end_region(m, vol, 30.0, top=True).mean(axis=0)
        bot_c = end_region(m, vol, 30.0, top=False).mean(axis=0)
        if ref is not None:
            head_is_top = np.linalg.norm(top_c - ref) < np.linalg.norm(bot_c - ref)
        else:
            head_is_top = True                                    # no girdle in the scan: assume arms down
        head_ok, elbow_ok = (top_ok, bot_ok) if head_is_top else (bot_ok, top_ok)
        head_c, elbow_c = (top_c, bot_c) if head_is_top else (bot_c, top_c)
        how = "superior" if head_is_top else "inferior (arm raised)"
        if head_ok:
            jt.set(f"humerus_{side}", head_c, 1.0, f"humeral head centre ({how} 30 mm, nearest shoulder girdle)")
        if elbow_ok:
            elb = end_region(m, vol, 20.0, top=not head_is_top).mean(axis=0)
            jt.set(f"ulna_{side}", elb, 0.7, f"elbow centre (20 mm at the end away from the girdle)")

    # ---- scapula joint ~ lateral clavicle (acromioclavicular region)
    for side, ts in (("r", "clavicula_right"), ("l", "clavicula_left")):
        if not has(ts):
            continue
        pts = _world_coords(masks[ts], vol)
        x = pts[:, 0]
        sel = x >= x.max() - 15.0 if side == "l" else x <= x.min() + 15.0
        jt.set(f"scapula_{side}", pts[sel].mean(axis=0), 0.5, "lateral clavicle end (15 mm)")

    # ---- spine joints.  Measured on the SKEL T-pose (see scripts/skel_joint_positions.py):
    #   lumbar_body joint sits ~40 mm below the top of the lumbar spine  -> L1 vertebral body
    #   thorax joint sits ~20 mm below the top of the thoracic cage      -> T1 vertebral body
    if has("vertebrae_L1"):
        jt.set("lumbar_body", centroid(masks["vertebrae_L1"], vol), 0.6, "L1 centroid (SKEL lumbar_body joint)")
    if has("vertebrae_T1"):
        jt.set("thorax", centroid(masks["vertebrae_T1"], vol), 0.6, "T1 centroid (SKEL thorax joint)")
    if has("vertebrae_C1"):
        jt.set("head", centroid(masks["vertebrae_C1"], vol), 0.5, "C1 (atlas) centroid ~ skull base")
    elif has("skull") and not _z_extent_truncated(masks["skull"], vol, top=False):
        jt.set("head", end_region(masks["skull"], vol, 20.0, top=False).mean(axis=0), 0.3, "skull base (inferior 20 mm)")

    # ---- pelvis joint ~ midpoint of the ASIS
    asis = []
    for ts in ("hip_left", "hip_right"):
        if not has(ts):
            continue
        pts = _world_coords(masks[ts], vol)
        z = pts[:, 2]
        upper = pts[z >= z.min() + 0.6 * (z.max() - z.min())]
        if len(upper) < 50:
            continue
        ant = upper[np.argsort(upper[:, 1])[:50]]           # smallest y (LPS) = most anterior
        asis.append(ant.mean(axis=0))
    if len(asis) == 2:
        jt.set("pelvis", 0.5 * (asis[0] + asis[1]), 0.5, "ASIS midpoint (anterior-most iliac points)")

    # ---- appendicular bones (un-lateralised): tibia -> ankle, carpals -> wrist
    for name in ("tibia", "carpals"):
        if not has(name) or name not in UNLATERALISED:
            continue
        sides = split_by_side(masks[name], vol, midline_x)
        for side, m in sides.items():
            if not m.any():
                continue
            if name == "tibia":
                if not _z_extent_truncated(m, vol, top=False):
                    jt.set(f"talus_{side}", end_region(m, vol, 15.0, top=False).mean(axis=0), 0.7,
                           "ankle centre (distal tibia 15 mm)")
                if jt.weights[PART_INDEX[f"tibia_{side}"]] == 0 and not _z_extent_truncated(m, vol, top=True):
                    jt.set(f"tibia_{side}", end_region(m, vol, 20.0, top=True).mean(axis=0), 0.7,
                           "knee centre (proximal tibia 20 mm)")
            elif name == "carpals":
                jt.set(f"hand_{side}", centroid(m, vol), 0.5, "carpal centroid ~ wrist")
    return jt
