"""Mapping between TotalSegmentator structure names and SKEL skeleton parts.

SKEL has 24 rigid bone parts named after its joints (``skel.kin_skel.skel_joints_name``).
TotalSegmentator ``total`` task names are lateralised (``_left`` / ``_right``);
the ``appendicular_bones`` task names are not, so those are split by side at
run time (see :func:`ct2skel.landmarks.split_by_side`).
"""
from __future__ import annotations

SKEL_PARTS = [
    "pelvis", "femur_r", "tibia_r", "talus_r", "calcn_r", "toes_r",
    "femur_l", "tibia_l", "talus_l", "calcn_l", "toes_l",
    "lumbar_body", "thorax", "head",
    "scapula_r", "humerus_r", "ulna_r", "radius_r", "hand_r",
    "scapula_l", "humerus_l", "ulna_l", "radius_l", "hand_l",
]

# TotalSegmentator structure -> SKEL part.  Lateralised names use {s} = l/r.
_TS_TO_SKEL = {
    "sacrum": "pelvis", "vertebrae_S1": "pelvis", "hip_{side}": "pelvis",
    "femur_{side}": "femur_{s}",
    "humerus_{side}": "humerus_{s}",
    "scapula_{side}": "scapula_{s}", "clavicula_{side}": "scapula_{s}",
    "skull": "head",
    "sternum": "thorax",                       # costal cartilage is not bone: exported separately (ct_cartilage)
    # appendicular_bones task (un-lateralised; side resolved at run time)
    "patella": "tibia_{s}", "tibia": "tibia_{s}", "fibula": "tibia_{s}",
    "tarsals": "calcn_{s}", "metatarsals": "calcn_{s}", "phalanges_feet": "toes_{s}",
    "ulna": "ulna_{s}", "radius": "radius_{s}",
    "carpals": "hand_{s}", "metacarpals": "hand_{s}", "phalanges_hand": "hand_{s}",
}
for _v in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"):
    _TS_TO_SKEL[f"vertebrae_{_v}"] = "head"
for _v in ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12"):
    _TS_TO_SKEL[f"vertebrae_{_v}"] = "thorax"
for _v in ("L1", "L2", "L3", "L4", "L5"):
    _TS_TO_SKEL[f"vertebrae_{_v}"] = "lumbar_body"
for _i in range(1, 13):
    _TS_TO_SKEL[f"rib_left_{_i}"] = "thorax"
    _TS_TO_SKEL[f"rib_right_{_i}"] = "thorax"

UNLATERALISED = {"patella", "tibia", "fibula", "tarsals", "metatarsals", "phalanges_feet",
                 "ulna", "radius", "carpals", "metacarpals", "phalanges_hand"}


def ts_to_skel_part(name: str, side: str | None = None) -> str | None:
    """Return the SKEL part for a TotalSegmentator structure name.

    ``side`` ('l' / 'r') is only needed for un-lateralised structures.
    """
    if name in _TS_TO_SKEL:
        tmpl = _TS_TO_SKEL[name]
        if "{s}" in tmpl:
            if side is None:
                return None
            return tmpl.format(s=side)
        return tmpl
    for suffix, s in (("_left", "l"), ("_right", "r")):
        if name.endswith(suffix):
            key = name[: -len(suffix)] + "_{side}"
            if key in _TS_TO_SKEL:
                return _TS_TO_SKEL[key].format(s=s)
    return None


# Colour per SKEL part for the viewer (hex strings), same palette for CT and SKEL parts.
PART_COLORS = {
    "pelvis": "#d98f4e", "lumbar_body": "#e0c060", "thorax": "#c7b8a0", "head": "#e8dcc4",
    "femur_r": "#7fb3d5", "femur_l": "#5d9cc9", "tibia_r": "#8fd3c4", "tibia_l": "#68c0ad",
    "talus_r": "#b9e2a1", "talus_l": "#9fd484", "calcn_r": "#a4c96e", "calcn_l": "#8bbd53",
    "toes_r": "#c9d96e", "toes_l": "#b6c853",
    "scapula_r": "#d7a3e0", "scapula_l": "#c58bd1", "humerus_r": "#f0a5a5", "humerus_l": "#e88b8b",
    "ulna_r": "#f2c28b", "ulna_l": "#ecad6a", "radius_r": "#f7d9a6", "radius_l": "#f0c98a",
    "hand_r": "#f9e6c8", "hand_l": "#f3dbb4",
}
