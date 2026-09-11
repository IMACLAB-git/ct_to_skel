"""Continuity checks of an output: CT skin vs estimated skin by height band, CT vs SKEL bone axes, chain gaps.

    python scripts/check_continuity.py out/s1397
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

out = Path(sys.argv[1])
man = json.loads((out / "manifest.json").read_text())
ct = trimesh.load(out / "stl" / "ct_skin.stl"); sk = trimesh.load(out / "stl" / "skel_skin.stl")
skp, _ = trimesh.sample.sample_surface(sk, 400000, seed=0)
d, _ = cKDTree(skp).query(ct.vertices); y = ct.vertices[:, 1]
lo, hi = y.min(), y.max()
edges = np.linspace(lo, hi, 9)
print("CT skin -> estimated skin by height band:")
for a, b in zip(edges[:-1], edges[1:]):
    sel = (y >= a) & (y < b)
    if sel.any():
        print(f"  y {a:6.0f}..{b:6.0f}: mean {d[sel].mean():5.1f} mm  p90 {np.percentile(d[sel], 90):5.1f}")
icp = man.get("metrics", {}).get("refine", {}).get("bone_icp", {})
print("bone long-axis agreement (CT vs SKEL part inside the CT range):")
for part in ["femur_r", "femur_l", "humerus_r", "humerus_l"]:
    pc, ps = out / "stl" / f"ct_bone_{part}.stl", out / "stl" / f"skel_bone_{part}.stl"
    if not (pc.exists() and ps.exists()):
        continue
    c, s = trimesh.load(pc), trimesh.load(ps)
    inside = s.vertices[(s.vertices[:, 1] >= c.bounds[0][1]) & (s.vertices[:, 1] <= c.bounds[1][1])]
    if len(inside) < 20:
        continue
    axis = lambda v: np.linalg.svd(v - v.mean(0), full_matrices=False)[2][0]
    ang = np.degrees(np.arccos(abs(np.dot(axis(np.asarray(c.vertices)), axis(inside)))))
    print(f"  {part}: angle {ang:4.1f} deg, ICP residual {icp.get(part, {}).get('residual_mm', float('nan')):.1f} mm")
print("chain gaps (child top -> parent):")
for child, parent in [("tibia_r", "femur_r"), ("tibia_l", "femur_l"), ("ulna_r", "humerus_r"), ("ulna_l", "humerus_l")]:
    pc, pp = out / "stl" / f"skel_bone_{child}.stl", out / "stl" / f"skel_bone_{parent}.stl"
    if pc.exists() and pp.exists():
        c, p = trimesh.load(pc), trimesh.load(pp)
        print(f"  {child}->{parent}: min {cKDTree(p.vertices).query(c.vertices)[0].min():.1f} mm")
print("seam offsets:", man.get("seam_offsets_mm"))
