"""How far the displayed bones are from the parametric SKEL model (re-posing consistency).

For every part: rotation angle and translation of the residual transform D_j that carries the
parametric part frame to the displayed (ICP/seam-corrected) one.  Small values (< 5 deg, < 10 mm)
mean that re-posing with the SKEL kinematics stays consistent.

    python scripts/check_residuals.py out/s1397
"""
import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
man = json.loads((out / "manifest.json").read_text())
T = man.get("metrics", {}).get("refine", {}).get("bone_transforms", {})
if not T:
    raise SystemExit("no bone_transforms in manifest")
rows = []
for name, M in T.items():
    M = np.array(M)
    R = M[:3, :3]
    R = R / np.cbrt(max(np.linalg.det(R), 1e-9))
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    j = np.array(man["joints"]["skel"].get(name, [0, 0, 0]))
    # translation of the joint itself (rotation about the origin folded in)
    t = M[:3, 3] + (M[:3, :3] - np.eye(3)) @ j if name in man["joints"]["skel"] else M[:3, 3]
    rows.append((name, ang, np.linalg.norm(t)))
rows.sort(key=lambda r: -r[1])
print(f"{'part':12s} {'rot deg':>8s} {'joint shift mm':>15s}")
for n, a, t in rows:
    print(f"{n:12s} {a:8.1f} {t:15.1f}")
print(f"mean rot {np.mean([r[1] for r in rows]):.1f} deg, max {max(r[1] for r in rows):.1f}; mean shift {np.mean([r[2] for r in rows]):.1f} mm")
