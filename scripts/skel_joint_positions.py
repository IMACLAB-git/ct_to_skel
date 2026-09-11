"""Print where each SKEL joint sits relative to its rigid bone part (T-pose, mm).

Used to derive the CT landmark rules in ct2skel/landmarks.py.
    python scripts/skel_joint_positions.py [male|female]
"""
import sys
import numpy as np
import torch
from ct2skel.skel_wrapper import load_skel, bone_part_labels
from ct2skel.labelmap import SKEL_PARTS

m = load_skel(sys.argv[1] if len(sys.argv) > 1 else "male", None, "cpu")
with torch.no_grad():
    o = m.forward(poses=torch.zeros(1, 46), betas=torch.zeros(1, 10), trans=torch.zeros(1, 3), poses_type="skel", skelmesh=True)
J = o.joints[0].numpy() * 1000
V = o.skel_verts[0].numpy() * 1000
lab = bone_part_labels(m)
np.set_printoptions(precision=0, suppress=True)
for i, n in enumerate(SKEL_PARTS):
    v = V[lab == i]
    lo, hi = v.min(0), v.max(0)
    rel = (J[i] - lo) / (hi - lo + 1e-9)
    print(f"{n:12s} joint={J[i]}  part y=[{lo[1]:.0f},{hi[1]:.0f}] z=[{lo[2]:.0f},{hi[2]:.0f}] x=[{lo[0]:.0f},{hi[0]:.0f}]  rel={rel.round(2)}")
