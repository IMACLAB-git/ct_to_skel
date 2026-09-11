"""Compare fitted SKEL parameters with the ground truth of a `synth --from-skel` phantom.

    python scripts/compare_ground_truth.py data/skel_phantom/ground_truth.json out/skel_phantom
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from ct2skel.skel_wrapper import load_skel

gt = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
man = json.loads((out / "manifest.json").read_text())
fit = man["fit"]

betas_gt, poses_gt = np.array(gt["betas"]), np.array(gt["poses"])
betas_fit, poses_fit = np.array(fit["betas"]), np.array(fit["poses"])
print("betas gt :", np.round(betas_gt, 2))
print("betas fit:", np.round(betas_fit, 2))
print("|dbeta| mean %.3f max %.3f" % (np.abs(betas_gt - betas_fit).mean(), np.abs(betas_gt - betas_fit).max()))
dq = np.degrees(np.arctan2(np.sin(poses_fit - poses_gt), np.cos(poses_fit - poses_gt)))
print("pose error (deg): mean %.2f  max %.2f  (global rot %.2f %.2f %.2f)" % (np.abs(dq).mean(), np.abs(dq).max(), *dq[:3]))
from skel.kin_skel import pose_param_names
worst = np.argsort(-np.abs(dq))[:8]
print("worst DOFs:", [(pose_param_names[i], round(float(dq[i]), 1)) for i in worst])

# joint position error: fitted joints vs ground-truth joints (both in SKEL frame, phantom frame centre subtracted)
m = load_skel(man["gender"], None, "cpu")
with torch.no_grad():
    o = m.forward(poses=torch.tensor(poses_gt)[None].float(), betas=torch.tensor(betas_gt)[None].float(),
                  trans=torch.tensor(gt["trans"])[None].float(), poses_type="skel", skelmesh=False)
j_gt = o.joints[0].numpy() * 1000.0
center = np.array(man["frame"]["center_mm"])
# CT frame centre -> SKEL frame: the phantom was generated with centre 0, so the fitted joints are offset by R*center
from ct2skel.frames import R_LPS_TO_SKEL
from ct2skel.labelmap import SKEL_PARTS
j_fit = np.array([man["joints"]["skel"][n] for n in SKEL_PARTS]) + R_LPS_TO_SKEL @ center
err = np.linalg.norm(j_fit - j_gt, axis=1)
print("joint position error (mm): mean %.1f  max %.1f" % (err.mean(), err.max()))
for n, e in sorted(zip(SKEL_PARTS, err), key=lambda x: -x[1])[:6]:
    print(f"   {n:12s} {e:6.1f}")
print("metrics:", json.dumps({k: v for k, v in man["metrics"].items() if k != "joints_mm"}))
