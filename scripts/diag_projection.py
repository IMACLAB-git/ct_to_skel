"""Quick 2-D diagnostic of a fit: coronal + sagittal projections of CT/SKEL skin & bone with joint pairs.

    python scripts/diag_projection.py out/s1397 [out.png]
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import trimesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = Path(sys.argv[1])
png = Path(sys.argv[2]) if len(sys.argv) > 2 else out / "diag_projection.png"
man = json.loads((out / "manifest.json").read_text())
load = lambda n: trimesh.load(out / "stl" / f"{n}.stl")
meshes = [(load("ct_skin"), "#5cc8ff", "CT skin"), (load("skel_skin"), "#ffb454", "SKEL skin"),
          (load("ct_bone_all"), "#1f6fb0", "CT bone"), (load("skel_bone_all"), "#c0392b", "SKEL bone")]
rng = np.random.default_rng(0)
J = man["joints"]
fig, ax = plt.subplots(1, 2, figsize=(14, 9))
for k, ((i, j), title) in enumerate([((0, 1), "coronal (x=left, y=up)"), ((2, 1), "sagittal (z=anterior, y=up)")]):
    for m, c, lab in meshes:
        p = m.vertices[rng.choice(len(m.vertices), min(15000, len(m.vertices)), replace=False)]
        ax[k].scatter(p[:, i], p[:, j], s=1, c=c, alpha=0.35, label=lab, linewidths=0)
    for name, d in J.get("ct", {}).items():
        p, q = d["pos_skel_mm"], J["skel"][name]
        ax[k].plot([p[i], q[i]], [p[j], q[j]], "k-", lw=1)
        ax[k].plot(p[i], p[j], "go", ms=5); ax[k].plot(q[i], q[j], "ro", ms=4)
        ax[k].annotate(name, (p[i], p[j]), fontsize=7)
    ax[k].set_aspect("equal"); ax[k].set_title(title); ax[k].legend(markerscale=6, fontsize=8)
m = man.get("metrics", {})
fig.suptitle(f"{man['case']}: skin {m.get('skin_ct_to_skel', {}).get('mean_mm', float('nan')):.1f} mm, "
             f"bone {m.get('bone_ct_to_skel', {}).get('mean_mm', float('nan')):.1f} mm, "
             f"joints {m.get('joints_mean_mm', float('nan')):.1f} mm (green = CT target, red = SKEL)")
plt.tight_layout(); plt.savefig(png, dpi=70)
print("wrote", png)
print(json.dumps(m.get("joints_mm", {}), indent=0))
