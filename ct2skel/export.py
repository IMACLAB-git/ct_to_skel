"""Export CT and SKEL meshes as STL (+ error-coloured PLY), manifest.json and the web viewer."""
from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from . import __version__
from .frames import FrameTransform
from .labelmap import PART_COLORS, SKEL_PARTS

VIEWER_DIR = Path(__file__).resolve().parent / "viewer"
SKIN_COLOR_CT, SKIN_COLOR_SKEL = "#d9b3a3", "#e7c8b8"


def _mesh_info(m: trimesh.Trimesh) -> dict:
    if len(m.faces) == 0:
        return {"faces": 0, "vertices": 0, "centroid": [0, 0, 0], "bbox": [[0, 0, 0], [0, 0, 0]]}
    return {"faces": int(len(m.faces)), "vertices": int(len(m.vertices)),
            "centroid": m.vertices.mean(0).tolist(), "bbox": m.bounds.tolist()}


def _turbo(t: np.ndarray) -> np.ndarray:
    """Turbo-like colormap without matplotlib dependency; t in [0,1] -> (N,3) uint8."""
    stops = np.array([[48, 18, 59], [70, 107, 227], [39, 200, 214], [122, 250, 100],
                      [232, 218, 43], [251, 106, 22], [165, 21, 4]], dtype=np.float64)
    x = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.floor(x).astype(int).clip(0, len(stops) - 2)
    f = (x - i)[:, None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def error_ply(mesh: trimesh.Trimesh, ref_pts: np.ndarray, err_max_mm: float, path: Path,
              bbox: np.ndarray | None = None) -> np.ndarray:
    """Colour mesh vertices by nearest distance (mm) to ``ref_pts`` and write a PLY.

    Vertices outside ``bbox`` (the CT coverage) are drawn grey and excluded from the returned
    distances, so parts that extend beyond the scan do not report meaningless errors.
    """
    if len(mesh.faces) == 0 or ref_pts is None or len(ref_pts) == 0:
        return np.zeros(0)
    d, _ = cKDTree(ref_pts).query(mesh.vertices)
    inside = np.ones(len(d), dtype=bool)
    if bbox is not None:
        inside = ((mesh.vertices >= bbox[0] - 10) & (mesh.vertices <= bbox[1] + 10)).all(1)
    colors = np.concatenate([_turbo(d / err_max_mm), np.full((len(d), 1), 255, np.uint8)], axis=1)
    colors[~inside, :3] = 120
    m = mesh.copy()
    m.visual.vertex_colors = colors
    m.export(path)
    return d[inside] if inside.any() else d


class CaseExporter:
    """Collect meshes (SKEL frame, mm) and write them together with the viewer."""

    def __init__(self, out_dir: str | Path, case: str, gender: str, frame: FrameTransform,
                 err_max_mm: float = 20.0, bbox_mm: np.ndarray | None = None):
        self.bbox = bbox_mm
        self.out = Path(out_dir)
        (self.out / "stl").mkdir(parents=True, exist_ok=True)
        (self.out / "ply").mkdir(exist_ok=True)
        self.case, self.gender, self.frame, self.err_max = case, gender, frame, err_max_mm
        self.parts: list[dict] = []
        self.extra: dict = {}

    def add_mesh(self, mesh: trimesh.Trimesh, pid: str, group: str, kind: str, name: str,
                 part: str | None = None, color: str | None = None, err_ref: np.ndarray | None = None,
                 hidden: bool = False, static: bool = False) -> None:
        if mesh is None or len(mesh.faces) == 0:
            return
        f = self.out / "stl" / f"{pid}.stl"
        mesh.export(f)
        entry = {"id": pid, "group": group, "kind": kind, "name": name, "part": part, "hidden": hidden, "static": static,
                 "file": f"stl/{f.name}",
                 "color": color or (PART_COLORS.get(part, "#cccccc") if kind in ("bone", "bone_tpl") else
                                    (SKIN_COLOR_CT if group == "ct" else SKIN_COLOR_SKEL)),
                 **_mesh_info(mesh)}
        if err_ref is not None and len(err_ref):
            p = self.out / "ply" / f"{pid}_err.ply"
            d = error_ply(mesh, err_ref, self.err_max, p, bbox=self.bbox)
            entry["err_file"] = f"ply/{p.name}"
            entry["err_mean_mm"] = float(d.mean())
            entry["err_p95_mm"] = float(np.percentile(d, 95))
        self.parts.append(entry)

    def write(self, metrics: dict | None = None, fit: dict | None = None, joints_skel: np.ndarray | None = None,
              joints_ct: dict | None = None, volume_meta: dict | None = None) -> Path:
        manifest = {
            "case": self.case, "gender": self.gender, "units": "mm", "ct2skel_version": __version__,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "frame": self.frame.to_dict(),
            "axes": {"x": "patient left", "y": "superior", "z": "anterior"},
            "error_range_mm": self.err_max,
            "parts": self.parts,
            "joints": {"skel": {n: joints_skel[i].tolist() for i, n in enumerate(SKEL_PARTS)} if joints_skel is not None else {},
                       "ct": joints_ct or {}},
            "metrics": metrics or {},
            "fit": fit or {},
            "volume": volume_meta or {},
            **self.extra,
        }
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
        if fit:
            np.savez(self.out / "skel_params.npz", **{k: np.asarray(v) for k, v in fit.items()
                                                       if k in ("betas", "poses", "trans", "scale")})
        for fn in ("index.html", "app.js"):
            shutil.copyfile(VIEWER_DIR / fn, self.out / fn)
        return self.out / "manifest.json"


def split_ct_bones_by_skel(ct_bone: trimesh.Trimesh, skel_verts_mm: np.ndarray, skel_labels: np.ndarray,
                           names: list[str]) -> dict[str, trimesh.Trimesh]:
    """Partition the CT bone mesh by the nearest fitted SKEL bone part."""
    from .skel_wrapper import split_mesh_by_label
    if len(ct_bone.faces) == 0:
        return {}
    _, idx = cKDTree(skel_verts_mm).query(ct_bone.vertices)
    labels = skel_labels[idx]
    return split_mesh_by_label(np.asarray(ct_bone.vertices), np.asarray(ct_bone.faces), labels, names)


# ---------------------------------------------------------------------- CT volume for the viewer
def _signed_permutation(M: np.ndarray, tol: float = 1e-3) -> bool:
    A = np.abs(M)
    return bool(np.allclose(A.sum(0), 1, atol=tol) and np.allclose(A.sum(1), 1, atol=tol) and np.allclose(A.max(0), 1, atol=tol))


def export_ct_volume(vol, frame: FrameTransform, out_dir: str | Path, max_inplane: int = 320,
                     max_slices: int = 512, hu_range: tuple[float, float] = (-1024.0, 1536.0)) -> dict:
    """Write the CT as an 8-bit volume re-oriented to the SKEL frame for slice overlays in the viewer.

    Output array order is (Y, Z, X) in SKEL axes with coordinates increasing along each axis:
    slice j is the axial image at ``origin_mm[1] + j * spacing_mm[1]``; within a slice, row r is
    anterior coordinate ``origin_mm[2] + r * spacing_mm[2]`` and column c is ``origin_mm[0] + c * spacing_mm[0]``.
    """
    import SimpleITK as sitk
    from skimage.measure import block_reduce
    from .dicom_io import Volume
    from .frames import R_LPS_TO_SKEL

    if not _signed_permutation(vol.direction):
        # oblique acquisition: resample onto an axis-aligned LPS grid with the same spacing
        img = vol.to_sitk()
        ref = sitk.Image(img.GetSize(), sitk.sitkFloat32)
        ref.SetSpacing(img.GetSpacing()); ref.SetOrigin(img.GetOrigin()); ref.SetDirection(np.eye(3).flatten().tolist())
        vol = Volume.from_sitk(sitk.Resample(img, ref, sitk.Transform(), sitk.sitkLinear, float(hu_range[0])), vol.meta)

    a_xyz = vol.array.transpose(2, 1, 0)                        # index order (x, y, z)
    M = R_LPS_TO_SKEL @ vol.direction                            # column a = SKEL direction of index axis a
    idx = [int(np.argmax(np.abs(M[s]))) for s in range(3)]      # index axis for SKEL X, Y, Z
    sign = [float(np.sign(M[s, idx[s]])) for s in range(3)]
    out = np.transpose(a_xyz, (idx[1], idx[2], idx[0]))         # (Y, Z, X)
    for n, s in enumerate((1, 2, 0)):
        if sign[s] < 0:
            out = np.flip(out, axis=n)
    spacing = np.array([vol.spacing[idx[1]], vol.spacing[idx[2]], vol.spacing[idx[0]]])   # (Y, Z, X)
    nx, ny, nz = vol.shape_xyz
    corners = np.array([[i, j, k] for i in (0, nx - 1) for j in (0, ny - 1) for k in (0, nz - 1)], dtype=float)
    skel_corners = frame.lps_to_skel(vol.index_to_world(corners)) * 1000.0
    origin_xyz = skel_corners.min(0)                             # (X, Y, Z) mm
    origin = np.array([origin_xyz[1], origin_xyz[2], origin_xyz[0]])   # (Y, Z, X)

    f = np.array([max(int(np.ceil(out.shape[0] / max_slices)), 1),
                  max(int(np.ceil(max(out.shape[1], out.shape[2]) / max_inplane)), 1)])
    factors = (int(f[0]), int(f[1]), int(f[1]))
    if any(k > 1 for k in factors):
        out = block_reduce(np.ascontiguousarray(out, dtype=np.float32), factors, np.mean, cval=float(hu_range[0]))
        origin = origin + (np.array(factors) - 1) / 2.0 * spacing
        spacing = spacing * np.array(factors)
    lo, hi = hu_range
    u8 = np.clip((out - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    path = Path(out_dir) / "ct_volume.bin"
    u8.tofile(path)
    return {"file": path.name, "dtype": "uint8", "order": "YZX", "shape": [int(v) for v in u8.shape],
            "spacing_mm": [float(spacing[2]), float(spacing[0]), float(spacing[1])],      # (X, Y, Z)
            "origin_mm": [float(origin[2]), float(origin[0]), float(origin[1])],          # (X, Y, Z)
            "hu_min": lo, "hu_max": hi, "bytes": int(u8.size)}
