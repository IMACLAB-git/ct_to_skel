"""Threshold-based skin and bone segmentation plus optional TotalSegmentator labels."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .dicom_io import Volume, save_volume


# ---------------------------------------------------------------------- helpers
def largest_component(mask: np.ndarray, n: int = 1) -> np.ndarray:
    lab, num = ndi.label(mask)
    if num == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, num + 1))
    keep = np.argsort(sizes)[::-1][:n] + 1
    return np.isin(lab, keep)


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    lab, num = ndi.label(mask)
    if num == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, num + 1))
    keep = np.where(sizes >= min_voxels)[0] + 1
    return np.isin(lab, keep)


def fill_holes_2d(mask: np.ndarray) -> np.ndarray:
    out = np.empty_like(mask)
    for k in range(mask.shape[0]):
        out[k] = ndi.binary_fill_holes(mask[k])
    return out


def _ball(radius_vox: np.ndarray) -> np.ndarray:
    r = np.maximum(np.round(radius_vox).astype(int), 0)
    zz, yy, xx = np.mgrid[-r[0]:r[0] + 1, -r[1]:r[1] + 1, -r[2]:r[2] + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (zz / max(r[0], 1e-6)) ** 2 + (yy / max(r[1], 1e-6)) ** 2 + (xx / max(r[2], 1e-6)) ** 2
    return d <= 1.0


def radius_mm_to_vox(vol: Volume, r_mm: float) -> np.ndarray:
    sx, sy, sz = vol.spacing
    return np.array([r_mm / sz, r_mm / sy, r_mm / sx])


# ---------------------------------------------------------------------- body / skin
def body_mask(vol: Volume, hu_threshold: float = -400.0, open_radius_mm: float = 3.0) -> np.ndarray:
    """Binary mask of the patient body (table and cushions removed, lungs filled).

    Steps: threshold -> morphological opening (detaches the table) -> largest
    3D connected component -> 2D hole filling per axial slice.
    """
    m = vol.array > hu_threshold
    if open_radius_mm > 0:
        m = ndi.binary_opening(m, structure=_ball(radius_mm_to_vox(vol, open_radius_mm)))
    m = largest_component(m)
    m = fill_holes_2d(m)
    return m


# ---------------------------------------------------------------------- bone
def bone_mask(vol: Volume, body: np.ndarray | None = None, hu_threshold: float = 250.0,
              min_component_mm3: float = 500.0, close_radius_mm: float = 1.0) -> np.ndarray:
    """Binary bone mask from HU threshold inside the body mask."""
    m = vol.array > hu_threshold
    if body is not None:
        m &= body
    if close_radius_mm > 0:
        m = ndi.binary_closing(m, structure=_ball(radius_mm_to_vox(vol, close_radius_mm)))
    min_vox = int(min_component_mm3 / vol.voxel_volume_mm3())
    m = remove_small_components(m, max(min_vox, 1))
    return m


# ---------------------------------------------------------------------- TotalSegmentator
BONE_STRUCTURES_TOTAL = (
    [f"vertebrae_{v}" for v in ("C1", "C2", "C3", "C4", "C5", "C6", "C7",
                                "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
                                "L1", "L2", "L3", "L4", "L5", "S1")]
    + ["sacrum", "hip_left", "hip_right", "femur_left", "femur_right",
       "humerus_left", "humerus_right", "scapula_left", "scapula_right",
       "clavicula_left", "clavicula_right", "skull", "sternum", "costal_cartilages"]
    + [f"rib_{s}_{i}" for s in ("left", "right") for i in range(1, 13)]
)

BONE_STRUCTURES_APPENDICULAR = [
    "patella", "tibia", "fibula", "tarsals", "metatarsals", "phalanges_feet",
    "ulna", "radius", "carpals", "metacarpals", "phalanges_hand",
]


def run_totalsegmentator(vol: Volume, work_dir: str | os.PathLike, fast: bool = False,
                         tasks: tuple[str, ...] = ("total", "appendicular_bones"),
                         device: str = "gpu") -> Path:
    """Run TotalSegmentator on the volume and return the directory with per-structure masks.

    Requires the ``TotalSegmentator`` CLI (``pip install TotalSegmentator``) on PATH.
    Output layout: ``<work_dir>/labels/<structure>.nii.gz``.
    """
    import sys
    # prefer the executable installed next to the running interpreter (venv Scripts/bin), then PATH
    exe = shutil.which("TotalSegmentator", path=str(Path(sys.executable).parent)) or shutil.which("TotalSegmentator")
    if exe is None:
        raise RuntimeError("TotalSegmentator executable not found on PATH. "
                           "Install with `pip install TotalSegmentator` or pass --labels <dir>.")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    nii = work / "ct.nii.gz"
    save_volume(vol, nii)
    out = work / "labels"
    out.mkdir(exist_ok=True)
    for task in tasks:
        cmd = [exe, "-i", str(nii), "-o", str(out), "--task", task, "--device", device]
        if fast and task == "total":
            cmd.append("--fast")
        if task == "total":
            cmd += ["--roi_subset", *BONE_STRUCTURES_TOTAL]
        print("[totalseg]", " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            if task == "total":
                raise
            # appendicular_bones needs a (free academic) TotalSegmentator license: totalseg_set_license -l <key>
            print(f"[totalseg] task '{task}' failed (exit {e.returncode}); continuing without it. "
                  "If it needs a license, register at https://backend.totalsegmentator.com/license-academic/ "
                  "and run `totalseg_set_license -l <key>`.", flush=True)
    return out


def load_label_masks(label_dir: str | os.PathLike, vol: Volume,
                     names: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load per-structure binary masks (TotalSegmentator layout) resampled onto ``vol``'s grid.

    Also accepts a single multilabel NIfTI accompanied by ``<file>.json`` mapping
    ``{"label_id": "name"}`` (or the reverse).
    """
    import SimpleITK as sitk

    p = Path(label_dir)
    ref = vol.to_sitk(np.zeros(vol.array.shape, dtype=np.uint8))
    masks: dict[str, np.ndarray] = {}

    def resample(img: sitk.Image) -> np.ndarray:
        if (img.GetSize() == ref.GetSize() and np.allclose(img.GetSpacing(), ref.GetSpacing(), atol=1e-3)
                and np.allclose(img.GetOrigin(), ref.GetOrigin(), atol=1e-2)):
            return sitk.GetArrayFromImage(img)
        return sitk.GetArrayFromImage(sitk.Resample(img, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0))

    if p.is_dir():
        files = sorted(list(p.glob("*.nii.gz")) + list(p.glob("*.nii")))
        for f in files:
            name = f.name.replace(".nii.gz", "").replace(".nii", "")
            if names is not None and name not in names:
                continue
            arr = resample(sitk.ReadImage(str(f)))
            if arr.any():
                masks[name] = arr > 0
        return masks

    # multilabel file + json
    img = sitk.ReadImage(str(p))
    arr = resample(img)
    side = p.with_suffix("").with_suffix("") if p.name.endswith(".nii.gz") else p.with_suffix("")
    js = Path(str(side) + ".json")
    if not js.exists():
        raise RuntimeError(f"Multilabel file {p} needs a sidecar {js} mapping ids to names")
    mapping = json.loads(js.read_text())
    for k, v in mapping.items():
        lid, name = (int(k), v) if str(k).isdigit() else (int(v), k)
        if names is not None and name not in names:
            continue
        m = arr == lid
        if m.any():
            masks[name] = m
    return masks


def claim_unlabeled_bone(part_masks: dict[str, np.ndarray], bone_hu_mask: np.ndarray, vol: Volume,
                         radius_mm: float = 8.0, exclude: np.ndarray | None = None,
                         max_component_mm3: float = 5000.0) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Attach bone-density voxels that no label claims to the nearest labelled part (within ``radius_mm``).

    Segmentation labels often stop short of joints (rib heads at the vertebrae, joint margins);
    the real cortex is still in the CT, so this closes such gaps with actual bone voxels.
    """
    any_label = np.zeros_like(bone_hu_mask)
    for m in part_masks.values():
        any_label |= m
    unl = bone_hu_mask & ~any_label
    if exclude is not None:
        unl &= ~exclude                      # e.g. calcified costal cartilage: not bone, never claimed
    # large unlabelled pieces are whole bones the labels do not cover (forearms, lower legs ...):
    # keep them separate instead of gluing them onto a neighbouring labelled bone
    big = np.zeros_like(unl)
    out = {k: v.copy() for k, v in part_masks.items()}
    lab, num = ndi.label(unl)
    if num:
        sizes = ndi.sum(unl, lab, index=np.arange(1, num + 1)) * vol.voxel_volume_mm3()
        big_ids = np.where(sizes >= max_component_mm3)[0] + 1
        if len(big_ids):
            big = np.isin(lab, big_ids)
            unl &= ~big
            # a big piece that lies inside the axial extent of one labelled long bone and touches it is a gap in
            # that label (segmentation dropped a stretch of the shaft): give it back to the bone instead of keeping
            # a bare cortex tube next to it
            r_vox = radius_mm_to_vox(vol, radius_mm)
            struct = _ball(r_vox)
            margin = int(np.ceil(15.0 / vol.spacing[2]))
            pad = [int(np.ceil(r)) + 1 for r in r_vox]
            for cid in big_ids:
                idx = np.where(lab == cid)
                z0, z1 = int(idx[0].min()), int(idx[0].max())
                sl = tuple(slice(max(int(idx[a].min()) - pad[a], 0), min(int(idx[a].max()) + pad[a] + 1, lab.shape[a])) for a in range(3))
                comp = lab[sl] == cid
                touch = ndi.binary_dilation(comp, structure=struct)
                owners = []
                for name, m in part_masks.items():
                    if not (touch & m[sl]).any():
                        continue
                    mz = np.where(m.any(axis=(1, 2)))[0]
                    if len(mz) and mz.min() + margin <= z0 and z1 <= mz.max() - margin:
                        owners.append(name)
                if len(owners) == 1:
                    out[owners[0]][sl] |= comp
                    big[sl] &= ~comp
    if not unl.any():
        return out, big
    best = np.full(bone_hu_mask.shape, np.inf, dtype=np.float32)
    owner = np.full(bone_hu_mask.shape, -1, dtype=np.int16)
    names = list(part_masks)
    r_vox = radius_mm_to_vox(vol, radius_mm)
    for i, name in enumerate(names):
        m = part_masks[name]
        idx = np.where(m)
        if len(idx[0]) == 0:
            continue
        lo = [max(int(idx[a].min() - np.ceil(r_vox[a]) - 1), 0) for a in range(3)]
        hi = [min(int(idx[a].max() + np.ceil(r_vox[a]) + 2), m.shape[a]) for a in range(3)]
        sl = tuple(slice(lo[a], hi[a]) for a in range(3))
        if not unl[sl].any():
            continue
        d = ndi.distance_transform_edt(~m[sl], sampling=vol.spacing[::-1]).astype(np.float32)
        cand = unl[sl] & (d <= radius_mm) & (d < best[sl])
        best[sl][cand] = d[cand]
        owner[sl][cand] = i
    for i, name in enumerate(names):
        add = owner == i
        if add.any():
            out[name] |= add
    return out, big


def trim_thin_structures(mask: np.ndarray, vol: Volume, min_thickness_mm: float = 4.0, core_min_mm3: float = 2000.0,
                         regrow_mm: float = 3.0) -> np.ndarray:
    """Remove thin appendages (wires, tape, contrast-filled vessels) from a bone-piece mask.

    Voxels thinner than ``min_thickness_mm`` are eroded away (binary opening); only thick cores of at least
    ``core_min_mm3`` survive, and the original mask is then re-grown ``regrow_mm`` around those cores so that thin
    cortex next to thick bone is kept while a long thin appendage keeps only a short stub."""
    r_open = radius_mm_to_vox(vol, min_thickness_mm / 2.0)
    opened = ndi.binary_opening(mask, structure=_ball(r_open))
    lab, num = ndi.label(opened)
    if not num:
        return np.zeros_like(mask)
    sizes = ndi.sum(opened, lab, index=np.arange(1, num + 1)) * vol.voxel_volume_mm3()
    core = np.isin(lab, np.where(sizes >= core_min_mm3)[0] + 1)
    if not core.any():
        return np.zeros_like(mask)
    grown = ndi.binary_dilation(core, structure=_ball(radius_mm_to_vox(vol, regrow_mm)))
    return mask & grown
