"""Patient bone surfaces from CT label masks with clinical-style clean-up.

Raw marching cubes on a label mask gives a noisy shell: voxel ripples, holes where the
cortex is thin, islands from partial volume.  :func:`smooth_bone_mesh` applies the usual
3D-printing pipeline per bone: morphological closing -> hole filling -> island removal ->
marching cubes -> Taubin smoothing -> decimation.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage as ndi

from .dicom_io import Volume
from .meshing import mask_to_mesh
from .segment import _ball, radius_mm_to_vox, remove_small_components


def clean_bone_mask(mask: np.ndarray, vol: Volume, close_mm: float = 2.5, min_component_mm3: float = 2000.0,
                    fill_holes: bool = True) -> np.ndarray:
    """Close small gaps, fill enclosed cavities and drop islands of a bone label mask."""
    m = mask.astype(bool)
    if not m.any():
        return m
    if close_mm > 0:
        # pad so that closing does not clip at the volume border
        r = np.maximum(np.round(radius_mm_to_vox(vol, close_mm)).astype(int), 1)
        pad = [(int(v), int(v)) for v in r]
        mp = np.pad(m, pad, mode="constant")
        mp = ndi.binary_closing(mp, structure=_ball(radius_mm_to_vox(vol, close_mm)))
        m = mp[pad[0][0]:mp.shape[0] - pad[0][1], pad[1][0]:mp.shape[1] - pad[1][1], pad[2][0]:mp.shape[2] - pad[2][1]]
    if fill_holes:
        m = ndi.binary_fill_holes(m)
    min_vox = max(int(min_component_mm3 / vol.voxel_volume_mm3()), 1)
    m = remove_small_components(m, min_vox)
    return m


def smooth_bone_mesh(mask: np.ndarray, vol: Volume, close_mm: float = 2.5, smooth_iterations: int = 30,
                     target_faces: int | None = None, min_component_mm3: float = 2000.0,
                     mc_step: int = 1, fill_holes: bool = True) -> trimesh.Trimesh:
    """Clean the mask and extract a smooth surface (LPS mm)."""
    m = clean_bone_mask(mask, vol, close_mm=close_mm, min_component_mm3=min_component_mm3, fill_holes=fill_holes)
    if not m.any():
        return trimesh.Trimesh()
    mesh = mask_to_mesh(m, vol, step=mc_step, smooth_iterations=0, target_faces=None, remove_border_caps=True)
    if len(mesh.faces) == 0:
        return mesh
    # Taubin keeps the volume while removing voxel ripples; more passes than for the skin
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iterations)
    if target_faces is not None and len(mesh.faces) > target_faces:
        from .meshing import decimate
        mesh = decimate(mesh, target_faces)
    return mesh
