"""Synthetic CT phantoms for testing the pipeline without patient data.

* :func:`make_phantom` — analytic torso + spine + femurs phantom with a CT table.
* :func:`make_from_skel` — voxelise a SKEL instance (skin + skeleton) into HU, giving a
  ground-truth (betas, poses, trans) for closed-loop validation of the fit.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .dicom_io import Volume

HU_AIR, HU_FAT, HU_SOFT, HU_BONE, HU_TABLE = -1000.0, -100.0, 40.0, 700.0, 250.0


def _grid(shape_xyz, spacing, origin):
    nx, ny, nz = shape_xyz
    sx, sy, sz = spacing
    x = origin[0] + np.arange(nx) * sx
    y = origin[1] + np.arange(ny) * sy
    z = origin[2] + np.arange(nz) * sz
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    return X, Y, Z


def make_phantom(shape_xyz=(128, 96, 140), spacing=(2.5, 2.5, 2.5), seed: int = 0) -> Volume:
    """Torso-like ellipsoid body with a spine, pelvis-like bone ring and two femurs.

    Geometry is in LPS mm with the body axis along +z (superior).
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape_xyz
    origin = (-shape_xyz[0] * spacing[0] / 2, -shape_xyz[1] * spacing[1] / 2, -shape_xyz[2] * spacing[2] / 2)
    X, Y, Z = _grid(shape_xyz, spacing, origin)
    hu = np.full(X.shape, HU_AIR, dtype=np.float32)

    # body: ellipsoidal cylinder (x half-width 150, y half-depth 100) tapering at the ends
    zr = (Z - Z.min()) / (Z.max() - Z.min())
    taper = 0.75 + 0.25 * np.sin(np.pi * zr)
    body = ((X / (150 * taper)) ** 2 + ((Y + 10) / (100 * taper)) ** 2) <= 1.0
    hu[body] = HU_SOFT
    # subcutaneous fat ring
    fat = body & (((X / (150 * taper)) ** 2 + ((Y + 10) / (100 * taper)) ** 2) > 0.85)
    hu[fat] = HU_FAT

    # spine: cylinder posterior of centre with vertebra-like density modulation
    spine = ((X) ** 2 + ((Y - 45)) ** 2) <= 18 ** 2
    spine &= body
    hu[spine] = HU_BONE * (0.8 + 0.2 * np.cos(2 * np.pi * Z / 30.0))[spine]
    # pelvis ring near the inferior third
    zc = origin[2] + 0.3 * nz * spacing[2]
    ring = (np.abs(Z - zc) < 25) & (((X / 110) ** 2 + ((Y - 15) / 70) ** 2) <= 1.0) & \
           (((X / 95) ** 2 + ((Y - 15) / 55) ** 2) >= 1.0)
    hu[ring & body] = HU_BONE
    # femurs: two cylinders below the pelvis
    for sx_ in (-70, 70):
        fem = ((X - sx_) ** 2 + (Y - 5) ** 2 <= 14 ** 2) & (Z < zc + 5)
        hu[fem & body] = HU_BONE
    # lungs: two low-density ellipsoids in the upper part
    zl = origin[2] + 0.72 * nz * spacing[2]
    for sx_ in (-60, 60):
        lung = (((X - sx_) / 45) ** 2 + ((Y + 5) / 55) ** 2 + ((Z - zl) / 90) ** 2) <= 1.0
        hu[lung & body] = -800.0
    # CT table: thin curved slab posterior to the body, touching it
    table = (np.abs(Y - 118 + 0.002 * X ** 2) < 6) & (np.abs(X) < 200)
    hu[table] = HU_TABLE
    hu += rng.normal(0, 8, hu.shape).astype(np.float32)   # mild noise
    return Volume(array=hu, spacing=spacing, origin=origin, direction=np.eye(3), meta={"phantom": "analytic"})


def make_from_meshes(skin: trimesh.Trimesh, bones: trimesh.Trimesh, spacing=(2.0, 2.0, 2.0),
                     margin_mm: float = 30.0) -> Volume:
    """Voxelise skin and skeleton meshes (LPS mm) into a HU volume."""
    from .meshing import mesh_to_mask

    lo = skin.bounds[0] - margin_mm
    hi = skin.bounds[1] + margin_mm
    shape_xyz = tuple(int(np.ceil((hi[i] - lo[i]) / spacing[i])) + 1 for i in range(3))
    vol = Volume(array=np.full(shape_xyz[::-1], HU_AIR, dtype=np.float32), spacing=tuple(spacing),
                 origin=tuple(lo), direction=np.eye(3), meta={"phantom": "mesh"})
    skin_m = mesh_to_mask(skin, vol)
    bone_m = mesh_to_mask(bones, vol) & skin_m
    vol.array[skin_m] = HU_SOFT
    vol.array[bone_m] = HU_BONE
    return vol


def make_from_skel(model, betas, poses, trans, spacing=(2.0, 2.0, 2.0)) -> tuple[Volume, dict]:
    """Voxelise a SKEL instance. Returns (volume, ground_truth dict)."""
    import torch
    from .frames import FrameTransform

    with torch.no_grad():
        out = model.forward(poses=torch.as_tensor(poses)[None].float(), betas=torch.as_tensor(betas)[None].float(),
                            trans=torch.as_tensor(trans)[None].float(), poses_type="skel", skelmesh=True)
    skin = trimesh.Trimesh(out.skin_verts[0].cpu().numpy(), model.skin_f.cpu().numpy(), process=False)
    skel = trimesh.Trimesh(out.skel_verts[0].cpu().numpy(), model.skel_f.cpu().numpy(), process=False)
    ft = FrameTransform(center_mm=np.zeros(3))
    skin.vertices = ft.skel_to_lps(skin.vertices)
    skel.vertices = ft.skel_to_lps(skel.vertices)
    vol = make_from_meshes(skin, skel, spacing=spacing)
    gt = {"betas": np.asarray(betas).tolist(), "poses": np.asarray(poses).tolist(), "trans": np.asarray(trans).tolist()}
    return vol, gt
