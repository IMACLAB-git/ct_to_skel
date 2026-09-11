"""Mask -> surface mesh (marching cubes), smoothing, decimation, mesh -> mask rasterisation."""
from __future__ import annotations

import numpy as np
import trimesh
from skimage import measure

from .dicom_io import Volume


def mask_to_mesh(mask: np.ndarray, vol: Volume, step: int = 1, smooth_iterations: int = 10,
                 target_faces: int | None = None, keep_largest: bool = False,
                 remove_border_caps: bool = True) -> trimesh.Trimesh:
    """Extract an iso-surface from a binary mask and return it in LPS millimetres.

    Structures cut by the scan border get a flat "cap" from the padding; with
    ``remove_border_caps`` those cap faces are dropped so that they cannot act as
    real anatomy (they are not) in the fit or in the exported STL.
    """
    if not mask.any():
        return trimesh.Trimesh()
    # pad so that the surface is closed at the volume border
    padded = np.pad(mask.astype(np.float32), 1, mode="constant")
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5, step_size=step, allow_degenerate=False)
    verts -= 1.0                                   # undo padding (still (z, y, x) index units)
    if remove_border_caps:
        # a cap face has all three vertices on the same border plane (index -0.5 or n-0.5)
        on_lo = verts <= -0.25
        on_hi = verts >= (np.array(mask.shape) - 0.75)
        cap = np.zeros(len(faces), dtype=bool)
        for ax in range(3):
            cap |= on_lo[faces, ax].all(axis=1) | on_hi[faces, ax].all(axis=1)
        faces = faces[~cap]
    verts_world = vol.zyx_to_world(verts)          # LPS mm
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()
    # marching_cubes on a positive-inside scalar field gives inward normals for
    # the (z,y,x)->world handedness used here; make normals point outward.
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()
    elif not mesh.is_watertight:
        # open mesh (caps removed): orient by comparing normals with the outward direction from the mask centroid
        c = vol.zyx_to_world(np.argwhere(mask).mean(axis=0, keepdims=True))[0]
        outward = ((mesh.triangles_center - c) * mesh.face_normals).sum(axis=1)
        if (outward < 0).mean() > 0.5:
            mesh.invert()
    if keep_largest and mesh.body_count > 1:
        parts = mesh.split(only_watertight=False)
        mesh = max(parts, key=lambda m: m.area)
    if smooth_iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iterations)
    if target_faces is not None and len(mesh.faces) > target_faces:
        mesh = decimate(mesh, target_faces)
    return mesh


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    try:
        import fast_simplification
    except ImportError:
        return mesh
    ratio = 1.0 - target_faces / max(len(mesh.faces), 1)
    if ratio <= 0:
        return mesh
    v, f = fast_simplification.simplify(np.asarray(mesh.vertices, dtype=np.float32),
                                        np.asarray(mesh.faces, dtype=np.int32),
                                        target_reduction=float(ratio))
    return trimesh.Trimesh(vertices=v, faces=f, process=True)


def sample_surface(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample ``n`` points (and face normals) on the mesh surface."""
    if len(mesh.faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts), np.asarray(mesh.face_normals[fid])


def mesh_to_mask(mesh: trimesh.Trimesh, vol: Volume) -> np.ndarray:
    """Rasterise a closed mesh (LPS mm) into a binary voxel mask on the volume grid.

    Uses per-slice plane intersection and polygon filling; robust for
    non-manifold marching-cubes-like meshes as long as sections are closed loops.
    """
    from skimage.draw import polygon as draw_polygon

    nz, ny, nx = vol.array.shape
    out = np.zeros((nz, ny, nx), dtype=bool)
    if len(mesh.faces) == 0:
        return out
    # slice planes along the world z axis (assumes axis-aligned direction matrix)
    z_world = vol.index_to_world(np.stack([np.zeros(nz), np.zeros(nz), np.arange(nz)], axis=1))[:, 2]
    origin = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 0.0, 1.0])
    # sections are 2D segments in the plane frame; ``to_3d`` maps them back to world
    sections, to_3d, _ = trimesh.intersections.mesh_multiplane(mesh, origin, normal, z_world)

    def fill(coords_2d: np.ndarray, k: int, value: bool) -> None:
        pts3 = trimesh.transform_points(np.column_stack([coords_2d, np.zeros(len(coords_2d))]), to_3d[k])
        ij = vol.world_to_index(pts3)
        rr, cc = draw_polygon(ij[:, 1], ij[:, 0], shape=(ny, nx))
        out[k, rr, cc] = value

    for k, seg in enumerate(sections):
        if seg is None or len(seg) == 0:
            continue
        path = trimesh.load_path(seg)
        try:
            polys = path.polygons_full
        except Exception:
            continue
        for poly in polys:
            if poly is None or poly.is_empty:
                continue
            fill(np.array(poly.exterior.coords), k, True)
            for hole in poly.interiors:
                fill(np.array(hole.coords), k, False)
    return out


def remove_vertices(mesh: trimesh.Trimesh, drop: np.ndarray, min_component_frac: float = 0.01) -> trimesh.Trimesh:
    """Remove every face touching a vertex flagged in ``drop`` and discard the small crumbs that are left
    behind (connected components with fewer than ``min_component_frac`` of the remaining faces)."""
    keep = ~np.asarray(drop, dtype=bool)[mesh.faces].any(axis=1)
    out = trimesh.Trimesh(mesh.vertices.copy(), mesh.faces[keep], process=False)
    out.remove_unreferenced_vertices()
    if len(out.faces) == 0:
        return out
    comps = out.split(only_watertight=False)
    big = [c for c in comps if len(c.faces) >= min_component_frac * len(out.faces)]
    if big and len(big) < len(comps):
        out = trimesh.util.concatenate(big) if len(big) > 1 else big[0]
    return out
