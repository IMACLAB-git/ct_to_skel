"""Patient-specific volumetric skinning weights computed on the CT itself (no statistical body model involved).

Every bone (labelled part or classified piece) is a heat source inside the body mask; diffusing the sources through
the soft tissue gives each voxel a smooth "which bone am I attached to" distribution (Dirichlet at the bones, Neumann
at the skin).  The patient's own CT skin is then skinned with those weights, so its deformation follows *this*
patient's bones and soft tissue - the SKEL skin topology and SKEL weights are no longer needed where the CT exists.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi


def _downsample(mask: np.ndarray, step: int) -> np.ndarray:
    if step == 1:
        return mask.astype(bool)
    z, y, x = mask.shape
    m = mask[: z - z % step, : y - y % step, : x - x % step].astype(np.uint8)
    m = m.reshape(z // step, step, y // step, step, x // step, step).max(axis=(1, 3, 5))
    return m.astype(bool)


def volumetric_weights(body: np.ndarray, sources: dict[int, np.ndarray], step: int = 2, iters: int = 400,
                       device: str | None = None, chunk: int = 8) -> tuple[np.ndarray, list[int]]:
    """Diffuse per-bone indicator fields through the body mask.

    ``body`` (Z,Y,X) bool at full resolution, ``sources`` part id -> bool mask (same grid).  Returns
    (W (K, Z', Y', X') float32 normalised per voxel, part ids) on the ``step``-downsampled grid.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    B = _downsample(body, step)
    ids = [k for k, m in sources.items() if m.any()]
    S = [_downsample(sources[k], step) & B for k in ids]
    for k, s in zip(ids, S):
        if not s.any():                                     # a source outside the body mask still has to exist
            S[ids.index(k)] = _downsample(sources[k], step)
    Bt = torch.from_numpy(B).to(dev, torch.float32)
    # number of in-body 6-neighbours per voxel (Neumann boundary at the skin)
    cnt = torch.zeros_like(Bt)
    for ax in (0, 1, 2):
        for sh in (1, -1):
            cnt += torch.roll(Bt, sh, dims=ax)
    cnt = cnt.clamp_min(1.0)
    any_src = torch.zeros_like(Bt, dtype=torch.bool)
    src_t = [torch.from_numpy(s).to(dev) for s in S]
    for s in src_t:
        any_src |= s
    W_out = np.zeros((len(ids),) + B.shape, dtype=np.float32)
    for c0 in range(0, len(ids), chunk):
        sl = slice(c0, min(c0 + chunk, len(ids)))
        W = torch.stack([s.float() for s in src_t[sl]])                  # (k, Z, Y, X)
        own = torch.stack(src_t[sl])
        for _ in range(iters):
            nb = torch.zeros_like(W)
            for ax in (1, 2, 3):
                for sh in (1, -1):
                    nb += torch.roll(W, sh, dims=ax)
            W = nb / cnt * Bt
            W = torch.where(own, torch.ones_like(W), W)                 # Dirichlet: 1 at the own bone
            W = torch.where(any_src[None] & ~own, torch.zeros_like(W), W)   # 0 at every other bone
        W_out[sl] = W.cpu().numpy()
        del W, nb, own
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    total = W_out.sum(0)
    reached = total > 1e-4
    W_out[:, reached] /= total[reached]
    # voxels the diffusion did not reach (thick soft tissue far from any bone): nearest source
    if (~reached & B).any():
        src_union = np.zeros(B.shape, dtype=bool)
        for s in S:
            src_union |= s
        _, idx = ndi.distance_transform_edt(~src_union, return_indices=True)
        lab = np.zeros(B.shape, dtype=np.int32)
        for k, s in enumerate(S):
            lab[s] = k
        near = lab[idx[0], idx[1], idx[2]]
        z, y, x = np.where(~reached & B)
        W_out[:, z, y, x] = 0.0
        W_out[near[z, y, x], z, y, x] = 1.0
    return W_out, ids


def sample_weights(W: np.ndarray, ids: list[int], idx_zyx: np.ndarray, step: int, top: int = 4):
    """Trilinear sample of the weight field at continuous full-resolution (z, y, x) indices -> (idx, val) top-k
    per point with SKEL part ids."""
    pts = np.asarray(idx_zyx, dtype=np.float64) / step
    out = np.zeros((len(pts), len(ids)), dtype=np.float32)
    for k in range(len(ids)):
        out[:, k] = ndi.map_coordinates(W[k], pts.T, order=1, mode="nearest")
    s = out.sum(1, keepdims=True)
    empty = s[:, 0] <= 1e-6
    if empty.any():                                          # outside the body mask: nearest voxel
        for k in range(len(ids)):
            out[empty, k] = ndi.map_coordinates(W[k], pts[empty].T, order=0, mode="nearest")
        s = out.sum(1, keepdims=True)
    out = out / np.maximum(s, 1e-9)
    order = np.argsort(-out, axis=1)[:, :top]
    val = np.take_along_axis(out, order, axis=1)
    val = val / np.maximum(val.sum(1, keepdims=True), 1e-9)
    idx = np.asarray(ids, dtype=np.int16)[order]
    return idx.astype(np.int16), val.astype(np.float32)


def rasterise_points(shape: tuple, idx_zyx: np.ndarray, radius_vox: int = 1) -> np.ndarray:
    """Mark the voxels around integer (z, y, x) indices; used to turn bone-piece surfaces into weight sources."""
    m = np.zeros(shape, dtype=bool)
    p = np.round(np.asarray(idx_zyx)).astype(int)
    ok = (p >= 0).all(1) & (p[:, 0] < shape[0]) & (p[:, 1] < shape[1]) & (p[:, 2] < shape[2])
    p = p[ok]
    m[p[:, 0], p[:, 1], p[:, 2]] = True
    if radius_vox > 0:
        m = ndi.binary_dilation(m, iterations=radius_vox)
    return m
