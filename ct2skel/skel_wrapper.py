"""Thin wrapper around the SKEL model (MarilynKeller/SKEL) plus a dummy stand-in for tests."""
from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh

from .labelmap import SKEL_PARTS

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "external" / "SKEL" / "data" / "skel"


def find_model_dir(model_dir: str | os.PathLike | None = None, gender: str = "male") -> Path | None:
    """Return a directory containing ``skel_<gender>.pkl`` or None."""
    candidates = []
    if model_dir:
        candidates.append(Path(model_dir))
    if os.environ.get("SKEL_MODELS"):
        candidates.append(Path(os.environ["SKEL_MODELS"]))
    candidates.append(DEFAULT_MODEL_DIR)
    for c in candidates:
        if (c / f"skel_{gender}.pkl").exists():
            return c
    return None


def load_skel(gender: str = "male", model_dir: str | os.PathLike | None = None, device: str = "cpu"):
    d = find_model_dir(model_dir, gender)
    if d is None:
        raise FileNotFoundError(
            f"skel_{gender}.pkl not found. Register at https://skel.is.tue.mpg.de/ , download the SKEL "
            f"models and place them in {DEFAULT_MODEL_DIR} (or set SKEL_MODELS / --skel-dir).")
    from skel.skel_model import SKEL  # noqa: WPS433  (external package)
    model = SKEL(gender=gender, model_path=str(d)).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------- pose presets
def init_pose(name: str, n_params: int = 46) -> np.ndarray:
    """Initial SKEL pose vector.

    ``arms_down``: arms along the body (typical CT with arms at the sides).
    ``arms_up``  : arms raised above the head (typical chest/abdomen CT).
    ``tpose``    : SKEL default.
    """
    q = np.zeros(n_params, dtype=np.float32)
    if name == "arms_down":
        q[29] = math.pi / 2       # shoulder_r_x
        q[39] = -math.pi / 2      # shoulder_l_x
    elif name == "arms_up":
        q[29] = -math.pi / 2
        q[39] = math.pi / 2
    elif name != "tpose":
        raise ValueError(f"unknown init pose {name}")
    return q


# ---------------------------------------------------------------------- part labels
def bone_part_labels(model) -> np.ndarray:
    """Per-skeleton-vertex rigid part id (0..23) from ``skel_weights_rigid``."""
    w = model.skel_weights_rigid
    w = w.to_dense() if w.is_sparse else w
    return w.argmax(dim=1).cpu().numpy()


def skin_part_labels(model) -> np.ndarray:
    """Per-skin-vertex dominant part id (0..23) from the skin skinning weights."""
    w = model.skin_weights
    w = w.to_dense() if w.is_sparse else w
    return w.argmax(dim=1).cpu().numpy()


def part_names(model) -> list[str]:
    names = getattr(model, "joints_name", None)
    if names is None or len(names) != 24:
        return list(SKEL_PARTS)
    return list(names)


def split_mesh_by_label(vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray,
                        names: list[str]) -> dict[str, trimesh.Trimesh]:
    """Split a mesh into sub-meshes by majority vertex label per face."""
    fl = labels[faces]                                 # (F, 3)
    # majority vote; ties -> first vertex label
    maj = np.where(fl[:, 1] == fl[:, 2], fl[:, 1], fl[:, 0])
    out = {}
    for pid in np.unique(maj):
        sub_faces = faces[maj == pid]
        m = trimesh.Trimesh(vertices=vertices, faces=sub_faces, process=False)
        m.remove_unreferenced_vertices()
        out[names[int(pid)]] = m
    return out


# ---------------------------------------------------------------------- dummy model (tests)
class DummySKEL(torch.nn.Module):
    """Interface-compatible stand-in for SKEL used in unit tests (no licensed data needed).

    Skin: ellipsoid whose size scales with betas[0] (height) and betas[1] (width).
    Skeleton: two smaller ellipsoids labelled 'pelvis' and 'thorax'.
    poses[:3] is a global axis-angle rotation; other pose entries are ignored.
    """

    def __init__(self):
        super().__init__()
        self.num_q_params, self.num_betas, self.num_joints = 46, 10, 24
        self.joints_name = list(SKEL_PARTS)
        skin = trimesh.creation.icosphere(subdivisions=3)
        self.register_buffer("skin_template_v", torch.tensor(skin.vertices, dtype=torch.float32))
        self.register_buffer("skin_f", torch.tensor(skin.faces, dtype=torch.long))
        a = trimesh.creation.icosphere(subdivisions=2)
        b = trimesh.creation.icosphere(subdivisions=2)
        av = a.vertices * np.array([0.10, 0.12, 0.08]) + np.array([0, -0.25, 0])
        bv = b.vertices * np.array([0.12, 0.25, 0.08]) + np.array([0, 0.25, 0])
        v = np.vstack([av, bv])
        f = np.vstack([a.faces, b.faces + len(av)])
        self.register_buffer("skel_template_v", torch.tensor(v, dtype=torch.float32))
        self.register_buffer("skel_f", torch.tensor(f, dtype=torch.long))
        labels = np.zeros((len(v), 24), dtype=np.float32)
        labels[: len(av), 0] = 1.0        # pelvis
        labels[len(av):, 12] = 1.0        # thorax
        self.register_buffer("skel_weights_rigid", torch.tensor(labels))
        self.register_buffer("skel_weights", torch.tensor(labels))         # skeleton skinning (rigid in the dummy)
        sw = np.zeros((len(skin.vertices), 24), dtype=np.float32); sw[:, 12] = 1.0
        self.register_buffer("skin_weights", torch.tensor(sw))
        joints = np.zeros((24, 3), dtype=np.float32)
        joints[:, 1] = np.linspace(-0.6, 0.6, 24)
        self.register_buffer("joints_template", torch.tensor(joints))

    @staticmethod
    def _rodrigues(aa: torch.Tensor) -> torch.Tensor:
        theta = torch.linalg.norm(aa, dim=-1, keepdim=True).clamp_min(1e-8)
        k = aa / theta
        K = torch.zeros(aa.shape[0], 3, 3, device=aa.device, dtype=aa.dtype)
        K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
        K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
        I = torch.eye(3, device=aa.device, dtype=aa.dtype)[None]
        s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
        return I + s * K + (1 - c) * (K @ K)

    def forward(self, poses, betas, trans, poses_type="skel", skelmesh=True, **kw):
        B = poses.shape[0]
        scale = torch.stack([1.0 + 0.10 * betas[:, 1], 1.0 + 0.15 * betas[:, 0], 1.0 + 0.10 * betas[:, 1]], dim=-1)
        base = torch.tensor([0.20, 0.85, 0.12], device=poses.device)
        R = self._rodrigues(poses[:, :3])
        skin = self.skin_template_v[None] * (base * scale)[:, None, :]
        skin = skin @ R.transpose(1, 2) + trans[:, None, :]
        skel = self.skel_template_v[None] * scale[:, None, :]
        skel = skel @ R.transpose(1, 2) + trans[:, None, :]
        joints = self.joints_template[None] * scale[:, None, :]
        joints = joints @ R.transpose(1, 2) + trans[:, None, :]
        return SimpleNamespace(skin_verts=skin, skel_verts=skel if skelmesh else None, joints=joints,
                               joints_ori=R[:, None].expand(B, 24, 3, 3), skin_f=self.skin_f, skel_f=self.skel_f)
