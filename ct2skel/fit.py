"""Fit SKEL (pose q, shape beta, translation) to CT-derived targets.

Targets (all in the SKEL frame, metres):
  * skin surface points sampled from the CT body iso-surface
  * bone surface points sampled from the CT bone iso-surface
  * optional anatomical joint centres with per-joint weights

Optimisation runs in stages (rigid -> shape+torso -> full pose) with Adam and
chamfer-style nearest-neighbour losses computed with chunked ``torch.cdist``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .labelmap import SKEL_PARTS

try:
    from skel.kin_skel import pose_limits as SKEL_POSE_LIMITS, pose_param_names as SKEL_POSE_NAMES
except Exception:  # pragma: no cover - SKEL not installed
    SKEL_POSE_LIMITS, SKEL_POSE_NAMES = {}, [f"q{i}" for i in range(46)]

DATA_SCALE = 1e4                              # data terms in cm^2 so that regularisers (rad^2, beta^2) are comparable

# pose DOF index -> SKEL joint (part) index whose bone it moves (from skel.kin_skel.pose_param_names)
_DOF_PREFIX_TO_PART = {
    "hip_": "femur", "knee_": "tibia", "ankle_": "talus", "subtalar_": "calcn", "mtp_": "toes",
    "lumbar_": "lumbar_body", "thorax_": "thorax", "head_": "head", "scapula_": "scapula",
    "shoulder_": "humerus", "elbow_": "ulna", "pro_sup_": "radius", "wrist_": "hand",
}


def dof_to_part() -> list[int]:
    """For each of the 46 pose DOFs, the SKEL part index it articulates (-1 for the 3 global DOFs)."""
    out = []
    for i, name in enumerate(SKEL_POSE_NAMES):
        if i < 3:
            out.append(-1)
            continue
        part = None
        for prefix, base in _DOF_PREFIX_TO_PART.items():
            if name.startswith(prefix):
                side = "_r" if name.endswith("_r") or "_r_" in name else ("_l" if name.endswith("_l") or "_l_" in name else "")
                part = base + side if base not in ("lumbar_body", "thorax", "head") else base
                break
        out.append(SKEL_PARTS.index(part) if part in SKEL_PARTS else -1)
    return out
SPINE_HEAD_IDX = list(range(17, 26))          # lumbar / thorax / head DOFs
SCAPULA_IDX = [26, 27, 28, 36, 37, 38]


@dataclass
class FitTargets:
    skin_pts: np.ndarray                          # (N, 3) m
    bone_pts: np.ndarray | None = None            # (M, 3) m
    joints: np.ndarray | None = None              # (24, 3) m
    joint_w: np.ndarray | None = None             # (24,)
    joint_rot: np.ndarray | None = None           # (24, 3, 3) target part orientations (from the bone ICP)
    joint_rot_w: np.ndarray | None = None         # (24,)
    bbox: np.ndarray | None = None                # (2, 3) m, CT coverage in SKEL frame
    limb_lengths: dict | None = None              # {(part_a, part_b): joint-to-joint length in m} anthropometric targets
    joint_rot_axis_only: np.ndarray | None = None  # (24,) bool: constrain only the bone axis (twist not observed by the ICP)

    def __post_init__(self):
        if self.bbox is None:
            self.bbox = np.stack([self.skin_pts.min(0), self.skin_pts.max(0)])


@dataclass
class Stage:
    name: str
    iters: int
    lr: float
    opt_trans: bool = True
    opt_rot: bool = True
    opt_betas: bool = False
    pose_idx: list = field(default_factory=list)      # pose DOFs (>=3) that are free
    w_skin: float = 1.0
    w_skin_rev: float = 0.5
    w_bone: float = 1.0
    w_bone_rev: float = 0.25
    w_joint: float = 50.0
    w_orient: float = 20.0           # part orientation targets (Frobenius^2 of the rotation difference)
    w_limb: float = 50.0             # limb-length targets (joint-to-joint distances, m^2 scaled like the joint term)
    w_pose_reg: float = 1e-2
    w_betas_reg: float = 1e-2
    w_limits: float = 10.0
    w_scapula: float = 1e-2
    trunc_m: float | None = None      # truncate surface distances at this value (m): robust to cut/missing anatomy
    opt_scale: bool = False           # optimise the global body scale (about the pelvis joint)


ALL_POSE_IDX = list(range(3, 46))

# long bones whose ICP orientation only fixes the axis (part -> child joint): femur, tibia, humerus, ulna, radius
LONG_BONE_CHILD = {SKEL_PARTS.index(a): SKEL_PARTS.index(b) for a, b in (
    ("femur_r", "tibia_r"), ("tibia_r", "talus_r"), ("femur_l", "tibia_l"), ("tibia_l", "talus_l"),
    ("humerus_r", "ulna_r"), ("ulna_r", "hand_r"), ("radius_r", "hand_r"),
    ("humerus_l", "ulna_l"), ("ulna_l", "hand_l"), ("radius_l", "hand_l"))}
LONG_BONE_MASK = torch.tensor([i in LONG_BONE_CHILD for i in range(24)])


def supine_prior_weights(n: int = 46) -> torch.Tensor:
    """Per-DOF weights for the pose prior (deviation from the initial pose).

    CT patients lie straight on the table: hips/knees/ankles rarely deviate much and are
    weakly observed when the legs leave the field of view, so they get a strong prior;
    spine and shoulders (arms up / down / crossed) are left comparatively free.
    """
    w = torch.full((n,), 1.0)
    for i, name in enumerate(SKEL_POSE_NAMES):
        if name.startswith(("hip_", "knee_", "ankle_", "subtalar_", "mtp_")):
            w[i] = 1000.0        # legs of a supine patient are straight unless the CT shows otherwise
        elif name.startswith(("lumbar_", "thorax_", "head_")):
            w[i] = 5.0
        elif name.startswith(("elbow_", "pro_sup_", "wrist_")):
            w[i] = 1000.0        # forearm/hand: like the legs, straight unless explicitly fitted (--fit-forearms)
        elif name.startswith(("scapula_", "shoulder_")):
            w[i] = 1.0
    return w


def default_stages(iters_scale: float = 1.0) -> list[Stage]:
    """rigid -> joints-only pose (fast limb alignment when landmarks exist) -> shape + pose -> refinement.

    DOFs of bones outside the CT coverage are removed from every stage by the fitter.
    """
    s = lambda n: max(int(n * iters_scale), 1)
    return [
        Stage("rigid", s(150), 0.02, opt_betas=False, pose_idx=[], w_skin_rev=0.0, w_bone_rev=0.0,
              w_pose_reg=0.0, w_betas_reg=0.0),
        Stage("joints", s(150), 0.05, opt_betas=False, pose_idx=ALL_POSE_IDX, w_skin=0.0, w_skin_rev=0.0,
              w_bone=0.0, w_bone_rev=0.0, w_joint=50.0, w_pose_reg=1e-3, w_betas_reg=0.0, w_limits=10.0),
        # after the joints stage the landmarks (5-15 mm heuristics) only guide; surfaces decide the shape
        Stage("shape+pose", s(250), 0.02, opt_betas=True, opt_scale=True, pose_idx=ALL_POSE_IDX, trunc_m=0.15, w_joint=5.0),
        Stage("refine", s(400), 0.006, opt_betas=True, opt_scale=True, pose_idx=ALL_POSE_IDX, trunc_m=0.08, w_joint=5.0),
    ]


@dataclass
class FitConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    init_pose: str = "arms_down"
    n_skin: int = 20000              # CT skin target points
    n_bone: int = 20000              # CT bone target points
    n_skin_model: int = 30000        # surface samples drawn on the SKEL skin each iteration
    n_bone_model: int = 60000        # surface samples drawn on the SKEL skeleton each iteration
    n_skel_sub: int = 60000          # skeleton faces eligible for sampling (memory bound)
    chunk: int = 1024
    bbox_margin: float = 0.01        # m; SKEL vertices outside CT coverage are ignored in reverse terms
    freeze_unsupported: bool = True  # freeze pose DOFs of joints whose bone lies outside the CT coverage
    betas_clip: float = 3.0          # |beta| bound (SMPL/SKEL shape space is trained on roughly +-3)
    betas_reg_w: tuple = (1.0,) * 10     # per-component multiplier of the beta L2 penalty (higher components = less anatomical)
    scale_range: tuple = (0.8, 1.25)  # bounds of the global scale
    w_scale_reg: float = 1.0         # (scale-1)^2 penalty
    min_support: float = 0.03        # fraction of a part's skeleton vertices inside the CT bbox to count as supported
    support_margin: float = 0.08     # m; bbox margin for the initial DOF-support test (model not yet placed)
    weak_prior_parts: list | None = None   # part ids whose DOFs get a weak pose prior (well covered by the CT)
    estimated_parts: list | None = None    # part ids treated as unobserved (e.g. arms): DOFs frozen at the init pose, no CT terms
    weak_prior_min_frac: float = 0.6       # ... otherwise: parts with at least this fraction of skeleton vertices inside the CT
    clamp_limits: bool = True              # project the pose onto the SKEL joint limits after every step
    stages: list = field(default_factory=default_stages)
    seed: int = 0
    verbose: bool = True


# ---------------------------------------------------------------------- distances
def nn_dist(a: torch.Tensor, b: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Distance from every point of ``a`` (N,3) to its nearest neighbour in ``b`` (M,3).

    The nearest-neighbour search runs without autograd (chunked cdist, argmin only),
    then the distance of the matched pairs is recomputed with gradients.  This gives
    the same gradient as differentiating ``cdist(...).min()`` but never stores the
    N x M matrix, so N and M can be tens of thousands on an 8 GB GPU.
    """
    if a.shape[0] == 0 or b.shape[0] == 0:
        return a.new_zeros(0)
    with torch.no_grad():
        idx = torch.cat([torch.cdist(a[i:i + chunk], b).argmin(dim=1) for i in range(0, a.shape[0], chunk)])
    return torch.linalg.norm(a - b[idx], dim=-1)


class FaceSampler:
    """Differentiable uniform surface sampling on a triangle mesh with fixed topology."""

    def __init__(self, template_v: torch.Tensor, faces: torch.Tensor, n: int, seed: int = 0,
                 face_subset: torch.Tensor | None = None, vertex_part: torch.Tensor | None = None):
        self.faces = faces if face_subset is None else faces[face_subset]
        self.face_part = vertex_part[self.faces[:, 0]] if vertex_part is not None else None
        self.part = None
        v = template_v[self.faces]                                        # (F, 3, 3)
        area = torch.linalg.norm(torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1), dim=-1)
        self.prob = (area / area.sum()).cpu()
        self.n = n
        self.gen = torch.Generator().manual_seed(seed)
        self.resample()

    def resample(self):
        fid = torch.multinomial(self.prob, self.n, replacement=True, generator=self.gen)
        r = torch.rand(self.n, 2, generator=self.gen)
        s = torch.sqrt(r[:, 0:1])
        self.bary = torch.cat([1 - s, s * (1 - r[:, 1:2]), s * r[:, 1:2]], dim=1)   # (n, 3)
        self.fid = fid
        if self.face_part is not None:
            self.part = self.face_part[fid.to(self.face_part.device)]

    def __call__(self, verts: torch.Tensor) -> torch.Tensor:
        tri = verts[self.faces[self.fid.to(verts.device)]]                # (n, 3, 3)
        return (tri * self.bary.to(verts.device)[:, :, None]).sum(dim=1)


def sq(d: torch.Tensor, trunc: float | None) -> torch.Tensor:
    """Squared distance, truncated at ``trunc`` metres when given (zero gradient beyond it)."""
    return d.pow(2) if trunc is None else d.clamp(max=trunc).pow(2)


def inside_bbox(pts: torch.Tensor, bbox: torch.Tensor, margin: float) -> torch.Tensor:
    lo, hi = bbox[0] - margin, bbox[1] + margin
    return ((pts >= lo) & (pts <= hi)).all(dim=-1)


def limits_penalty(poses: torch.Tensor) -> torch.Tensor:
    if not SKEL_POSE_LIMITS:
        return poses.new_zeros(())
    pen = poses.new_zeros(())
    for name, (lo, hi) in SKEL_POSE_LIMITS.items():
        if name not in SKEL_POSE_NAMES:
            continue
        i = SKEL_POSE_NAMES.index(name)
        lo_, hi_ = (min(lo, hi), max(lo, hi))
        q = poses[:, i]
        pen = pen + torch.relu(lo_ - q).pow(2).sum() + torch.relu(q - hi_).pow(2).sum()
    return pen


def skel_forward(model, poses, betas, trans, skelmesh=True, scale=None):
    """SKEL forward pass with an optional global scale about the pelvis joint."""
    out = model.forward(poses=poses, betas=betas, trans=trans, poses_type="skel", skelmesh=skelmesh)
    if scale is not None:
        c = out.joints[:, 0:1, :]
        sc = scale.view(-1, 1, 1)
        out.skin_verts = c + sc * (out.skin_verts - c)
        out.joints = c + sc * (out.joints - c)
        if out.skel_verts is not None:
            out.skel_verts = c + sc * (out.skel_verts - c)
    return out


# ---------------------------------------------------------------------- fitter
class SkelCTFitter:
    def __init__(self, model, cfg: FitConfig | None = None):
        self.cfg = cfg or FitConfig()
        self.dev = torch.device(self.cfg.device)
        self.model = model.to(self.dev)
        g = torch.Generator().manual_seed(self.cfg.seed)
        skin_f, skel_f = model.skin_f.to(self.dev), model.skel_f.to(self.dev)
        from .skel_wrapper import bone_part_labels, skin_part_labels
        self.skel_part = torch.as_tensor(bone_part_labels(model), device=self.dev)
        self.skin_part = torch.as_tensor(skin_part_labels(model), device=self.dev)
        self.skin_sampler = FaceSampler(model.skin_template_v.to(self.dev), skin_f, self.cfg.n_skin_model, self.cfg.seed,
                                        vertex_part=self.skin_part)
        nf = int(skel_f.shape[0])
        subset = torch.randperm(nf, generator=g)[: self.cfg.n_skel_sub].to(self.dev) if nf > self.cfg.n_skel_sub else None
        self.skel_sampler = FaceSampler(model.skel_template_v.to(self.dev), skel_f, self.cfg.n_bone_model, self.cfg.seed + 1,
                                        face_subset=subset, vertex_part=self.skel_part)
        self.part_support = torch.ones(24, dtype=torch.bool, device=self.dev)
        self.part_estimated = torch.zeros(24, dtype=torch.bool, device=self.dev)
        self._betas_reg_w = torch.as_tensor(list(self.cfg.betas_reg_w) + [1.0] * 20, dtype=torch.float32, device=self.dev)
        self._limit_table = [(SKEL_POSE_NAMES.index(n), min(lo, hi), max(lo, hi)) for n, (lo, hi) in SKEL_POSE_LIMITS.items()
                             if n in SKEL_POSE_NAMES]
        self._limit_table = [(i, lo, hi) for i, lo, hi in self._limit_table]

    # ------------------------------------------------------------------ helpers
    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.dev)

    def forward(self, poses, betas, trans, skelmesh=True, scale=None):
        return skel_forward(self.model, poses, betas, trans, skelmesh=skelmesh, scale=scale)

    def _subsample(self, pts: np.ndarray, n: int) -> torch.Tensor:
        if pts is None or len(pts) == 0:
            return None
        if len(pts) > n:
            rng = np.random.default_rng(self.cfg.seed)
            pts = pts[rng.choice(len(pts), n, replace=False)]
        return self._t(pts)

    # ------------------------------------------------------------------ init
    @torch.no_grad()
    def initialise(self, targets: FitTargets, poses, betas, trans):
        """Translate SKEL so its skin matches the CT skin; 1-D search along the body axis."""
        out = self.forward(poses, betas, trans, skelmesh=False)
        skin = out.skin_verts[0]
        ct = self._t(targets.skin_pts)
        j_ct = self._t(targets.joints) if targets.joints is not None else None
        w = self._t(targets.joint_w) if targets.joint_w is not None else None
        if j_ct is not None and w is not None and (w > 0).sum() >= 2:
            sel = w > 0
            delta = ((j_ct[sel] - out.joints[0][sel]) * w[sel, None]).sum(0) / w[sel].sum()
            trans += delta[None]
            return trans
        # x/z: centroid of the CT skin vs SKEL skin restricted to the CT y-range after each y shift
        ct_c = ct.mean(0)
        best, best_d = None, float("inf")
        y_full = (targets.bbox[1, 1] - targets.bbox[0, 1]) > 1.4
        cands = [0.0] if y_full else np.arange(-0.7, 0.71, 0.05)
        for dy in cands:
            t = torch.zeros(3, device=self.dev)
            t[1] = dy
            v = skin + t
            m = inside_bbox(v, self._t(targets.bbox), 0.02)
            if m.sum() < 50:
                continue
            t2 = t.clone()
            c = v[m].mean(0)
            t2[0] += ct_c[0] - c[0]
            t2[2] += ct_c[2] - c[2]
            v = skin + t2
            m = inside_bbox(v, self._t(targets.bbox), 0.02)
            d = nn_dist(ct[::4], v[m], self.cfg.chunk).mean() + nn_dist(v[m][::2], ct, self.cfg.chunk).mean()
            if d < best_d:
                best_d, best = float(d), t2.clone()
        if best is not None:
            trans += best[None]
        return trans

    @torch.no_grad()
    def _dof_support(self, poses, betas, trans, targets: FitTargets, margin: float | None = None,
                     scale=None) -> torch.Tensor:
        """1 for DOFs whose articulated bone has skeleton vertices inside the CT bbox, else 0.

        ``margin`` (m) widens the bbox test: before the model is placed, limbs may still stick out of the scan by a
        few centimetres although the CT contains them; support is re-evaluated after every stage."""
        from .skel_wrapper import bone_part_labels
        with torch.no_grad():
            out = self.forward(poses.detach(), betas.detach(), trans.detach(), skelmesh=True, scale=scale)
        labels = torch.as_tensor(bone_part_labels(self.model), device=self.dev)
        inside = inside_bbox(out.skel_verts[0], self._t(targets.bbox), self.cfg.bbox_margin if margin is None else margin)
        support = torch.ones(self.model.num_q_params, device=self.dev)
        self.part_frac = torch.zeros(24, device=self.dev)
        for part in range(24):
            sel = labels == part
            frac = inside[sel].float().mean() if sel.any() else torch.tensor(0.0)
            self.part_frac[part] = float(frac)
            self.part_support[part] = bool(float(frac) >= self.cfg.min_support)
        for part in (self.cfg.estimated_parts or []):
            self.part_support[part] = False                       # estimated: kept at the init pose, no chamfer terms
            self.part_estimated[part] = True
        for i, part in enumerate(dof_to_part()):
            if part >= 0 and not self.part_support[part]:
                support[i] = 0.0
        return support

    # ------------------------------------------------------------------ main loop
    def fit(self, targets: FitTargets, init_pose: np.ndarray | None = None,
            init_betas: np.ndarray | None = None, init_trans: np.ndarray | None = None) -> dict:
        cfg = self.cfg
        from .skel_wrapper import init_pose as preset
        q0 = preset(cfg.init_pose, self.model.num_q_params) if init_pose is None else np.asarray(init_pose)
        poses = self._t(q0)[None].clone()
        betas = torch.zeros(1, self.model.num_betas, device=self.dev) if init_betas is None else self._t(init_betas)[None].clone()
        trans = torch.zeros(1, 3, device=self.dev) if init_trans is None else self._t(init_trans)[None].clone()
        trans = self.initialise(targets, poses, betas, trans)
        poses_ref = poses.clone()
        scale = torch.ones(1, device=self.dev)

        skin_t = self._subsample(targets.skin_pts, cfg.n_skin)
        bone_t = self._subsample(targets.bone_pts, cfg.n_bone) if targets.bone_pts is not None else None
        j_ct = self._t(targets.joints) if targets.joints is not None else None
        j_w = self._t(targets.joint_w) if targets.joint_w is not None else None
        bbox = self._t(targets.bbox)
        use_joints = j_ct is not None and j_w is not None and float(j_w.sum()) > 0
        R_t = self._t(targets.joint_rot) if targets.joint_rot is not None else None
        R_w = self._t(targets.joint_rot_w) if targets.joint_rot_w is not None else None
        use_orient = R_t is not None and R_w is not None and float(R_w.sum()) > 0
        # long bones whose ICP could not discriminate the twist (multi-start margin small) constrain only their axis
        if targets.joint_rot_axis_only is not None:
            axis_only_mask = torch.as_tensor(np.asarray(targets.joint_rot_axis_only, dtype=bool), device=self.dev) & LONG_BONE_MASK.to(self.dev)
        else:
            axis_only_mask = torch.zeros(24, dtype=torch.bool, device=self.dev)

        supported = torch.ones(self.model.num_q_params, device=self.dev)
        if cfg.freeze_unsupported:
            supported = self._dof_support(poses, betas, trans, targets, margin=cfg.support_margin)
            frozen = [SKEL_POSE_NAMES[i] for i in range(self.model.num_q_params) if supported[i] == 0]
            if cfg.verbose and frozen:
                print(f"[fit] freezing {len(frozen)} pose DOFs without CT support: {frozen}", flush=True)
        prior_w = supine_prior_weights(self.model.num_q_params).to(self.dev)
        # the supine prior only has to hold where the CT gives no (or too little) data
        if cfg.weak_prior_parts is not None:
            weak = torch.tensor([p in set(cfg.weak_prior_parts) for p in dof_to_part()], device=self.dev)
        elif hasattr(self, "part_frac"):
            # a bone mostly outside the scan (proximal femur only) keeps the supine prior: its joint centre is
            # observed but its rotation is not
            weak = torch.tensor([p >= 0 and float(self.part_frac[p]) >= cfg.weak_prior_min_frac for p in dof_to_part()], device=self.dev)
        else:
            weak = supported > 0
        prior_w = torch.where(weak, torch.ones_like(prior_w), prior_w)
        poses.requires_grad_(True)
        betas.requires_grad_(True)
        trans.requires_grad_(True)
        scale.requires_grad_(True)
        history = []
        t0 = time.time()
        for k_stage, stage in enumerate(cfg.stages):
            if stage.w_skin == 0 and stage.w_bone == 0 and not use_joints and not use_orient and not (targets.limb_lengths and stage.w_limb > 0):
                continue                                   # joints-only stage without joint targets
            if cfg.freeze_unsupported and k_stage > 0:
                # the model has been placed by now: re-evaluate which bones the scan actually contains
                new_sup = self._dof_support(poses, betas, trans, targets, margin=cfg.bbox_margin, scale=scale.detach())
                if cfg.verbose and bool((new_sup != supported).any()):
                    changed = [SKEL_POSE_NAMES[i] + ("+" if new_sup[i] > supported[i] else "-") for i in range(len(new_sup)) if new_sup[i] != supported[i]]
                    print(f"[fit] DOF support re-evaluated before '{stage.name}': {changed}", flush=True)
                supported = new_sup
            pose_mask = torch.zeros(self.model.num_q_params, device=self.dev)
            if stage.opt_rot:
                pose_mask[:3] = 1
            if stage.pose_idx:
                pose_mask[list(stage.pose_idx)] = 1
            pose_mask = pose_mask * supported
            pose_mask[:3] = 1.0 if stage.opt_rot else 0.0
            params = []
            if stage.opt_trans:
                params.append({"params": [trans], "lr": stage.lr})
            if stage.opt_rot or stage.pose_idx:
                params.append({"params": [poses], "lr": stage.lr})
            if stage.opt_betas:
                params.append({"params": [betas], "lr": stage.lr})
            if stage.opt_scale:
                params.append({"params": [scale], "lr": stage.lr * 0.25})
            opt = torch.optim.Adam(params)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stage.iters, eta_min=stage.lr * 0.1)
            need_skel = bone_t is not None and (stage.w_bone > 0 or stage.w_bone_rev > 0)
            for it in range(stage.iters):
                opt.zero_grad(set_to_none=True)
                q = poses * pose_mask + poses.detach() * (1 - pose_mask)
                b = betas if stage.opt_betas else betas.detach()
                sc = scale if stage.opt_scale else scale.detach()
                out = self.forward(q, b, trans, skelmesh=need_skel, scale=sc)
                if it % 10 == 0:
                    self.skin_sampler.resample()
                    self.skel_sampler.resample()
                skin = self.skin_sampler(out.skin_verts[0])           # surface samples (differentiable)
                losses = {}
                if stage.w_skin > 0:
                    mf = ~self.part_estimated[self.skin_sampler.part]  # estimated parts never attract CT points
                    losses["skin"] = stage.w_skin * DATA_SCALE * sq(nn_dist(skin_t, skin[mf], cfg.chunk), stage.trunc_m).mean()
                if stage.w_skin_rev > 0:
                    m = inside_bbox(skin.detach(), bbox, cfg.bbox_margin) & self.part_support[self.skin_sampler.part]
                    if m.sum() > 10:
                        losses["skin_rev"] = stage.w_skin_rev * DATA_SCALE * sq(nn_dist(skin[m], skin_t, cfg.chunk), stage.trunc_m).mean()
                if need_skel:
                    skel = self.skel_sampler(out.skel_verts[0])
                    if stage.w_bone > 0:
                        mf = ~self.part_estimated[self.skel_sampler.part]
                        losses["bone"] = stage.w_bone * DATA_SCALE * sq(nn_dist(bone_t, skel[mf], cfg.chunk), stage.trunc_m).mean()
                    if stage.w_bone_rev > 0:
                        m = inside_bbox(skel.detach(), bbox, cfg.bbox_margin) & self.part_support[self.skel_sampler.part]
                        if m.sum() > 10:
                            losses["bone_rev"] = stage.w_bone_rev * DATA_SCALE * sq(nn_dist(skel[m], bone_t, cfg.chunk), stage.trunc_m).mean()
                if targets.limb_lengths and stage.w_limb > 0 and (stage.opt_betas or stage.opt_scale):
                    lim = torch.zeros((), device=self.dev)
                    for (pa, pb), L in targets.limb_lengths.items():
                        d = (out.joints[0, SKEL_PARTS.index(pa)] - out.joints[0, SKEL_PARTS.index(pb)]).norm()
                        lim = lim + (d - float(L)).pow(2)
                    losses["limb"] = stage.w_limb * DATA_SCALE * lim / len(targets.limb_lengths)
                if use_joints and stage.w_joint > 0:
                    losses["joint"] = stage.w_joint * DATA_SCALE * ((out.joints[0] - j_ct).pow(2).sum(-1) * j_w).sum() / j_w.sum()
                if use_orient and stage.w_orient > 0 and (stage.opt_rot or stage.pose_idx):
                    # long bones are nearly cylindrical: an ICP transform observes their axis, not their twist.
                    # Their target therefore constrains only the axis direction (joint -> child joint, expressed in
                    # the part frame); the twist comes from the pose prior and from the extremities' full targets.
                    R_fit = out.joints_ori[0]
                    full = (R_fit - R_t).pow(2).sum(dim=(1, 2))
                    J = out.joints[0].detach()
                    ax = torch.zeros_like(J)
                    for pi, ci in LONG_BONE_CHILD.items():
                        d = J[ci] - J[pi]
                        ax[pi] = R_fit[pi].detach().transpose(0, 1) @ (d / d.norm().clamp_min(1e-6))
                    axis = ((R_fit @ ax[:, :, None]) - (R_t @ ax[:, :, None])).squeeze(-1).pow(2).sum(dim=1) * 3.0
                    per_part = torch.where(axis_only_mask, axis, full)
                    losses["orient"] = stage.w_orient * (per_part * R_w).sum() / R_w.sum()
                if stage.w_pose_reg > 0 and stage.pose_idx:
                    losses["pose_reg"] = stage.w_pose_reg * (prior_w[3:] * (q[:, 3:] - poses_ref[:, 3:]).pow(2)).sum()
                if stage.w_betas_reg > 0 and stage.opt_betas:
                    losses["betas_reg"] = stage.w_betas_reg * (b.pow(2) * self._betas_reg_w[: b.shape[1]]).sum()
                if stage.opt_scale and cfg.w_scale_reg > 0:
                    losses["scale_reg"] = cfg.w_scale_reg * (sc - 1.0).pow(2).sum()
                if stage.w_limits > 0 and stage.pose_idx:
                    losses["limits"] = stage.w_limits * limits_penalty(q)
                if stage.w_scapula > 0 and stage.pose_idx:
                    losses["scapula"] = stage.w_scapula * q[:, SCAPULA_IDX].pow(2).sum()
                loss = sum(losses.values())
                loss.backward()
                opt.step()
                sched.step()
                with torch.no_grad():
                    betas.clamp_(-cfg.betas_clip, cfg.betas_clip)
                    scale.clamp_(*cfg.scale_range)
                    if cfg.clamp_limits:
                        for i, lo_, hi_ in self._limit_table:
                            poses[:, i].clamp_(lo_, hi_)
                    poses[:, 3:] = torch.atan2(poses[:, 3:].sin(), poses[:, 3:].cos())
                if cfg.verbose and (it % 50 == 0 or it == stage.iters - 1):
                    msg = " ".join(f"{k}={v.item():.5f}" for k, v in losses.items())
                    print(f"[fit:{stage.name}] it {it:4d} loss={loss.item():.5f} {msg}  ({time.time() - t0:.0f}s)", flush=True)
                history.append({"stage": stage.name, "it": it, "loss": loss.item(), **{k: v.item() for k, v in losses.items()}})
            with torch.no_grad():
                poses[:] = poses * pose_mask + poses.detach() * (1 - pose_mask)

        with torch.no_grad():
            out = self.forward(poses, betas, trans, skelmesh=True, scale=scale)
        return {
            "scale": float(scale.detach().cpu()[0]),
            "dof_support": [bool(v) for v in supported.detach().cpu().numpy()],
            "poses": poses.detach().cpu().numpy()[0],
            "betas": betas.detach().cpu().numpy()[0],
            "trans": trans.detach().cpu().numpy()[0],
            "skin_verts": out.skin_verts[0].cpu().numpy(),
            "skel_verts": out.skel_verts[0].cpu().numpy(),
            "skin_faces": self.model.skin_f.cpu().numpy(),
            "skel_faces": self.model.skel_f.cpu().numpy(),
            "joints": out.joints[0].cpu().numpy(),
            "joints_ori": out.joints_ori[0].cpu().numpy(),
            "history": history,
            "seconds": time.time() - t0,
        }


def surface_points(vertices: np.ndarray, faces: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Uniform surface samples of a mesh (numpy), for point-to-surface distance evaluation."""
    import trimesh
    m = trimesh.Trimesh(vertices, faces, process=False)
    pts, _ = trimesh.sample.sample_surface(m, n, seed=seed)
    return np.asarray(pts)


# ---------------------------------------------------------------------- metrics
def _stats(d: np.ndarray) -> dict:
    if d is None or len(d) == 0:
        return {}
    d = np.asarray(d) * 1000.0
    return {"mean_mm": float(d.mean()), "rms_mm": float(np.sqrt((d ** 2).mean())),
            "p95_mm": float(np.percentile(d, 95)), "max_mm": float(d.max()), "n": int(len(d))}


def evaluate(result: dict, targets: FitTargets, bbox_margin: float = 0.01,
             n_surface: int = 300000) -> dict:
    """Point-to-surface distances (mm) between CT targets and the fitted SKEL surfaces.

    SKEL surfaces are densely sampled (``n_surface`` points) so that distances are
    not biased by the coarse SKEL skin topology.
    """
    from scipy.spatial import cKDTree

    m = {}
    skel_skin_pts = surface_points(result["skin_verts"], result["skin_faces"], n_surface)
    d, _ = cKDTree(skel_skin_pts).query(targets.skin_pts)
    m["skin_ct_to_skel"] = _stats(d)
    ct_tree = cKDTree(targets.skin_pts)
    lo, hi = targets.bbox[0] - bbox_margin, targets.bbox[1] + bbox_margin
    inside = ((skel_skin_pts >= lo) & (skel_skin_pts <= hi)).all(1)
    if inside.any():
        d, _ = ct_tree.query(skel_skin_pts[inside][::10])
        m["skin_skel_to_ct"] = _stats(d)
    if targets.bone_pts is not None and len(targets.bone_pts):
        skel_bone_pts = surface_points(result["skel_verts"], result["skel_faces"], n_surface * 2, seed=1)
        d, _ = cKDTree(skel_bone_pts).query(targets.bone_pts)
        m["bone_ct_to_skel"] = _stats(d)
        ctb = cKDTree(targets.bone_pts)
        inside = ((skel_bone_pts >= lo) & (skel_bone_pts <= hi)).all(1)
        if inside.any():
            d, _ = ctb.query(skel_bone_pts[inside][::10])
            m["bone_skel_to_ct"] = _stats(d)
        result["_skel_bone_surface_pts"] = skel_bone_pts
    result["_skel_skin_surface_pts"] = skel_skin_pts
    if targets.joints is not None and targets.joint_w is not None:
        per = {}
        for i, n in enumerate(SKEL_PARTS):
            if targets.joint_w[i] > 0:
                per[n] = float(np.linalg.norm(result["joints"][i] - targets.joints[i]) * 1000.0)
        if per:
            m["joints_mm"] = per
            m["joints_mean_mm"] = float(np.mean(list(per.values())))
    return m


# ---------------------------------------------------------------------- limb initialisation by grid search
def init_arm_from_targets(model, poses: np.ndarray, betas: np.ndarray, trans: np.ndarray, scale: float,
                          targets: FitTargets, side: str, device: str = "cpu", iters: int = 60,
                          free_forearm: bool = True) -> tuple[np.ndarray, float]:
    """Choose the arm DOFs (shoulder x/y/z, elbow, pronation) that best reproduce the target joint positions
    and bone orientations, from a grid of initial guesses optimised together as one batch (the arm pose is
    non-convex: a wrong axial rotation makes elbow flexion move the forearm the wrong way).

    With ``free_forearm=False`` the elbow and pronation keep their current values (the arm hangs from the
    shoulder like the leg from the hip) and only the shoulder DOFs are searched.

    Returns the updated pose vector and the best loss.
    """
    names = SKEL_POSE_NAMES
    dofs = [names.index(f"shoulder_{side}_x"), names.index(f"shoulder_{side}_y"), names.index(f"shoulder_{side}_z"),
            names.index(f"elbow_flexion_{side}"), names.index(f"pro_sup_{side}")]
    parts = [SKEL_PARTS.index(f"humerus_{side}"), SKEL_PARTS.index(f"ulna_{side}"), SKEL_PARTS.index(f"radius_{side}"),
             SKEL_PARTS.index(f"hand_{side}")]
    dev = torch.device(device)
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)
    J_w = t(targets.joint_w)[parts] if targets.joint_w is not None else torch.zeros(len(parts), device=dev)
    R_w = t(targets.joint_rot_w)[parts] if targets.joint_rot_w is not None else torch.zeros(len(parts), device=dev)
    if float(J_w.sum()) + float(R_w.sum()) <= 0:
        return poses, float("nan")
    J_t = t(targets.joints)[parts] if targets.joints is not None else None
    R_t = t(targets.joint_rot)[parts] if targets.joint_rot is not None else None
    limits = {names.index(n): (min(lo, hi), max(lo, hi)) for n, (lo, hi) in SKEL_POSE_LIMITS.items() if n in names}
    if free_forearm:
        grid = [(ex, sz, ps) for ex in (0.3, 1.0, 1.7) for sz in (-1.5, -0.75, 0.0, 0.75, 1.5) for ps in (-0.8, 0.0, 0.8)]
    else:
        e0, p0 = float(poses[dofs[3]]), float(poses[dofs[4]])
        grid = [(e0, sz, p0) for sz in (-1.5, -0.75, 0.0, 0.75, 1.5)]
        dofs = dofs[:3]
    G = len(grid)
    q = t(poses)[None].repeat(G, 1)
    for g, (ex, sz, ps) in enumerate(grid):
        q[g, names.index(f"elbow_flexion_{side}")] = ex; q[g, dofs[2]] = q[g, dofs[2]] + sz; q[g, names.index(f"pro_sup_{side}")] = ps
    var = q[:, dofs].clone().requires_grad_(True)
    b, tr, sc = t(betas)[None].repeat(G, 1), t(trans)[None].repeat(G, 1), torch.full((G,), float(scale), device=dev)
    opt = torch.optim.Adam([var], lr=0.05)

    def losses(qq):
        out = skel_forward(model, qq, b, tr, skelmesh=False, scale=sc)
        l = torch.zeros(G, device=dev)
        if J_t is not None:
            l = l + 50.0 * DATA_SCALE * ((out.joints[:, parts] - J_t).pow(2).sum(-1) * J_w).sum(1) / max(float(J_w.sum()), 1e-6)
        if R_t is not None:
            l = l + 20.0 * ((out.joints_ori[:, parts] - R_t).pow(2).sum(dim=(2, 3)) * R_w).sum(1) / max(float(R_w.sum()), 1e-6)
        return l

    for _ in range(iters):
        opt.zero_grad()
        qq = q.clone(); qq[:, dofs] = var
        losses(qq).sum().backward()
        opt.step()
        with torch.no_grad():
            for k, i in enumerate(dofs):
                if i in limits:
                    var[:, k].clamp_(*limits[i])
    with torch.no_grad():
        qq = q.clone(); qq[:, dofs] = var
        l = losses(qq)
        g = int(l.argmin())
    return qq[g].detach().cpu().numpy().copy(), float(l[g])


# ---------------------------------------------------------------------- anthropometric limb lengths
# Trotter & Gleser (1958) stature regressions (cm, maximum bone lengths); joint-to-joint offsets convert the
# anatomical bone lengths to SKEL joint distances (femoral head top / condyles, tibial plateau / malleolus ...)
_TG = {"male": {"humerus": (3.08, 70.45), "radius": (3.78, 79.01), "femur": (2.38, 61.41), "tibia": (2.52, 78.62)},
       "female": {"humerus": (3.36, 57.97), "radius": (4.74, 54.93), "femur": (2.47, 54.10), "tibia": (2.90, 61.53)}}
_JOINT_OFFSET_CM = {"humerus": 3.0, "radius": 1.0, "femur": 3.5, "tibia": 2.0}


def stature_from_bone(gender: str, bone: str, length_cm: float) -> float:
    a, b = _TG["female" if gender.startswith("f") else "male"][bone]
    return a * length_cm + b


def limb_lengths_from_stature(gender: str, stature_cm: float) -> dict:
    """SKEL joint-to-joint limb lengths (m) expected for a stature: femur (hip->knee), tibia (knee->ankle),
    humerus (shoulder->elbow), forearm (elbow->wrist) on both sides."""
    tg = _TG["female" if gender.startswith("f") else "male"]
    bone_cm = {b: (stature_cm - c) / a for b, (a, c) in tg.items()}
    joint_cm = {b: bone_cm[b] - _JOINT_OFFSET_CM[b] for b in bone_cm}
    out = {}
    for side in "rl":
        out[(f"femur_{side}", f"tibia_{side}")] = joint_cm["femur"] / 100.0
        out[(f"tibia_{side}", f"talus_{side}")] = joint_cm["tibia"] / 100.0
        out[(f"humerus_{side}", f"ulna_{side}")] = joint_cm["humerus"] / 100.0
        out[(f"ulna_{side}", f"hand_{side}")] = joint_cm["radius"] / 100.0
    return out


def bone_extent_mm(mesh_vertices: np.ndarray) -> float:
    """Length of a bone mesh along its principal axis (mm)."""
    v = np.asarray(mesh_vertices, dtype=np.float64)
    c = v.mean(0)
    ax = np.linalg.svd(v - c, full_matrices=False)[2][0]
    return float(np.ptp((v - c) @ ax))
