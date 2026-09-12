"""Command line interface.

    python -m ct2skel run   --input <dicom_dir|volume.nii.gz> --out out/case1 [--gender male] [--labels DIR | --totalseg]
    python -m ct2skel synth --out data/phantom            # analytic phantom DICOM series
    python -m ct2skel synth --out data/skel_phantom --from-skel --gender male   # SKEL-derived phantom (needs SKEL data)
    python -m ct2skel serve out/case1 [--port 8000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from . import __version__
from .dicom_io import load_volume, save_volume, write_dicom_series
from .frames import FrameTransform
from .labelmap import SKEL_PARTS, ts_to_skel_part, UNLATERALISED


def _log(msg: str) -> None:
    print(f"[ct2skel] {msg}", flush=True)


# ---------------------------------------------------------------------- run
def cmd_run(a: argparse.Namespace) -> int:
    from .segment import body_mask, bone_mask, load_label_masks, run_totalsegmentator
    from .meshing import mask_to_mesh, sample_surface
    from .landmarks import derive_joint_targets, split_by_side, centroid
    from .export import CaseExporter, split_ct_bones_by_skel
    from .skel_wrapper import find_model_dir, load_skel, bone_part_labels, part_names, split_mesh_by_label

    t0 = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    case = a.case or Path(a.input).stem or "case"

    # 1. volume -------------------------------------------------------------
    vol = load_volume(a.input)
    meta = vol.meta or {}
    _log(f"volume {vol.shape_xyz} spacing {tuple(round(s, 3) for s in vol.spacing)} mm  meta={meta}")
    if a.resample_mm:
        from .dicom_io import resample_volume
        vol = resample_volume(vol, a.resample_mm)
        meta = vol.meta or {}
        _log(f"resampled to {a.resample_mm} mm: {vol.shape_xyz}")
    gender = a.gender or {"M": "male", "F": "female"}.get(str(meta.get("patient_sex", "")).upper(), "male")
    _log(f"gender = {gender}")
    if a.save_nifti:
        save_volume(vol, out / "ct.nii.gz")

    # 2. segmentation -------------------------------------------------------
    body = body_mask(vol, hu_threshold=a.skin_hu)
    bone = bone_mask(vol, body, hu_threshold=a.bone_hu)
    _log(f"body voxels {int(body.sum())}, bone voxels {int(bone.sum())}")
    labels = {}
    from .segment import BONE_STRUCTURES_TOTAL, BONE_STRUCTURES_APPENDICULAR
    bone_names = list(BONE_STRUCTURES_TOTAL) + list(BONE_STRUCTURES_APPENDICULAR)   # only bones are loaded (memory)
    if a.totalseg:
        lab_dir = run_totalsegmentator(vol, out / "totalseg", fast=a.totalseg_fast)
        labels = load_label_masks(lab_dir, vol, names=bone_names)
    elif a.labels:
        labels = load_label_masks(a.labels, vol, names=bone_names)
    if labels:
        _log(f"loaded {len(labels)} label masks: {sorted(labels)[:8]} ...")
        if not a.bone_from_hu:
            # anatomical bone labels are far more reliable than a HU threshold on real CT
            # (contrast vessels, partial volume at low resolution, metal); keep a loose HU gate
            union = np.zeros_like(bone)
            for name, m in labels.items():
                if ts_to_skel_part(name) or name in UNLATERALISED:
                    union |= m
            bone = union & (vol.array > a.bone_hu_gate) & body
            _log(f"bone mask from labels: {int(bone.sum())} voxels (HU gate > {a.bone_hu_gate})")

    # 3. frame + meshes ---------------------------------------------------------
    center = centroid(body, vol)
    frame = FrameTransform(center)
    to_skel_mm = lambda m: _transform_mesh(m, frame)
    ct_skin = to_skel_mm(mask_to_mesh(body, vol, step=a.mc_step, smooth_iterations=10,
                                      target_faces=a.skin_faces, keep_largest=True))
    ct_bone = to_skel_mm(mask_to_mesh(bone, vol, step=a.mc_step, smooth_iterations=6, target_faces=a.bone_faces))
    _log(f"CT skin mesh {len(ct_skin.faces)} faces, CT bone mesh {len(ct_bone.faces)} faces")

    ct_bone_parts: dict[str, trimesh.Trimesh] = {}
    ct_cartilage = None
    ct_unlabeled: list[trimesh.Trimesh] = []
    if labels:
        midline = center[0]
        per_part_masks: dict[str, np.ndarray] = {}
        for name, m in labels.items():
            if name in UNLATERALISED:
                for side, sm in split_by_side(m, vol, midline).items():
                    p = ts_to_skel_part(name, side)
                    if p and sm.any():
                        per_part_masks[p] = per_part_masks.get(p, np.zeros_like(m)) | sm
            else:
                p = ts_to_skel_part(name)
                if p:
                    per_part_masks[p] = per_part_masks.get(p, np.zeros_like(m)) | m
        from .bones import smooth_bone_mesh
        from .refine import COMPACT_PARTS
        from .segment import claim_unlabeled_bone
        cartilage = labels.get("costal_cartilages")
        big_unlabeled = None
        if a.claim_mm > 0:
            n0 = sum(int(m.sum()) for m in per_part_masks.values())
            per_part_masks, big_unlabeled = claim_unlabeled_bone(per_part_masks, (vol.array > a.bone_hu) & body, vol,
                                                                 radius_mm=a.claim_mm, exclude=cartilage)
            _log(f"unlabelled bone voxels attached to nearest part (<= {a.claim_mm} mm): +{sum(int(m.sum()) for m in per_part_masks.values()) - n0}")
        for p, m in per_part_masks.items():
            compact = p in COMPACT_PARTS
            # thin, multi-piece parts (ribs, scapula, hands, feet): light closing, no cavity filling
            mesh = smooth_bone_mesh(m, vol, close_mm=a.bone_close_mm if compact else min(a.bone_close_mm, 1.0),
                                    smooth_iterations=a.bone_smooth if compact else max(a.bone_smooth // 3, 6),
                                    fill_holes=compact, min_component_mm3=2000.0 if compact else 300.0,
                                    target_faces=max(int(a.bone_faces * m.sum() / max(bone.sum(), 1)), 4000),
                                    mc_step=a.mc_step)
            ct_bone_parts[p] = to_skel_mm(mesh)
        _log(f"smooth CT bone parts: {len(ct_bone_parts)} (closing {a.bone_close_mm} mm, Taubin x{a.bone_smooth})")
        ct_unlabeled: list[trimesh.Trimesh] = []
        if big_unlabeled is not None and big_unlabeled.any():
            from scipy import ndimage as _ndi
            lab_u, n_u = _ndi.label(big_unlabeled)
            for i in range(1, n_u + 1):
                comp = lab_u == i
                mesh = smooth_bone_mesh(comp, vol, close_mm=1.0, smooth_iterations=10, fill_holes=False,
                                        min_component_mm3=300.0, target_faces=40000, mc_step=a.mc_step)
                from .meshing import drop_small_components
                mesh = drop_small_components(mesh)          # specks / vessel strings hanging off the piece
                if len(mesh.faces):
                    ct_unlabeled.append(to_skel_mm(mesh))
            _log(f"unlabelled CT bones kept as separate pieces: {len(ct_unlabeled)} (e.g. forearms without appendicular labels)")
        ct_cartilage = None
        if cartilage is not None and cartilage.any():
            ct_cartilage = to_skel_mm(smooth_bone_mesh(cartilage & (vol.array > -50), vol, close_mm=1.0, smooth_iterations=10,
                                                       fill_holes=False, min_component_mm3=300.0, target_faces=40000, mc_step=a.mc_step))

    # 4. targets ---------------------------------------------------------------
    skin_pts, _ = sample_surface(ct_skin, 40000)
    bone_pts, _ = sample_surface(ct_bone, 40000)
    joints_ct = None
    if labels:
        jt = derive_joint_targets(labels, vol, midline_x=center[0])
        joints_ct = jt
        _log(f"joint targets: {sorted(jt.as_dict())}")

    # 5. SKEL fit -------------------------------------------------------------------
    fit_res, metrics, model, model_device = None, {}, None, None
    model_dir = find_model_dir(a.skel_dir, gender)
    if a.no_fit:
        _log("--no-fit: skipping SKEL fitting")
    elif model_dir is None:
        _log(f"SKEL model skel_{gender}.pkl not found (looked in --skel-dir / $SKEL_MODELS / external/SKEL/data/skel). "
             "Exporting CT meshes only. Register at https://skel.is.tue.mpg.de/ to download the model.")
    else:
        import torch
        from .fit import FitConfig, FitTargets, SkelCTFitter, evaluate, default_stages
        device = model_device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = load_skel(gender, model_dir, device)
        # arms (default): estimated from the body model like the legs.  The first fit keeps the arms free so that the
        # fitted model can tell which CT points belong to the arms; those are then dropped, the arm DOFs are reset to
        # the init pose and frozen for the re-fit rounds (FitConfig.estimated_parts)
        est_names = [] if a.fit_arms else [n for n in SKEL_PARTS if n.split("_")[0] in ("humerus", "ulna", "radius", "hand")]
        est_ids = [SKEL_PARTS.index(n) for n in est_names]
        arm_pieces: set[int] = set()
        cfg = FitConfig(device=device, init_pose=a.init_pose, stages=default_stages(a.iters_scale))
        targets = FitTargets(skin_pts=skin_pts / 1000.0, bone_pts=bone_pts / 1000.0,
                             joints=frame.lps_to_skel(joints_ct.positions) if joints_ct else None,
                             joint_w=joints_ct.weights if joints_ct else None)
        _log(f"fitting SKEL ({gender}) on {device} ...")
        fitter = SkelCTFitter(model, cfg)
        fit_res = fitter.fit(targets)
        refine_info = {}
        ex_seams = {}
        if ct_bone_parts and not a.no_icp:
            from dataclasses import replace
            from .refine import align_bones, refine_skin
            names, labels_v = part_names(model), bone_part_labels(model)
            bbox_mm = targets.bbox * 1000.0
            if est_names:
                from scipy.spatial import cKDTree as _KD
                from .fit import dof_to_part
                from .skel_wrapper import init_pose as _preset, skin_part_labels
                is_est = np.zeros(len(names), dtype=bool); is_est[est_ids] = True
                skin_tree = _KD(fit_res["skin_verts"] * 1000.0); skel_tree = _KD(fit_res["skel_verts"] * 1000.0)
                skin_lab = skin_part_labels(model)
                keep_s = ~is_est[skin_lab[skin_tree.query(skin_pts)[1]]]
                keep_b = ~is_est[labels_v[skel_tree.query(bone_pts)[1]]]
                for i, m in enumerate(ct_unlabeled):
                    if is_est[int(np.bincount(labels_v[skel_tree.query(np.asarray(m.vertices))[1]], minlength=len(names)).argmax())]:
                        arm_pieces.add(i)
                grew = True                                    # small fragments next to an arm piece belong to it too
                while grew:
                    grew = False
                    trees = [_KD(np.asarray(ct_unlabeled[i].vertices)) for i in arm_pieces]
                    for i, m in enumerate(ct_unlabeled):
                        if i not in arm_pieces and any(t.query(np.asarray(m.vertices))[0].min() < 30.0 for t in trees):
                            arm_pieces.add(i); grew = True
                if arm_pieces:
                    arm_tree = _KD(np.vstack([np.asarray(ct_unlabeled[i].vertices) for i in arm_pieces]))
                    keep_b &= arm_tree.query(bone_pts)[0] > 6.0
                # the patient's arm skin is removed from the CT skin as well (the estimated arm replaces it): a skin
                # vertex is arm skin when its nearest CT bone is an arm bone (humerus / arm piece) or its nearest
                # fitted SKEL skin vertex belongs to an arm part
                bone_v = [np.asarray(m.vertices) for m in ct_bone_parts.values()] + [np.asarray(m.vertices) for m in ct_unlabeled]
                bone_arm = [np.full(len(m.vertices), n in est_names) for n, m in ct_bone_parts.items()] + \
                           [np.full(len(m.vertices), i in arm_pieces) for i, m in enumerate(ct_unlabeled)]
                bone_tree = _KD(np.vstack(bone_v)); bone_arm = np.concatenate(bone_arm)
                arm_skin_v = bone_arm[bone_tree.query(ct_skin.vertices)[1]] | is_est[skin_lab[skin_tree.query(ct_skin.vertices)[1]]]
                from .meshing import remove_vertices
                n_faces0 = len(ct_skin.faces)
                ct_skin = remove_vertices(ct_skin, arm_skin_v)
                keep_s &= ~bone_arm[bone_tree.query(skin_pts)[1]]
                _log(f"CT skin: removed the arm skin ({int(arm_skin_v.sum())} vertices, {n_faces0 - len(ct_skin.faces)} faces)")
                skin_pts, bone_pts = skin_pts[keep_s], bone_pts[keep_b]
                targets = FitTargets(skin_pts=skin_pts / 1000.0, bone_pts=bone_pts / 1000.0, joints=targets.joints,
                                     joint_w=targets.joint_w, bbox=targets.bbox)
                q0 = np.asarray(fit_res["poses"]).copy(); q_init = _preset(a.init_pose, len(q0))
                arm_dofs = [i for i, pt in enumerate(dof_to_part()) if pt in est_ids]
                q0[arm_dofs] = q_init[arm_dofs]
                fit_res["poses"] = q0
                _log(f"arms estimated (not fitted): dropped {int((~keep_s).sum())} arm skin and {int((~keep_b).sum())} arm bone "
                     f"target points; unlabelled arm pieces {sorted(arm_pieces)}; arm DOFs reset to '{a.init_pose}'")
            # pass 1: bones onto CT bones -> accurate joint centres -> re-fit the parametric model with them
            # estimated arms hang from the shoulder: the humerus is not ICP'd (its joint centre is carried by the
            # scapula as a soft target) and the unlabelled pieces are not used; --fit-arms enables both
            icp_unlab = ct_unlabeled if a.fit_arms else None
            icp_parts = {n: m for n, m in ct_bone_parts.items() if n not in est_names}
            al = align_bones(fit_res["skel_verts"] * 1000.0, labels_v, names, fit_res["joints"] * 1000.0,
                             icp_parts, bbox=bbox_mm, unlabeled=icp_unlab)
            aligned = [n for n, w in zip(names, al.joint_weight) if w >= 1.0]
            _log(f"bone ICP pass 1: {len(aligned)} parts aligned; residuals (mm): "
                 + ", ".join(f"{n} {al.stats[n].get('residual_mm', float('nan')):.1f}" for n in aligned)
                 + f"; soft child-joint targets: {[n for n, w in zip(names, al.joint_weight) if 0 < w < 1]}")
            # how much of each SKEL bone the CT actually covers (along y): a bone that is mostly outside the scan
            # (proximal femur only) has a well-defined joint centre but an ill-defined ICP rotation
            skel_v_mm = fit_res["skel_verts"] * 1000.0
            coverage = {}
            for i, n in enumerate(names):
                sel = labels_v == i
                if n in icp_parts and sel.any():
                    y = skel_v_mm[sel][:, 1]; cy = icp_parts[n].vertices[:, 1]
                    ov = max(0.0, min(y.max(), cy.max()) - max(y.min(), cy.min()))
                    coverage[n] = float(ov / max(y.max() - y.min(), 1e-6))
            well = {n for n, c in coverage.items() if c >= 0.6}
            _log("CT coverage of SKEL bones: " + ", ".join(f"{n} {c:.2f}" for n, c in coverage.items()))
            # anthropometric prior for the estimated limbs: stature from the CT humerus (or --stature-cm) ->
            # expected femur / tibia / humerus / forearm lengths (the legs and arms are outside the CT)
            limb_targets, stature_cm = None, a.stature_cm
            if a.limb_prior or a.stature_cm is not None:
                from .fit import bone_extent_mm, limb_lengths_from_stature, stature_from_bone
                if stature_cm is None:
                    hum = []
                    for side in "rl":
                        m = ct_bone_parts.get(f"humerus_{side}")
                        y = skel_v_mm[labels_v == names.index(f"humerus_{side}")][:, 1]
                        if m is not None and len(y) and (min(y.max(), m.vertices[:, 1].max()) - max(y.min(), m.vertices[:, 1].min())) / np.ptp(y) >= 0.9:
                            hum.append(bone_extent_mm(m.vertices))
                    if hum:
                        stature_cm = stature_from_bone(gender, "humerus", float(np.mean(hum)) / 10.0)
                        _log(f"stature estimated from the CT humerus ({np.mean(hum):.0f} mm): {stature_cm:.0f} cm")
                if stature_cm is not None:
                    limb_targets = limb_lengths_from_stature(gender, stature_cm)
                    _log("limb-length prior (joint-to-joint, mm): " + ", ".join(f"{k[0]}-{k[1]} {v * 1000:.0f}" for k, v in limb_targets.items() if k[0].endswith("_r")))
                    refine_info["stature_cm"] = round(float(stature_cm), 1)
                    refine_info["limb_lengths_mm"] = {f"{k[0]}-{k[1]}": round(v * 1000, 1) for k, v in limb_targets.items()}

            def targets_from(al_):
                # joint positions of the ICP-placed bones are targets for the parametric fit; bone orientations only
                # where the CT covers most of the bone (otherwise the ICP rotation is not reliable).  Soft child joints
                # (knee/elbow/wrist carried by the parent) likewise only below well-covered or unlabelled-aligned parents.
                R_param = fit_res["joints_ori"]
                R_tgt = np.array([al_.transforms.get(n, np.eye(4))[:3, :3] @ R_param[i] for i, n in enumerate(names)])
                R_w = np.zeros(len(names)); J_w = al_.joint_weight.copy()
                for i, n in enumerate(names):
                    src = al_.stats.get(n, {}).get("source")
                    if al_.joint_weight[i] >= 1.0:
                        R_w[i] = 1.0 if n in well else (0.5 if src == "unlabeled" else 0.0)
                    elif 0 < al_.joint_weight[i] < 1:
                        if src == "unlabeled_axis":                  # direction of the unlabelled forearm pieces only
                            R_w[i] = 0.3
                            continue
                        par = al_.stats.get(n, {}).get("follows")
                        if par not in well and al_.stats.get(par, {}).get("source") not in ("unlabeled", "unlabeled_axis"):
                            J_w[i] = 0.0
                return FitTargets(skin_pts=targets.skin_pts, bone_pts=targets.bone_pts, joints=al_.joints / 1000.0,
                                  joint_w=J_w, bbox=targets.bbox, joint_rot=R_tgt, joint_rot_w=R_w, limb_lengths=limb_targets)

            strong = [replace(st, w_joint=max(st.w_joint, 50.0) if st.w_joint > 0 else 0.0) for st in cfg.stages]
            weak_parts = [names.index(n) for n in well] + [names.index(n) for n in names
                                                             if al.stats.get(n, {}).get("source") in ("unlabeled", "unlabeled_axis")]
            cfg2 = FitConfig(device=device, init_pose=a.init_pose, stages=strong, weak_prior_parts=weak_parts,
                             estimated_parts=est_ids or None)
            from .fit import init_arm_from_targets
            for it_ in range(2):
                tg = targets_from(al)
                q0 = fit_res["poses"]
                for side in (("r", "l") if a.fit_arms else ()):
                    q0, l_ = init_arm_from_targets(model, q0, fit_res["betas"], fit_res["trans"], fit_res.get("scale", 1.0), tg, side,
                                                  device=device)
                    _log(f"  arm_{side} initialised by grid search (loss {l_:.2f})")
                fit_res["poses"] = q0
                _log(f"re-fitting SKEL with ICP-derived joint centres and bone orientations (round {it_ + 1}) ...")
                fit_res = SkelCTFitter(model, cfg2).fit(tg, init_pose=fit_res["poses"],
                                                       init_betas=fit_res["betas"], init_trans=fit_res["trans"])
                al = align_bones(fit_res["skel_verts"] * 1000.0, labels_v, names, fit_res["joints"] * 1000.0,
                                 icp_parts, bbox=bbox_mm, unlabeled=icp_unlab)
                res = {n: al.stats[n].get("residual_mm") for n in names if "residual_mm" in al.stats.get(n, {})}
                _log(f"  ICP residuals after round {it_ + 1}: " + ", ".join(f"{n} {v:.1f}" for n, v in res.items()))
            # the displayed SKEL geometry is the parametric model itself (consistent re-posing); the final ICP is
            # reported as the residual between the parametric bones and the CT bones
            refine_info["bone_icp"] = {n: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in al.stats[n].items()}
                                       for n in al.stats}
            refine_info["bone_transforms"] = {n: np.round(T, 5).tolist() for n, T in al.transforms.items()}
            refine_info["display"] = "parametric"
            ex_seams = {}
            if not a.no_skin_refine:
                from .skel_wrapper import skin_part_labels
                is_est = np.zeros(len(names), dtype=bool); is_est[est_ids] = True
                constrain = ((al.joint_weight > 0) & ~is_est)[skin_part_labels(model)]
                V, st = refine_skin(fit_res["skin_verts"] * 1000.0, model.skin_f.cpu().numpy(), ct_skin, constrain=constrain)
                fit_res["skin_verts"] = V / 1000.0
                refine_info["skin_refine"] = st
                _log(f"skin refinement: {st}")
        metrics = evaluate(fit_res, targets)
        if refine_info.get("bone_icp") and "joints_mm" in metrics:
            # after ICP the landmark heuristics are no longer the reference: report them as a consistency check only
            metrics["landmark_vs_icp_joints_mm"] = metrics.pop("joints_mm")
            metrics["landmark_vs_icp_mean_mm"] = metrics.pop("joints_mean_mm")
        metrics["refine"] = refine_info
        _log("metrics: " + json.dumps({k: v for k, v in metrics.items() if k not in ("joints_mm", "refine")}, indent=None))

    # 6. export ----------------------------------------------------------------------
    ex = CaseExporter(out, case, gender, frame, err_max_mm=a.err_max,
                      bbox_mm=np.stack([skin_pts.min(0), skin_pts.max(0)]) if len(skin_pts) else None)
    if not a.no_volume:
        from .export import export_ct_volume
        ex.extra["ct_volume"] = export_ct_volume(vol, frame, out, max_inplane=a.volume_res)
        _log(f"CT volume for slice overlay: {ex.extra['ct_volume']['shape']} -> ct_volume.bin")
    # dense surface samples of the fitted SKEL (mm) are the reference for CT error maps
    skel_skin_mm = fit_res["_skel_skin_surface_pts"] * 1000.0 if fit_res else None
    skel_bone_mm = fit_res["_skel_bone_surface_pts"] * 1000.0 if fit_res else None
    ex.add_mesh(ct_skin, "ct_skin", "ct", "skin", "CT skin", err_ref=skel_skin_mm)
    if ct_bone_parts:
        for p, m in ct_bone_parts.items():
            ex.add_mesh(m, f"ct_bone_{p}", "ct", "bone", f"CT {p}", part=p, err_ref=skel_bone_mm, hidden=p in est_names)
        ex.extra["ct_bone_partition"] = "totalsegmentator"
    elif fit_res is not None:
        names = part_names(model)
        parts = split_ct_bones_by_skel(ct_bone, fit_res["skel_verts"] * 1000.0, bone_part_labels(model), names)
        for p, m in parts.items():
            ex.add_mesh(m, f"ct_bone_{p}", "ct", "bone", f"CT {p}", part=p, err_ref=skel_bone_mm)
        ex.extra["ct_bone_partition"] = "nearest_skel_part"
    else:
        ex.extra["ct_bone_partition"] = "none"
    if ct_unlabeled:
        from scipy.spatial import cKDTree as _KD
        lab_tree = _KD(fit_res["skel_verts"] * 1000.0) if fit_res is not None else None
        skel_lab = bone_part_labels(model) if fit_res is not None else None
        for i, m in enumerate(ct_unlabeled):
            part = None
            if lab_tree is not None:
                _, nn = lab_tree.query(np.asarray(m.vertices))
                part = SKEL_PARTS[int(np.bincount(skel_lab[nn]).argmax())]
            ex.add_mesh(m, f"ct_bone_unlab_{i}", "ct", "bone", f"CT bone (unlabelled{' near ' + part if part else ''})", part=part,
                        color="#d8d0c0", err_ref=skel_bone_mm, hidden=i in arm_pieces)
    ex.extra["estimated_parts"] = est_names
    ex.extra["run_id"] = run_id
    if ct_cartilage is not None and len(ct_cartilage.faces):
        ex.add_mesh(ct_cartilage, "ct_cartilage", "ct", "bone", "CT costal cartilage", part=None, color="#e6e0d0", hidden=True)
    # whole CT bone surface is always exported (used when no per-part partition exists)
    ex.add_mesh(ct_bone, "ct_bone_all", "ct", "bone_all", "CT bones (all)", color="#cfc6b8", err_ref=skel_bone_mm)

    if fit_res is not None:
        skel_skin_v, skel_bone_v = fit_res["skin_verts"] * 1000.0, fit_res["skel_verts"] * 1000.0
        skin = trimesh.Trimesh(skel_skin_v, model.skin_f.cpu().numpy(), process=False)
        skel = trimesh.Trimesh(skel_bone_v, model.skel_f.cpu().numpy(), process=False)
        ct_skin_v, ct_bone_v = np.asarray(ct_skin.vertices), np.asarray(ct_bone.vertices)
        refined = "skin_verts_param" in fit_res
        skel_parts = split_mesh_by_label(skel_bone_v, model.skel_f.cpu().numpy(), bone_part_labels(model), part_names(model))
        if refine_info.get("shaft") is not None:
            ex.extra["seam_offsets_mm"] = ex_seams
        ex.add_mesh(skin, "skel_skin", "skel", "skin", "SKEL skin (refined)" if refined else "SKEL skin", err_ref=ct_skin_v)
        ex.add_mesh(skel, "skel_bone_all", "skel", "bone_all", "SKEL skeleton (all)", color="#e8e2d6", err_ref=ct_bone_v)
        if refined:
            ex.add_mesh(trimesh.Trimesh(fit_res["skin_verts_param"] * 1000.0, model.skin_f.cpu().numpy(), process=False),
                        "skel_skin_param", "skel", "skin", "SKEL skin (parametric)", err_ref=ct_skin_v, hidden=True)
        if "skel_verts_param" in fit_res:
            ex.add_mesh(trimesh.Trimesh(fit_res["skel_verts_param"] * 1000.0, model.skel_f.cpu().numpy(), process=False),
                        "skel_bone_param", "skel", "bone_all", "SKEL skeleton (parametric)", color="#d6c9b8",
                        err_ref=ct_bone_v, hidden=True)
        for p, m in skel_parts.items():
            ex.add_mesh(m, f"skel_bone_{p}", "skel", "bone", f"SKEL {p}", part=p, err_ref=ct_bone_v)
    if fit_res is not None and not a.no_poses:
        from .pose import PRESETS, apply_pose_spec, write_poses, skin_weight_table, ct_skin_corner_weights
        fit_params = {"poses": fit_res["poses"], "betas": fit_res["betas"], "trans": fit_res["trans"],
                      "scale": fit_res.get("scale", 1.0)}
        from .pose import preset_pose
        from .skel_wrapper import skin_part_labels
        poses = {n: preset_pose(model, fit_params, n, ct_skin=ct_skin if (est_ids and ct_bone_parts) else None,
                                ct_bones=ct_bone_parts or None) for n in PRESETS}
        idx, val = skin_weight_table(model)
        allowed = ~np.isin(skin_part_labels(model), est_ids) if (est_ids and ct_bone_parts) else None
        cw = ct_skin_corner_weights(ct_skin, fit_res["skin_verts"] * 1000.0, idx, val, allowed=allowed)
        write_poses(out, model, fit_params, poses, ct_skin_corner_weights=cw,
                    bone_entries=ex.parts, skel_verts_mm=fit_res["skel_verts"] * 1000.0, run_id=run_id)
        np.savez_compressed(out / "model_verts.npz", skel_verts_mm=(fit_res["skel_verts"] * 1000.0).astype(np.float32),
                            skin_verts_mm=(fit_res["skin_verts"] * 1000.0).astype(np.float32))
        ex.extra["poses_file"] = "poses.json"
        _log(f"poses.json written with presets: {list(PRESETS)}")
    manifest = ex.write(
        metrics=metrics,
        fit={"betas": fit_res["betas"].tolist(), "poses": fit_res["poses"].tolist(), "trans": fit_res["trans"].tolist(),
             "scale": fit_res.get("scale", 1.0),
             "dof_support": fit_res.get("dof_support"),
             "init_pose": a.init_pose, "seconds": fit_res["seconds"], "device": str(model_device)} if fit_res else None,
        joints_skel=fit_res["joints"] * 1000.0 if fit_res else None,
        joints_ct={n: {**d, "pos_skel_mm": (frame.lps_to_skel(np.array(d["pos_lps_mm"])) * 1000.0).tolist()}
                   for n, d in joints_ct.as_dict().items()} if joints_ct else None,
        volume_meta={**meta, "shape_xyz": list(vol.shape_xyz), "spacing_mm": list(vol.spacing)},
    )
    _log(f"done in {time.time() - t0:.0f}s -> {manifest}")
    _log(f"view with:  python -m ct2skel serve {out}")
    return 0


def _transform_mesh(m: trimesh.Trimesh, frame: FrameTransform) -> trimesh.Trimesh:
    if len(m.faces) == 0:
        return m
    m = m.copy()
    m.vertices = frame.lps_to_skel(m.vertices) * 1000.0
    return m


# ---------------------------------------------------------------------- synth
def cmd_synth(a: argparse.Namespace) -> int:
    from .synthetic import make_phantom, make_from_skel
    out = Path(a.out)
    if a.from_skel:
        from .skel_wrapper import load_skel, init_pose
        model = load_skel(a.gender, a.skel_dir, "cpu")
        rng = np.random.default_rng(a.seed)
        betas = rng.uniform(-1.5, 1.5, 10).astype(np.float32)
        poses = init_pose(a.init_pose)
        poses[3:] += rng.normal(0, 0.05, 43).astype(np.float32)
        trans = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        vol, gt = make_from_skel(model, betas, poses, trans, spacing=(a.spacing,) * 3)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ground_truth.json").write_text(json.dumps(gt, indent=1))
    else:
        vol = make_phantom(spacing=(a.spacing,) * 3, seed=a.seed)
    files = write_dicom_series(vol, out / "dicom", patient_sex="M" if a.gender == "male" else "F")
    save_volume(vol, out / "phantom.nii.gz")
    _log(f"wrote {len(files)} DICOM slices to {out / 'dicom'}")
    return 0


# ---------------------------------------------------------------------- volume
def cmd_volume(a: argparse.Namespace) -> int:
    """Add (or refresh) the CT slice volume of an existing output directory without re-running the fit."""
    from .export import export_ct_volume, VIEWER_DIR
    import shutil
    out = Path(a.out)
    mf = out / "manifest.json"
    if not mf.exists():
        raise SystemExit(f"{mf} not found - run `ct2skel run` first")
    man = json.loads(mf.read_text())
    vol = load_volume(a.input)
    frame = FrameTransform(np.array(man["frame"]["center_mm"]))
    man["ct_volume"] = export_ct_volume(vol, frame, out, max_inplane=a.volume_res)
    mf.write_text(json.dumps(man, indent=1))
    for fn in ("index.html", "app.js"):
        shutil.copyfile(VIEWER_DIR / fn, out / fn)
    _log(f"CT volume {man['ct_volume']['shape']} written to {out / 'ct_volume.bin'}; manifest + viewer updated")
    return 0


# ---------------------------------------------------------------------- templates
def cmd_refresh(a: argparse.Namespace) -> int:
    """Refresh the viewer files and the bone skinning weights of an existing output."""
    import shutil
    from .export import VIEWER_DIR
    out = Path(a.out)
    mf = out / "manifest.json"
    man = json.loads(mf.read_text())
    mv, pj = out / "model_verts.npz", out / "poses.json"
    if mv.exists() and pj.exists():
        from .pose import write_bone_weights, skel_weight_table
        from .skel_wrapper import load_skel
        model = load_skel(man["gender"], None, "cpu")
        bidx, bval = skel_weight_table(model)
        data = json.loads(pj.read_text())
        from .skel_wrapper import bone_part_labels
        data["bone_weights"] = write_bone_weights(out, man["parts"], np.load(mv)["skel_verts_mm"], bidx, bval,
                                                  part_labels=bone_part_labels(model))
        ct_entry = next((e for e in man["parts"] if e["id"] == "ct_skin"), None)
        if ct_entry is not None and (out / ct_entry["file"]).exists():
            # CT skin skinning weights follow the current ct_skin mesh (it may have been edited, e.g. arms removed)
            from .pose import ct_skin_corner_weights, skin_weight_table
            sidx, sval = skin_weight_table(model)
            from .skel_wrapper import skin_part_labels
            est = [SKEL_PARTS.index(n) for n in man.get("estimated_parts", []) if n in SKEL_PARTS]
            allowed = ~np.isin(skin_part_labels(model), est) if est else None
            cidx, cval = ct_skin_corner_weights(trimesh.load(out / ct_entry["file"]), np.load(mv)["skin_verts_mm"], sidx, sval,
                                                allowed=allowed)
            (out / "ct_skin_weights.bin").write_bytes(cidx.astype(np.uint8).tobytes() + cval.astype(np.float32).tobytes())
            data["ct_skin_weights"] = {"file": "ct_skin_weights.bin", "n": int(len(cidx)), "top": int(cidx.shape[1])}
        pj.write_text(json.dumps(data))
        _log("bone and CT skin skinning weights refreshed")
    for fn in ("index.html", "app.js"):
        shutil.copyfile(VIEWER_DIR / fn, out / fn)
    _log("viewer refreshed")
    return 0


# ---------------------------------------------------------------------- pose
def cmd_pose(a: argparse.Namespace) -> int:
    """Add a custom pose (SKEL DOFs in degrees) to poses.json of an existing output."""
    from .pose import apply_pose_spec, build_pose_entry, parse_set, PRESETS
    from .skel_wrapper import load_skel
    out = Path(a.out)
    man = json.loads((out / "manifest.json").read_text())
    pj = out / "poses.json"
    if not pj.exists():
        raise SystemExit("poses.json not found - run `ct2skel run` (without --no-poses) first")
    data = json.loads(pj.read_text())
    model = load_skel(man["gender"], a.skel_dir, "cpu")
    fit = man["fit"]
    if a.preset:
        from .pose import preset_pose
        name = a.name or a.preset
        ct_entry = next((e for e in man["parts"] if e["id"] == "ct_skin"), None)
        ct_skin = trimesh.load(out / ct_entry["file"]) if ct_entry and man.get("estimated_parts") else None
        ct_bones = {e["part"]: trimesh.load(out / e["file"]) for e in man["parts"]
                    if e["id"].startswith("ct_bone_") and e.get("part") and "unlab" not in e["id"]}
        q = preset_pose(model, fit, a.preset, ct_skin=ct_skin, ct_bones=ct_bones or None)
    else:
        spec = parse_set(a.set); name = a.name or a.set
        q = apply_pose_spec(fit["poses"], spec, absolute=not a.relative)
    data["poses"] = [p for p in data["poses"] if p["name"] != name] + [build_pose_entry(model, fit, name, q)]
    pj.write_text(json.dumps(data))
    _log(f"pose '{name}' added to {pj} ({len(data['poses'])} poses)")
    return 0


# ---------------------------------------------------------------------- serve
# IMaC Lab text logo (same look as the lab's other GitHub Pages apps): red bar + "MaC Lab", links to the lab site
LAB_URL = "https://imac.super.site"
LAB_LOGO_CSS = (".lab-logo{display:inline-flex;align-items:baseline;white-space:nowrap;font:700 21px/1 Calibri,Carlito,"
                "'Segoe UI','Helvetica Neue',Arial,sans-serif;letter-spacing:-.01em;color:#f8fafc;text-decoration:none;"
                "padding:4px 7px;margin:-4px -1px -4px -7px;border-radius:7px;transition:background .12s}"
                ".lab-logo .lab-i{display:inline-block;width:.14em;height:.66em;margin-right:.1em;background:#c00000}"
                ".lab-logo:hover{background:rgba(255,255,255,.08)}")
LAB_LOGO_HTML = (f'<a class="lab-logo" href="{LAB_URL}" target="_blank" rel="noopener" title="IMaC Lab 홈페이지 (새 창)" '
                 'aria-label="IMaC Lab 홈페이지"><span class="lab-i" aria-hidden="true"></span>MaC Lab</a>')


def cmd_publish(a: argparse.Namespace) -> int:
    """Copy the static viewer of an output directory into a web-publishable folder (e.g. for GitHub Pages).

    Only the files the viewer needs are copied (manifest, STL parts, error PLYs, poses + skinning weights, CT
    volume); debug meshes, the pose exports and the model parameters stay local.  Without the FK server the viewer
    falls back to the saved preset poses (blend slider) instead of live sliders.
    """
    import shutil
    from .export import VIEWER_DIR
    out = Path(a.out)
    man = json.loads((out / "manifest.json").read_text())
    name = a.name or out.name
    dest = Path(a.dest) / name
    dest.mkdir(parents=True, exist_ok=True)
    files = ["manifest.json"]
    for e in man["parts"]:
        files.append(e["file"])
        if e.get("err_file") and not a.no_err:
            files.append(e["err_file"])
    if man.get("poses_file"):
        files.append(man["poses_file"])
        pj = json.loads((out / man["poses_file"]).read_text())
        for key in ("ct_skin_weights", "bone_weights"):
            if pj.get(key, {}).get("file"):
                files.append(pj[key]["file"])
    if man.get("ct_volume", {}).get("file") and not a.no_volume:
        files.append(man["ct_volume"]["file"])
    total = 0
    for rel in files:
        src = out / rel
        if not src.exists():
            _log(f"  missing (skipped): {rel}")
            continue
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest / rel)
        total += src.stat().st_size
    if a.no_err:                                            # the manifest must not reference files we did not copy
        for e in man["parts"]:
            e.pop("err_file", None)
    if a.no_volume:
        man.pop("ct_volume", None)
    (dest / "manifest.json").write_text(json.dumps(man))
    for fn in ("index.html", "app.js"):
        shutil.copyfile(VIEWER_DIR / fn, dest / fn)
    # landing page listing every published case
    site = Path(a.dest)
    cases = sorted(d.name for d in site.iterdir() if d.is_dir() and (d / "manifest.json").exists())
    items = "\n".join(f'<li><a href="{c}/index.html">{c}</a></li>' for c in cases)
    (site / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>ct2skel cases</title>"
        "<style>body{font-family:system-ui;margin:2rem;background:#14171c;color:#e6e8eb}a{color:#7cc4ff}" + LAB_LOGO_CSS + "</style>"
        f"<header style='display:flex;align-items:center;gap:14px;margin-bottom:1.2rem'>{LAB_LOGO_HTML}"
        "<span style='opacity:.5'>|</span><strong>ct2skel — CT ↔ SKEL viewer</strong></header>"
        f"<ul>{items}</ul>"
        "<p>Static build: saved preset poses only (live joint sliders need <code>ct2skel serve</code>).</p>\n", encoding="utf-8")
    (site / ".nojekyll").write_text("")
    _log(f"published {len(files)} files ({total / 1e6:.0f} MB) to {dest}  (site index: {site / 'index.html'})")
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .export import VIEWER_DIR
    from .server import serve
    import shutil
    d = Path(a.dir).resolve()
    for fn in ("index.html", "app.js"):
        if not (d / fn).exists() or a.refresh_viewer:
            shutil.copyfile(VIEWER_DIR / fn, d / fn)
    serve(d, a.port, open_browser=not a.no_browser)
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ct2skel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=__version__)
    sp = p.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("run", help="CT -> SKEL fit -> STL + viewer")
    r.add_argument("--input", "-i", required=True, help="DICOM directory or NIfTI/NRRD/MHA file")
    r.add_argument("--out", "-o", required=True)
    r.add_argument("--case", default=None)
    r.add_argument("--resample-mm", type=float, default=None,
                   help="resample the CT to this isotropic spacing before processing (e.g. 1.5 for sub-mm whole-body scans)")
    r.add_argument("--gender", choices=["male", "female"], default=None, help="default: from DICOM PatientSex")
    r.add_argument("--skel-dir", default=None, help="directory with skel_male.pkl / skel_female.pkl")
    r.add_argument("--labels", default=None, help="TotalSegmentator output directory (per-structure NIfTIs)")
    r.add_argument("--totalseg", action="store_true", help="run TotalSegmentator (must be installed)")
    r.add_argument("--totalseg-fast", action="store_true")
    r.add_argument("--init-pose", choices=["arms_down", "arms_up", "tpose"], default="arms_down")
    r.add_argument("--device", default=None)
    r.add_argument("--iters-scale", type=float, default=1.0, help="scale optimisation iterations (0.2 = quick test)")
    r.add_argument("--no-fit", action="store_true", help="segmentation + CT STL only")
    r.add_argument("--no-icp", action="store_true", help="skip the CT-driven bone ICP / re-fit (needs labels)")
    r.add_argument("--stature-cm", type=float, default=None,
                   help="patient stature; default: estimated from the CT humerus length (Trotter-Gleser). Sets the expected "
                        "limb lengths of the estimated legs/arms (anthropometric prior)")
    r.add_argument("--limb-prior", action="store_true",
                   help="constrain the estimated limb lengths to the stature (implied by --stature-cm). Off by default: the SKEL "
                        "shape space couples femur and tibia, so the prior mostly shortens the whole leg")
    r.add_argument("--fit-arms", action="store_true",
                   help="fit the arms to the CT (humerus ICP, elbow/forearm onto unlabelled bone pieces). Default: the whole arm "
                        "is estimated from the body model and hangs from the CT-placed shoulder in the init pose, like the legs")
    r.add_argument("--no-skin-refine", action="store_true", help="skip the non-rigid skin refinement onto the CT surface")
    r.add_argument("--no-poses", action="store_true", help="do not write poses.json (re-posing presets)")
    r.add_argument("--skin-hu", type=float, default=-400.0)
    r.add_argument("--bone-hu", type=float, default=250.0)
    r.add_argument("--bone-from-hu", action="store_true", help="use the HU threshold for bones even when labels exist")
    r.add_argument("--bone-hu-gate", type=float, default=100.0, help="loose HU gate applied inside label-derived bone masks")
    r.add_argument("--mc-step", type=int, default=1, help="marching cubes step (2 = faster, coarser)")
    r.add_argument("--skin-faces", type=int, default=150000)
    r.add_argument("--bone-faces", type=int, default=300000)
    r.add_argument("--bone-close-mm", type=float, default=2.5, help="morphological closing radius for bone label masks")
    r.add_argument("--bone-smooth", type=int, default=30, help="Taubin smoothing passes for per-part CT bones")
    r.add_argument("--claim-mm", type=float, default=8.0, help="attach unlabelled bone voxels within this distance to the nearest labelled bone (0 = off)")
    r.add_argument("--err-max", type=float, default=20.0, help="colour range for error maps (mm)")
    r.add_argument("--save-nifti", action="store_true")
    r.add_argument("--no-volume", action="store_true", help="do not export the 8-bit CT volume for slice overlays")
    r.add_argument("--volume-res", type=int, default=320, help="max in-plane size of the exported CT volume")
    r.set_defaults(func=cmd_run)

    s = sp.add_parser("synth", help="write a synthetic phantom DICOM series")
    s.add_argument("--out", "-o", required=True)
    s.add_argument("--from-skel", action="store_true", help="voxelise a random SKEL body (needs SKEL data)")
    s.add_argument("--gender", choices=["male", "female"], default="male")
    s.add_argument("--skel-dir", default=None)
    s.add_argument("--init-pose", choices=["arms_down", "arms_up", "tpose"], default="arms_down")
    s.add_argument("--spacing", type=float, default=2.5)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    vo = sp.add_parser("volume", help="add the CT slice volume to an existing output directory")
    vo.add_argument("--input", "-i", required=True, help="DICOM directory or volume file used for the run")
    vo.add_argument("--out", "-o", required=True)
    vo.add_argument("--volume-res", type=int, default=320)
    vo.set_defaults(func=cmd_volume)

    rf = sp.add_parser("refresh", help="rebuild the estimated bones/skin outside the CT (seams, skinning weights) from an existing output")
    rf.add_argument("--out", "-o", required=True)
    rf.set_defaults(func=cmd_refresh)

    pb = sp.add_parser("publish", help="copy the static viewer of an output into a web folder (GitHub Pages etc.)")
    pb.add_argument("--out", "-o", required=True, help="ct2skel output directory")
    pb.add_argument("--dest", default="site", help="site root; the case goes to <dest>/<name>/ (default: site)")
    pb.add_argument("--name", default=None, help="case folder name (default: output directory name)")
    pb.add_argument("--no-volume", action="store_true", help="skip the CT slice volume (ct_volume.bin, ~30 MB)")
    pb.add_argument("--no-err", action="store_true", help="skip the per-part error colour PLYs (~50 MB)")
    pb.set_defaults(func=cmd_publish)

    po = sp.add_parser("pose", help="add a re-posing preset to an existing output (degrees)")
    po.add_argument("--out", "-o", required=True)
    po.add_argument("--set", default="", help='e.g. "hip_flexion_r=45,knee_angle_r=90" (see skel.kin_skel.pose_param_names)')
    po.add_argument("--preset", default=None, help="one of the built-in presets")
    po.add_argument("--name", default=None)
    po.add_argument("--relative", action="store_true", help="add to the fitted angles instead of setting them")
    po.add_argument("--skel-dir", default=None)
    po.set_defaults(func=cmd_pose)

    v = sp.add_parser("serve", help="serve the viewer for an output directory")
    v.add_argument("dir")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--no-browser", action="store_true")
    v.add_argument("--refresh-viewer", action="store_true", help="overwrite index.html/app.js with the current viewer")
    v.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
