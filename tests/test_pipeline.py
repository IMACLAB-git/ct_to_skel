import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from ct2skel.dicom_io import load_volume, write_dicom_series
from ct2skel.export import CaseExporter, split_ct_bones_by_skel
from ct2skel.fit import ALL_POSE_IDX, FitConfig, FitTargets, SkelCTFitter, Stage, evaluate
from ct2skel.labelmap import SKEL_PARTS
from ct2skel.frames import FrameTransform, R_LPS_TO_SKEL
from ct2skel.landmarks import derive_joint_targets
from ct2skel.meshing import mask_to_mesh, mesh_to_mask, sample_surface
from ct2skel.segment import body_mask, bone_mask
from ct2skel.skel_wrapper import DummySKEL, bone_part_labels, init_pose, split_mesh_by_label
from ct2skel.synthetic import make_phantom, make_from_meshes


@pytest.fixture(scope="module")
def phantom():
    return make_phantom(shape_xyz=(96, 72, 100), spacing=(3.0, 3.0, 3.0))


def test_frame_roundtrip():
    ft = FrameTransform([10.0, -20.0, 30.0])
    p = np.random.default_rng(0).normal(size=(50, 3)) * 100
    assert np.allclose(ft.skel_to_lps(ft.lps_to_skel(p)), p)
    assert np.isclose(np.linalg.det(R_LPS_TO_SKEL), 1.0)
    # superior (LPS +z) maps to SKEL +y, anterior (LPS -y) maps to SKEL +z
    assert np.allclose(ft.rotate_lps_to_skel([0, 0, 1]), [0, 1, 0])
    assert np.allclose(ft.rotate_lps_to_skel([0, -1, 0]), [0, 0, 1])


def test_dicom_roundtrip(tmp_path, phantom):
    write_dicom_series(phantom, tmp_path / "dcm")
    vol = load_volume(tmp_path / "dcm")
    assert vol.array.shape == phantom.array.shape
    assert np.allclose(vol.spacing, phantom.spacing)
    assert np.allclose(vol.origin, phantom.origin, atol=1e-3)
    assert np.abs(vol.array - np.round(phantom.array)).max() < 1.0
    assert vol.meta["patient_sex"] == "M"


def test_segmentation_removes_table_and_fills_lungs(phantom):
    body = body_mask(phantom)
    # table voxels (HU 250 slab at y ~ +118 mm) must not be in the body mask
    zyx = np.argwhere(body)
    world = phantom.zyx_to_world(zyx)
    assert world[:, 1].max() < 110.0
    # lungs (HU -800) are inside the body mask
    lungs = (phantom.array < -700) & (phantom.array > -900)     # exclude outside air (-1000)
    assert (lungs & body).sum() / max(lungs.sum(), 1) > 0.9
    bone = bone_mask(phantom, body)
    assert bone.sum() > 0 and (bone & ~body).sum() == 0


def test_meshing_and_rasterisation(phantom):
    body = body_mask(phantom)
    mesh = mask_to_mesh(body, phantom, smooth_iterations=3, target_faces=20000, remove_border_caps=False)
    assert mesh.is_watertight
    assert mesh.volume > 0                                    # outward normals
    # with caps removed the mesh is open at the scan borders but still oriented outward
    open_mesh = mask_to_mesh(body, phantom, smooth_iterations=0, target_faces=None)
    assert not open_mesh.is_watertight
    c = open_mesh.vertices.mean(0)
    assert (((open_mesh.triangles_center - c) * open_mesh.face_normals).sum(1) > 0).mean() > 0.9
    back = mesh_to_mask(mesh, phantom)
    inter = (back & body).sum()
    union = (back | body).sum()
    assert inter / union > 0.9


def test_split_mesh_by_label():
    m = DummySKEL()
    labels = bone_part_labels(m)
    parts = split_mesh_by_label(m.skel_template_v.numpy(), m.skel_f.numpy(), labels, m.joints_name)
    assert set(parts) == {"pelvis", "thorax"}
    assert all(len(p.faces) > 0 for p in parts.values())


def test_landmarks_from_masks(phantom):
    # femur-like cylinders fully inside the volume -> hip and knee targets
    X = np.zeros(phantom.array.shape, dtype=bool)
    zyx = np.indices(phantom.array.shape).reshape(3, -1).T
    w = phantom.zyx_to_world(zyx)
    fem_l = ((w[:, 0] - 70) ** 2 + (w[:, 1] - 5) ** 2 <= 14 ** 2) & (w[:, 2] > -100) & (w[:, 2] < 60)
    fem_r = ((w[:, 0] + 70) ** 2 + (w[:, 1] - 5) ** 2 <= 14 ** 2) & (w[:, 2] > -100) & (w[:, 2] < 60)
    masks = {"femur_left": fem_l.reshape(X.shape), "femur_right": fem_r.reshape(X.shape)}
    jt = derive_joint_targets(masks, phantom, midline_x=0.0)
    d = jt.as_dict()
    assert {"femur_l", "femur_r", "tibia_l", "tibia_r"} <= set(d)
    assert d["femur_l"]["pos_lps_mm"][0] > 0 > d["femur_r"]["pos_lps_mm"][0]
    assert d["femur_l"]["pos_lps_mm"][2] > d["tibia_l"]["pos_lps_mm"][2]      # hip above knee


def _dummy_targets(model, betas, trans, rot):
    poses = torch.zeros(1, 46)
    poses[0, :3] = torch.tensor(rot)
    with torch.no_grad():
        out = model(poses, torch.tensor(betas)[None], torch.tensor(trans)[None])
    skin = trimesh.Trimesh(out.skin_verts[0].numpy(), model.skin_f.numpy(), process=False)
    skel = trimesh.Trimesh(out.skel_verts[0].numpy(), model.skel_f.numpy(), process=False)
    sp, _ = sample_surface(skin, 4000)
    bp, _ = sample_surface(skel, 4000)
    return FitTargets(skin_pts=sp, bone_pts=bp, joints=out.joints[0].numpy(), joint_w=np.ones(24) * 0.5), skin, skel


def test_fit_recovers_dummy_parameters():
    model = DummySKEL()
    betas_gt = np.zeros(10, dtype=np.float32); betas_gt[0], betas_gt[1] = 1.0, -0.8
    trans_gt = np.array([0.05, -0.10, 0.03], dtype=np.float32)
    rot_gt = [0.0, 0.3, 0.0]
    targets, _, _ = _dummy_targets(model, betas_gt, trans_gt, rot_gt)
    cfg = FitConfig(device="cpu", init_pose="tpose", n_skin=2000, n_bone=2000, n_skin_model=3000, n_bone_model=3000, verbose=False,
                    stages=[Stage("rigid", 120, 0.02, pose_idx=[], w_pose_reg=0, w_betas_reg=0, w_limits=0, w_scapula=0),
                            Stage("shape", 200, 0.02, opt_betas=True, pose_idx=[], w_pose_reg=0, w_betas_reg=0,
                                  w_limits=0, w_scapula=0)])
    res = SkelCTFitter(model, cfg).fit(targets)
    assert np.allclose(res["trans"], trans_gt, atol=0.01)
    assert abs(res["poses"][1] - 0.3) < 0.02
    # width beta has only ~3 mm leverage on the dummy body, so it is recovered less precisely
    assert np.allclose(res["betas"][:2], betas_gt[:2], atol=0.25)
    m = evaluate(res, targets)
    assert m["skin_ct_to_skel"]["mean_mm"] < 5.0
    assert m["bone_ct_to_skel"]["mean_mm"] < 5.0


def test_export_manifest(tmp_path):
    model = DummySKEL()
    targets, skin, skel = _dummy_targets(model, np.zeros(10, np.float32), np.zeros(3, np.float32), [0, 0, 0])
    frame = FrameTransform([0, 0, 0])
    ex = CaseExporter(tmp_path, "dummy", "male", frame)
    skin_mm = skin.copy(); skin_mm.vertices *= 1000
    skel_mm = skel.copy(); skel_mm.vertices *= 1000
    ex.add_mesh(skin_mm, "ct_skin", "ct", "skin", "CT skin", err_ref=skel_mm.vertices)
    parts = split_ct_bones_by_skel(skel_mm, np.asarray(skel_mm.vertices), bone_part_labels(model), model.joints_name)
    for p, m in parts.items():
        ex.add_mesh(m, f"ct_bone_{p}", "ct", "bone", f"CT {p}", part=p)
    mf = ex.write(metrics={"skin_ct_to_skel": {"mean_mm": 1.0, "p95_mm": 2.0}},
                  fit={"betas": [0] * 10, "poses": [0] * 46, "trans": [0, 0, 0]},
                  joints_skel=np.zeros((24, 3)))
    data = json.loads(Path(mf).read_text())
    ids = {p["id"] for p in data["parts"]}
    assert {"ct_skin", "ct_bone_pelvis", "ct_bone_thorax"} <= ids
    assert (tmp_path / "stl" / "ct_skin.stl").exists()
    assert (tmp_path / "ply" / "ct_skin_err.ply").exists()
    assert (tmp_path / "index.html").exists() and (tmp_path / "app.js").exists()
    assert (tmp_path / "skel_params.npz").exists()


def test_init_pose_presets():
    q = init_pose("arms_down")
    assert math.isclose(q[29], math.pi / 2, rel_tol=1e-6) and math.isclose(q[39], -math.pi / 2, rel_tol=1e-6)
    assert not init_pose("tpose").any()


def test_make_from_meshes_roundtrip():
    sphere = trimesh.creation.icosphere(3, radius=60.0)
    bone = trimesh.creation.icosphere(2, radius=20.0)
    vol = make_from_meshes(sphere, bone, spacing=(3.0, 3.0, 3.0), margin_mm=10)
    body = body_mask(vol, open_radius_mm=0)
    vol_mm3 = body.sum() * vol.voxel_volume_mm3()
    assert abs(vol_mm3 - sphere.volume) / sphere.volume < 0.1
    assert bone_mask(vol, body, min_component_mm3=100).sum() > 0


def test_cli_run_end_to_end_with_dummy_model(tmp_path, phantom, monkeypatch):
    """Full `run` path (segment -> fit -> export) using DummySKEL in place of the licensed model."""
    import ct2skel.skel_wrapper as sw
    from ct2skel.cli import main

    write_dicom_series(phantom, tmp_path / "dcm")
    monkeypatch.setattr(sw, "find_model_dir", lambda *a, **k: Path("dummy"))
    monkeypatch.setattr(sw, "load_skel", lambda gender, model_dir, device: DummySKEL().to(device))
    out = tmp_path / "out"
    rc = main(["run", "--input", str(tmp_path / "dcm"), "--out", str(out), "--device", "cpu",
               "--iters-scale", "0.05", "--skin-faces", "20000", "--bone-faces", "20000", "--mc-step", "2"])
    assert rc == 0
    data = json.loads((out / "manifest.json").read_text())
    ids = {p["id"] for p in data["parts"]}
    assert {"ct_skin", "ct_bone_all", "skel_skin", "skel_bone_all", "skel_bone_pelvis", "skel_bone_thorax"} <= ids
    assert data["ct_bone_partition"] == "nearest_skel_part"
    assert "skin_ct_to_skel" in data["metrics"] and "bone_ct_to_skel" in data["metrics"]
    assert len(data["fit"]["betas"]) == 10 and len(data["fit"]["poses"]) == 46
    assert (out / "skel_params.npz").exists()
    assert any(p.get("err_file") for p in data["parts"] if p["id"] == "ct_skin")


def test_export_ct_volume_orientation(tmp_path, phantom):
    """The exported 8-bit volume must be (Y, Z, X) in the SKEL frame with increasing coordinates."""
    from ct2skel.export import export_ct_volume

    frame = FrameTransform([0.0, 0.0, 0.0])
    meta = export_ct_volume(phantom, frame, tmp_path, max_inplane=48, max_slices=50)
    data = np.fromfile(tmp_path / "ct_volume.bin", dtype=np.uint8).reshape(meta["shape"])
    nY, nZ, nX = meta["shape"]
    ox, oy, oz = meta["origin_mm"]
    sx, sy, sz = meta["spacing_mm"]
    hu = lambda v: meta["hu_min"] + v / 255.0 * (meta["hu_max"] - meta["hu_min"])
    # the phantom spine is a cylinder at LPS (x=0, y=45) -> SKEL (x=0, z=-45), running along the whole body
    j = nY // 2
    r = int(round((-45.0 - oz) / sz)); c = int(round((0.0 - ox) / sx))
    assert hu(data[j, r, c]) > 300                    # bone at the spine location
    # anterior of the spine (SKEL z = +60) is soft tissue; the posterior end of the volume (row 0) is air
    r_soft = int(round((60.0 - oz) / sz))
    assert -200 < hu(data[j, r_soft, c]) < 200
    assert hu(data[j, 0, c]) < -800
    # superior end of the volume (last slice) corresponds to LPS z max
    z_top_lps = phantom.origin[2] + (phantom.array.shape[0] - 1) * phantom.spacing[2]
    assert abs((oy + (nY - 1) * sy) - z_top_lps) <= sy


def test_icp_recovers_similarity_transform():
    from ct2skel.refine import icp, apply_T
    from scipy.spatial.transform import Rotation
    src = trimesh.creation.icosphere(4, radius=50.0).vertices * np.array([1.0, 1.6, 0.8])
    R = Rotation.from_rotvec([0.1, -0.2, 0.15]).as_matrix()
    T_true = np.eye(4); T_true[:3, :3] = 1.05 * R; T_true[:3, 3] = [8.0, -5.0, 3.0]
    dst = apply_T(T_true, src)
    dst = dst[dst[:, 1] < 40.0]                                  # partially scanned target
    T, st = icp(src, dst, allow_scale=True)
    # a smooth ellipsoid can slide along itself a little, so allow a few mm; rotation/scale must be close
    assert np.abs(T - T_true).max() < 5.0
    assert abs(np.linalg.norm(T[:3, 0]) - 1.05) < 0.03
    assert st["residual_mm"] < 3.0


def test_refine_skin_moves_sphere_onto_target():
    from ct2skel.refine import refine_skin
    src = trimesh.creation.icosphere(3, radius=100.0)
    dst = trimesh.creation.icosphere(4, radius=100.0)
    dst.vertices *= np.array([1.15, 1.0, 0.9])                   # ellipsoid target
    V, st = refine_skin(np.asarray(src.vertices), np.asarray(src.faces), dst, iters=6)
    _, d_after, _ = dst.nearest.on_surface(V)
    _, d_before, _ = dst.nearest.on_surface(np.asarray(src.vertices))
    assert d_after.mean() < 0.25 * d_before.mean()
    assert st["mean_dist_after_mm"] < 3.0


def test_pose_transforms_dummy():
    from ct2skel.pose import part_frames, relative_transforms, apply_pose_spec, parse_set
    m = DummySKEL()
    q = np.zeros(46); b = np.zeros(10); t = np.zeros(3)
    G0 = part_frames(m, q, b, t, 1.0)
    q2 = q.copy(); q2[:3] = [0, 0.5, 0]                      # dummy: global rotation only
    G1 = part_frames(m, q2, b, t, 1.0)
    M = relative_transforms(G0, G1)
    ang = np.degrees(np.arccos(np.clip((np.trace(M[0][:3, :3]) - 1) / 2, -1, 1)))
    assert abs(ang - np.degrees(0.5)) < 0.1
    spec = parse_set("hip_flexion_r=45,knee_angle_r=90")
    q3 = apply_pose_spec(q, spec)
    assert abs(q3[3] - np.radians(45)) < 1e-9 and abs(q3[6] - np.radians(90)) < 1e-9


def test_align_bones_propagates_to_unscanned_children():
    """A child part without CT (e.g. tibia) must follow its ICP-moved parent (femur) rigidly."""
    from ct2skel.refine import align_bones, apply_T
    from ct2skel.labelmap import SKEL_PARTS
    femur = trimesh.creation.box(extents=(30.0, 60.0, 20.0)).subdivide().subdivide().subdivide().subdivide()   # dense box: rotation observable
    femur.vertices += [0, -300, 0]
    tibia = trimesh.creation.icosphere(2, radius=20.0); tibia.vertices += [0, -700, 0]
    verts = np.vstack([femur.vertices, tibia.vertices])
    labels = np.concatenate([np.full(len(femur.vertices), 1), np.full(len(tibia.vertices), 2)])   # femur_r, tibia_r
    joints = np.zeros((24, 3)); joints[1] = [0, -280, 0]; joints[2] = [0, -690, 0]
    shift = np.array([12.0, -5.0, 8.0])
    ct = {"femur_r": trimesh.Trimesh(femur.vertices + shift, femur.faces)}       # CT femur is shifted; no CT tibia
    al = align_bones(verts, labels, list(SKEL_PARTS), joints, ct, bbox=None)
    assert al.joint_weight[1] == 1 and al.joint_weight[2] == 0.5    # child joint becomes a soft target
    moved_tibia = al.verts[labels == 2]
    femur_disp = al.verts[labels == 1].mean(0) - femur.vertices.mean(0)
    assert np.allclose(moved_tibia.mean(0) - tibia.vertices.mean(0), femur_disp, atol=4.0)   # followed the femur
    assert np.allclose(femur_disp, shift, atol=5.0)   # ICP slides a little along the long axis
    assert al.stats["tibia_r"]["follows"] == "femur_r"


def test_fit_orientation_targets_dummy():
    """Orientation targets alone must recover the global rotation of the dummy model."""
    from scipy.spatial.transform import Rotation
    model = DummySKEL()
    targets, _, _ = _dummy_targets(model, np.zeros(10, np.float32), np.zeros(3, np.float32), [0, 0, 0])
    R = Rotation.from_rotvec([0, 0.4, 0]).as_matrix().astype(np.float32)
    cfg = FitConfig(device="cpu", init_pose="tpose", n_skin=500, n_bone=500, n_skin_model=800, n_bone_model=800, verbose=False,
                    stages=[Stage("rot", 150, 0.03, pose_idx=[], w_skin=0, w_skin_rev=0, w_bone=0, w_bone_rev=0, w_joint=0,
                                  w_pose_reg=0, w_betas_reg=0, w_limits=0, w_scapula=0, w_orient=1.0)])
    t2 = FitTargets(skin_pts=targets.skin_pts, bone_pts=None, joints=None, joint_w=None,
                    joint_rot=np.tile(R, (24, 1, 1)), joint_rot_w=np.ones(24))
    res = SkelCTFitter(model, cfg).fit(t2)
    assert abs(res["poses"][1] - 0.4) < 0.05


def test_fit_estimated_parts_stay_frozen():
    """Parts listed in FitConfig.estimated_parts (e.g. the arms) keep their init-pose DOFs whatever the targets say."""
    model = DummySKEL()
    targets, _, _ = _dummy_targets(model, np.zeros(10, np.float32), np.zeros(3, np.float32), [0, 0, 0])
    arm = [SKEL_PARTS.index(n) for n in ("humerus_r", "ulna_r", "radius_r", "hand_r")]
    cfg = FitConfig(device="cpu", init_pose="tpose", n_skin=500, n_bone=500, n_skin_model=800, n_bone_model=800, verbose=False,
                    estimated_parts=arm,
                    stages=[Stage("pose", 40, 0.05, pose_idx=ALL_POSE_IDX, w_pose_reg=0, w_betas_reg=0, w_limits=0,
                                  w_scapula=0, w_joint=50.0)])
    # pull the right shoulder/elbow towards a rotated target: the estimated arm must not follow
    tg = FitTargets(skin_pts=targets.skin_pts, bone_pts=targets.bone_pts, joints=targets.joints + np.array([0, 0.1, 0]),
                    joint_w=np.array([1.0 if i in arm else 0.0 for i in range(24)]))
    res = SkelCTFitter(model, cfg).fit(tg)
    from skel.kin_skel import pose_param_names as names
    for n in ("shoulder_r_x", "shoulder_r_y", "shoulder_r_z", "elbow_flexion_r", "pro_sup_r", "wrist_flexion_r"):
        assert res["poses"][names.index(n)] == 0.0, n
    assert res["dof_support"][names.index("elbow_flexion_r")] == 0


def test_fit_limb_length_prior_dummy():
    """A limb-length target pulls the joint-to-joint distance of the estimated leg towards the target."""
    from ct2skel.fit import limb_lengths_from_stature, stature_from_bone
    model = DummySKEL()
    targets, _, _ = _dummy_targets(model, np.zeros(10, np.float32), np.zeros(3, np.float32), [0, 0, 0])
    with torch.no_grad():
        out0 = model(torch.zeros(1, 46), torch.zeros(1, 10), torch.zeros(1, 3))
    a, b = SKEL_PARTS.index("femur_r"), SKEL_PARTS.index("tibia_r")
    L0 = float((out0.joints[0, a] - out0.joints[0, b]).norm())
    target = L0 * 1.3
    cfg = FitConfig(device="cpu", init_pose="tpose", n_skin=300, n_bone=300, n_skin_model=500, n_bone_model=500, verbose=False,
                    stages=[Stage("shape", 80, 0.05, opt_betas=True, pose_idx=[], w_skin=0, w_skin_rev=0, w_bone=0, w_bone_rev=0,
                                  w_joint=0, w_pose_reg=0, w_betas_reg=0, w_limits=0, w_scapula=0, w_limb=50.0)])
    tg = FitTargets(skin_pts=targets.skin_pts, bone_pts=None, joints=None, joint_w=None,
                    limb_lengths={("femur_r", "tibia_r"): target})
    res = SkelCTFitter(model, cfg).fit(tg)
    L1 = float(np.linalg.norm(res["joints"][a] - res["joints"][b]))
    assert abs(L1 - target) < abs(L0 - target) * 0.5
    # regression sanity: a 26.5 cm humerus is a short adult male
    st = stature_from_bone("male", "humerus", 26.5)
    assert 145 < st < 160
    assert 0.3 < limb_lengths_from_stature("male", st)[("femur_r", "tibia_r")] < 0.4


def test_pose_engine_export_dummy(tmp_path, phantom, monkeypatch):
    """The FK server's pose / export API works on a pipeline output (DummySKEL): transforms + q, posed STLs."""
    import ct2skel.skel_wrapper as sw
    from ct2skel.cli import main
    write_dicom_series(phantom, tmp_path / "dcm")
    monkeypatch.setattr(sw, "find_model_dir", lambda *a, **k: Path("dummy"))
    monkeypatch.setattr(sw, "load_skel", lambda gender, model_dir, device: DummySKEL().to(device))
    out = tmp_path / "out"
    assert main(["run", "--input", str(tmp_path / "dcm"), "--out", str(out), "--device", "cpu", "--iters-scale", "0.05",
                 "--skin-faces", "20000", "--bone-faces", "20000", "--mc-step", "2"]) == 0
    from ct2skel.server import PoseEngine
    eng = PoseEngine(out)
    assert eng.available
    q = list(eng.fit["poses"]); q[3] += 0.3
    r = eng.transforms(q, scapula_auto=True)
    assert len(r["M"]) == 24 and len(r["q"]) == 46
    res = eng.export("test_pose", q)
    assert "ct_skin.stl" in res["files"]
