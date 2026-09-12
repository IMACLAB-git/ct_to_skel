"""Viewer HTTP server with a small SKEL forward-kinematics API for interactive re-posing.

Endpoints (all JSON):
    GET  /api/info                -> pose parameter names, limits (deg), fitted q, DOF support, saved poses
    POST /api/pose   {"q": [46], "scapula_auto": bool} -> {"M": [24 x 16], "q": [46]} rigid part transforms (mm)
                     relative to the fitted pose; with scapula_auto the scapula DOFs follow the arm (returned in q)
    POST /api/save   {"name", "q"} -> appends/replaces the pose in poses.json (with interpolation frames)
    POST /api/export {"name", "q"} -> writes posed STLs to <out>/poses/<name>/ and returns the file list
Static files are served from the output directory as before.
"""
from __future__ import annotations

import functools
import http.server
import json
import math
import threading
from pathlib import Path

import numpy as np

# default ranges (deg) for DOFs that SKEL does not limit
_DEFAULT_RANGES = {
    "pelvis_tilt": (-45, 45), "pelvis_list": (-45, 45), "pelvis_rotation": (-45, 45),
    "hip_flexion": (-30, 120), "hip_adduction": (-45, 45), "hip_rotation": (-45, 45),
    "shoulder_x": (-180, 180), "shoulder_z": (-180, 180),
}


def _limits_deg(names):
    try:
        from skel.kin_skel import pose_limits
    except Exception:
        pose_limits = {}
    out = {}
    for n in names:
        if n in pose_limits:
            lo, hi = pose_limits[n]
            out[n] = [math.degrees(min(lo, hi)), math.degrees(max(lo, hi))]
        else:
            base = n[:-2] if n.endswith(("_r", "_l")) else n
            if base.startswith("shoulder_") and base.endswith(("_x", "_z")):
                base = "shoulder_" + base[-1]
            out[n] = list(_DEFAULT_RANGES.get(base, (-90, 90)))
    return out


class PoseEngine:
    """Holds the fitted SKEL model of one output directory and answers FK requests."""

    def __init__(self, out_dir: Path):
        from .pose import POSE_NAMES, part_frames
        from .skel_wrapper import load_skel

        self.out = out_dir
        self.man = json.loads((out_dir / "manifest.json").read_text())
        self.fit = self.man.get("fit") or {}
        self.names = list(POSE_NAMES)
        self.lock = threading.Lock()
        self.model = None
        self.G_fit = None
        if self.fit.get("poses"):
            import torch
            self.model = load_skel(self.man["gender"], None, "cuda" if torch.cuda.is_available() else "cpu")
            self.G_fit = part_frames(self.model, self.fit["poses"], self.fit["betas"], self.fit["trans"], self.fit.get("scale", 1.0))

    @property
    def available(self) -> bool:
        return self.model is not None

    def info(self) -> dict:
        poses = []
        pj = self.out / "poses.json"
        if pj.exists():
            poses = [{"name": p["name"], "q": p["q"]} for p in json.loads(pj.read_text()).get("poses", [])]
        return {"available": self.available, "names": self.names, "limits_deg": _limits_deg(self.names),
                "q_fit": self.fit.get("poses"), "dof_support": self.fit.get("dof_support"), "poses": poses,
                "parts": self.man.get("parts") and [p["id"] for p in self.man["parts"]]}

    def transforms(self, q, scapula_auto: bool = False) -> dict:
        from .pose import apply_scapula_rhythm, part_frames, relative_transforms
        q = np.asarray(q, dtype=np.float64)
        with self.lock:
            if scapula_auto:
                q = apply_scapula_rhythm(self.model, self.fit, q)
            G = part_frames(self.model, q, self.fit["betas"], self.fit["trans"], self.fit.get("scale", 1.0))
        return {"M": relative_transforms(self.G_fit, G).reshape(24, 16).round(5).tolist(), "q": [float(v) for v in q]}

    def save(self, name: str, q) -> dict:
        from .pose import build_pose_entry
        pj = self.out / "poses.json"
        data = json.loads(pj.read_text()) if pj.exists() else {"poses": []}
        with self.lock:
            entry = build_pose_entry(self.model, self.fit, name, np.asarray(q, dtype=np.float64))
        data["poses"] = [p for p in data.get("poses", []) if p["name"] != name] + [entry]
        pj.write_text(json.dumps(data))
        return {"saved": name, "n_poses": len(data["poses"])}

    def export(self, name: str, q) -> dict:
        """Write every part of the manifest in the requested pose as STL."""
        import trimesh
        from scipy.spatial import cKDTree
        from .labelmap import SKEL_PARTS
        from .pose import skin_weight_table

        M = np.asarray(self.transforms(q)["M"]).reshape(24, 4, 4)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "pose"
        dst = self.out / "poses" / safe
        dst.mkdir(parents=True, exist_ok=True)
        from .pose import skel_weight_table
        idx, val = skin_weight_table(self.model)
        bidx, bval = skel_weight_table(self.model)
        mv = self.out / "model_verts.npz"
        if mv.exists():                                        # model-ordered skin vertices, estimated parts excluded
            from .skel_wrapper import skin_part_labels
            est = [SKEL_PARTS.index(n) for n in self.man.get("estimated_parts", []) if n in SKEL_PARTS]
            sub = np.where(~np.isin(skin_part_labels(self.model), est))[0] if est else np.arange(len(idx))
            skin_v = np.load(mv)["skin_verts_mm"]
            _full = cKDTree(skin_v)
            _stree = cKDTree(skin_v[sub])
            tree = type("T", (), {"query": staticmethod(lambda V: (lambda d, n: (d, sub[n]))(*_stree.query(V)))})()
            full_tree = _full                                       # the SKEL skin itself keeps its own arm weights
            btree = cKDTree(np.load(mv)["skel_verts_mm"])
        else:
            skel_skin = trimesh.load(self.out / "stl" / "skel_skin.stl") if (self.out / "stl" / "skel_skin.stl").exists() else None
            tree = cKDTree(np.asarray(skel_skin.vertices)) if skel_skin is not None else None
            full_tree = tree
            btree = None

        def lbs(V, nn, widx, wval):
            W = np.zeros((len(V), 24)); np.put_along_axis(W, widx[nn].astype(int), wval[nn], axis=1)
            Vh = np.concatenate([V, np.ones((len(V), 1))], axis=1)
            return np.einsum("vj,jab,vb->va", W, M[:, :3, :], Vh)

        written = []
        for e in self.man["parts"]:
            if e["kind"] == "bone_all" or e.get("hidden"):
                continue
            m = trimesh.load(self.out / e["file"])
            V = np.asarray(m.vertices, dtype=np.float64)
            if e["kind"] in ("bone", "bone_tpl") and e.get("part") in SKEL_PARTS:
                if btree is not None:                       # SKEL skeleton skinning: spine bends, joints stay together
                    from .pose import bone_corner_weights, part_vertex_mask
                    from .skel_wrapper import bone_part_labels
                    ci, cv = bone_corner_weights(m, np.load(mv)["skel_verts_mm"], bidx, bval,
                                                 allowed=part_vertex_mask(bone_part_labels(self.model), e["part"]))
                    faces = np.asarray(m.faces).reshape(-1)
                    Vc = V[faces]                                # per-corner skinning (matches the viewer)
                    Pc = lbs(Vc, np.arange(len(Vc)), ci, cv)
                    V = V.copy(); V[faces] = Pc
                else:
                    T = M[SKEL_PARTS.index(e["part"])]; V = V @ T[:3, :3].T + T[:3, 3]
            elif e.get("static"):
                continue
            elif e["kind"] == "skin" and e["group"] == "ct" and mv.exists() and "patient_skin_widx" in np.load(mv).files:
                npz = np.load(mv)
                corners = npz["patient_skin_faces"].reshape(-1)
                if len(corners) == len(m.faces) * 3:
                    Vc = V[np.asarray(m.faces).reshape(-1)]
                    Pc = lbs(Vc, np.arange(len(Vc)), npz["patient_skin_widx"][corners], npz["patient_skin_wval"][corners])
                    V = V.copy(); V[np.asarray(m.faces).reshape(-1)] = Pc
                else:
                    continue
            elif e["kind"] == "skin" and e["group"] == "ct" and mv.exists():
                from .pose import ct_skin_vertex_weights
                from .skel_wrapper import skin_part_labels
                est_ = [SKEL_PARTS.index(n) for n in self.man.get("estimated_parts", []) if n in SKEL_PARTS]
                allowed = ~np.isin(skin_part_labels(self.model), est_) if est_ else None
                cidx, cval = ct_skin_vertex_weights(m, np.load(mv)["skin_verts_mm"], idx, val, allowed)
                V = lbs(V, np.arange(len(V)), cidx, cval)
            elif e["kind"] == "skin" and tree is not None:
                _, nn = full_tree.query(V); V = lbs(V, nn, idx, val)
            else:
                continue
            out_f = dst / f"{e['id']}.stl"
            trimesh.Trimesh(V, np.asarray(m.faces), process=False).export(out_f)
            written.append(out_f.name)
        (dst / "pose.json").write_text(json.dumps({"name": name, "q": list(map(float, q)), "names": self.names}, indent=1))
        return {"dir": str(dst), "files": written}


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    engine: PoseEngine | None = None

    def log_message(self, fmt, *args):      # quieter than the default
        if "/api/" in (args[0] if args else ""):
            return
        super().log_message(fmt, *args)

    def _json(self, code: int, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # outputs are rewritten in place by `ct2skel run` / `refresh`: never let the browser reuse stale STL / weights
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/api/info":
            if self.engine is None:
                return self._json(200, {"available": False})
            return self._json(200, self.engine.info())
        return super().do_GET()

    def do_POST(self):
        route = self.path.split("?")[0]
        if not route.startswith("/api/") or self.engine is None or not self.engine.available:
            return self._json(404, {"error": "no pose engine"})
        n = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
            if route == "/api/pose":
                return self._json(200, self.engine.transforms(payload["q"], bool(payload.get("scapula_auto", False))))
            if route == "/api/save":
                return self._json(200, self.engine.save(payload["name"], payload["q"]))
            if route == "/api/export":
                return self._json(200, self.engine.export(payload.get("name", "pose"), payload["q"]))
        except Exception as e:  # noqa: BLE001
            return self._json(500, {"error": repr(e)})
        return self._json(404, {"error": "unknown route"})


def serve(out_dir: Path, port: int, open_browser: bool = True) -> None:
    engine = None
    try:
        engine = PoseEngine(out_dir)
        print(f"[ct2skel] pose engine: {'ready (' + str(next(engine.model.buffers()).device) + ')' if engine.available else 'no fit in manifest'}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[ct2skel] pose engine unavailable: {e!r}", flush=True)
    ViewerHandler.engine = engine
    handler = functools.partial(ViewerHandler, directory=str(out_dir))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/index.html"
    print(f"[ct2skel] serving {out_dir} at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
