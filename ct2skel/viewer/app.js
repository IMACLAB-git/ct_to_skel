// CT <-> SKEL comparison viewer.  Loads manifest.json (written by ct2skel) and the STL/PLY parts.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const $ = (id) => document.getElementById(id);
const status = (t) => { $('status').textContent = t; };

// ------------------------------------------------------------------ scene
const main = document.querySelector('main');
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.localClippingEnabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
main.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);
const camera = new THREE.PerspectiveCamera(35, 1, 1, 20000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 0.9));
const key = new THREE.DirectionalLight(0xffffff, 1.2); key.position.set(1, 2, 3); scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.5); fill.position.set(-2, -1, -2); scene.add(fill);
const root = new THREE.Group(); scene.add(root);
const gCT = new THREE.Group(), gSKEL = new THREE.Group(), gJoints = new THREE.Group();
root.add(gCT, gSKEL, gJoints);

function resize() {
  const w = main.clientWidth, h = main.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize); resize();

// ------------------------------------------------------------------ state
const state = {
  mode: 'overlay', contrast: 'part', explode: 0, layer: 0, gap: 0,
  opCT: 0.35, opSKEL: 0.35, wire: false, clip: 1, joints: false,
  slice: 0, window: 'soft', sliceOp: 0.95, sliceShow: true, sliceClip: false, slicePanel: true, contours: 'all',
  pose: null, poseT: 0, fillMissing: true,
};
const items = [];          // {entry, mesh, errMesh, group, kind, centroid, base, visible}
let manifest = null, bodyCenter = new THREE.Vector3(), bodySize = new THREE.Vector3(), bbox = null;
const clipPlane = new THREE.Plane(new THREE.Vector3(0, -1, 0), 1e9);

const SRC_COLORS = { ct: new THREE.Color('#5cc8ff'), skel: new THREE.Color('#ffb454') };

function makeMaterial(entry) {
  const isSkin = entry.kind === 'skin';
  return new THREE.MeshStandardMaterial({
    color: new THREE.Color(entry.color || '#cccccc'),
    roughness: isSkin ? 0.75 : 0.55, metalness: 0.0,
    transparent: isSkin, opacity: isSkin ? 0.35 : 1, depthWrite: !isSkin,
    side: THREE.DoubleSide, clippingPlanes: [clipPlane], clipShadows: true,
    flatShading: entry.group === 'skel' && !isSkin,
  });
}

// ------------------------------------------------------------------ loading
async function load() {
  const res = await fetch('manifest.json', { cache: 'no-store' });
  manifest = await res.json();
  $('case').textContent = `${manifest.case} · ${manifest.gender} · ${manifest.created}`;
  $('legend_max').textContent = `${manifest.error_range_mm} mm`;
  const stl = new STLLoader(), ply = new PLYLoader();
  let n = 0;
  for (const entry of manifest.parts) {
    status(`loading ${entry.id} (${++n}/${manifest.parts.length})`);
    const geom = await stl.loadAsync(entry.file + '?v=' + encodeURIComponent(manifest.created || ''));   // cache-bust per run
    geom.computeVertexNormals();
    const mesh = new THREE.Mesh(geom, makeMaterial(entry));
    mesh.userData.entry = entry;
    mesh.userData.basePos = geom.attributes.position.array.slice();
    const it = { entry, mesh, errMesh: null, group: entry.group, kind: entry.kind,
      centroid: new THREE.Vector3(...entry.centroid), visible: entry.kind !== 'bone_all' && !entry.hidden };
    if (entry.err_file) {
      try {
        const g = await ply.loadAsync(entry.err_file + '?v=' + encodeURIComponent(manifest.created || ''));
        g.computeVertexNormals();
        it.errMesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
          vertexColors: true, roughness: 0.7, side: THREE.DoubleSide, clippingPlanes: [clipPlane],
          transparent: entry.kind === 'skin', opacity: entry.kind === 'skin' ? 0.9 : 1 }));
      } catch (e) { console.warn('no error mesh for', entry.id, e); }
    }
    (entry.group === 'ct' ? gCT : gSKEL).add(mesh);
    if (it.errMesh) (entry.group === 'ct' ? gCT : gSKEL).add(it.errMesh);
    items.push(it);
  }
  // patient model available (CT bones refined): the SKEL reference group starts hidden (?skel=1 shows it)
  const q0 = new URLSearchParams(location.search);
  if (manifest.ct_bone_partition && manifest.ct_bone_partition !== 'none' && q0.get('skel') !== '1') {
    for (const it of items) if (it.group === 'skel') it.visible = false;
  }
  // the union bone mesh is only shown by default when a group has no per-part bones
  for (const g of ['ct', 'skel']) {
    const hasParts = items.some((it) => it.group === g && it.kind === 'bone');
    for (const it of items) if (it.group === g && it.kind === 'bone_all') it.visible = !hasParts && !it.entry.hidden;
  }
  computeBounds();
  buildJoints();
  buildMetrics();
  // optional features must never take the whole viewer down
  for (const [name, fn] of [['CT volume', loadVolume], ['poses', loadPoses], ['pose UI', initPoseUI]]) {
    try { await fn(); } catch (e) { console.error(name, e); status(`${name} unavailable: ${e}`); }
  }
  try { buildFillItems(); } catch (e) { console.error('fill', e); }
  buildTree();
  applyQuery();
  applyAll();
  resetView();
  status(`${items.length} parts · units mm · axes: x=left y=superior z=anterior`);
}

function computeBounds() {
  bbox = new THREE.Box3();
  for (const it of items) if (it.group === 'ct' || it.kind === 'skin') bbox.expandByObject(it.mesh);
  if (bbox.isEmpty()) for (const it of items) bbox.expandByObject(it.mesh);
  bbox.getCenter(bodyCenter); bbox.getSize(bodySize);
}

function buildJoints() {
  const sph = new THREE.SphereGeometry(1, 16, 12);
  const add = (pos, color, r) => {
    const m = new THREE.Mesh(sph, new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35 }));
    m.position.set(pos[0], pos[1], pos[2]); m.scale.setScalar(r); gJoints.add(m);
  };
  const js = manifest.joints || {};
  for (const [name, p] of Object.entries(js.skel || {})) add(p, 0xff4d4d, 6);
  for (const [name, d] of Object.entries(js.ct || {})) if (d.pos_skel_mm) add(d.pos_skel_mm, 0x4dff88, 5);
  // connecting lines between corresponding joints
  const pts = [];
  for (const [name, d] of Object.entries(js.ct || {})) {
    if (d.pos_skel_mm && js.skel && js.skel[name]) pts.push(new THREE.Vector3(...d.pos_skel_mm), new THREE.Vector3(...js.skel[name]));
  }
  if (pts.length) gJoints.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6 })));
}

// ------------------------------------------------------------------ UI: tree & metrics
function buildTree() {
  const tree = $('tree'); tree.innerHTML = '';
  const groups = [['ct', 'Patient (CT)'], ['fill', 'Estimated from body model (outside CT)'], ['skel', 'SKEL reference (hidden by default)']];
  for (const [g, title] of groups) {
    const det = document.createElement('details'); det.open = true;
    const sum = document.createElement('summary');
    if (!items.some((it) => it.group === g)) continue;
    const cb = document.createElement('input'); cb.type = 'checkbox';
    cb.checked = items.some((it) => it.group === g && it.visible);
    cb.addEventListener('change', () => { for (const it of items) if (it.group === g && it.kind !== 'bone_all') { it.visible = cb.checked; it.cb.checked = cb.checked; } applyAll(); });
    sum.appendChild(cb); sum.appendChild(document.createTextNode(' ' + title));
    det.appendChild(sum);
    const order = (it) => (it.kind === 'skin' ? 0 : it.kind === 'bone_all' ? 1 : 2);
    for (const it of items.filter((x) => x.group === g).sort((a, b) => order(a) - order(b) || a.entry.name.localeCompare(b.entry.name))) {
      const row = document.createElement('div'); row.className = 'item';
      const c = document.createElement('input'); c.type = 'checkbox'; c.checked = it.visible;
      c.addEventListener('change', () => { it.visible = c.checked; applyAll(); });
      it.cb = c;
      const sw = document.createElement('span'); sw.className = 'sw'; sw.style.background = it.entry.color;
      const lab = document.createElement('span'); lab.textContent = it.entry.name + (it.kind === 'bone_all' ? ' [union]' : '');
      it.row = row;
      row.append(c, sw, lab);
      if (it.entry.err_mean_mm !== undefined) {
        const e = document.createElement('span'); e.className = 'err';
        const cov = it.entry.err_coverage === undefined ? 1 : it.entry.err_coverage;
        // a part the CT does not contain (or an estimated limb) has no meaningful error: say so instead of a number
        e.textContent = it.entry.estimated ? 'estimated' : (cov < 0.15 ? 'outside CT' : `${it.entry.err_mean_mm.toFixed(1)} mm`);
        e.title = `coverage ${(cov * 100).toFixed(0)} %`;
        row.appendChild(e);
      }
      det.appendChild(row);
    }
    tree.appendChild(det);
  }
}

function buildMetrics() {
  const t = $('metrics'); t.innerHTML = '';
  const m = manifest.metrics || {};
  const row = (k, v) => { const tr = document.createElement('tr'); tr.innerHTML = `<td>${k}</td><td>${v}</td>`; t.appendChild(tr); };
  if (!Object.keys(m).length) { row('SKEL fit', 'not run'); return; }
  for (const k of ['skin_ct_to_skel', 'skin_skel_to_ct', 'bone_ct_to_skel', 'bone_skel_to_ct']) {
    if (m[k]) row(k.replace(/_/g, ' '), `${m[k].mean_mm.toFixed(1)} / p95 ${m[k].p95_mm.toFixed(1)} mm`);
  }
  if (m.joints_mean_mm !== undefined) row('joints mean', `${m.joints_mean_mm.toFixed(1)} mm`);
  for (const [k, v] of Object.entries(m.joints_mm || {})) row('  ' + k, `${v.toFixed(1)} mm`);
  if (manifest.fit && manifest.fit.betas) row('betas', manifest.fit.betas.map((b) => b.toFixed(2)).join(', '));
  if (manifest.fit && manifest.fit.scale) row('global scale', manifest.fit.scale.toFixed(3));
  if (manifest.fit && manifest.fit.seconds) row('fit time', `${manifest.fit.seconds.toFixed(0)} s`);
}

// ------------------------------------------------------------------ apply state
function applyAll() {
  const errMode = state.mode === 'error';
  const sideGap = state.mode === 'side' ? bodySize.x * 1.25 : 0;
  const gapExtra = state.gap * bodySize.x * 0.8;
  gSKEL.position.set(sideGap + gapExtra, 0, 0);
  for (const it of items) {
    const showErr = errMode && it.errMesh;
    it.mesh.visible = it.visible && !showErr;
    if (it.errMesh) it.errMesh.visible = it.visible && showErr;
    const mat = it.mesh.material;
    if (it.kind === 'skin') {
      mat.opacity = it.group === 'ct' ? state.opCT : it.group === 'fill' ? Math.max(state.opCT, 0.25) : state.opSKEL;
      mat.visible = mat.opacity > 0.005;
    }
    if (it.group === 'fill') it.mesh.visible = it.mesh.visible && state.fillMissing;
    if (it.group === 'fill') { mat.color.set(it.kind === 'skin' ? '#b9c3d1' : '#aeb7c4'); }
    else if (state.contrast === 'source') { mat.color.copy(SRC_COLORS[it.group]); if (it.kind === 'skin') mat.color.lerp(new THREE.Color(0xffffff), 0.5); }
    else mat.color.set(it.entry.color);
    mat.wireframe = state.wire && it.group === 'skel';
    // exploded offsets
    const off = explodeOffset(it);
    it.mesh.position.copy(off);
    if (it.errMesh) it.errMesh.position.copy(off);
  }
  gJoints.visible = state.joints;
  gJoints.position.set(0, 0, 0);
  // clipping plane along y (superior axis): slider 1 = off
  if (state.clip >= 0.999) { clipPlane.constant = 1e9; $('clip_v').textContent = 'off'; }
  else { const y = bbox.min.y + state.clip * (bbox.max.y - bbox.min.y); clipPlane.constant = y; $('clip_v').textContent = `${y.toFixed(0)}`; }
  $('legend').style.display = errMode ? 'block' : 'none';
  applyPose();
  updateSlice();
}

function explodeOffset(it) {
  const off = new THREE.Vector3();
  if (it.kind === 'skin') {
    // skin layer moves outward (anterior) so that the skeleton becomes visible
    off.z += state.layer * bodySize.z * 1.5;
    return off;
  }
  if (it.kind === 'bone_all') return off;
  const d = it.centroid.clone().sub(bodyCenter);
  const scale = state.explode * 0.9;
  off.set(d.x * 1.0 * scale, d.y * 0.35 * scale, d.z * 1.2 * scale);
  // small deterministic lateral spread for midline parts so that the spine "opens" in a column
  if (Math.abs(d.x) < bodySize.x * 0.05) off.y += (it.entry.part || '').length * 0.0;
  return off;
}

function resetView() {
  // frame the whole body (plus side-by-side offset) in the vertical field of view
  const extra = state.mode === 'side' ? bodySize.x * 1.25 : 0;
  const h = Math.max(bodySize.y, (bodySize.x + extra) / camera.aspect, bodySize.z) || 1000;
  const dist = (h / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.25;
  const c = bodyCenter.clone(); c.x += extra / 2;
  camera.position.set(c.x + dist * 0.35, c.y + dist * 0.15, c.z + dist * 0.92);
  camera.near = dist * 0.01; camera.far = dist * 40; camera.updateProjectionMatrix();
  controls.target.copy(c); controls.update();
}

// URL query -> initial state, e.g. index.html?mode=side&bones=0.6&layer=0.3&opct=0.2&contrast=source
function applyQuery() {
  const q = new URLSearchParams(location.search);
  const num = (k, key, id) => { if (q.has(k)) { state[key] = parseFloat(q.get(k)); $(id).value = state[key]; const o = $(id + '_v'); if (o) o.textContent = state[key].toFixed(2); } };
  num('bones', 'explode', 'explode'); num('layer', 'layer', 'layer'); num('gap', 'gap', 'gap');
  num('opct', 'opCT', 'op_ct'); num('opskel', 'opSKEL', 'op_skel'); num('clip', 'clip', 'clip');
  if (q.has('mode')) setMode(q.get('mode'));
  if (q.has('contrast')) { state.contrast = q.get('contrast'); for (const x of $('contrast').querySelectorAll('button')) x.classList.toggle('active', x.dataset.c === state.contrast); }
  if (q.has('wire')) { state.wire = q.get('wire') === '1'; $('wire').checked = state.wire; }
  if (q.has('pose') && posesData) { state.pose = q.get('pose'); $('pose_select').value = state.pose; }
  if (q.has('poset')) { state.poseT = parseFloat(q.get('poset')); $('pose_t').value = state.poseT; $('pose_t_v').textContent = state.poseT.toFixed(1); }
  if (q.has('slice') && volume) { state.slice = Math.round(parseFloat(q.get('slice')) * (volume.shape[0] - 1)); $('slice').value = state.slice; }
  if (q.has('window')) { state.window = q.get('window'); $('window').value = state.window; }
  if (q.has('sliceclip')) { state.sliceClip = q.get('sliceclip') === '1'; $('slice_clip').checked = state.sliceClip; }
  if (q.has('panel')) { state.slicePanel = q.get('panel') === '1'; $('slice_panel').checked = state.slicePanel; }
}

// ------------------------------------------------------------------ controls wiring
function bindRange(id, key, fmt = (v) => v.toFixed(2)) {
  const el = $(id), out = $(id + '_v');
  el.addEventListener('input', () => { state[key] = parseFloat(el.value); if (out) out.textContent = fmt(state[key]); applyAll(); });
}
bindRange('explode', 'explode'); bindRange('layer', 'layer'); bindRange('gap', 'gap');
bindRange('op_ct', 'opCT'); bindRange('op_skel', 'opSKEL'); bindRange('clip', 'clip', (v) => v.toFixed(2));
$('wire').addEventListener('change', (e) => { state.wire = e.target.checked; applyAll(); });
$('joints').addEventListener('change', (e) => { state.joints = e.target.checked; applyAll(); });
for (const b of $('mode').querySelectorAll('button')) b.addEventListener('click', () => setMode(b.dataset.mode));
for (const b of $('contrast').querySelectorAll('button')) b.addEventListener('click', () => {
  state.contrast = b.dataset.c; for (const x of $('contrast').querySelectorAll('button')) x.classList.toggle('active', x === b); applyAll();
});
function setMode(m) {
  state.mode = m; for (const x of $('mode').querySelectorAll('button')) x.classList.toggle('active', x.dataset.mode === m); applyAll();
}
$('reset').addEventListener('click', resetView);
$('shot').addEventListener('click', () => {
  renderer.render(scene, camera);
  const a = document.createElement('a'); a.download = `${manifest ? manifest.case : 'view'}_${state.mode}.png`;
  a.href = renderer.domElement.toDataURL('image/png'); a.click();
});
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'e' || e.key === 'E') { const v = state.explode > 0 ? 0 : 0.6; $('explode').value = v; state.explode = v; $('explode_v').textContent = v.toFixed(2); applyAll(); }
  if (e.key === '1') setMode('overlay'); if (e.key === '2') setMode('side'); if (e.key === '3') setMode('error');
  if (e.key === 'r' || e.key === 'R') resetView();
  if (volume && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
    e.preventDefault();
    const step = e.shiftKey ? 10 : 1;
    state.slice = Math.max(0, Math.min(volume.shape[0] - 1, state.slice + (e.key === 'ArrowUp' ? step : -step)));
    $('slice').value = state.slice; applyAll();
  }
});

// ------------------------------------------------------------------ fill outside the CT with the fitted body model
// SKEL parts (fitted to this patient's shape) are shown only where the CT has no coverage,
// cut at the CT skin's superior/inferior extent, styled as "estimated".
function buildFillItems() {
  const ctSkin = items.find((it) => it.group === 'ct' && it.kind === 'skin');
  const skelSkin = items.find((it) => it.group === 'skel' && it.kind === 'skin');
  if (!ctSkin || !skelSkin) return;
  const [lo, hi] = [ctSkin.entry.bbox[0][1] + 5, ctSkin.entry.bbox[1][1] - 5];
  const full = skelSkin.entry.bbox;
  const estNames = manifest.estimated_parts || [];                 // parts estimated in full (e.g. arms), not clipped
  if (full[0][1] > lo - 20 && full[1][1] < hi + 20 && !estNames.length) return;   // CT covers the whole body
  $('fill_row').hidden = false;
  const cw = posesData ? skelSkinCorners() : null;
  const estIdx = new Set(estNames.map((n) => partIndex[n]).filter((i) => i !== undefined));
  // parts whose bone was aligned to the CT (ICP residual recorded) are patient bones already: never fill them, even
  // where the SKEL bone pokes a few millimetres past the CT skin extent (e.g. the neutral SKEL foot below a CT foot)
  const icp = ((manifest.metrics || {}).refine || {}).bone_icp || {};
  const aligned = new Set(Object.keys(icp).filter((n) => icp[n] && icp[n].residual_mm !== undefined));
  // with a SKEL-topology patient skin the skin already covers the whole body (parametric where the CT ends)
  const src = items.filter((it) => it.group === 'skel' && (it.kind === 'skin' || it.kind === 'bone') && !it.entry.hidden
                                   && !(it.kind === 'skin' && manifest.patient_skin === 'skel_topology'));
  for (const it of src) {
    const pos = it.mesh.geometry.attributes.position.array;
    const nTri = pos.length / 9, keep = [];
    if (it.kind === 'bone' && aligned.has(it.entry.part) && !estNames.includes(it.entry.part)) continue;
    const wholePart = it.kind === 'bone' && estNames.includes(it.entry.part);
    const skinW = it.kind === 'skin' && estIdx.size && cw && cw.idx.length === nTri * 3 * cw.top ? cw : null;
    for (let t = 0; t < nTri; t++) {
      let outside = wholePart;
      if (!outside) for (let k = 0; k < 3; k++) { const y = pos[t * 9 + k * 3 + 1]; if (y < lo || y > hi) { outside = true; break; } }
      if (!outside && skinW) for (let k = 0; k < 3; k++) { if (estIdx.has(skinW.idx[(t * 3 + k) * skinW.top])) { outside = true; break; } }
      if (outside) keep.push(t);
    }
    if (keep.length < 4) continue;
    const arr = new Float32Array(keep.length * 9);
    keep.forEach((t, i) => arr.set(pos.subarray(t * 9, t * 9 + 9), i * 9));
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(arr, 3)); g.computeVertexNormals();
    const entry = { ...it.entry, id: 'fill_' + it.entry.id, group: 'fill', name: it.entry.name.replace('SKEL ', 'est. '), hidden: false, err_file: undefined, err_mean_mm: undefined };
    const mesh = new THREE.Mesh(g, makeMaterial({ ...entry, group: 'skel' }));
    mesh.userData.entry = entry; mesh.userData.basePos = arr.slice();
    const fi = { entry, mesh, errMesh: null, group: 'fill', kind: it.kind, centroid: it.centroid.clone(), visible: true };
    const srcW = it.kind === 'skin' ? cw : (boneW && boneW[it.entry.id]);
    if (srcW && srcW.idx.length === nTri * 3 * srcW.top) {
      const top = srcW.top, idx = new Uint8Array(keep.length * 3 * top), w = new Float32Array(keep.length * 3 * top);
      keep.forEach((t, i) => { for (let k = 0; k < 3; k++) for (let j = 0; j < top; j++) { idx[(i * 3 + k) * top + j] = srcW.idx[(t * 3 + k) * top + j]; w[(i * 3 + k) * top + j] = srcW.w[(t * 3 + k) * top + j]; } });
      fi.cornerW = { idx, w, top };
    }
    gCT.add(mesh);
    items.push(fi);
  }
}
$('fill_missing').addEventListener('change', (e) => { state.fillMissing = e.target.checked; applyAll(); });

// ------------------------------------------------------------------ re-posing (SKEL kinematics)
let posesData = null, ctSkinW = null, boneW = null;   // boneW: {partId: {idx, w, top}} skeleton skinning per bone corner
const partIndex = {};
async function loadPoses() {
  if (!manifest.poses_file) return;
  try {
    posesData = await (await fetch(manifest.poses_file, { cache: 'no-store' })).json();
  } catch (e) { console.warn('no poses', e); return; }
  if (posesData.run_id && manifest.run_id && posesData.run_id !== manifest.run_id) {
    posesData = null;
    status('output is being rewritten (poses.json and manifest.json come from different runs): reload later');
    return;
  }
  posesData.parts.forEach((n, i) => { partIndex[n] = i; });
  if (posesData.ct_skin_weights) {
    const meta = posesData.ct_skin_weights;
    const buf = await (await fetch(meta.file, { cache: 'no-store' })).arrayBuffer();
    const n = meta.n, top = meta.top;
    ctSkinW = { idx: new Uint8Array(buf, 0, n * top), w: new Float32Array(buf, n * top, n * top), top };
  }
  if (posesData.bone_weights) {
    const meta = posesData.bone_weights;
    const buf = await (await fetch(meta.file, { cache: 'no-store' })).arrayBuffer();
    boneW = {};
    for (const [pid, seg] of Object.entries(meta.parts)) {
      const off = seg.offset * meta.top, n = seg.n * meta.top;
      // layout per part: uint8 idx[n] then float32 w[n]; parts are concatenated
      const base = off * 5;                                    // 1 byte idx + 4 bytes w per entry, per part block
      boneW[pid] = { idx: new Uint8Array(buf, base, n), w: new Float32Array(buf.slice(base + n, base + n + n * 4)), top: meta.top };
    }
  }
  const sel = $('pose_select'); sel.innerHTML = '';
  const o = document.createElement('option'); o.value = ''; o.textContent = 'fitted (CT pose)'; sel.appendChild(o);
  for (const p of posesData.poses) { const op = document.createElement('option'); op.value = p.name; op.textContent = p.name; sel.appendChild(op); }
  $('pose_section').hidden = false;
}
$('pose_select').addEventListener('change', (e) => { state.pose = e.target.value || null; applyAll(); });
$('pose_t').addEventListener('input', (e) => { state.poseT = parseFloat(e.target.value); $('pose_t_v').textContent = state.poseT.toFixed(1); applyAll(); });

let liveM = null;          // transforms from the FK server (interactive sliders)
function poseFrame() {
  if (liveM) return liveM;
  if (!posesData || !state.pose || state.poseT <= 0) return null;
  const p = posesData.poses.find((x) => x.name === state.pose);
  if (!p) return null;
  const k = Math.round(state.poseT * (posesData.n_steps - 1));
  return p.frames[k].map((m) => new THREE.Matrix4().set(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]));
}

const _v = new THREE.Vector3(), _acc = new THREE.Vector3();
function skinCorners(mesh, M, cornerIdx, cornerW, top) {
  const pos = mesh.geometry.attributes.position, base = mesh.userData.basePos;
  const n = pos.count;
  for (let c = 0; c < n; c++) {
    _acc.set(0, 0, 0);
    for (let k = 0; k < top; k++) {
      const w = cornerW[c * top + k]; if (w === 0) continue;
      _v.set(base[c * 3], base[c * 3 + 1], base[c * 3 + 2]).applyMatrix4(M[cornerIdx[c * top + k]]);
      _acc.addScaledVector(_v, w);
    }
    pos.array[c * 3] = _acc.x; pos.array[c * 3 + 1] = _acc.y; pos.array[c * 3 + 2] = _acc.z;
  }
  pos.needsUpdate = true; mesh.geometry.computeVertexNormals();
}

let skelCornerW = null;   // per-corner weights of the SKEL skin STL (from faces + vertex weights)
function skelSkinCorners() {
  if (skelCornerW || !posesData) return skelCornerW;
  const f = posesData.skin.faces, wi = posesData.skin.w_idx, ww = posesData.skin.w, top = wi[0].length;
  const n = f.length * 3, idx = new Uint8Array(n * top), w = new Float32Array(n * top);
  for (let t = 0; t < f.length; t++) for (let k = 0; k < 3; k++) {
    const c = t * 3 + k, v = f[t][k];
    for (let j = 0; j < top; j++) { idx[c * top + j] = wi[v][j]; w[c * top + j] = ww[v][j]; }
  }
  skelCornerW = { idx, w, top };
  return skelCornerW;
}

let posed = false;
function applyPose() {
  const M = poseFrame();
  if (!M) {
    if (posed) {           // restore
      for (const it of items) {
        const pos = it.mesh.geometry.attributes.position;
        pos.array.set(it.mesh.userData.basePos); pos.needsUpdate = true; it.mesh.geometry.computeVertexNormals();
        it.mesh.matrix.identity(); it.mesh.matrixAutoUpdate = true; it.mesh.rotation.set(0, 0, 0);
        if (it.errMesh) it.errMesh.visible = it.errMesh.visible;
      }
      gJoints.visible = state.joints; posed = false;
    }
    return;
  }
  posed = true;
  for (const it of items) {
    const e = it.entry;
    if (e.kind === 'bone_all') { it.mesh.visible = false; continue; }
    if (e.static) { it.mesh.visible = false; continue; }          // raw CT surface: reference in the CT pose only
    if (e.part && partIndex[e.part] !== undefined && (e.kind === 'bone' || e.kind === 'bone_tpl')) {
      const bw = it.cornerW || (boneW && boneW[e.id]);
      if (bw && bw.idx.length === it.mesh.geometry.attributes.position.count * bw.top) { skinCorners(it.mesh, M, bw.idx, bw.w, bw.top); if (it.errMesh) it.errMesh.visible = false; continue; }
      // rigid fallback: bake M into the geometry so that explode offsets (mesh.position) still apply
      const pos = it.mesh.geometry.attributes.position, base = it.mesh.userData.basePos, m = M[partIndex[e.part]];
      for (let c = 0; c < pos.count; c++) { _v.set(base[c * 3], base[c * 3 + 1], base[c * 3 + 2]).applyMatrix4(m); pos.array[c * 3] = _v.x; pos.array[c * 3 + 1] = _v.y; pos.array[c * 3 + 2] = _v.z; }
      pos.needsUpdate = true; it.mesh.geometry.computeVertexNormals();
    } else if (e.kind === 'skin' && (e.group === 'skel' || e.group === 'fill') && posesData) {
      const cw = it.cornerW || skelSkinCorners(); skinCorners(it.mesh, M, cw.idx, cw.w, cw.top);
    } else if (e.kind === 'skin' && e.group === 'ct' && ctSkinW && it.mesh.geometry.attributes.position.count === ctSkinW.idx.length / ctSkinW.top) {
      skinCorners(it.mesh, M, ctSkinW.idx, ctSkinW.w, ctSkinW.top);
    }
    if (it.errMesh) it.errMesh.visible = false;   // error maps refer to the fitted pose
  }
  gJoints.visible = false;
}

// ------------------------------------------------------------------ interactive posing (FK server)
const DOF_GROUPS = [
  ['Global orientation', ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation']],
  ['Hips', ['hip_flexion_r', 'hip_flexion_l', 'hip_adduction_r', 'hip_adduction_l', 'hip_rotation_r', 'hip_rotation_l']],
  ['Knees', ['knee_angle_r', 'knee_angle_l']],
  ['Ankles / feet', ['ankle_angle_r', 'ankle_angle_l', 'subtalar_angle_r', 'subtalar_angle_l', 'mtp_angle_r', 'mtp_angle_l']],
  ['Spine', ['lumbar_bending', 'lumbar_extension', 'lumbar_twist', 'thorax_bending', 'thorax_extension', 'thorax_twist']],
  ['Head / neck', ['head_bending', 'head_extension', 'head_twist']],
  ['Shoulders', ['shoulder_r_x', 'shoulder_l_x', 'shoulder_r_y', 'shoulder_l_y', 'shoulder_r_z', 'shoulder_l_z',
                 'scapula_abduction_r', 'scapula_abduction_l', 'scapula_elevation_r', 'scapula_elevation_l', 'scapula_upward_rot_r', 'scapula_upward_rot_l']],
  ['Elbows / wrists', ['elbow_flexion_r', 'elbow_flexion_l', 'pro_sup_r', 'pro_sup_l', 'wrist_flexion_r', 'wrist_flexion_l', 'wrist_deviation_r', 'wrist_deviation_l']],
];
// DOFs whose sign is opposite between sides (mirroring flips the sign)
const MIRROR_FLIP = (n) => /^shoulder_[rl]_x$/.test(n) || /^scapula_(elevation|abduction)_[rl]$/.test(n);
const sideOf = (n) => (/(_r$|_r_)/.test(n) ? 'R' : /(_l$|_l_)/.test(n) ? 'L' : '');
const counterpart = (n) => sideOf(n) === 'R' ? n.replace(/_r$/, '_l').replace(/_r_/, '_l_') : sideOf(n) === 'L' ? n.replace(/_l$/, '_r').replace(/_l_/, '_r_') : null;
const dofLabel = (n) => n.replace(/_[rl]$/, '').replace(/_[rl]_/, '_').replace(/_/g, ' ');

let poseApi = null, qLive = null, sliders = {}, ghostGroup = null, poseTimer = null, poseBusy = false, poseQueued = false;
const rad = (d) => d * Math.PI / 180, deg = (r) => r * 180 / Math.PI;

async function initPoseUI() {
  try {
    const info = await (await fetch('api/info', { cache: 'no-store' })).json();
    if (!info.available) return;
    poseApi = info;
  } catch (e) { return; }
  $('pose_section').hidden = false;
  $('pose_blend_row').hidden = true;                 // live mode: sliders instead of blending
  $('pose_actions').hidden = false;
  qLive = poseApi.q_fit.slice();
  const support = poseApi.dof_support || poseApi.names.map(() => true);
  const box = $('pose_groups'); box.innerHTML = '';
  for (const [title, dofs] of DOF_GROUPS) {
    const det = document.createElement('details');
    const sum = document.createElement('summary'); sum.textContent = title; det.appendChild(sum);
    let anySupported = false;
    for (const name of dofs) {
      const i = poseApi.names.indexOf(name); if (i < 0) continue;
      const [lo, hi] = poseApi.limits_deg[name];
      const row = document.createElement('div'); row.className = 'dof' + (support[i] ? '' : ' off');
      row.title = support[i] ? name : `${name} — no CT data for this bone (moves freely)`;
      anySupported = anySupported || support[i];
      const lab = document.createElement('span'); lab.textContent = dofLabel(name);
      const side = document.createElement('span'); side.className = 'side'; side.textContent = sideOf(name);
      const sl = document.createElement('input'); sl.type = 'range'; sl.min = Math.min(lo, deg(qLive[i]) - 1); sl.max = Math.max(hi, deg(qLive[i]) + 1); sl.step = 1; sl.value = deg(qLive[i]).toFixed(0);
      const val = document.createElement('span'); val.className = 'val'; val.textContent = `${deg(qLive[i]).toFixed(0)}°`;
      sl.addEventListener('input', () => setDof(i, rad(parseFloat(sl.value)), true));
      sl.addEventListener('dblclick', () => setDof(i, poseApi.q_fit[i], true));
      row.append(lab, side, sl, val); det.appendChild(row);
      sliders[name] = { sl, val, i };
    }
    det.open = title === 'Hips' || title === 'Knees';
    if (dofs.some((n) => poseApi.names.includes(n))) box.appendChild(det);
  }
  // saved poses -> select
  const sel = $('pose_select'); sel.innerHTML = '';
  const o = document.createElement('option'); o.value = ''; o.textContent = 'fitted (CT pose)'; sel.appendChild(o);
  for (const p of poseApi.poses) { const op = document.createElement('option'); op.value = p.name; op.textContent = p.name; sel.appendChild(op); }
  sel.onchange = () => {
    const p = poseApi.poses.find((x) => x.name === sel.value);
    setQ(p ? p.q.slice() : poseApi.q_fit.slice());
  };
  $('pose_reset').addEventListener('click', () => { sel.value = ''; setQ(poseApi.q_fit.slice()); });
  $('pose_save').addEventListener('click', async () => {
    const name = prompt('Pose name', sel.value || 'pose 1'); if (!name) return;
    const r = await (await fetch('api/save', { method: 'POST', body: JSON.stringify({ name, q: qLive }) })).json();
    if (r.saved) { poseApi.poses = poseApi.poses.filter((p) => p.name !== name).concat([{ name, q: qLive.slice() }]);
      const op = document.createElement('option'); op.value = name; op.textContent = name; sel.appendChild(op); sel.value = name;
      $('pose_status').textContent = `saved "${name}" to poses.json`; }
  });
  $('pose_export').addEventListener('click', async () => {
    const name = prompt('Export folder name (out/poses/<name>)', sel.value || 'pose 1'); if (!name) return;
    $('pose_status').textContent = 'exporting STL…';
    const r = await (await fetch('api/export', { method: 'POST', body: JSON.stringify({ name, q: qLive }) })).json();
    $('pose_status').textContent = r.dir ? `exported ${r.files.length} STL files to ${r.dir}` : `export failed: ${r.error}`;
  });
  $('pose_ghost').addEventListener('change', (e) => { setGhost(e.target.checked); });
  if ($('pose_scapula')) $('pose_scapula').addEventListener('change', () => requestPose());
  // ?pose=<name> in live mode: apply the saved pose through the FK server
  const qp = new URLSearchParams(location.search).get('pose');
  if (qp) { const p = poseApi.poses.find((x) => x.name === qp); if (p) { sel.value = qp; setQ(p.q.slice()); } }
}

function refreshSliders() {
  for (const [name, o] of Object.entries(sliders)) { o.sl.value = deg(qLive[o.i]).toFixed(0); o.val.textContent = `${deg(qLive[o.i]).toFixed(0)}°`; }
}
function setDof(i, value, fromSlider) {
  qLive[i] = value;
  const name = poseApi.names[i];
  if ($('pose_mirror').checked) {
    const other = counterpart(name);
    if (other) { const j = poseApi.names.indexOf(other); if (j >= 0) qLive[j] = MIRROR_FLIP(name) ? -value : value; }
  }
  refreshSliders();
  requestPose();
}
function setQ(q) { qLive = q; refreshSliders(); requestPose(); }

function requestPose() {
  clearTimeout(poseTimer);
  poseTimer = setTimeout(async () => {
    if (poseBusy) { poseQueued = true; return; }
    poseBusy = true;
    try {
      const same = qLive.every((v, i) => Math.abs(v - poseApi.q_fit[i]) < 1e-6);
      if (same) { liveM = null; }
      else {
        const scap = $('pose_scapula') && $('pose_scapula').checked;
        const r = await (await fetch('api/pose', { method: 'POST', body: JSON.stringify({ q: qLive, scapula_auto: scap }) })).json();
        if (scap && r.q) { qLive = r.q; refreshSliders(); }      // scapula DOFs were adjusted by the server
        liveM = r.M.map((m) => new THREE.Matrix4().set(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]));
      }
      applyAll();
    } catch (e) { $('pose_status').textContent = 'pose server error: ' + e; }
    poseBusy = false;
    if (poseQueued) { poseQueued = false; requestPose(); }
  }, 40);
}

// translucent copy of the patient model in the CT pose, for comparison while posing
function setGhost(on) {
  if (ghostGroup) { root.remove(ghostGroup); ghostGroup = null; }
  if (!on) return;
  ghostGroup = new THREE.Group();
  for (const it of items) {
    if (it.group !== 'ct' || !it.visible || it.kind === 'bone_all') continue;
    const g = it.mesh.geometry.clone();
    g.attributes.position.array.set(it.mesh.userData.basePos); g.attributes.position.needsUpdate = true; g.computeVertexNormals();
    const m = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: 0x9aa0ab, transparent: true, opacity: it.kind === 'skin' ? 0.08 : 0.25, depthWrite: false, side: THREE.DoubleSide }));
    ghostGroup.add(m);
  }
  root.add(ghostGroup);
}

// ------------------------------------------------------------------ CT slice overlay
let volume = null;           // {shape:[nY,nZ,nX], spacing:[sx,sy,sz], origin:[x0,y0,z0], data:Uint8Array, hu_min, hu_max}
let slicePlane = null, slicePlane2 = null, sliceTex = null, sliceRGBA = null;
const WINDOWS = { soft: [40, 400], bone: [400, 1800], lung: [-600, 1500], full: null };

async function loadVolume() {
  const meta = manifest.ct_volume;
  if (!meta || !meta.file) return;
  status(`loading CT volume ${meta.shape.join('x')} …`);
  const buf = await (await fetch(meta.file, { cache: 'no-store' })).arrayBuffer();
  volume = { shape: meta.shape, spacing: meta.spacing_mm, origin: meta.origin_mm, data: new Uint8Array(buf),
             hu_min: meta.hu_min, hu_max: meta.hu_max };
  const [nY, nZ, nX] = volume.shape;
  const [sx, sy, sz] = volume.spacing;
  sliceRGBA = new Uint8Array(nX * nZ * 4);
  sliceTex = new THREE.DataTexture(sliceRGBA, nX, nZ, THREE.RGBAFormat, THREE.UnsignedByteType);
  sliceTex.colorSpace = THREE.SRGBColorSpace;
  sliceTex.magFilter = THREE.LinearFilter; sliceTex.minFilter = THREE.LinearFilter;
  const geom = new THREE.PlaneGeometry(nX * sx, nZ * sz);
  geom.rotateX(-Math.PI / 2);                     // lies in the XZ plane; local +y -> world -z
  const mat = new THREE.MeshBasicMaterial({ map: sliceTex, transparent: true, opacity: 0.95, side: THREE.DoubleSide,
                                           depthWrite: true, alphaTest: 0.5, polygonOffset: true, polygonOffsetFactor: 1 });
  slicePlane = new THREE.Mesh(geom, mat); slicePlane.renderOrder = -1; root.add(slicePlane);
  slicePlane2 = new THREE.Mesh(geom, mat); slicePlane2.renderOrder = -1; gSKEL.add(slicePlane2);
  $('slice_section').hidden = false;
  $('slice').max = nY - 1;
  state.slice = Math.floor(nY / 2); $('slice').value = state.slice;     // start mid-volume
  $('panel2d').hidden = false;
}

function sliceLUT() {
  const lut = new Uint8Array(256);
  const w = WINDOWS[state.window];
  const span = volume.hu_max - volume.hu_min;
  for (let v = 0; v < 256; v++) {
    const hu = volume.hu_min + (v / 255) * span;
    const t = w ? (hu - (w[0] - w[1] / 2)) / w[1] : v / 255;
    lut[v] = Math.max(0, Math.min(255, Math.round(t * 255)));
  }
  return lut;
}

function sliceY() { return volume.origin[1] + state.slice * volume.spacing[1]; }

function updateSlice() {
  if (!volume) return;
  const [nY, nZ, nX] = volume.shape;
  const [sx, sy, sz] = volume.spacing;
  const j = Math.max(0, Math.min(nY - 1, state.slice));
  const y = sliceY();
  $('slice_v').textContent = `${y.toFixed(0)} mm`;
  const lut = sliceLUT();
  const base = j * nZ * nX;
  // air (< -900 HU) is transparent in the 3D plane so that the plane does not hide the scene
  const airCut = Math.round(((-900 - volume.hu_min) / (volume.hu_max - volume.hu_min)) * 255);
  // texture rows are bottom-first (v=0) and the plane's local +y maps to world -z,
  // so texture row t holds slice row (nZ-1-t): world z then increases with the slice row index
  for (let t = 0; t < nZ; t++) {
    const r = nZ - 1 - t;
    const src = base + r * nX, dst = t * nX * 4;
    for (let c = 0; c < nX; c++) {
      const raw = volume.data[src + c], g = lut[raw];
      const o = dst + c * 4;
      sliceRGBA[o] = g; sliceRGBA[o + 1] = g; sliceRGBA[o + 2] = g; sliceRGBA[o + 3] = raw < airCut ? 0 : 255;
    }
  }
  sliceTex.needsUpdate = true;
  const cx = volume.origin[0] + (nX * sx) / 2 - sx / 2, cz = volume.origin[2] + (nZ * sz) / 2 - sz / 2;
  slicePlane.position.set(cx, y, cz);
  slicePlane2.position.set(cx, y, cz);
  slicePlane.material.opacity = state.sliceOp;
  slicePlane.visible = state.sliceShow;
  slicePlane2.visible = state.sliceShow && state.mode === 'side';
  if (state.sliceClip) { clipPlane.constant = y + 0.5; $('clip_v').textContent = `${y.toFixed(0)} (slice)`; }
  $('panel2d').hidden = !state.slicePanel;
  if (state.slicePanel) drawPanel(j, y, lut);
}

// mesh ∩ plane(y = h) -> flat array [x1,z1,x2,z2, ...] of segments in the mesh's own (un-exploded) coordinates
function meshContour(geometry, h) {
  const pos = geometry.attributes.position.array;
  const idx = geometry.index ? geometry.index.array : null;
  const nTri = idx ? idx.length / 3 : pos.length / 9;
  const segs = [];
  const px = [0, 0, 0], py = [0, 0, 0], pz = [0, 0, 0];
  for (let t = 0; t < nTri; t++) {
    for (let k = 0; k < 3; k++) {
      const vi = idx ? idx[t * 3 + k] : t * 3 + k;
      px[k] = pos[vi * 3]; py[k] = pos[vi * 3 + 1]; pz[k] = pos[vi * 3 + 2];
    }
    const ymin = Math.min(py[0], py[1], py[2]), ymax = Math.max(py[0], py[1], py[2]);
    if (h < ymin || h > ymax || ymin === ymax) continue;
    let n = 0;
    for (let k = 0; k < 3; k++) {
      const a = k, b = (k + 1) % 3;
      if ((py[a] - h) * (py[b] - h) > 0 || py[a] === py[b]) continue;
      const s = (h - py[a]) / (py[b] - py[a]);
      segs.push(px[a] + s * (px[b] - px[a]), pz[a] + s * (pz[b] - pz[a]));
      if (++n === 2) break;
    }
    if (n === 1) segs.length -= 2;
  }
  return segs;
}

function drawPanel(j, y, lut) {
  const [nY, nZ, nX] = volume.shape;
  const [sx, sy, sz] = volume.spacing;
  const cv = $('slice_canvas');
  const W = 360, H = Math.round(W * (nZ * sz) / (nX * sx));
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const ctx = cv.getContext('2d');
  // slice image: anterior (max z) at the top, patient left (+x) on the right (radiological convention)
  const off = document.createElement('canvas'); off.width = nX; off.height = nZ;
  const img = off.getContext('2d').createImageData(nX, nZ);
  const base = j * nZ * nX;
  for (let r = 0; r < nZ; r++) {
    const row = nZ - 1 - r;
    for (let c = 0; c < nX; c++) {
      const g = lut[volume.data[base + row * nX + c]], o = (r * nX + c) * 4;
      img.data[o] = g; img.data[o + 1] = g; img.data[o + 2] = g; img.data[o + 3] = 255;
    }
  }
  off.getContext('2d').putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, H);
  const x0 = volume.origin[0] - sx / 2, z0 = volume.origin[2] - sz / 2, wx = nX * sx, wz = nZ * sz;
  const toPx = (x) => ((x - x0) / wx) * W, toPy = (z) => H - ((z - z0) / wz) * H;
  if (state.contours !== 'none') {
    for (const it of items) {
      if (!it.visible || it.kind === 'bone_all') continue;
      if (posed) continue;                                          // the CT slice is in the CT pose: no posed contours
      if (state.contours === 'ct' && it.group !== 'ct') continue;
      if (state.contours === 'skel' && it.group !== 'skel') continue;
      const segs = meshContour(it.mesh.geometry, y);
      if (!segs.length) continue;
      ctx.strokeStyle = it.group === 'ct' ? '#5cc8ff' : '#ffb454';
      ctx.lineWidth = it.kind === 'skin' ? 2 : 1.2;
      ctx.setLineDash(it.group === 'skel' && it.kind === 'skin' ? [5, 3] : []);
      ctx.beginPath();
      for (let i = 0; i < segs.length; i += 4) { ctx.moveTo(toPx(segs[i]), toPy(segs[i + 1])); ctx.lineTo(toPx(segs[i + 2]), toPy(segs[i + 3])); }
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }
  const js = (manifest.joints && manifest.joints.skel) || {};
  for (const [name, p] of Object.entries(js)) {
    if (Math.abs(p[1] - y) > 15) continue;                      // joints within 15 mm of the slice
    ctx.fillStyle = '#ff4d4d'; ctx.beginPath(); ctx.arc(toPx(p[0]), toPy(p[2]), 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.font = '10px system-ui'; ctx.fillText(name, toPx(p[0]) + 5, toPy(p[2]) - 4);
  }
  ctx.fillStyle = 'rgba(255,255,255,.7)'; ctx.font = '11px system-ui';
  ctx.fillText('A', W / 2 - 4, 12); ctx.fillText('P', W / 2 - 4, H - 4); ctx.fillText('R', 4, H / 2); ctx.fillText('L', W - 12, H / 2);
  $('panel_info').textContent = `axial y = ${y.toFixed(0)} mm · slice ${j + 1}/${nY} · ${state.window}`;
}

$('slice').addEventListener('input', () => { state.slice = parseInt($('slice').value, 10); applyAll(); });
$('window').addEventListener('change', (e) => { state.window = e.target.value; applyAll(); });
$('slice_op').addEventListener('input', (e) => { state.sliceOp = parseFloat(e.target.value); $('slice_op_v').textContent = state.sliceOp.toFixed(2); applyAll(); });
$('slice_show').addEventListener('change', (e) => { state.sliceShow = e.target.checked; applyAll(); });
$('slice_clip').addEventListener('change', (e) => { state.sliceClip = e.target.checked; if (!state.sliceClip) { state.clip = 1; $('clip').value = 1; } applyAll(); });
$('slice_panel').addEventListener('change', (e) => { state.slicePanel = e.target.checked; applyAll(); });
$('contour_set').addEventListener('change', (e) => { state.contours = e.target.value; applyAll(); });

// ------------------------------------------------------------------ loop
function animate() { requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
animate();
load().catch((e) => { console.error(e); status('failed to load manifest.json: ' + e); });
