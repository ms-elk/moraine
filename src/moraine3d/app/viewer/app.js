// Moraine three.js viewer.
// Speaks the Viewer Protocol (DESIGN 7.2) over WebSocket; renders cached
// GLBs (PBR materials included); shows image-output previews;
// auto-generates the parameter panel from param_schema and sends set_param;
// shows the project picker from the workspace message and sends set_asset;
// answers screenshot_request; places deixis pins with DCC selection
// gestures (a plain click that is not an orbit drag selects the
// clicked primitive alone, a background click clears, shift-click
// adds/removes) and renders the pins broadcast as occludable
// highlights — the pinned primitive, or a predicate pin's matched
// polygons — derived from the displayed geometry so they follow shape
// changes, with a marker sphere only where no highlight can resolve,
// plus a jump-to-pin list.
// WebGLRenderer for now (WebGPURenderer swap is a client-internal
// concern, invisible to the protocol).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const hud = document.getElementById('hud');
const picker = document.getElementById('picker');
const pickerBox = document.getElementById('picker-box');
const pickerMark = document.getElementById('picker-mark');
// Roster projects changed on disk since last displayed (the workspace
// message's `changed`); rendered beside the closed picker against the
// server-reported active project, which arrives after the roster on
// connect — so the mark is derived, never computed once on receipt.
let changedProjects = new Set();

function renderPickerMark() {
  const others = [...changedProjects].filter(n => n !== currentProject);
  pickerMark.textContent = others.length ? '●' : '';
  pickerMark.title = others.length
    ? `changed since last displayed: ${others.join(', ')}` : '';
}
const panel = document.getElementById('panel');
const acceptedBox = document.getElementById('accepted');
const pinsEl = document.getElementById('pins');
const timelineEl = document.getElementById('timeline');
const imagesEl = document.getElementById('images');
const canvas = document.getElementById('app');

// The rig's one fixed exposure (DESIGN 7.1): the lights of `addLightRig`
// sum to about 1.5x on a Lambert surface facing the key, so this scale
// puts base color 1.0 there just below clipping and 0.5 at 0.5 -
// reflectance reads as itself in every shot. Linear, not a filmic
// curve: a curve's shoulder or black offset would move the mid and dark
// tones the human compares against the authored factors. Re-measure
// with three matte spheres (base color 0.18 / 0.5 / 1.0) whenever a
// light changes.
const RIG_EXPOSURE = 0.64;

function calibrateExposure(r) {
  r.toneMapping = THREE.LinearToneMapping;
  r.toneMappingExposure = RIG_EXPOSURE;
}

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
enableShadows(renderer);
calibrateExposure(renderer);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x23252a);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
camera.position.set(2.5, 2, 3);

const controls = new OrbitControls(camera, renderer.domElement);
// Undamped, stock mapping — the DESIGN 7.1 camera contract. Do not
// re-enable damping: the camera must stop when the hand stops.
// Modifiers never remap the camera (same contract): stock OrbitControls
// turns a ctrl/meta/shift left-drag into a pan, colliding with
// shift-click additive selection — a multi-pick with a few pixels of
// hand jitter would silently pan instead of picking. Shadow the
// modifier getters on the event before the controls' bubble-phase
// handler reads them (capture runs first at the target); the picker
// reads its modifiers off pointerup, so selection is untouched, and
// pinch zoom (ctrl+wheel) is a separate path this never sees.
canvas.addEventListener('pointerdown', (e) => {
  for (const k of ['shiftKey', 'ctrlKey', 'metaKey']) {
    Object.defineProperty(e, k, { value: false });
  }
}, true);

// Lighting rig (the standard rig of DESIGN 7.1/7.2): hemisphere + key from
// the front quadrant (assets face +Z, DESIGN 4.3) + a soft fill from the
// back quadrant — the fill kills the hard terminator band the single key
// showed on smooth cylinders without flattening the form — plus a neutral
// environment map, so metallic and glossy surfaces reflect something and
// read as material rather than as silhouettes, and a ground shadow
// catcher at y = 0: the key casts, the catcher receives, and a shot gains
// the contact and scale cues a render with a ground plane has. One
// constructor for both the interactive view and the screenshot scene:
// the agent's evidence renders must be lit exactly as the human sees, by
// construction. Returns the parts `fitRig` re-aims per load.
const KEY_DIR = new THREE.Vector3(3, 5, 2).normalize();

function enableShadows(r) {
  r.shadowMap.enabled = true;
  r.shadowMap.type = THREE.PCFSoftShadowMap;
  // The casters and the key move only on load (`fitRig`), so the depth
  // pass is rendered on demand there, not on every animation frame.
  r.shadowMap.autoUpdate = false;
}

function addLightRig(target, r) {
  target.add(new THREE.HemisphereLight(0xdde3ee, 0x44464c, 1.0));
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.copy(KEY_DIR).multiplyScalar(6);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  target.add(key);
  target.add(key.target);
  const fill = new THREE.DirectionalLight(0xbfd0e6, 0.8);
  fill.position.set(-4, 2.5, -3);
  target.add(fill);
  // A visible, lit ground: a shade above the background, so the shadow
  // it receives reads (a shadow-only catcher on this dark background
  // was invisible) and its horizon gives the shot a ground line. A hair
  // below y = 0 so an asset's flush underside never z-fights it.
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    // Chrome, like the grid: lit, but outside the asset exposure, so it
    // keeps the shade above the background it was tuned to.
    new THREE.MeshStandardMaterial({
      color: 0x33363c, roughness: 0.95, metalness: 0.0, toneMapped: false,
    }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -5e-4;
  ground.receiveShadow = true;
  target.add(ground);
  target.environment = environmentFor(r);
  target.environmentIntensity = 0.8;
  return { key, ground };
}

// The key's orthographic shadow frustum and the ground catcher follow
// the asset bounds on every load: a fixed frustum would blur a fitting
// or clip a yard. Direction is fixed (the rig is one rig); only the
// framing moves. The ground is viewer chrome, never an occluder of file
// state: it hides while the asset extends below y = 0 (a waived
// foundation, a terrain slab) so nothing in the file is drawn behind it.
const GROUND_TOL = 1e-3;

function fitRig(rig, box, r) {
  const center = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getBoundingSphere(new THREE.Sphere()).radius, 0.5);
  rig.key.target.position.copy(center);
  rig.key.position.copy(center).addScaledVector(KEY_DIR, radius * 3);
  const cam = rig.key.shadow.camera;
  cam.left = cam.bottom = -radius * 1.25;
  cam.right = cam.top = radius * 1.25;
  cam.near = radius * 0.5;
  cam.far = radius * 6;
  cam.updateProjectionMatrix();
  // The bias scales with the frustum it depth-tests in: a couple of
  // shadow-map texels (2.5 * radius over 2048), not a fixed distance,
  // so a contact shadow meets its caster at every asset size instead
  // of detaching by centimeters on a small part (a lit gap under a
  // grounded object reads as floating).
  rig.key.shadow.normalBias = radius * 0.003;
  rig.key.shadow.bias = -0.0002;
  rig.ground.scale.set(radius * 12, radius * 12, 1);
  rig.ground.position.x = center.x;
  rig.ground.position.z = center.z;
  rig.ground.visible = box.min.y >= -GROUND_TOL;
  r.shadowMap.needsUpdate = true;
}

function markShadowCasters(root) {
  root.traverse((obj) => {
    if (obj.isMesh) { obj.castShadow = true; obj.receiveShadow = true; }
  });
}

// A neutral studio for the environment map: gray walls and three soft
// panels (overhead, front-left, right), pre-filtered by PMREMGenerator.
// One map per renderer — a GL texture belongs to the context that made
// it, and the shot renderer is a second context.
const environments = new Map();

function environmentFor(r) {
  if (!environments.has(r)) {
    // Light-gray walls over a slightly darker floor, with soft panels:
    // a metallic side seen from above mirrors the floor and lower
    // walls, so those must carry light too, or the part reads as a
    // silhouette with a bright lid; a mild top-to-bottom gradient is
    // what makes metal read as metal.
    const room = new THREE.Scene();
    const walls = new THREE.Mesh(
      new THREE.BoxGeometry(20, 12, 20),
      new THREE.MeshBasicMaterial({ color: 0xa4a8b0, side: THREE.BackSide }),
    );
    walls.position.y = 5;
    room.add(walls);
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(19.9, 19.9),
      new THREE.MeshBasicMaterial({ color: 0x74787f }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.95;
    room.add(floor);
    const panel = (w, h, pos, level) => {
      const m = new THREE.Mesh(
        new THREE.PlaneGeometry(w, h),
        new THREE.MeshBasicMaterial({ color: new THREE.Color().setScalar(level) }),
      );
      m.position.copy(pos);
      m.lookAt(0, 1.0, 0);
      room.add(m);
    };
    panel(6, 6, new THREE.Vector3(0, 10.5, 0.01), 2.5);   // overhead
    panel(4, 4, new THREE.Vector3(-7, 4, 6), 4.0);        // front-left key
    panel(3, 5, new THREE.Vector3(8, 3, 2), 2.5);         // right fill
    panel(6, 3, new THREE.Vector3(2, 3, -9), 2.0);        // back rim
    const pmrem = new THREE.PMREMGenerator(r);
    environments.set(r, pmrem.fromScene(room, 0.04).texture);
    pmrem.dispose();
  }
  return environments.get(r);
}

const mainRig = addLightRig(scene, renderer);
const grid = new THREE.GridHelper(10, 10, 0x55585f, 0x3a3d44);
grid.material.toneMapped = false;
scene.add(grid);
const axes = new THREE.AxesHelper(0.5);
axes.material.toneMapped = false;
scene.add(axes);

const loaded = new THREE.Group();
scene.add(loaded);

// Pin markers and their face-region highlights live beside the loaded parts,
// never under them: excluded from the deixis raycast, the auto-frame
// bounds, and (like all viewer chrome) from screenshots, which render
// GLBs in a throwaway scene. Markers are screen-scaled per frame;
// highlights are world-scale overlays, so they sit in separate groups.
const pinsGroup = new THREE.Group();
scene.add(pinsGroup);
const pinHighlights = new THREE.Group();
scene.add(pinHighlights);

const loader = new GLTFLoader();
let framed = false;
let lastAsset = null;
let currentProject = null;

function resize() {
  // A tab can be laid out at 0x0 (backgrounded/hidden at load): a 0/0
  // aspect is NaN and poisons the projection matrix for every later
  // raycast and render — clamp to 1px until a real size arrives.
  const w = Math.max(1, window.innerWidth);
  const h = Math.max(1, window.innerHeight);
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

renderer.setAnimationLoop(() => {
  controls.update();
  // Markers keep a constant screen size: unit geometry scaled by
  // camera distance every frame (scaleK carries the hover emphasis).
  for (const m of pinsGroup.children) {
    m.scale.setScalar(
      m.position.distanceTo(camera.position) * 0.008 * (m.userData.scaleK || 1),
    );
  }
  renderer.render(scene, camera);
});

// Sticky once set: the server's watch task died, so this display stops
// updating on file edits — say so on every HUD repaint until restart.
let watchDead = false;
// Sticky per outage: the server's listener died while this socket
// still streams — a half-dead server only its open tabs can hear.
let listenDead = false;
// Sticky file-state failure: the display is the last good state and
// this says why it is not the file. Its own slot, never the state line:
// a reload's GLB load finishing after the error would otherwise repaint
// the state line to "connected" over a file that is still broken.
// Cleared only by an asset_updated whose load lands at the current
// generation, or by a fresh connection (the server replays a standing
// error; a restarted server without one replays nothing to retract).
let assetError = '';
// Last state line, so a repaint without a new state (e.g. watch_dead
// arriving) preserves what is showing — an asset-error must survive it.
let hudState = 'connecting';
// Transient HUD notice, rendered alongside whatever state follows it:
// setHud('connected') after a reload's GLB load must not swallow the
// "your preview was reset" announcement that raced ahead of it.
let hudNotice = '';
let hudNoticeTimer = null;

function setNotice(text) {
  hudNotice = text;
  clearTimeout(hudNoticeTimer);
  hudNoticeTimer = setTimeout(() => { hudNotice = ''; setHud(); }, 8000);
  setHud();
}

// Asset totals, not a per-output list (DESIGN 7.2): the list is
// declaration-ordered and runs off the window long before it runs out
// of outputs, so at asset scale the HUD shows an arbitrary handful.
// The three totals stay true at every scale. Image outputs are absent
// by the same rule that keeps them out of `parts` — the thumbnail
// strip is their surface.
function assetTotals(s) {
  if (!s) return '';
  const n = (v) => v.toString().replace(/\B(?=(\d{3})+$)/g, ',');
  const parts = `${n(s.parts)} ${s.parts === 1 ? 'part' : 'parts'}`;
  if (!s.parts) return parts;
  const mats = s.materials ? `  ·  ${n(s.materials.count)} ` +
    `${s.materials.count === 1 ? 'material' : 'materials'}` : '';
  return `${parts}  ·  ${n(s.tris)} tris${mats}`;
}

function setHud(state) {
  if (state !== undefined) hudState = state;
  const stats = assetTotals(lastAsset);
  const project = currentProject ? `${currentProject} — ` : '';
  const note = hudNotice ? ` — ${hudNotice}` : '';
  const broken = assetError ? `\n${assetError}` : '';
  const dead = watchDead ? '\nwatch dead — re-arming; display may be stale' : '';
  const closed = listenDead
    ? '\nlistener dead — re-arming; new tabs and commands cannot connect' : '';
  hud.textContent =
    `Moraine — ${project}${hudState}${note}${broken}${dead}${closed}\n${stats}`;
}

function assetBounds(group = loaded) {
  const box = new THREE.Box3();
  if (group.children.length) box.setFromObject(group);
  if (box.isEmpty()) { box.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(1, 1, 1)); }
  return box;
}

function frameCamera(cam, dir, up) {
  const box = assetBounds();
  const center = box.getCenter(new THREE.Vector3());
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const dist = Math.max(sphere.radius, 0.1) / Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)) * 1.35;
  cam.position.copy(center).addScaledVector(dir.clone().normalize(), dist);
  cam.up.copy(up);
  cam.lookAt(center);
  cam.updateProjectionMatrix();
  return center;
}

let loadGen = 0;
// A failed load asks again on this schedule (ms), abandoned the moment a
// newer generation lands: a record can vanish under a running server (a
// machine-wide `cache clear` from another process), and the server
// re-materializes it on the 404 — but a tab whose load already failed
// has to fetch again to see the rebuilt file.
const LOAD_RETRY_MS = [1000, 3000, 8000];
let loadRetry = null;

async function loadOutputs(glbUrls, retry = null, attempt = 0) {
  // Generation guard: rapid edits can interleave loads, and without it a
  // slow older load finishing last would overwrite the newer asset.
  // Returns whether this load is still the current generation, so the
  // caller can gate everything else derived from the same message.
  const gen = ++loadGen;
  clearTimeout(loadRetry);
  const next = [];
  try {
    for (const [name, url] of Object.entries(glbUrls)) {
      const gltf = await loader.loadAsync(url);
      gltf.scene.name = name;
      next.push(gltf.scene);
    }
  } catch (err) {
    if (gen === loadGen) {
      setHud(`load error: ${err.message || err}`);
      if (retry && attempt < LOAD_RETRY_MS.length) {
        loadRetry = setTimeout(() => {
          if (gen === loadGen) retry(attempt + 1);
        }, LOAD_RETRY_MS[attempt]);
      }
    }
    return false;
  }
  if (gen !== loadGen) return false;
  // Each GLTFLoader parse creates its own geometries/materials/textures
  // and Object3D.clear() only detaches: release the outgoing generation's
  // GPU resources, or every reload (edit, slider step) pins them for the
  // life of the tab.
  disposeDeep(loaded);
  loaded.clear();
  for (const obj of next) { markShadowCasters(obj); loaded.add(obj); }
  fitRig(mainRig, assetBounds(), renderer);
  // Loaded materials carry the last-evaluated factors: re-apply the live
  // factor-tier uniforms so a load never undoes an active scrub, and a
  // dropped overlay's file values land (the store mirrors the latest
  // schema; idempotent when the GLB already bakes them).
  applyStoredUniforms();
  if (!framed && next.length) {
    const center = frameCamera(camera, new THREE.Vector3(1, 0.8, 1), new THREE.Vector3(0, 1, 0));
    controls.target.copy(center);
    framed = true;
  }
  return true;
}

function renderImages(imageUrls) {
  imagesEl.replaceChildren();
  for (const [name, url] of Object.entries(imageUrls || {})) {
    const fig = document.createElement('figure');
    const img = document.createElement('img');
    img.src = url;
    img.alt = name;
    const cap = document.createElement('figcaption');
    cap.textContent = name;
    fig.append(img, cap);
    imagesEl.append(fig);
  }
}

// -- parameter panel (param_schema -> controls -> set_param) ----------------
// Tier dispatch (DESIGN 7.2): factor knobs drive material uniforms live
// and send one coalesced set_param per gesture (no evaluation
// server-side); evaluation knobs stage into a pending map behind an
// explicit Apply — one batch set_param, one evaluation; seed knobs are
// reroll-with-history controls, because a seed is unordered and a
// slider is the wrong control for it.

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

function fmt(v) {
  return Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
}

function valueEq(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function outsideRange(p) {
  // The asset's own value is reported, not refused (DESIGN 7.2): the
  // schema can carry a number or vector knob outside its declared range.
  if (typeof p.min !== 'number' || typeof p.max !== 'number') return false;
  return [].concat(p.value).some(c => c < p.min || c > p.max);
}

function intoRange(p, v) {
  // One component pulled into the declared range; a knob with an
  // integral step rounds, so a typed 2.5 cannot stage a value the op's
  // integer validation rejects at Apply.
  const c = Math.min(p.max, Math.max(p.min, v));
  return Number.isInteger(p.step) ? Math.round(c) : c;
}

function vectorInRange(p, base) {
  // The server judges a vector whole: a component left outside the
  // range would refuse the edit of any other, and a refused batch
  // reverts every knob staged with it. So the first edit takes every
  // component into range, and the row shows the ones that moved.
  return base.map(c => intoRange(p, c));
}

function renderRangeNote(p, row) {
  // A control's range attributes pull the shown value to the nearest
  // end, so the note carries the number the asset holds. Once a value
  // is staged the note says what the Apply replaces.
  row.querySelector('.note')?.remove();
  const staged = pending.has(p.param_id);
  const outside = outsideRange(p);
  row.classList.toggle('outside', outside && !staged);
  if (!outside) return;
  const note = document.createElement('div');
  note.className = 'note';
  const held = [].concat(p.value).map(fmt).join(', ');
  const range = `the declared ${fmt(p.min)} to ${fmt(p.max)}`;
  note.textContent = staged
    ? `replaces ${held}, outside ${range}`
    : `${held} is outside ${range}`;
  row.append(note);
}

function clearRangeNote(row) {
  // A live scrub has already moved the value into range.
  row.classList.remove('outside');
  row.querySelector('.note')?.remove();
}

// Live factor uniforms: material -> channel -> value, refreshed from
// every schema and every scrub, and re-applied after every GLB load —
// loaded materials carry the last-evaluated factors, which must not
// undo an active factor preview.
const factorUniforms = new Map();

function noteUniform(uniform, value) {
  let channels = factorUniforms.get(uniform.material);
  if (!channels) { channels = new Map(); factorUniforms.set(uniform.material, channels); }
  channels.set(uniform.channel, value);
}

function applyUniform(materialName, channel, value) {
  loaded.traverse((obj) => {
    const mats = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
    for (const m of mats) {
      if (m.name !== materialName) continue;
      if (channel === 'base_color') {
        m.color.setRGB(value[0], value[1], value[2]);
        const a = value.length > 3 ? value[3] : 1.0;
        // Accepted approximation (DESIGN 7.2): live alpha rides
        // transparent+opacity; the next evaluation trues up the real
        // glTF alphaMode.
        const wasTransparent = m.transparent;
        m.opacity = a;
        m.transparent = a < 1.0;
        if (m.transparent !== wasTransparent) m.needsUpdate = true;
      } else if (channel === 'metallic') {
        m.metalness = value;
      } else if (channel === 'roughness') {
        m.roughness = value;
      } else if (channel === 'emissive') {
        // The loader carries a >1 factor as a unit color times
        // emissiveIntensity (KHR_materials_emissive_strength); the live
        // value is the raw factor, so the intensity must not stay on.
        m.emissive.setRGB(value[0], value[1], value[2]);
        m.emissiveIntensity = 1.0;
      } else if (channel === 'normal_scale') {
        // Preserve the loader's component signs (GLTFLoader encodes
        // the UV flip in the sign of normalScale.y). Negative zero
        // counts as negative: a scrub through 0 stores -0, and a falsy
        // check (`|| 1`) would read it as positive on the way back out,
        // flipping the map until the next evaluation.
        const sgn = (c) => (c < 0 || Object.is(c, -0) ? -1 : 1);
        m.normalScale.set(
          value * sgn(m.normalScale.x),
          value * sgn(m.normalScale.y),
        );
      }
    }
  });
}

function applyStoredUniforms() {
  for (const [material, channels] of factorUniforms) {
    for (const [channel, value] of channels) applyUniform(material, channel, value);
  }
}

// Staged evaluation-tier edits (param_id -> value), sent only by the
// Apply button as one batch set_param. Kept across schema rebuilds and
// pruned passively: a schema whose value matches a staged edit means
// the edit landed (or is redundant) — drop it; a rejected Apply snaps
// the schema back to the old values, so its edits stay staged.
const pending = new Map();
let applyBar = null;
let applyButton = null;
// The Apply in flight (DESIGN 7.2): the ids it sent, until its reply —
// the schema that lands them, or the snap-back after a failure. The
// busy state lives on the control itself and on the fields it sent.
const inflight = new Set();
// Staged rows by param id, so an Apply can mark the fields it sent
// without a rebuild (a rebuild under the pointer is deferred).
const stagedRows = new Map();

// Outstanding set_param requests of this tab, oldest first (DESIGN
// 7.2): the server answers one socket's requests in order, one
// terminal reply each — the schema that lands the values, or a
// asset_error whose snap-back schema follows and is not a second
// reply. Requests settle by position, never by value: an earlier
// request's reply (a reroll sent before an Apply) must not settle a
// later one. A dropped overlay or a fresh connection owes nothing.
const ledger = [];
let snapBackPending = false;

function sendParam(msg, entry) {
  send(msg);
  ledger.push(entry);
}

function settle(entry) {
  if (!entry) return;  // a file-state schema, not a reply
  if (entry.kind === 'apply') settleInflight();
  else if (entry.kind === 'seed'
           && !ledger.some(e => e.kind === 'seed' && e.id === entry.id)) {
    seedBusy.delete(entry.id);  // no later pick of this knob is pending
    const sync = seedSync.get(entry.id);
    if (sync) sync();
  }
}

function replyArrived() {
  if (snapBackPending) { snapBackPending = false; return; }
  settle(ledger.shift());
}

function replyFailed() {
  if (!ledger.length) return;  // a file-state error, not a reply
  settle(ledger.shift());
  snapBackPending = true;
}

// The wire asset_error is the same shape for a broadcast file-state
// failure and a per-client set_param rejection, so a non-empty ledger
// alone cannot prove "this is my reply": a file edit can fail between
// this tab's send and the server reading it. Only an error that names a
// param or the overlay is a reply; anything else is file state, and the
// ledger entry it would have settled settles on the snap-back schema.
function isReplyError(error) {
  if (!error) return false;
  const code = String(error.code || '');
  if (code.startsWith('param_')) return true;
  const detail = error.detail || {};
  if ('param' in detail || 'param_id' in detail || 'param_ids' in detail) return true;
  return /\boverride\b/.test(String(error.message || ''));
}


function resetLedger() {
  ledger.length = 0;
  snapBackPending = false;
  seedBusy.clear();
  for (const sync of seedSync.values()) sync();
  settleInflight();
}

function stagedValue(p) {
  // What a staged field shows: the edit, else the asset's value.
  return pending.has(p.param_id) ? pending.get(p.param_id) : p.value;
}

// Reset to defaults (DESIGN 7.1): preview the recipe's default for
// every knob the human's file holds, the unset of DESIGN 7.2, in one
// evaluation. Accept then removes their lines; Discard drops it.
function resetToDefaults() {
  const held = (lastSchema || []).filter(p => p.from === 'accepted');
  if (!held.length || !(ws && ws.readyState === WebSocket.OPEN)) return;
  for (const p of held) {
    if (p.kind === 'vector') vectorState.set(p.param_id, [...p.default]);
  }
  sendParam(
    { type: 'set_param', params: Object.fromEntries(held.map(p => [p.param_id, null])) },
    { kind: 'reset' },
  );
  resetBusy = true;
  renderLanding();
  setHud('evaluating…');
}

function stage(p, value, row) {
  if (valueEq(value, p.value)) pending.delete(p.param_id);
  else pending.set(p.param_id, value);
  // The pending signal lives on the field itself — which knobs an Apply
  // will land — not as a count in the bar (the button's enabled state
  // already says "something is staged").
  if (row) {
    row.classList.toggle('staged', pending.has(p.param_id));
    renderRangeNote(p, row);
  }
  renderApplyBar();
}

function renderApplyBar() {
  // Always present, disabled while nothing is staged: an
  // appearing-on-first-edit control would leave the staged workflow
  // undiscoverable and shift the layout under the pointer. Busy while
  // its evaluation runs: disabled and relabeled, so the click reads
  // as taken.
  if (!applyButton) return;
  applyButton.disabled = !pending.size || inflight.size > 0;
  applyButton.textContent = inflight.size ? 'Applying…' : 'Apply';
}

function settleInflight() {
  // A reply arrived for this tab: the Apply is no longer in flight,
  // whatever it landed (a failed batch snaps every field back and
  // leaves the edits staged).
  for (const id of inflight) stagedRows.get(id)?.classList.remove('inflight');
  inflight.clear();
  renderApplyBar();
}

function buildApplyBar() {
  applyBar = document.createElement('div');
  applyBar.className = 'apply';
  applyButton = document.createElement('button');
  applyButton.textContent = 'Apply';
  applyButton.addEventListener('click', () => {
    if (pending.size && !inflight.size) {
      sendParam({ type: 'set_param', params: Object.fromEntries(pending) }, { kind: 'apply' });
      for (const id of pending.keys()) {
        inflight.add(id);
        stagedRows.get(id)?.classList.add('inflight');
      }
      renderApplyBar();
      setHud('evaluating…');
    }
    applyButton.blur();
  });
  applyBar.append(applyButton);
  return applyBar;
}

// Reroll history per seed knob, client-local like the overlay it feeds:
// values (cap 32) plus a cursor for back/forward.
const seedHistory = new Map();
// Seed knobs with a pick outstanding (DESIGN 7.1): the readout shows
// the pick at the click, and the dice carries busy and the row's three
// buttons stay disabled until the pick's reply arrives. A pick is
// judged by its result, and a seed is unordered, so a pick made before
// the last one shows would only delay the one that gets looked at.
const seedBusy = new Set();
// Each seed row's own refresh, by knob: the reply may land while the
// panel's rebuild is deferred (a focused control elsewhere), and a row
// left disabled until then would take the next pick away.
const seedSync = new Map();

function seedRow(p) {
  let h = seedHistory.get(p.param_id);
  if (!h) {
    h = { values: [p.value], idx: 0 };
    seedHistory.set(p.param_id, h);
  } else if (h.values[h.idx] !== p.value) {
    const at = h.values.lastIndexOf(p.value);
    if (at >= 0) {
      // A value already in the walk: a step back or forward landing
      // (each reply re-renders the panel with the value it evaluated).
      // Move the cursor, keep the walk: resetting here would kill the
      // back/forward history exactly when it is being used.
      h.idx = at;
    } else {
      // A value from outside the walk (a file edit, another tab): the
      // old walk no longer describes reality.
      h = { values: [p.value], idx: 0 };
      seedHistory.set(p.param_id, h);
    }
  }
  const wrap = document.createElement('div');
  wrap.className = 'seed';
  const out = document.createElement('span');
  out.textContent = fmt(h.values[h.idx]);
  const back = document.createElement('button');
  back.textContent = '◀';
  back.title = 'previous seed';
  back.setAttribute('aria-label', 'previous seed');
  // The dice: the procedural-tool convention for "draw a new seed".
  // While a pick is awaited it becomes a spinning glyph — an added,
  // moving thing, the same busy signal the staged fields carry.
  const reroll = document.createElement('button');
  const glyph = document.createElement('span');
  reroll.append(glyph);
  reroll.title = 'reroll';
  reroll.setAttribute('aria-label', 'reroll');
  const syncDice = () => {
    const busy = seedBusy.has(p.param_id);
    glyph.textContent = busy ? '⟳' : '🎲';
    reroll.classList.toggle('busy', busy);
    reroll.disabled = busy;
    back.disabled = busy || h.idx === 0;
    fwd.disabled = busy || h.idx === h.values.length - 1;
  };
  seedSync.set(p.param_id, syncDice);
  const fwd = document.createElement('button');
  fwd.textContent = '▶';
  fwd.title = 'next seed';
  fwd.setAttribute('aria-label', 'next seed');
  // A pick that cannot be sent is not made: the row would wait on a
  // reply nobody owes, and show a seed the server never saw.
  const online = () => ws && ws.readyState === WebSocket.OPEN;
  const sync = (btn) => {
    out.textContent = fmt(h.values[h.idx]);
    sendParam(
      { type: 'set_param', param_id: p.param_id, value: h.values[h.idx] },
      { kind: 'seed', id: p.param_id },
    );
    seedBusy.add(p.param_id);
    syncDice();
    setHud('evaluating…');
    btn.blur();
  };
  syncDice();
  reroll.addEventListener('click', () => {
    if (!online()) return;
    // A user gesture picking an unordered value — client-side
    // Math.random is fine (op-side determinism, DESIGN 4.4, is about
    // streams, and the picked value travels as an explicit set_param).
    const lo = Math.ceil(p.min);
    const hi = Math.floor(p.max);
    let v = h.values[h.idx];
    for (let i = 0; i < 4 && v === h.values[h.idx]; i++) {
      v = lo + Math.floor(Math.random() * (hi - lo + 1));
    }
    h.values = h.values.slice(0, h.idx + 1);
    h.values.push(v);
    if (h.values.length > 32) h.values.shift();
    h.idx = h.values.length - 1;
    sync(reroll);
  });
  back.addEventListener('click', () => {
    if (online() && h.idx > 0) { h.idx--; sync(back); }
  });
  fwd.addEventListener('click', () => {
    if (online() && h.idx < h.values.length - 1) { h.idx++; sync(fwd); }
  });
  wrap.append(out, back, reroll, fwd);
  return wrap;
}

function numberField(p, initial, onCommit) {
  // Evaluation-tier numeric entry (DESIGN 7.2): typed or stepped, never
  // a slider — a slider's drag affordance promises the live response
  // only the factor tier has. Commits clamp to the declared range
  // (the same client-side rule the sliders follow); knobs with any
  // integral step round — the schema gives every integer-valued knob
  // an integral step (declared, or the injected default 1), so a typed
  // 2.5 cannot stage a value the op's integer validation rejects at
  // Apply, taking the rest of the batch down with it.
  const input = document.createElement('input');
  input.type = 'number';
  input.min = p.min;
  input.max = p.max;
  input.step = p.step ?? (p.max - p.min) / 100;
  input.value = initial;
  let last = initial;
  input.addEventListener('change', () => {
    let v = parseFloat(input.value);
    if (!Number.isFinite(v)) { input.value = last; return; }
    v = intoRange(p, v);
    input.value = v;
    last = v;
    onCommit(v);
  });
  return input;
}

function sliderRow(p, initial, onChange, onCommit) {
  const wrap = document.createElement('div');
  wrap.className = 'slider';
  const input = document.createElement('input');
  input.type = 'range';
  input.min = p.min;
  input.max = p.max;
  input.step = p.step ?? (p.max - p.min) / 100;
  input.value = initial;
  const out = document.createElement('span');
  out.textContent = fmt(initial);
  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    out.textContent = fmt(v);
    onChange(v);
  });
  if (onCommit) {
    input.addEventListener('change', () => onCommit(parseFloat(input.value)));
  }
  wrap.append(input, out);
  return wrap;
}

// Latest schema, kept even when a rebuild is deferred; vectorState holds
// each vector param's send base and is refreshed from every arriving
// schema, so a component drag never resurrects values the server already
// replaced (e.g. an overlay dropped by a file edit). Pending edits win
// over the schema until applied or un-staged.
let lastSchema = [];
let panelStale = false;
const vectorState = new Map();

// Param ids this connection's preview overlay holds, from the latest
// schema (DESIGN 7.2 `overridden`): Accept and Discard are enabled by
// it, not by a client-side guess at what the server holds.
let overriddenIds = [];
// The footer's landings, kept across schema rebuilds like the Apply
// bar: their enabled state follows every schema in place, since a
// schema arriving while a knob keeps focus (the tab after a drag)
// defers the panel rebuild until focus leaves - and a human reaching
// for Accept right after a scrub must not find it still disabled.
let revertButton = null;
let resetButton = null;
let acceptButton = null;
// A reset awaiting its reply: one evaluation per click.
let resetBusy = false;

function renderLanding() {
  if (!revertButton) return;
  const nothing = !overriddenIds.length;
  const idle = 'nothing previewed: scrub, apply, or reroll a knob first';
  revertButton.disabled = nothing;
  revertButton.title = nothing ? idle : "discard this tab's preview and show the asset file";
  const held = (lastSchema || []).some(p => p.from === 'accepted');
  resetButton.disabled = !held || resetBusy;
  resetButton.title = held
    ? "preview the recipe's defaults for every accepted knob; Accept then "
      + "removes their lines from params.json"
    : 'no knob holds an accepted value';
  acceptButton.disabled = nothing;
  acceptButton.title = nothing
    ? idle : "write the previewed values to the project's params.json";
}

// The accept warning (DESIGN 7.2): what the server said of the
// project's accepted state, one line, kept up while it is red. The
// server holds the warning with the project and re-sends it on every
// reload, so the box shows the last message and decides nothing. What
// is red and whose move it is are the agent's to tell: the fix is a
// recipe edit, and `validate` gives the agent the entries.
function renderAccepted(msg) {
  acceptedBox.replaceChildren();
  if (!msg || msg.ok) {
    acceptedBox.style.display = 'none';
    return;
  }
  const names = (msg.params || []).join(', ');
  const el = document.createElement('div');
  if (msg.checking) {
    // Between the write and its verdict: validate over a large asset
    // takes a while, and silence would read as a verdict.
    el.className = 'means';
    el.textContent = `accepted: ${names} — validating`;
  } else if (msg.checked === false) {
    // The verdict waits for a state that builds.
    el.className = 'title';
    el.textContent = `⚠ accepted ${names} — the asset does not build ` +
      `(${msg.error.code}) · ask the agent to resolve`;
  } else {
    // The failure leads, the act follows: the line is a warning first.
    const n = (msg.failed || []).length;
    el.className = 'title';
    el.textContent = `⚠ accepted ${names} fails ` +
      `${n === 1 ? 'a check' : `${n} checks`} · ask the agent to resolve`;
  }
  acceptedBox.append(el);
  acceptedBox.style.display = 'block';
}

function schemaArrived(params, overridden) {
  lastSchema = params;
  overriddenIds = overridden || [];
  resetBusy = false;
  renderLanding();
  const ids = new Set(params.map(p => p.param_id));
  for (const id of [...vectorState.keys()]) { if (!ids.has(id)) vectorState.delete(id); }
  for (const id of [...pending.keys()]) { if (!ids.has(id)) pending.delete(id); }
  for (const id of [...seedHistory.keys()]) { if (!ids.has(id)) seedHistory.delete(id); }
  for (const id of [...seedBusy]) { if (!ids.has(id)) seedBusy.delete(id); }
  for (const id of [...seedSync.keys()]) { if (!ids.has(id)) seedSync.delete(id); }
  factorUniforms.clear();
  for (const p of params) {
    if (pending.has(p.param_id) && valueEq(pending.get(p.param_id), p.value)) {
      pending.delete(p.param_id);
    }
    if (p.kind === 'vector') {
      vectorState.set(p.param_id, [...stagedValue(p)]);
    }
    if (p.tier === 'factor' && p.uniform) noteUniform(p.uniform, p.value);
  }
  applyStoredUniforms();
  renderApplyBar();
  renderPanel(params);
}

panel.addEventListener('focusout', () => {
  // After focus settles elsewhere, apply any schema that arrived mid-drag.
  setTimeout(() => {
    if (panelStale && !panel.contains(document.activeElement)) renderPanel(lastSchema);
  }, 0);
});

function renderPanel(params) {
  // Don't rebuild under the user's pointer: while a control is focused
  // (mid-drag), defer; the focusout handler above replays the latest
  // schema once the drag ends.
  if (panel.contains(document.activeElement)) { panelStale = true; return; }
  panelStale = false;
  panel.replaceChildren();
  stagedRows.clear();
  panel.style.display = params.length ? 'block' : 'none';
  // Sections by tier, so the cost model is legible from the layout
  // alone: live knobs first, seeds (instant, one evaluation per pick),
  // then the staged group with its Apply control directly beneath the
  // fields it applies.
  const sections = [
    { caption: 'live', rows: [] },
    { caption: 'seeds', rows: [] },
    { caption: 'staged', apply: true, rows: [] },
  ];
  for (const p of params) {
    const row = document.createElement('div');
    row.className = 'param';
    const label = document.createElement('label');
    label.textContent = p.label;
    // The declared range beside the name, so a field's clamp and a
    // slider's ends are read before the hand moves. A seed carries
    // none: its range is a pick's domain, not a scale.
    if (p.tier !== 'seed' && typeof p.min === 'number' && typeof p.max === 'number') {
      const range = document.createElement('span');
      range.className = 'range';
      range.textContent = `${fmt(p.min)} – ${fmt(p.max)}`;
      label.append(range);
    }
    row.append(label);
    let section = 2;
    if (p.tier === 'seed' && p.kind === 'number') {
      section = 1;
      row.append(seedRow(p));
    } else if (p.tier === 'factor' && p.uniform && p.kind === 'number') {
      section = 0;
      row.append(sliderRow(
        p, p.value,
        v => {
          clearRangeNote(row);
          noteUniform(p.uniform, v);
          applyUniform(p.uniform.material, p.uniform.channel, v);
        },
        v => sendParam({ type: 'set_param', param_id: p.param_id, value: v }, { kind: 'factor' }),
      ));
    } else if (p.tier === 'factor' && p.uniform && p.kind === 'vector') {
      section = 0;
      p.value.forEach((component, i) => {
        row.append(sliderRow(
          p, component,
          v => {
            const value = vectorInRange(p, vectorState.get(p.param_id) || [...p.value]);
            value[i] = v;
            vectorState.set(p.param_id, value);
            row.querySelectorAll('.slider span').forEach((el, j) => {
              el.textContent = fmt(value[j]);
            });
            clearRangeNote(row);
            noteUniform(p.uniform, [...value]);
            applyUniform(p.uniform.material, p.uniform.channel, [...value]);
          },
          () => sendParam({
            type: 'set_param', param_id: p.param_id,
            value: [...(vectorState.get(p.param_id) || p.value)],
          }, { kind: 'factor' }),
        ));
      });
    } else if (p.kind === 'number') {
      row.append(numberField(p, stagedValue(p), v => stage(p, v, row)));
    } else if (p.kind === 'vector') {
      const base = stagedValue(p);
      const fields = document.createElement('div');
      fields.className = 'fields';
      base.forEach((component, i) => {
        fields.append(numberField(p, component, v => {
          const value = vectorInRange(p, vectorState.get(p.param_id) || [...base]);
          value[i] = v;
          vectorState.set(p.param_id, value);
          fields.querySelectorAll('input').forEach((el, j) => { el.value = value[j]; });
          stage(p, [...value], row);
        }));
      });
      row.append(fields);
    } else if (p.kind === 'boolean') {
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = stagedValue(p);
      input.addEventListener('change', () => stage(p, input.checked, row));
      label.prepend(input);
    } else {
      const input = document.createElement('input');
      input.type = 'text';
      input.value = stagedValue(p);
      input.addEventListener('change', () => stage(p, input.value, row));
      row.append(input);
    }
    if (section === 2) {
      row.classList.toggle('staged', pending.has(p.param_id));
      row.classList.toggle('inflight', inflight.has(p.param_id));
      stagedRows.set(p.param_id, row);
    }
    renderRangeNote(p, row);
    if (p.from || p.accepted !== undefined) {
      // The values behind the row (DESIGN 7.1): what the human's file
      // holds, by its number, and the recipe's default. A preview
      // moves the field and leaves both standing; under the unset the
      // entry carries no default, since its value is the default.
      const layer = document.createElement('div');
      layer.className = 'layer';
      const shown = v => [].concat(v)
        .map(c => typeof c === 'number' ? fmt(c) : String(c)).join(', ');
      const source = p.accepted !== undefined ? `accepted ${shown(p.accepted)}` : p.from;
      layer.textContent = `${source} · default ${shown(p.default !== undefined ? p.default : p.value)}`;
      row.append(layer);
    }
    sections[section].rows.push({ row, mat: section === 0 ? p.uniform.material : null });
  }
  applyBar = null;
  applyButton = null;
  const captioned = sections.filter(s => s.rows.length).length > 1;
  for (const s of sections) {
    if (!s.rows.length) continue;
    const box = document.createElement('div');
    box.className = 'section';
    if (captioned) {
      const cap = document.createElement('div');
      cap.className = 'caption';
      cap.textContent = s.caption;
      box.append(cap);
    }
    // Live knobs cluster under their material once more than one
    // material is represented — with several materials, "roughness"
    // alone no longer says whose. Clustering keys on the schema's
    // uniform target, so interleaved declarations still group.
    const mats = [...new Set(s.rows.map(r => r.mat))].filter(Boolean);
    if (mats.length > 1) {
      for (const mat of mats) {
        const sub = document.createElement('div');
        sub.className = 'subcaption';
        sub.textContent = mat;
        box.append(sub);
        box.append(...s.rows.filter(r => r.mat === mat).map(r => r.row));
      }
    } else {
      box.append(...s.rows.map(r => r.row));
    }
    if (s.apply) box.append(buildApplyBar());
    panel.append(box);
  }
  // Panel footer: the overlay-scoped landings (DESIGN 7.2). Accept
  // promotes this tab's whole preview overlay to the asset file —
  // factor scrubs, applied values, and seed picks alike; Discard
  // drops it and shows the file again; Reset to defaults previews the
  // state without the human's lines. They live here rather than
  // beside Apply, whose scope is the staged group alone: a Discard
  // under the staged fields would read as governing them.
  // Asymmetric: Accept is the one emphasized control, alone at the
  // right; the other two are quiet at the left, so none are read as
  // a pair of alike buttons and a hand reaching for the landing
  // cannot slip onto the discard. Always present, disabled while
  // there is nothing to act on (Apply's rule) — the state reads
  // before a click; the server's refusal stays the backstop for the
  // moment between a reload dropping the overlay and the click.
  const footer = document.createElement('div');
  footer.className = 'accept';
  const quiet = document.createElement('div');
  quiet.className = 'quiet-group';
  const revertBtn = document.createElement('button');
  revertBtn.className = 'quiet';
  revertBtn.textContent = 'Discard';
  revertBtn.addEventListener('click', () => {
    send({ type: 'revert' });
    revertBtn.blur();
  });
  const resetBtn = document.createElement('button');
  resetBtn.className = 'quiet';
  resetBtn.textContent = 'Reset to defaults';
  resetBtn.addEventListener('click', () => {
    resetBtn.blur();
    resetToDefaults();
  });
  quiet.append(revertBtn, resetBtn);
  footer.append(quiet);
  const acceptBtn = document.createElement('button');
  acceptBtn.className = 'land';
  acceptBtn.textContent = 'Accept';
  acceptBtn.addEventListener('click', () => {
    send({ type: 'accept' });
    acceptBtn.blur();
  });
  footer.append(acceptBtn);
  panel.append(footer);
  revertButton = revertBtn;
  resetButton = resetBtn;
  acceptButton = acceptBtn;
  renderLanding();
  renderApplyBar();
}

// -- deixis pins (DESIGN 7.2) ------------------------------------------------

// Selection gestures, the DCC convention: a left click that is not an
// orbit drag (pointerup within a small slop of its pointerdown)
// selects the clicked primitive alone; a background click clears;
// shift-click adds or removes without touching the rest.
const CLICK_SLOP_PX = 5;
const raycaster = new THREE.Raycaster();
let pressAt = null;

canvas.addEventListener('pointerdown', (e) => {
  pressAt = e.button === 0 ? [e.clientX, e.clientY] : null;
});

canvas.addEventListener('pointerup', (e) => {
  if (e.button !== 0 || !pressAt) return;
  const [x0, y0] = pressAt;
  pressAt = null;
  if ((e.clientX - x0) ** 2 + (e.clientY - y0) ** 2 > CLICK_SLOP_PX ** 2) return;
  const rect = canvas.getBoundingClientRect();
  // Picking must not depend on the render loop having run (a
  // backgrounded tab's rAF is paused, leaving matrixWorld stale).
  camera.updateMatrixWorld();
  raycaster.setFromCamera(new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1,
  ), camera);
  const additive = e.shiftKey;
  const hit = raycaster.intersectObject(loaded, true)[0];
  if (!hit || hit.faceIndex == null) {
    // DCC convention: a plain click on empty space clears the
    // selection; shift-click leaves it alone.
    if (!additive && pinData.length) {
      selectAhead(new Set(), null);
      send({ type: 'select', action: 'clear' });
    }
    return;
  }
  // The output is the asset group's direct child the hit sits under
  // (GLB scenes are named by output on load).
  let obj = hit.object;
  while (obj && obj.parent !== loaded) obj = obj.parent;
  if (!obj || !obj.name) return;
  const instance = hit.instanceId ?? null;
  const samePrim = (p) =>
    p.output === obj.name && (p.instance ?? null) === instance;
  const clicked = {
    output: obj.name, instance, face: hit.faceIndex,
    position: [hit.point.x, hit.point.y, hit.point.z],
  };
  const pinMsg = (action) => {
    const msg = { type: 'select', action, ...clicked };
    if (instance === null) delete msg.instance;
    send(msg);
  };
  // Selection is primitive-unit, DCC-style: a plain click makes the
  // clicked primitive the sole selection; shift-click toggles it in or
  // out of the set without touching the rest.
  const onPrim = pinData.filter(samePrim);
  const ids = (ps) => new Set(ps.map(p => p.id));
  if (additive) {
    if (onPrim.length) {
      const last = onPrim[onPrim.length - 1];
      selectAhead(ids(pinData.filter(p => p !== last)), null);
      send({ type: 'select', action: 'unpin', id: last.id });
    } else {
      selectAhead(null, clicked);
      pinMsg('pin');
    }
    return;
  }
  if (onPrim.length) {
    // Already pinned: keep it (ids preserved), drop the rest.
    if (onPrim.length === pinData.length) return;
    // One unpin per other pin, each answered by its own broadcast.
    selectAhead(ids(onPrim), null, pinData.length - onPrim.length);
    for (const p of pinData) {
      if (!samePrim(p)) send({ type: 'select', action: 'unpin', id: p.id });
    }
  } else {
    // One atomic server-side mutation — no unpin burst that could race
    // a concurrent pin writer (agent clear, another tab).
    selectAhead(new Set(), clicked);
    pinMsg('replace');
  }
});

// The pins broadcast is full state: replace markers and list wholesale.
let pinData = [];
let pinsAssetHash = null;

// What a gesture asks for, shown from the click's own raycast before
// the server answers (DESIGN 7.1): ``keep`` is the recorded pins still
// shown (null: all), ``extra`` the clicked primitive not yet recorded.
// Display only — the gesture logic above reads the recorded pins, so
// a second click before the answer cannot toggle against a guess. The
// layer stands for as many pins broadcasts as the gesture sends
// messages (``answers``; the unpin burst is answered one by one, and
// a layer dropped by its first answer would flash the rest back), and
// a refused select (an asset_error while it stands) restores the
// recorded selection. Any other broadcast counting down early only
// shows the record sooner. The list draws the record alone: its ids
// are the agent's currency.
let pendingSelection = null;

function selectAhead(keep, extra, answers = 1) {
  pendingSelection = { keep, extra, answers };
  refreshPinVisuals();
}

function displayedMeshFor(output) {
  const root = loaded.children.find(o => o.name === output);
  if (!root) return null;
  let found = null;
  root.traverse(o => { if (!found && o.isMesh) found = o; });
  return found;
}

// One selection accent for every pin — the DCC convention: whatever
// the material colors are, this one hue always means "pinned", so a
// highlight can never be mistaken for a material tint (per-pin palette
// colors were tried and read as materials). Which pin is which is the
// list's job, via row hover.
const PIN_ACCENT = 0xff8a2a;

function highlightMaterial() {
  return new THREE.MeshBasicMaterial({
    color: PIN_ACCENT, transparent: true, opacity: 0.35, toneMapped: false,
    // Pulled toward the camera just enough to win the z-fight with
    // the surface it covers; still occluded by anything in front.
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
    side: THREE.DoubleSide, depthWrite: false,
  });
}

// Where a pin sits and what it highlights, derived from the *displayed*
// geometry every refresh — so markers and highlights follow
// topology-stable parameter changes (this tab's own preview included).
// Selection granularity is primitive-unit: a click pin highlights the
// whole clicked primitive (the output, or that one copy of an
// instanced output); a predicate pin highlights exactly its matched
// polygons from the pins broadcast — the fine-grained tool. The record
// itself stays the placement-time file-state resolution; whatever no
// longer resolves (output gone, topology changed, an oversized matched
// set) falls back to the recorded position with no highlight.
function pinSpot(p) {
  const fallback = {
    position: new THREE.Vector3(...p.position),
    regionKey: null,
    buildHighlight: null,
  };
  const mesh = displayedMeshFor(p.output);
  if (!mesh) return fallback;
  const geom = mesh.geometry;
  const index = geom.index ? geom.index.array : null;
  const posAttr = geom.attributes.position;
  const faceCount = Math.floor((index ? index.length : posAttr.count) / 3);
  const world = mesh.matrixWorld.clone();
  if (mesh.isInstancedMesh) {
    if (p.instance == null || p.instance >= mesh.count) return fallback;
    const im = new THREE.Matrix4();
    mesh.getMatrixAt(p.instance, im);
    world.multiply(im);
  }
  const v = new THREE.Vector3();
  let regionKey;
  let buildHighlight;
  if (p.predicate != null) {
    // Predicate pin: its matched polygons, an owned geometry. An
    // oversized matched set (faces omitted past the server cap) falls
    // back to the marker — a whole-primitive highlight would claim the
    // predicate matched everything.
    if (!Array.isArray(p.faces)) return fallback;
    if (p.faces.some((f) => f >= faceCount)) return fallback;
    // Keyed on the predicate itself: first-face + count is not an
    // identity — overlapping selections can tie on both and collide.
    regionKey = `${p.output}|${p.instance ?? ''}|${JSON.stringify(p.predicate)}`;
    buildHighlight = () => {
      const flat = new Float32Array(p.faces.length * 9);
      let k = 0;
      for (const f of p.faces) {
        for (let j = 0; j < 3; j++) {
          v.fromBufferAttribute(posAttr, index ? index[f * 3 + j] : f * 3 + j)
            .applyMatrix4(world);
          flat[k++] = v.x;
          flat[k++] = v.y;
          flat[k++] = v.z;
        }
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(flat, 3));
      const h = new THREE.Mesh(g, highlightMaterial());
      h.userData.ownsGeometry = true;
      return h;
    };
  } else {
    // Click pin: the whole primitive, sharing the displayed geometry
    // (never disposed by the highlight teardown).
    regionKey = `${p.output}|${p.instance ?? ''}|prim`;
    buildHighlight = () => {
      const h = new THREE.Mesh(geom, highlightMaterial());
      h.matrixAutoUpdate = false;
      h.matrix.copy(world);
      h.userData.ownsGeometry = false;
      return h;
    };
  }
  const spotPos = new THREE.Vector3();
  if (p.face != null && p.face < faceCount) {
    // The marker sits at the exact pointed spot, re-derived on the
    // current shape via the record's barycentric coordinates — two
    // pins on one primitive stay visually apart. Without bary (an
    // agent's face-form pin), the face centroid stands in.
    const weights = Array.isArray(p.bary) && p.bary.length === 3
      ? p.bary
      : [1 / 3, 1 / 3, 1 / 3];
    for (let j = 0; j < 3; j++) {
      v.fromBufferAttribute(posAttr, index ? index[p.face * 3 + j] : p.face * 3 + j)
        .applyMatrix4(world);
      spotPos.addScaledVector(v, weights[j]);
    }
  } else if (Array.isArray(p.faces) && p.faces.length) {
    // A predicate pin's marker follows as the mean of its matched
    // polygons' vertices on the current shape.
    let count = 0;
    for (const f of p.faces) {
      for (let j = 0; j < 3; j++) {
        v.fromBufferAttribute(posAttr, index ? index[f * 3 + j] : f * 3 + j)
          .applyMatrix4(world);
        spotPos.add(v);
        count += 1;
      }
    }
    spotPos.multiplyScalar(1 / count);
  } else {
    spotPos.copy(fallback.position);
  }
  return { position: spotPos, regionKey, buildHighlight };
}

// id -> {marker|null, regionKey}; regionKey -> highlight mesh. A click
// pin's highlight is the whole pinned primitive; a predicate pin's is
// its matched polygons. Identical targets (same key) draw once at one
// opacity — stacked translucent fills would darken, faking a meaning
// opacity does not have. The highlight is the pin's whole visual: a
// marker sphere appears only for a pin the highlight cannot show
// (output gone, topology changed, oversized matched set) — id chips
// are gone, identification flows through the list and its hover.
const pinVisuals = new Map();
const pinRegions = new Map();
let emphasizedPin = null;

// Hovering a list row singles its pin out — its highlight brightens
// while other pins' dim (a fallback marker grows instead).
function applyEmphasis() {
  const focusedVis = emphasizedPin === null ? null : pinVisuals.get(emphasizedPin);
  const focusedKey = focusedVis ? focusedVis.regionKey : null;
  for (const [key, mesh] of pinRegions) {
    mesh.material.opacity =
      emphasizedPin === null ? 0.35 : (key === focusedKey ? 0.55 : 0.1);
  }
  for (const [pid, vis] of pinVisuals) {
    if (!vis.marker) continue;
    // Emphasis by size only: the color is the pin's identity.
    vis.marker.userData.scaleK =
      emphasizedPin !== null && pid === emphasizedPin ? 1.6 : 1.0;
  }
}

function emphasizePin(id) {
  emphasizedPin = id;
  applyEmphasis();
}

function clearPinHighlights() {
  // Not disposeDeep: a primitive highlight *shares* the displayed
  // mesh's geometry — disposing it would break the asset itself.
  for (const h of pinHighlights.children) {
    if (h.userData.ownsGeometry) h.geometry.dispose();
    h.material.dispose();
  }
  pinHighlights.clear();
}

function refreshPinVisuals() {
  disposeDeep(pinsGroup);
  pinsGroup.clear();
  clearPinHighlights();
  pinVisuals.clear();
  pinRegions.clear();
  const shown = pendingSelection === null ? pinData : pinData.filter(
    p => pendingSelection.keep === null || pendingSelection.keep.has(p.id));
  if (pendingSelection !== null && pendingSelection.extra) {
    shown.push({ id: null, ...pendingSelection.extra });
  }
  if (!shown.length) return;
  loaded.updateMatrixWorld(true);
  for (const p of shown) {
    const spot = pinSpot(p);
    let marker = null;
    if (spot.buildHighlight) {
      if (!pinRegions.has(spot.regionKey)) {
        const highlight = spot.buildHighlight();
        pinHighlights.add(highlight);
        pinRegions.set(spot.regionKey, highlight);
      }
    } else {
      // No highlight to show: the marker is this pin's only visual.
      marker = new THREE.Mesh(
        new THREE.SphereGeometry(1, 12, 8),
        new THREE.MeshBasicMaterial({ color: PIN_ACCENT, toneMapped: false }),
      );
      marker.position.copy(spot.position);
      pinsGroup.add(marker);
    }
    pinVisuals.set(p.id, {
      marker, regionKey: spot.regionKey, position: spot.position,
    });
  }
  // A hovered row can vanish under the pointer (a remote unpin rebuilds
  // the list; removed DOM nodes fire no mouseleave) — a dead id must
  // not leave every highlight dimmed with none focused.
  if (emphasizedPin !== null && !pinVisuals.has(emphasizedPin)) {
    emphasizedPin = null;
  }
  applyEmphasis(); // keep a live hover's emphasis across the rebuild
}

function pinsArrived(msg) {
  if (pendingSelection !== null && --pendingSelection.answers <= 0) {
    pendingSelection = null;
  }
  pinData = msg.pins;
  pinsAssetHash = msg.asset_hash;
  refreshPinVisuals();
  renderPinList();
}

function jumpToPin(p) {
  // Retarget only: the camera keeps its offset, the orbit re-centers —
  // on the spot re-derived from the displayed geometry (the marker and
  // highlight follow shape changes; the jump must land where they are),
  // falling back to the recorded position when nothing resolves.
  const vis = pinVisuals.get(p.id);
  if (vis && vis.position) {
    controls.target.copy(vis.position);
  } else {
    controls.target.set(...p.position);
  }
}

function renderPinList() {
  pinsEl.replaceChildren();
  pinsEl.style.display = pinData.length ? 'block' : 'none';
  for (const p of pinData) {
    const row = document.createElement('div');
    row.className = 'pin';
    row.title = 'click to jump';
    // The label is the primitive (what a click selects); the id trails
    // dimmed, the currency for talking to the agent. Which highlight
    // is which row is answered by hovering the row. Tag membership
    // stays in the record — on the label it would come and go with the
    // clicked triangle.
    const label = document.createElement('span');
    label.className = 'label';
    const inst = p.instance == null ? '' : `[${p.instance}]`;
    label.textContent = `${p.output}${inst}`;
    row.append(label);
    if (pinsAssetHash && p.asset_hash !== pinsAssetHash) {
      const stale = document.createElement('span');
      stale.className = 'stale';
      stale.textContent = 'stale';
      stale.title = 'placed against an earlier revision';
      row.append(stale);
    }
    const pid = document.createElement('span');
    pid.className = 'pid';
    pid.textContent = `p${p.id}`;
    row.append(pid);
    // No delete control: shift-click toggles a primitive out, a
    // background click clears, and `moraine pins clear` covers the rest.
    row.addEventListener('click', () => jumpToPin(p));
    row.addEventListener('mouseenter', () => emphasizePin(p.id));
    row.addEventListener('mouseleave', () => emphasizePin(null));
    pinsEl.append(row);
  }
}

// -- session timeline (timeline message -> panel) ---------------------------
// Renders the timeline message; the display contract — a progress +
// attention header over a newest-first round list, measured units in
// plain words — is normative at DESIGN 7.2 ``timeline``. The payload
// is root-scoped and survives a project switch; only the rendered
// per-project slice follows the active project.

let timelineData = null;
let timelineCollapsed = true;

function count(n, noun) {
  return `${n} ${noun}${n === 1 ? '' : 's'}`;
}

function renderTimeline() {
  timelineEl.replaceChildren();
  timelineEl.style.display = timelineData ? 'block' : 'none';
  if (!timelineData) return;
  const s = timelineData.session;
  const proj = (s.projects || {})[currentProject];
  const rounds = (proj && proj.rounds_detail) || [];
  const hdr = document.createElement('div');
  hdr.className = 'hdr';
  hdr.title = 'one round = one asset revision that got verified '
    + '(edit → verify cycle)\n'
    + (timelineCollapsed ? 'click to expand rounds' : 'click to collapse');
  // Count, amber, and rows all read the active project's slice — a
  // session-wide count over per-project rows would let the two header
  // signals disagree in a multi-project session.
  hdr.textContent = `${timelineCollapsed ? '▸' : '▾'} `
    + `${count(rounds.length, 'verify round')}`;
  // Amber means red NOW, never a cumulative failure count (DESIGN 7.2
  // ``timeline``); it clears when a green round lands.
  const last = rounds[rounds.length - 1];
  if (last && last.validate_ok === false) {
    const red = document.createElement('span');
    red.className = 'warn';
    red.title = 'the newest verified revision failed validate';
    const checks = (last.failed_checks || []).join(',');
    red.textContent = checks ? ` · ✗ ${checks}` : ' · ✗ validate failed';
    hdr.append(red);
  }
  hdr.addEventListener('click', () => {
    timelineCollapsed = !timelineCollapsed;
    renderTimeline();
  });
  timelineEl.append(hdr);
  if (timelineData.stale) {
    const stale = document.createElement('div');
    stale.className = 'warn';
    stale.textContent = 'stale — the log stopped updating; restart serve';
    timelineEl.append(stale);
  }
  if (timelineCollapsed) return;
  if (!rounds.length) {
    const none = document.createElement('div');
    none.className = 'meta';
    none.textContent = `no rounds yet for ${currentProject || 'this project'}`;
    timelineEl.append(none);
    return;
  }
  for (let i = rounds.length - 1; i >= 0; i--) {
    const r = rounds[i];
    const row = document.createElement('div');
    row.className = 'round';
    row.title = 'the named check closed the round — the last one that ran '
      + 'before the next edit\n'
      + 'shots = screenshots the agent requested\n'
      + 'lines edited = lines added + removed since the last verified '
      + 'revision (a rewritten line counts twice)\n'
      + 'commands this round: ' + (Object.entries(r.commands || {})
        .map(([k, n]) => `${k}×${n}`).join('  ') || 'none');
    // A row with no check/cross glyph is a round that was never
    // validated.
    row.append(`#${i + 1} ${r.ended_by || '…'}`);
    if (r.validate_ok === false) {
      const bad = document.createElement('span');
      bad.className = 'warn';
      // No check names = validate failed before its checks could run
      // (load/evaluation error); the cross alone carries the state.
      const checks = (r.failed_checks || []).join(',');
      bad.textContent = checks ? ` ✗ ${checks}` : ' ✗';
      row.append(bad);
    } else if (r.validate_ok === true) {
      row.append(' ✓');
    }
    const meta = document.createElement('span');
    meta.className = 'meta';
    // Shots lead the tail; one command is already implied by the tier
    // name, so the count appears only past it.
    const ncmd = Object.values(r.commands || {}).reduce((a, b) => a + b, 0);
    meta.textContent = (r.shots ? ` · ${count(r.shots, 'shot')}` : '')
      + (ncmd > 1 ? ` · ${ncmd} commands` : '')
      + (r.changed_lines ? ` · ${count(r.changed_lines, 'line')} edited` : '');
    row.append(meta);
    timelineEl.append(row);
  }
}

// Dedicated render session (DESIGN 7.2): the server sends file-state GLB
// urls and a fully resolved camera pose, and the shot renders them in a
// throwaway scene — never this tab's loaded generation, preview overlay,
// camera, or helpers (the grid and axes are viewer chrome, not file
// state). All camera math lives server-side.

function disposeDeep(root) {
  root.traverse((obj) => {
    // A shadow-casting light owns a render target the renderer
    // allocated for it; nothing but its dispose() releases it.
    if (obj.isLight) obj.dispose();
    if (obj.geometry) obj.geometry.dispose();
    const mats = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
    for (const m of mats) {
      for (const v of Object.values(m)) { if (v && v.isTexture) v.dispose(); }
      m.dispose();
    }
  });
}

// Lazily created on the first screenshot_request and reused for the
// tab's lifetime: shots render on their own never-attached canvas, so
// the interactive view is never resized or repainted by a shot. One
// persistent second renderer, not one per shot — browsers cap live
// WebGL contexts, and churning them would eventually evict the
// interactive view's.
let shotRenderer = null;

async function takeScreenshot({ glb_urls, camera: pose, size }) {
  const [w, h] = size;
  if (!shotRenderer) {
    shotRenderer = new THREE.WebGLRenderer({
      canvas: document.createElement('canvas'), antialias: true,
    });
    shotRenderer.setPixelRatio(1);
    enableShadows(shotRenderer);
    calibrateExposure(shotRenderer);
  }
  const shot = new THREE.Scene();
  shot.background = new THREE.Color(0x23252a);
  // Same lighting rig as the interactive view (environment and ground
  // shadow included), chrome-free: no grid, no axes, no pins.
  const rig = addLightRig(shot, shotRenderer);
  const parts = new THREE.Group();
  shot.add(parts);
  try {
    for (const [name, url] of Object.entries(glb_urls)) {
      const gltf = await loader.loadAsync(url);
      gltf.scene.name = name;
      markShadowCasters(gltf.scene);
      parts.add(gltf.scene);
    }
    fitRig(rig, assetBounds(parts), shotRenderer);
    const cam = new THREE.PerspectiveCamera(pose.fov, w / h, 0.01, 1000);
    cam.position.set(...pose.position);
    cam.up.set(...pose.up);
    cam.lookAt(new THREE.Vector3(...pose.target));
    cam.updateProjectionMatrix();

    shotRenderer.setSize(w, h, false);
    shotRenderer.render(shot, cam);
    // toDataURL immediately after render, same task: no
    // preserveDrawingBuffer needed.
    const dataUrl = shotRenderer.domElement.toDataURL('image/png');
    return dataUrl.slice(dataUrl.indexOf(',') + 1);
  } finally {
    disposeDeep(shot);
  }
}

let ws = null;
let reconnectTimer = null;

function scheduleReconnect(delay) {
  clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(connect, delay);
}

// A hidden tab's retry timer is throttled by the browser — to about one
// wake a minute once its socket has been closed a while — so a returning
// human would otherwise face a frozen tab for up to that long after a
// restart. Retry the moment the tab is looked at.
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && (!ws || ws.readyState === WebSocket.CLOSED)) {
    scheduleReconnect(0);
  }
});

function setActiveProject(name) {
  if (!name || name === currentProject) return;
  currentProject = name;
  renderPickerMark();
  framed = false;
  // Client-local per-asset state dies with the asset it described:
  // param ids are asset-scoped names, so a staged edit or a seed walk
  // surviving into a project that happens to reuse an id would apply
  // one project's values to another. (The server drops its per-client
  // overlays on the switch; these maps are the client-side
  // counterparts it cannot reach.)
  pending.clear();
  seedHistory.clear();
  vectorState.clear();
  factorUniforms.clear();
  // The accept warning belongs to the project it was raised in; the
  // server ends it with the switch.
  renderAccepted(null);
  renderApplyBar();
  // The timeline payload is root-scoped and survives the switch; only
  // its rendered per-project slice follows.
  renderTimeline();
}

picker.addEventListener('change', () => {
  send({ type: 'set_asset', project: picker.value });
});

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    // Hidden until this connection's snapshot proves workspace mode:
    // a reconnect may land on a restarted single-project server, whose
    // snapshot carries no workspace message to retract a stale picker.
    pickerBox.style.display = 'none';
    // Same rule for pins: a restarted server has none and replays no
    // pins message, so stale markers must not linger — nor a gesture
    // the old socket never delivered.
    pendingSelection = null;
    pinsArrived({ pins: [], asset_hash: null });
    // And for the timeline: a restarted logging-off server replays
    // nothing to retract the panel.
    timelineData = null;
    renderTimeline();
    // And for the accept warning: the server replays the standing
    // one with the snapshot, and a restarted server has none.
    renderAccepted(null);
    // And for watch death: a restarted server's watch is alive and
    // replays no watch_dead, so the sticky staleness warning must not
    // outlive the connection that observed it.
    watchDead = false;
    listenDead = false;  // this connection just proved the listener alive
    assetError = '';  // the snapshot replays a standing error, if any
    // A fresh connection has a fresh, empty overlay: whatever the old
    // socket was still awaiting is never answered on this one.
    resetLedger();
    setHud('connected');
  };
  ws.onclose = () => { setHud('disconnected — retrying'); scheduleReconnect(1000); };
  ws.onmessage = async (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    try {
      await dispatch(msg);
    } catch (err) {
      // A payload the client cannot handle surfaces on the HUD; an
      // unhandled rejection from an async handler is silent.
      setHud(`client error: ${(err && err.message) || err}`);
    }
  };
  async function dispatch(msg) {
    if (msg.type === 'asset_updated') {
      lastAsset = msg.asset;
      // Project change handling runs before the awaited load so the
      // re-frame decision is covered by the same generation gate.
      setActiveProject(msg.project);
      picker.value = msg.project;
      // Everything derived from this message rides the same generation
      // gate: a stale load must not repaint the image strip or hud either.
      const show = async (attempt) => {
        if (await loadOutputs(msg.glb_urls, show, attempt)) {
          renderImages(msg.image_urls);
          // The displayed geometry changed: re-derive marker and
          // highlight placement so pins follow the new shape.
          refreshPinVisuals();
          // The display is the file again: only a load that lands at
          // the current generation may retract a standing error — an
          // error that arrived mid-load bumped the generation, so the
          // superseded load never reaches here.
          assetError = '';
          setHud('connected');
        }
      };
      await show(0);
    } else if (msg.type === 'workspace') {
      // Roster plus the projects that changed on disk since they were
      // last displayed (DESIGN 7.2): with several sessions authoring,
      // the mark is how the human knows where to look next. Full
      // state, idempotent; selection comes only from
      // asset_updated/asset_error, so rebuilding keeps the value.
      const selected = picker.value;
      const changed = new Set(msg.changed || []);
      picker.replaceChildren();
      for (const name of msg.projects) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = changed.has(name) ? `${name} ●` : name;
        picker.append(opt);
      }
      if (selected) picker.value = selected;
      // The mark also sits beside the closed control: the marked
      // project is never the displayed one, so the option list alone
      // would show it only once opened.
      changedProjects = changed;
      renderPickerMark();
      pickerBox.style.display = 'flex';
    } else if (msg.type === 'param_schema') {
      replyArrived();
      schemaArrived(msg.params, msg.overridden);
    } else if (msg.type === 'preview_dropped') {
      // This tab's preview overlay is gone. Announce why: a silently
      // vanished preview reads as a glitch, and a wrong cause is the
      // same failure — the notice is the only account the human gets.
      // One line per trigger the server drops an overlay on; the
      // fallback is the watch's, the drop nobody asked for.
      // Staged (un-applied) edits are client-local and survive all of
      // them. The factor cache goes with the overlay it previewed: the
      // file-state pair that follows reloads file materials, and a
      // surviving cache would re-apply the dropped scrub over them
      // until the schema arrives.
      factorUniforms.clear();
      // The landings follow the overlay at the drop itself: a reload
      // that fails sends no schema after it, and Accept and Discard
      // would stay lit over an overlay the server no longer holds.
      overriddenIds = [];
      renderLanding();
      // Nothing of this tab's is in flight once its overlay is gone:
      // the requests it awaited will never be answered.
      resetLedger();
      setNotice({
        revert: 'preview reverted — showing the file',
        set_asset: 'project changed — preview reset',
        watch: 'file changed — preview reset',
        cache_lost: 'cache rebuilt — preview reset',
      }[msg.trigger] || 'file changed — preview reset');
    } else if (msg.type === 'accepted') {
      // The durable confirmation is the watch reload that follows; this
      // is the accepting tab's immediate feedback (and the only one
      // when the watch is dead). A notice, not the state line: the
      // reload's own "connected" would overwrite the state line within
      // the watch latency, before the human has read it.
      const dropped = (msg.dropped || []).length
        ? ` (dropped: ${msg.dropped.join(', ')})` : '';
      setNotice(`accepted: ${msg.params.join(', ')}${dropped}`);
      // The verdict follows the reload, which a dead watch never sends.
      if (!watchDead) renderAccepted({ checking: true, params: msg.params });
    } else if (msg.type === 'accept_validated') {
      if (msg.moves_no_check) {
        // The pick can move no check, so none ran. The notice says
        // that, and not that the state is green.
        renderAccepted(null);
        setNotice(`accepted: ${(msg.params || []).join(', ')} — changes no check, not validated`);
      } else {
        renderAccepted(msg);
        if (msg.ok) setNotice('accepted state passes validate');
      }
    } else if (msg.type === 'asset_error') {
      if (pendingSelection !== null) {
        // A refused select answers here (DESIGN 7.2): the display goes
        // back to what the server records.
        pendingSelection = null;
        refreshPinVisuals();
      }
      // The picker mirrors the server's reported active project: a
      // rejected switch snaps back (the server did not move), an
      // accepted-but-broken one stays on the new project — so
      // re-picking the old project always fires a real change event.
      // The server deliberately does not re-broadcast asset_updated
      // for a no-op (it would force every viewer through a GLB reload).
      if (msg.project) {
        picker.value = msg.project;
        // A re-root that lands broken still moved the server — and a
        // server whose startup evaluation failed sends no
        // asset_updated at all: the active project must follow the
        // error report so the timeline slice is not empty exactly
        // when its red state matters.
        setActiveProject(msg.project);
      }
      const m = msg.error.message ? `: ${msg.error.message}` : '';
      const text = `asset error — ${msg.error.code}${m}`.slice(0, 160);
      if (ledger.length && isReplyError(msg.error)) {
        // A failed preview snaps back: the oldest outstanding request
        // is the one that failed (an Apply's edits stay staged, a seed
        // pick was reverted); the snap-back schema that follows is not
        // a second reply. The display stays current with the file, so
        // the state line carries it, not the sticky slot.
        replyFailed();
        setHud(text);
      } else {
        // File state failed: the display is now the last good state.
        // Superseding any load in flight keeps it exactly that — a
        // reload whose GLBs were still loading when the error arrived
        // must not install them after it, and must not repaint the
        // state line over the error when it finishes.
        loadGen++;
        clearTimeout(loadRetry);
        assetError = text;
        setHud();
      }
    } else if (msg.type === 'pins') {
      pinsArrived(msg);
    } else if (msg.type === 'timeline') {
      timelineData = msg;
      renderTimeline();
    } else if (msg.type === 'watch_dead') {
      watchDead = true;
      setHud();
    } else if (msg.type === 'watch_alive') {
      // The watch re-armed: the staleness warning ends here, and the
      // catch-up reload that follows repaints the display.
      watchDead = false;
      setHud();
    } else if (msg.type === 'listen_dead') {
      // A half-dead server: this tab still streams, but nothing new can
      // connect — the human's only channel is the tab already open.
      listenDead = true;
      setHud();
    } else if (msg.type === 'listen_alive') {
      listenDead = false;
      setHud();
    } else if (msg.type === 'screenshot_request') {
      // Echo the protocol stamp: the server refuses results from clients
      // that predate the request shape (a reload fixes those tabs).
      try {
        const png = await takeScreenshot(msg);
        send({ type: 'screenshot_result', request_id: msg.request_id, protocol: msg.protocol, png });
      } catch (err) {
        send({
          type: 'screenshot_result', request_id: msg.request_id, protocol: msg.protocol,
          error: String((err && err.message) || err),
        });
      }
    }
  }
}
connect();
