'use strict';
/* PocketVina workspace (pocketvina/workspace.py, README.md): the PocketVina Atlas website's protein page (components/ProteinWorkspace.tsx,
 * Viewer.tsx and cells.tsx, file names from lib/keys.ts; commit 40e8d59) rebuilt in plain JavaScript for a self-contained page: the 3D
 * view (viewer.js of the site in a frame, see bridge.js) with its legend and protein style buttons, the pocket chips, Reset view and
 * Save image under it, beside the tabs Top hits, Selected pair, Pockets and Download (the files, and Python lines that select the shown
 * pair and save it: pocketvina.select(protein=<P.id>, ligand=<L.id>)); a protein menu when the page holds several proteins. The
 * page's data (workspace.py: workspace_data) and the frame's code are in the script elements #pv-data and #pv-assets (JSON, gzip,
 * base64). Every pose is in the data: no request is made (with molstar="url" the frame loads the Mol* bundle from the Atlas website).
 * A page of the PocketVina Atlas (atlas.py: Atlas._page) has optional fields, drawn only when present, with their wording in the data:
 * the protein's information (P.info: title, name, links, facts under Annotations, associated diseases with their scores; the site's
 * protein header, app/protein/[id]/page.tsx and components/OpenTargets.tsx), the pockets' pLDDT (a column in Pockets), the ligands'
 * identifiers (L.ids: rows of label, value, link, tooltip, short text of a collapsed row; ChEMBL ID, ATC, synonyms) and the stored
 * PoseBusters result of each pair's highest-affinity pose (r.pb); the text above Top hits (P.topNote: its rows are the site's Top hits,
 * the PoseBusters-valid drugs; r.rank counts the drugs and is null for a row that is not ranked) and the protein named by its gene in
 * the texts (P.info.title), also in the Python lines: atlas.select(protein=<gene, or P.id for a title of several genes>, ligand=<the
 * drug's name, L.id>). Links open in a new tab (target=_blank, rel=noopener noreferrer), only http(s) addresses.
 * Test hooks: window.__pvState() (what is selected and shown), __pvViewerLog (viewer-ready / -frame / -drawn), __pocketvinaDraws,
 * __pvLastImage (the last image saved: name, width, height, scale, bytes) and __pvFrame (the 3D frame element). */
(() => {
  const POCKET_COLORS = ['#5c93e6', '#6a47b3', '#e65ce0', '#b34761', '#e69e5c'];
  const pocketColor = k => POCKET_COLORS[(k - 1) % POCKET_COLORS.length];   // ranks beyond 5: the colours again (viewer.js)
  const LIGAND_C = '#E3B35A';
  // colours as Mol* draws them (Viewer.tsx)
  const ELEMENT = { N: '#3050F8', O: '#FF0D0D', S: '#FFFF30', P: '#FF8000', F: '#90E050', Cl: '#1FF01F', Br: '#A62929', I: '#940094', B: '#FFB5B5',
    Se: '#FFA100', Si: '#F0C8A0', Fe: '#E06633', Zn: '#7D80B0', Mg: '#8AFF00', Na: '#AB5CF2', K: '#8F40D4', Ca: '#3DFF00', Pt: '#D0D0E0', As: '#BD80E3' };
  const INTERACTIONS = [['H-bond', '#2B83BA'], ['hydrophobic', '#808080'], ['π-stacking', '#8CB366'], ['cation–π', '#FF8000'],
    ['halogen bond', '#40FFBF'], ['ionic', '#F0C814'], ['metal', '#8C4099'], ['weak H-bond', '#C5DDEC']];
  const STYLES = [['cartoon', 'Cartoon', 'Protein as cartoon'], ['surface', 'Surface', 'Protein as a molecular surface (see-through, so the pose stays visible)'],
    ['sticks', 'Sticks', 'Protein atoms as ball-and-stick']];
  const ICON = {
    down: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>',
    left: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m15 18-6-6 6-6"/></svg>',
    right: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg>',
    eye: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
    eyeOff: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><path d="m3 3 18 18"/></svg>',
    copy: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>',
  };
  const root = document.getElementById('pv-root');
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
  const fmt = (x, d = 1) => (x == null || Number.isNaN(Number(x)) ? '–' : Number(x).toFixed(d));     // cells.tsx
  const signed = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1);
  const int = x => (x == null ? '–' : Number(x).toLocaleString('en-US'));
  const pad2 = m => String(m).padStart(2, '0');
  const byNum = (a, b) => a - b;
  // a link of the data (Atlas pages) in a new tab; text only for anything but an http(s) address
  const extLink = (href, html, cls = 'link') => (/^https?:\/\//i.test(String(href || ''))
    ? `<a class="${cls}" href="${esc(href)}" target="_blank" rel="noopener noreferrer">${html}</a>` : html);

  // ------------------------------------------------------------ data: JSON, gzip, base64 (workspace.py _pack)
  async function unpack(id) {
    const el = document.getElementById(id);
    if (typeof DecompressionStream === 'undefined') throw new Error('this browser cannot unpack the page (no DecompressionStream): use a current Chrome, Edge, Firefox or Safari');
    const bin = Uint8Array.from(atob(el.textContent.trim()), c => c.charCodeAt(0));
    const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'));
    return JSON.parse(await new Response(stream).text());
  }

  // ------------------------------------------------------------ state (ProteinWorkspace: sel, show, focus, ...; Viewer: status, legend)
  let D = null, A = null;
  const S = {
    prot: null, row: null,                       // the protein shown, the selected pair (a row of prot.rows) or null
    view: 'best', pk: null, mode: null,          // 'best': the pair's highest-affinity pose (its SDF); 'modes': mode `mode` of pocket `pk` (PDBQT)
    allModes: false,
    show: [], userShow: false, selShow: [],      // surfaces asked for; a chip was used (selections keep the surfaces); the selection's own
    focusPocket: null, focusRes: null, hidden: [], pstyle: 'cartoon',
    tab: 'top', q: '', openPocket: null, theme: 'light', themeFixed: false,
    contacts: null, contactsAll: null, focus3d: { active: false, key: null, line: false }, surfaces: [], names: null,
    status: 'Loading 3D…', framed: false, drawn: false, legendOpen: true, imgScale: 2, imgTransparent: false, imageBusy: false,
  };
  const ligOf = r => D.ligands[String(r.lig)];
  const protName = P => (P.info && P.info.title) || P.id;          // the protein in texts: its gene on an Atlas page (the site's wording)
  const pocketOf = (r, k) => (r && r.pockets.find(p => p.rank === k)) || null;
  const bestIn = p => Math.min(...p.modes.map(m => m[0]));
  const poseKey = () => (S.row ? `${S.row.lig}:${S.view}:${S.pk}:${S.mode}` : '');
  // file names (lib/keys.ts with "PV_"): PV_<protein>_<ligand>_top.sdf, ..._pk2_modes.pdbqt, ..._pk2_m03.png
  const pairStem = () => `PV_${S.prot.name}_${ligOf(S.row).name}`;
  const poseName = () => `${pairStem()}_pk${S.pk}_m${pad2(S.mode)}`;
  const imageName = () => (S.row ? `${poseName()}.png` : `PV_${S.prot.name}.png`);

  // ------------------------------------------------------------ selection
  function choose(row, k, m, view) {
    if (!row) { S.row = null; S.view = 'best'; S.pk = S.mode = null; }
    else if (view === 'best') { S.row = row; S.view = 'best'; S.pk = row.pocket; S.mode = row.mode; }
    else {
      const pk = pocketOf(row, k) ? k : row.pocket, P = pocketOf(row, pk), n = P ? P.modes.length : 1;
      S.row = row; S.view = 'modes'; S.pk = pk;
      S.mode = Math.max(1, Math.min(m || (pk === row.pocket ? row.mode : 1), n));
    }
    // each selection shows its own pocket until a chip was used; then the surfaces stay, with the new pocket's added
    if (!S.userShow) S.show = S.pk ? [S.pk] : S.prot.docked.slice();
    else if (S.pk && !S.show.includes(S.pk)) S.show = [...S.show, S.pk].sort(byNum);
    S.selShow = S.pk ? [S.pk] : S.prot.docked.slice();
    S.focusPocket = null; S.focusRes = null;
    if (row) S.tab = 'pair';
    update();
  }
  // from the highest-affinity pose the arrows move into its pocket's modes; within the modes they step and stop at both ends
  function step(d) {
    if (!S.row) return;
    if (S.view === 'best') { choose(S.row, S.pk, Math.max(1, S.mode + d), 'modes'); return; }
    const P = pocketOf(S.row, S.pk); if (!P) return;
    const m = Math.max(1, Math.min(P.modes.length, S.mode + d));
    if (m !== S.mode) choose(S.row, S.pk, m, 'modes');
  }
  // a pocket chip under the view: another pocket with poses of the pair shows them (and its surface); else its surface is shown or hidden
  function chip(k) {
    if (S.row && k !== S.pk && pocketOf(S.row, k)) { choose(S.row, k, 1, 'modes'); return; }
    S.userShow = true;
    const on = S.surfaces.includes(k);
    S.show = on ? S.show.filter(x => x !== k) : S.show.includes(k) ? S.show : [...S.show, k].sort(byNum);
    update();
  }
  function setProtein(id) {
    const P = D.proteins.find(p => p.id === id); if (!P || P === S.prot) return;
    S.prot = P; S.row = null; S.view = 'best'; S.pk = S.mode = null;
    S.show = P.docked.slice(); S.selShow = P.docked.slice(); S.userShow = false;
    S.focusPocket = S.focusRes = null; S.hidden = []; S.tab = 'top'; S.q = ''; S.openPocket = null;
    S.contacts = S.contactsAll = null; S.names = null; S.surfaces = []; S.focus3d = { active: false, key: null, line: false };
    update();
  }
  function clearFocus3d() {
    S.focus3d = { active: false, key: null, line: false }; S.focusRes = null;
    toFrame({ type: 'clear-focus' });
    lastSpec = JSON.stringify(spec());            // the focus is cleared in 3D already: no new drawing for it
    update();
  }
  function reset() {
    S.userShow = false; S.show = S.selShow.slice(); S.focusPocket = null; S.focusRes = null;
    S.focus3d = { active: false, key: null, line: false };
    toFrame({ type: 'reset' });
    update();
  }

  // ------------------------------------------------------------ the 3D frame (viewer.js of the site; bridge.js)
  let frame = null, frameReady = false, lastSpec = '', sentStructure = '', lastPoseKey = '';
  const toFrame = m => { if (frame && frame.contentWindow) frame.contentWindow.postMessage({ pv: 'page', m }, '*'); };
  function spec() {
    const P = S.prot;
    return { theme: S.theme, protein: P.key, name: P.name, pockets: P.pockets.map(p => ({ rank: p.rank, residues: p.residues })), showPockets: S.show.slice(),
      proteinStyle: S.pstyle, focusPocket: S.focusPocket, focusResidue: S.focusRes, hiddenInteractions: S.hidden.slice(), ligand: ligandSpec() };
  }
  function ligandSpec() {
    const r = S.row; if (!r) return null;
    const key = poseKey(), name = poseName();
    if (S.view === 'best' && r.sdf) return { text: r.sdf, format: 'sdf', model: 0, nModels: 1, allModes: false, key, name };
    const P = pocketOf(r, S.pk); if (!P) return null;
    return { text: P.pdbqt, format: 'pdbqt', model: S.mode - 1, nModels: P.modes.length, allModes: S.allModes && S.view === 'modes', key, name };
  }
  // the spec when it changed (force: also when not, e.g. a Contacts chip of the residue focused before: the viewer draws and focuses again)
  function send(force) {
    if (!frameReady) return;
    const s = spec(), k = JSON.stringify(s);
    if (!force && k === lastSpec) return;
    if (sentStructure !== S.prot.key) { toFrame({ type: 'structure', path: S.prot.key, text: S.prot.pdb }); sentStructure = S.prot.key; }
    lastSpec = k;
    toFrame({ type: 'show', spec: s });
    if (!S.framed && !S.drawn) setStatus('Drawing…');
  }
  function frameDoc() {
    const molstar = A.molstarJs != null ? `<script>${A.molstarJs}<\/script>`
      : `<script src="${esc(A.molstarUrl)}" onerror="parent.postMessage({pv: 'frame', m: {type: 'viewer-error', message: 'cannot load the 3D viewer from the PocketVina Atlas website (offline?)'}}, '*')"><\/script>`;
    return `<!DOCTYPE html><html lang="en" data-theme="${S.theme}"><head><meta charset="utf-8"><meta name="color-scheme" content="light dark"><title>3D view</title>`
      + `<style>${A.molstarCss}\n${A.viewerCss}</style></head><body><div id="app"></div><script>${A.bridgeJs}<\/script>${molstar}<script>${A.viewerJs}<\/script></body></html>`;
  }
  const viewerLog = window.__pvViewerLog = [];
  window.__pocketvinaDraws = 0;
  const images = new Map(); let imageSeq = 0;
  function fromFrame(m) {
    if (m.type === 'viewer-ready' || m.type === 'viewer-frame' || m.type === 'viewer-drawn') viewerLog.push({ type: m.type, t: performance.now() });
    switch (m.type) {
      case 'bridge-ready': frameReady = true; send(true); break;
      case 'need-structure': { const P = D.proteins.find(p => p.key === m.path); if (P) { toFrame({ type: 'structure', path: P.key, text: P.pdb }); sentStructure = P.key; } break; }
      case 'viewer-frame': S.framed = true; setStatus(''); frame.classList.remove('pv-hidden'); break;
      case 'viewer-drawn': S.drawn = true; setStatus(''); frame.classList.remove('pv-hidden'); window.__pocketvinaDraws++; update(); break;
      case 'viewer-error': setStatus('3D error: ' + (m.message || '')); break;
      case 'key':
        if (m.key === 'ArrowLeft' || m.key === 'ArrowRight') step(m.key === 'ArrowLeft' ? -1 : 1);
        else if (m.key === 'Escape') closeShot();
        break;
      case 'focus': S.focus3d = { active: !!m.active, key: typeof m.key === 'string' ? m.key : null, line: !!m.line }; update(); break;
      case 'contacts':
        if (!Array.isArray(m.contacts)) break;
        S.contacts = { key: String(m.key || ''), source: m.source || 'none', contacts: m.contacts, elements: Array.isArray(m.elements) ? m.elements.map(String) : [],
          counts: m.counts || {}, atomCounts: m.atomCounts || {}, hidden: Array.isArray(m.hidden) ? m.hidden.map(String) : [] };
        if (!S.contacts.hidden.length) S.contactsAll = S.contacts;
        update(); break;
      case 'residues': if (m.residues && m.protein === S.prot.key) { S.names = m.residues; update(); } break;
      case 'surfaces': if (Array.isArray(m.shown)) { S.surfaces = m.shown.map(Number).filter(k => Number.isInteger(k) && k >= 1); update(); } break;
      case 'image': { const w = images.get(m.id); if (w) { images.delete(m.id); if (m.error) w.reject(new Error(m.error)); else w.resolve(m); } break; }
      default: break;
    }
  }
  window.addEventListener('message', ev => {
    if (!frame || ev.source !== frame.contentWindow || !ev.data || ev.data.pv !== 'frame') return;
    const m = ev.data.m; if (m && typeof m === 'object') fromFrame(m);
  });

  // ------------------------------------------------------------ files: a download, and what to do where downloads are blocked
  function download(blob, name) {
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = name; a.rel = 'noopener'; a.style.display = 'none';
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }
  let toastTimer = 0;
  function toast(text) {
    const t = document.getElementById('pv-toast'); t.textContent = text; t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(hideToast, 6000);
  }
  function hideToast() { clearTimeout(toastTimer); document.getElementById('pv-toast').hidden = true; }
  function saveText(text, name, type) {
    download(new Blob([text], { type }), name);
    toast(`Saved ${name}. If nothing was downloaded, downloads are blocked here: use the Python lines in the Download tab.`);
  }
  const dataUrlBlob = url => { const [head, b64] = url.split(','), bin = atob(b64), a = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i); return new Blob([a], { type: head.slice(5).split(';')[0] }); };
  async function saveImage() {
    if (!S.drawn || S.imageBusy) return;
    const name = imageName(), id = ++imageSeq;
    S.imageBusy = true; update();
    try {
      const x = await new Promise((resolve, reject) => { images.set(id, { resolve, reject }); toFrame({ type: 'image', id, scale: S.imgScale, transparent: S.imgTransparent }); });
      const blob = dataUrlBlob(x.url);
      download(blob, name);
      window.__pvLastImage = { name, width: x.width, height: x.height, scale: x.scale, bytes: blob.size, transparent: S.imgTransparent };
      showShot(x, name);
    } catch (e) { toast('Save image failed: ' + (e && e.message || e)); }
    finally { S.imageBusy = false; update(); }
  }
  // the image just saved and a hint, also for where downloads are blocked (right-click > Save image as)
  function showShot(x, name) {
    const box = document.getElementById('pv-shot');
    box.innerHTML = `<div class="pv-shot-box" role="dialog" aria-label="Image saved"><div class="pv-shot-head"><span>Saved <b class="mono">${esc(name)}</b> `
      + `(${x.width} × ${x.height} px). Nothing downloaded? Right-click the image &gt; <i>Save image as…</i></span>`
      + `<button type="button" class="btn" data-act="shot-close">Close</button></div><img alt="${esc(name)}" src="${x.url}"></div>`;
    box.hidden = false;
  }
  function closeShot() { const box = document.getElementById('pv-shot'); if (!box.hidden) { box.hidden = true; box.innerHTML = ''; } }

  // ------------------------------------------------------------ rendering (each part's HTML is replaced only when it changed)
  const set = (el, h) => { if (el && el.__pv !== h) { el.__pv = h; el.innerHTML = h; } };
  const $ = id => document.getElementById(id);
  function setStatus(t) { S.status = t; const el = $('pv-status'); if (el) el.textContent = t; }
  function layout() {
    root.innerHTML = `<header class="pv-head"><span class="pv-brand">PocketVina</span><span id="pv-proteins"></span><span class="pv-title" id="pv-title"></span>`
      + `<button type="button" class="btn" data-act="theme" id="pv-theme"></button></header>`
      + (D.proteins.some(p => p.info) ? `<div class="pv-info" id="pv-info"></div>` : '')
      + `<div class="pv-main"><div class="pv-left"><div class="pv-viewer" id="pv-viewer"><span class="pv-status" id="pv-status" role="status"></span>`
      + `<div class="pv-overlay" id="pv-overlay"></div></div><div class="pv-chips" id="pv-chips"></div></div>`
      + `<div class="pv-panel"><div class="pv-tabs" role="tablist" aria-label="Panels" id="pv-tabs"></div><div class="pv-body" id="pv-body"></div></div></div>`
      + `<div class="pv-shot" id="pv-shot" hidden></div><div class="pv-toast" id="pv-toast" role="status" hidden></div>`;
    frame = document.createElement('iframe');
    frame.className = 'pv-hidden'; frame.title = '3D view'; frame.setAttribute('allow', 'xr-spatial-tracking *; fullscreen *');   // Mol* asks for WebXR (any origin: a srcdoc frame's may be opaque)
    frame.srcdoc = frameDoc();
    $('pv-viewer').prepend(frame);
    window.__pvFrame = frame;
    setStatus(S.status);
  }
  function renderHead() {
    const many = D.proteins.length > 1;
    set($('pv-proteins'), many ? `<label>Protein <select data-act="protein" aria-label="Protein">${D.proteins.map(p =>
      `<option value="${esc(p.id)}"${p === S.prot ? ' selected' : ''}>${esc(p.id)} (${int(p.n)} ligands)</option>`).join('')}</select></label>` : `<b>${esc(S.prot.id)}</b>`);
    set($('pv-title'), S.row ? `${esc(ligOf(S.row).id)} · ${S.view === 'best' ? 'highest-affinity pose' : `pocket ${S.pk}, mode ${S.mode}`}` : '');
    $('pv-theme').textContent = S.theme === 'dark' ? 'Light theme' : 'Dark theme';
  }
  // the protein's information (Atlas pages): title, name and links in one line, then Annotations and the associated diseases, each a
  // panel that opens over the page (the site's protein header)
  function renderInfo() {
    const el = $('pv-info'); if (!el) return;
    const I = S.prot.info;
    el.hidden = !I;
    if (!I) { set(el, ''); return; }
    const pop = (attr, label, body) => `<details class="pv-pop" ${attr}><summary>${label}<span class="pv-chev">${ICON.down}</span></summary><div class="pv-pop-body">${body}</div></details>`;
    const facts = (I.facts || []).length ? pop('data-info-facts', 'Annotations', `<dl class="kv">${I.facts.map(x => `<dt>${esc(x.label)}</dt><dd>${extLink(x.href, esc(x.value))}</dd>`).join('')}</dl>`) : '';
    const Z = I.diseases;
    const diseases = Z && Z.items && Z.items.length ? pop('data-info-diseases', esc(Z.title), `<ol class="pv-dis">${Z.items.map(d => `<li data-disease>`
      + `${extLink(d.href, `<span>${esc(d.name)}</span>`, 'link pv-dis-name')}<span class="num">${fmt(d.score, 2)}</span>`
      + `<span class="pv-dis-bar" aria-hidden="true"><i style="width:${Math.max(1, Math.min(100, Number(d.score) * 100 || 0))}%"></i></span></li>`).join('')}</ol>`
      + (Z.all ? `<p class="small" style="margin-top:8px">${extLink(Z.all.href, esc(Z.all.text))}</p>` : '')
      + (Z.note ? `<p class="small mut"${Z.tip ? ` title="${esc(Z.tip)}"` : ''}>${esc(Z.note)}</p>` : '')) : '';
    set(el, `<span class="pv-info-title">${esc(I.title)}</span>${I.name ? `<span class="mut">${esc(I.name)}</span>` : ''}`
      + (I.links || []).map(x => `<span><span class="mut">${esc(x.label)} </span>${extLink(x.href, esc(x.value))}</span>`).join('') + facts + diseases);
  }
  function renderOverlay() {
    // Viewer.tsx: the legend of what is drawn now and, above it, the protein's style buttons; once the view is drawn
    if (!S.drawn) { set($('pv-overlay'), ''); return; }
    const hasLig = !!S.row, lg = hasLig && S.contacts && S.contacts.key === poseKey() ? S.contacts : null;
    const types = lg ? INTERACTIONS.filter(([t]) => lg.counts[t] || lg.hidden.includes(t)) : [];
    const proteinC = S.theme === 'dark' ? '#8C96A3' : '#A9B1BB';
    const colours = (hasLig ? `<li data-carbon="ligand"><i class="pv-sw" style="background:${LIGAND_C}"></i><b>C</b><span class="mut">ligand</span></li>` : '')
      + `<li data-carbon="protein"><i class="pv-sw" style="background:${proteinC}"></i><b>C</b><span class="mut">protein</span></li>`
      + (lg ? lg.elements.map(el => `<li data-element="${esc(el)}"><i class="pv-sw" style="background:${ELEMENT[el] || '#FF1493'}"></i>${esc(el)}</li>`).join('') : '')
      + `<li><span>${POCKET_COLORS.map(c => `<i class="pv-sw-pk" style="background:${c}"></i>`).join('')}</span>pocket surface</li>`;
    const typeList = types.length ? `<ul class="pv-types" aria-label="Interactions">${types.map(([t, c]) => {
      const off = lg.hidden.includes(t), n = lg.counts[t] || 0, atoms = lg.atomCounts[t] || 0;
      const tip = `Show / hide ${t} in 3D.` + (off ? '' : ` ${n} ${n === 1 ? 'residue' : 'residues'}, ${atoms} atom-level ${atoms === 1 ? 'contact' : 'contacts'}.`);
      return `<li><button type="button" data-act="type" data-v="${esc(t)}" aria-pressed="${!off}" title="${esc(tip)}" data-interaction="${esc(t)}">`
        + `${off ? ICON.eyeOff : ICON.eye}<i class="pv-dash" style="border-top-color:${c}"></i><span>${esc(t)}${off ? '' : `<span class="mut"> · ${n} ${n === 1 ? 'residue' : 'residues'}</span>`}</span></button></li>`;
    }).join('')}</ul>` : '';
    set($('pv-overlay'), `<details class="pv-legend" data-legend${S.legendOpen ? ' open' : ''}><summary>Legend <span class="pv-chev">${ICON.down}</span></summary>`
      + `<div class="pv-legend-body"><ul aria-label="Colours">${colours}</ul>${typeList}</div></details>`
      + `<div class="pv-styles" role="group" aria-label="Protein style in 3D">${STYLES.map(([v, label, tip]) =>
        `<button type="button" class="tag" data-act="style" data-v="${v}" aria-pressed="${S.pstyle === v}" title="${esc(tip)}">${label}</button>`).join('')}</div>`);
  }
  function renderChips() {
    const P = S.prot, r = S.row;
    const chips = P.docked.map(k => {
      const on = S.surfaces.includes(k), pk = pocketOf(r, k), best = pk ? bestIn(pk) : null, star = r && k === r.pocket;
      const switches = r && pk && k !== S.pk;
      const tip = switches ? `Pocket ${k}: show the pair's poses there (highest affinity ${signed(best)} kcal/mol)`
        : `Pocket ${k}: ${on ? 'hide' : 'show'} its surface in 3D` + (best != null ? ` · highest affinity of the pair here ${signed(best)} kcal/mol` : '');
      return `<button type="button" class="tag" data-act="chip" data-v="${k}" aria-pressed="${on}" data-surface="${on ? 'on' : 'off'}" data-current="${!!(r && k === S.pk)}"`
        + ` style="--pk:${pocketColor(k)}" title="${esc(tip)}"${star ? ' data-best' : ''}><i class="dot" style="background:${pocketColor(k)}"></i>${k}</button>`;
    }).join('');
    set($('pv-chips'), `<span class="pv-label">Pockets</span><div class="pv-chips-list" role="group" aria-label="Pockets" style="display:flex;flex-wrap:wrap;gap:8px">${chips}</div>`
      + `<div class="pv-actions"><button type="button" class="btn" data-act="reset" title="The selection's pocket surfaces, no focus, the whole view">Reset view</button>`
      + `<button type="button" class="btn" data-act="image" data-save-image${S.drawn && !S.imageBusy ? '' : ' disabled'} title="A PNG of exactly this view">${S.imageBusy ? 'Saving…' : 'Save image'}</button>`
      + `<select data-act="img-scale" aria-label="Image size" title="Image size: 2 or 4 times the view">${[2, 4].map(k => `<option value="${k}"${S.imgScale === k ? ' selected' : ''}>${k}×</option>`).join('')}</select>`
      + `<label title="PNG without the background"><input type="checkbox" data-act="img-transparent"${S.imgTransparent ? ' checked' : ''}>transparent</label></div>`);
  }
  function renderTabs() {
    set($('pv-tabs'), [['top', 'Top hits'], ['pair', 'Selected pair'], ['pockets', 'Pockets'], ['download', 'Download']].map(([k, l]) =>
      `<button type="button" role="tab" data-act="tab" data-v="${k}" aria-selected="${S.tab === k}">${l}</button>`).join(''));
  }
  function renderBody() {
    const body = $('pv-body');
    if (S.tab === 'top') {
      const key = `top:${S.prot.id}`;
      if (body.__pv !== key) {
        body.__pv = key;
        const P = S.prot, listed = P.rows.filter(r => D.maxLigands == null || r.rank <= D.maxLigands).length;
        body.innerHTML = `<div role="tabpanel" aria-label="Top hits"><p class="pv-note">${P.topNote != null ? esc(P.topNote)
          : `Ligands docked into ${esc(protName(P))}, highest affinity (lowest Vina score) first. Pick one to see its poses.`}</p>`
          + `<input type="search" class="pv-search" id="pv-q" placeholder="Search ligands (name or SMILES)" aria-label="Search ligands" value="${esc(S.q)}">`
          + `<div class="scroll-x"><table class="dt"><thead><tr><th class="r">#</th><th>Ligand</th><th class="r">Vina</th><th>Pocket</th><th class="r">Mode</th><th class="r" title="Ligand efficiency: −Vina / heavy atoms">LE</th></tr></thead>`
          + `<tbody id="pv-rows"></tbody></table></div>`
          + (P.topNote == null && P.n > listed ? `<p class="pv-note">The top ${int(listed)} of ${int(P.n)} ligands docked into ${esc(protName(P))} (max_ligands=${int(D.maxLigands)}).</p>` : '')
          + `<p class="pv-note">Vina: highest affinity of the pair (its lowest AutoDock Vina score over all pockets and modes, kcal/mol; more negative is better). Pocket and mode of that pose. LE: ligand efficiency.</p></div>`;
      }
      const t = S.q.trim().toLowerCase();
      const text = L => `${L.id} ${L.smiles}` + (L.ids ? ` ${L.ids.map(x => x.value).join(' ')}` : '');   // Atlas pages: also the identifiers
      const rows = S.prot.rows.filter(r => !t || text(ligOf(r)).toLowerCase().includes(t));
      set($('pv-rows'), rows.length ? rows.map(r => `<tr data-act="row" data-v="${r.lig}" data-active="${S.row === r}"><td class="r num mut">${r.rank == null ? '–' : r.rank}</td>`
        + `<td class="trunc"><span class="link" title="${esc(ligOf(r).id)}">${esc(ligOf(r).id)}</span></td><td class="r num">${fmt(r.vina)}</td>`
        + `<td><span style="display:inline-flex;align-items:center;gap:6px" class="mut"><i class="dot"></i>${r.pocket}</span></td><td class="r num">${r.mode}</td>`
        + `<td class="r num">${fmt(r.le, 2)}</td></tr>`).join('') : `<tr><td colspan="6" class="mut">${S.prot.rows.length ? `No ligand matches “${esc(S.q)}”.` : S.prot.topNote != null ? 'No drug with a PoseBusters-valid pose.' : `No ligand has a pose in ${esc(protName(S.prot))}.`}</td></tr>`);
      return;
    }
    // the Python lines keep the focus and their scroll when the tab is drawn again (e.g. once the 3D view has drawn), only while the page
    // has the focus: else focus() would take it from where the user is now (Firefox: a notebook cell outside the page's frame)
    const a = document.hasFocus() ? document.activeElement : null, py = a && a.matches('[data-python]') ? a : null, x = py ? py.scrollLeft : 0;
    set(body, S.tab === 'pair' ? pairPanel() : S.tab === 'download' ? downloadPanel() : pocketsPanel());
    const now = py && !py.isConnected ? body.querySelector('[data-python]') : null;
    if (now) { now.focus({ preventScroll: true }); now.scrollLeft = x; }
  }
  // the 2D drawing (RDKit's SVG, made from the user's SMILES) as an image: nothing in it can run in the page
  function pairPanel() {
    const r = S.row, P = S.prot;
    if (!r) return `<div role="tabpanel" aria-label="Selected pair"><p class="mut">Pick a ligand in Top hits to see its poses in each pocket.</p></div>`;
    const L = ligOf(r), cur = pocketOf(r, S.pk), modes = cur ? cur.modes : [], m = modes[S.mode - 1];
    const pct = Math.max(1, Math.round(100 * r.rank / P.n));
    const pockets = P.docked.map(k => {
      const pk = pocketOf(r, k), v = pk ? bestIn(pk) : null, bad = v != null && v >= 0, best = k === r.pocket, on = S.view === 'modes' ? S.pk === k : best;
      const tip = (best ? 'Pocket with the highest-affinity pose. ' : '') + (!pk ? `Pocket ${k}: no pose of this ligand` : bad
        ? `Pocket ${k}: no favourable pose (lowest score ${signed(v)} kcal/mol, not below 0). Show its poses.` : `Pocket ${k}: highest affinity ${signed(v)} kcal/mol. Show its poses.`);
      return `<button type="button" data-act="pk" data-v="${k}"${pk ? '' : ' disabled'} aria-pressed="${on}" title="${esc(tip)}" aria-label="${esc(tip)}"${best ? ' data-best' : ''}>`
        + `<span><i class="dot" style="background:${pocketColor(k)}"></i>${k}</span><span><b${bad ? ' style="font-weight:normal;opacity:.75"' : ''}>${v == null ? '–' : signed(v)}</b>${best ? '<span aria-hidden="true" data-top>★</span>' : ''}</span></button>`;
    }).join('');
    const affs = modes.map(x => x[0]), lo = Math.min(...affs), hi = Math.max(...affs);
    const bars = modes.length > 1 ? `<div class="pv-bars" aria-label="Vina score of each mode; pick one">${modes.map((x, i) =>
      `<button type="button" data-act="mode" data-v="${i + 1}" aria-pressed="${i + 1 === S.mode}" aria-label="Mode ${i + 1}: ${fmt(x[0])} kcal/mol${x[0] >= 0 ? ', no favourable pose' : ''}" `
      + `title="mode ${i + 1}: ${fmt(x[0])} kcal/mol${x[0] >= 0 ? ' (no favourable pose)' : ''}"><i style="height:${hi === lo ? 100 : 30 + 70 * (hi - x[0]) / (hi - lo)}%"></i></button>`).join('')}</div>` : '';
    const modeText = S.view === 'best' ? `mode ${S.mode} · highest affinity` : `mode ${S.mode}/${modes.length}`;
    const note = S.view === 'modes' && m ? `${fmt(m[0])} kcal/mol${m[0] >= 0 ? ' (no favourable pose)' : ''} · RMSD to mode 1: ${fmt(m[1], 2)}–${fmt(m[2], 2)} Å`
      : `pocket ${r.pocket} · the pair's highest-affinity pose`;
    // an Atlas page (P.info) in the site's words: "drugs on", "compound", and a line when the protein is outside the drug's best targets
    const across = r.across ? `<dt>Across proteins</dt><dd>${esc(protName(P))} is ${r.across[0] == null ? `not among the top ${int(r.across[2])} of ${int(r.across[1])}`
      : `#${r.across[0]} of ${P.info ? int(r.across[1]) : r.across[1]}`} docked proteins for this ${P.info ? 'compound' : 'ligand'}</dd>` : '';
    return `<div role="tabpanel" aria-label="Selected pair"><div class="pv-drug" data-pair-drug><div class="pv-svg">${L.svg ? `<img alt="2D structure of ${esc(L.id)}" src="data:image/svg+xml;charset=utf-8,${encodeURIComponent(L.svg)}">` : ''}</div>`
      + `<div style="min-width:0"><h3>${esc(L.id)}</h3>`
      + `<p class="small mut">Docked with PocketVina into ${P.docked.length} pocket${P.docked.length === 1 ? '' : 's'} of ${esc(protName(P))}</p></div></div>`
      + identifiers(L)
      + `<dl class="kv" data-pair-score><dt>Highest affinity (Vina, kcal/mol)</dt><dd class="num">${fmt(r.vina)} kcal/mol${r.rank == null ? '' : ` · rank ${int(r.rank)} of ${int(P.n)}${P.info ? ' drugs' : ''} on ${esc(protName(P))} (top ${pct}%)`}</dd>`
      + `${across}<dt>Ligand efficiency</dt><dd class="num">${fmt(r.le, 2)}</dd><dt>Highest-affinity pose</dt><dd>pocket ${r.pocket}, mode ${r.mode}</dd>${posebusters(r.pb)}</dl>`
      + `<h4>Highest affinity in each pocket</h4><div class="pv-pk-grid" role="group" aria-label="Pocket">${pockets}</div>`
      + `<div class="pv-modes" data-pair-modes><button type="button" class="btn" data-act="step" data-v="-1" aria-label="Previous mode"${S.mode <= 1 ? ' disabled' : ''}>${ICON.left}</button>`
      + `<span class="pv-mode">${modeText}</span><button type="button" class="btn" data-act="step" data-v="1" aria-label="Next mode"${S.view === 'modes' && S.mode >= modes.length ? ' disabled' : ''}>${ICON.right}</button>`
      + `<span class="small mut">${note}</span></div>${bars}`
      + `<div class="pv-row"><button type="button" class="tag" data-act="best" aria-pressed="${S.view === 'best'}" title="The pair's highest-affinity pose">Highest-affinity pose</button>`
      + `<label><input type="checkbox" data-act="allmodes"${S.allModes && S.view === 'modes' ? ' checked' : ''}${cur ? '' : ' disabled'}>Show all modes</label></div>`
      + contactsList() + `</div>`;
  }
  // the ligand's identifiers the run has, each with a copy button (no empty rows): the docked SMILES only when it is not the input.
  // Atlas pages: the rows of L.ids (a link, a tooltip; a long list collapsed to its short text)
  const copyButton = (label, v) => `<button type="button" class="pv-copy" data-act="copy" data-v="${esc(v)}" title="Copy ${esc(label)}" aria-label="Copy ${esc(label)}">${ICON.copy}</button>`;
  function identifiers(L) {
    if (L.ids) {
      return `<dl class="kv pv-ids" data-pair-ids>${L.ids.filter(x => x && x.value).map(x => {
        const v = String(x.value), shown = extLink(x.href, `<span class="mono pv-id">${esc(v)}</span>`);
        const value = x.short ? `<details class="pv-fold"><summary><span class="mono pv-id">${esc(x.short)}</span></summary>${shown}</details>` : shown;
        return `<dt${x.tip ? ` title="${esc(x.tip)}"` : ''}>${esc(x.label)}</dt><dd data-id="${esc(x.label)}">${value}${copyButton(x.label, v)}</dd>`;
      }).join('')}</dl>`;
    }
    const rows = [['Ligand ID', L.id], ['SMILES', L.smiles], ['Docked SMILES', L.docked && L.docked !== L.smiles ? L.docked : null,
      'The fragment docked, e.g. without the counter-ion of a salt'], ['InChIKey', L.inchikey, 'Of the docked structure']].filter(x => x[1]);
    return `<dl class="kv pv-ids" data-pair-ids>${rows.map(([k, v, tip]) => `<dt${tip ? ` title="${esc(tip)}"` : ''}>${k}</dt>`
      + `<dd><span class="mono pv-id">${esc(v)}</span><button type="button" class="pv-copy" data-act="copy" data-v="${esc(v)}" title="Copy ${k}" aria-label="Copy ${k}">${ICON.copy}</button></dd>`).join('')}</dl>`;
  }
  // PoseBusters of the shown pair's highest-affinity pose, only when the run has checked it (PoseBusters is optional)
  function posebusters(pb) {
    if (!pb) return '';
    const ok = pb.valid === true, bad = pb.valid === false;
    const what = pb.total ? `${pb.passed}/${pb.total} checks passed` : esc(pb.status);
    const failed = pb.failed && pb.failed.length ? `<details class="pv-pb-failed"><summary>${pb.failed.length} failed</summary><span class="mono small">${pb.failed.map(esc).join(', ')}</span></details>` : '';
    return `<dt>PoseBusters</dt><dd data-pair-pb><i class="dot" style="background:${ok ? 'var(--ok, #2e9d5b)' : bad ? 'var(--bad, #c0392b)' : 'var(--mut, #888)'}"></i> ${what}`
      + `<span class="small mut"> · highest-affinity pose only</span>${failed}</dd>`;
  }
  // the Python lines of the Download tab, for where downloads are blocked: they select the shown pair by the ids the page shows (an
  // Atlas page: the gene and the drug's name) and save it. An id as a Python string (a JSON string is a valid Python string)
  const pyStr = s => JSON.stringify(String(s));
  // select() takes a gene, not a title of several ("A; B"): then the protein's id (the UniProt accession)
  const pyProtein = P => (/[;,/]/.test(protName(P)) ? P.id : protName(P));
  const pythonLines = (P, r) => `pair = ${P.info ? 'atlas' : 'pocketvina'}.select(protein=${pyStr(pyProtein(P))}, ligand=${pyStr(r ? ligOf(r).id : '<ligand>')})`
    + '\npair.export(protein=True)   # the poses and the protein';
  // the Download tab (as on the Atlas site): the pair's poses, the protein, the pockets' residues and the image, then the Python lines
  // with a copy button (a line too long for its box is cut where scroll bars are hidden, e.g. on a phone). The box takes the focus on a
  // click, so the arrow keys scroll it and do not step the modes
  function downloadPanel() {
    const r = S.row, P = S.prot, cur = r ? pocketOf(r, S.pk) : null, py = pythonLines(P, r);
    const item = (act, label, file, note) => `<button type="button" class="link" data-act="${act}">${label}</button><span class="pv-file mono">${esc(file)}</span>`
      + (note ? `<span class="small mut pv-dl-note">${note}</span>` : '');
    const pair = r ? (r.sdf ? item('save-sdf', 'Highest-affinity pose (SDF)', `${pairStem()}_top.sdf`) : '')
      + (cur ? item('save-pdbqt', `Pocket ${S.pk}: all modes (PDBQT)`, `${pairStem()}_pk${S.pk}_modes.pdbqt`) : '') : '';
    return `<div role="tabpanel" aria-label="Download"><div class="pv-dl" data-downloads>`
      + (r ? `<h4>${esc(ligOf(r).id)} on ${esc(protName(P))}</h4>${pair}` : `<p class="mut">Pick a ligand in Top hits for its poses.</p>`)
      + `<h4>${esc(protName(P))}</h4>`
      + item('save-protein', 'Protein (PDB)', `PV_${P.name}.pdb`, 'the structure the ligands were docked into')
      + item('save-residues', 'Pocket residues (CSV)', `PV_${P.name}_pockets.csv`, 'every P2Rank pocket: rank, score, centre, residues')
      + `<button type="button" class="link" data-act="image"${S.drawn && !S.imageBusy ? '' : ' disabled'}>Image of the view (PNG)</button><span class="pv-file mono">${esc(imageName())}</span>`
      + `</div><div class="pv-py"><p class="small mut">In Python:${copyButton('the Python lines', py)}</p><pre class="mono" data-python tabindex="0" role="region" aria-label="Python lines">${esc(py)}</pre></div></div>`;
  }
  function pocketsCsv(P) {
    const q = v => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    const rows = P.pockets.map(p => [p.rank, p.score, p.probability, ...(p.center || []), (p.residues || []).join(' '), P.docked.includes(p.rank) ? 'yes' : 'no']);
    return ['rank,score,probability,center_x,center_y,center_z,residues,docked', ...rows.map(x => x.map(q).join(','))].join('\n') + '\n';
  }
  async function copy(text) {
    try { await navigator.clipboard.writeText(text); toast('Copied.'); return; } catch (e) { /* a sandboxed frame: below */ }
    const t = document.createElement('textarea'); t.value = text; t.setAttribute('readonly', ''); t.style.position = 'fixed'; t.style.opacity = '0';
    document.body.appendChild(t); t.select();
    let ok = false; try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    t.remove(); toast(ok ? 'Copied.' : 'Copy blocked here: select the text and copy it.');
  }
  function contactsList() {
    const c = S.contactsAll && S.contactsAll.key === poseKey() ? S.contactsAll : null;
    const list = c ? c.contacts : [], near = list.filter(x => x.dist != null), far = list.filter(x => x.dist == null);
    const label = x => `${x.name}${x.seq}`;
    const head = `<h4>Contacts</h4>`;
    if (!c) return `<div data-contacts="pending">${head}<p class="small mut">Computing…</p></div>`;
    if (!list.length) return `<div data-contacts="${esc(c.source)}">${head}<p class="small mut">No residue within 4 Å of this pose.</p></div>`;
    return `<div data-contacts="${esc(c.source)}">${head}<div class="pv-contacts">${[...near, ...far].map(x => {
      const on = S.focus3d.key === `${x.chain}:${x.seq}`;
      return `<button type="button" class="tag" data-act="contact" data-v="${esc(x.chain)}:${x.seq}" aria-pressed="${on}" data-contact="${esc(label(x))}" `
        + `title="${esc(`${label(x)}${x.dist != null ? ` · ${fmt(x.dist, 1)} Å` : ' · beyond 4 Å'} · ${x.types.length ? x.types.join(', ') : 'contact (no interaction found)'}`)}">`
        + `<span class="mono">${esc(label(x))}</span><span class="mut"> · ${esc(x.types.length ? x.types.join(', ') : 'contact')}</span></button>`;
    }).join('')}</div><p class="small mut" style="margin-top:8px">${near.length} within 4 Å${far.length ? `, ${far.length} more with an interaction` : ''}`
      + `${c.source === 'distance' ? ' (distances only)' : ''}. Pick a residue to look at it.</p></div>`;
  }
  function pocketsPanel() {
    const P = S.prot, n = {};
    for (const r of P.rows) n[r.pocket] = (n[r.pocket] || 0) + 1;
    const plddt = P.pockets.some(p => p.plddt != null), cols = plddt ? 7 : 6;   // Atlas pages: the pockets' pLDDT
    const resOf = id => { const [ch, num] = String(id).split('_'), nm = S.names && S.names[id]; return { chain: ch, seq: parseInt(num, 10), name: nm ? nm[0] : '' }; };
    const rows = P.pockets.map(p => {
      const open = S.openPocket === p.rank;
      const head = `<tr data-act="pocket" data-v="${p.rank}" data-active="${S.focusPocket === p.rank}"><td><span style="display:inline-flex;align-items:center;gap:6px">`
        + `<span style="display:inline-flex;${open ? 'transform:rotate(180deg)' : ''}" class="mut">${ICON.down}</span><i class="dot" style="background:${pocketColor(p.rank)}"></i>${p.rank}</span></td>`
        + `<td class="r num">${fmt(p.score, 1)}</td><td class="r num">${fmt(p.probability, 2)}</td><td class="r num">${p.residues.length}</td>`
        + (plddt ? `<td class="r num" data-plddt>${fmt(p.plddt)}</td>` : '')
        + `<td>${p.docked ? 'yes' : '<span class="mut">no</span>'}</td><td class="r num">${p.docked ? int(n[p.rank] || 0) : '–'}</td></tr>`;
      if (!open) return head;
      const best = P.rows.filter(r => r.pocket === p.rank).sort((a, b) => a.vina - b.vina)[0];
      const rs = p.residues.map(resOf).sort((a, b) => (a.chain < b.chain ? -1 : a.chain > b.chain ? 1 : a.seq - b.seq));
      const chips = rs.map(x => { const on = S.focusRes && S.focusRes.chain === x.chain && S.focusRes.seq === x.seq;
        return `<button type="button" class="tag mono" data-act="res" data-v="${esc(x.chain)}:${x.seq}" aria-pressed="${!!on}">${esc(x.name ? `${x.name}${x.seq}` : `${x.chain}_${x.seq}`)}</button>`; }).join('');
      return head + `<tr class="pv-detail" data-pocket-detail="${p.rank}"><td colspan="${cols}"><dl class="kv" style="margin-top:0"><dt>Centre</dt><dd class="num">${p.center.map(x => fmt(x, 1)).join(', ')} Å</dd>`
        + (plddt ? `<dt>Mean pLDDT</dt><dd class="num">${fmt(p.plddt)}</dd>` : '')
        + `<dt>Docked</dt><dd>${p.docked ? 'yes' : 'no (not among the pockets docked)'}</dd>`
        + (best ? `<dt>Highest-affinity ligand here</dt><dd><button type="button" class="link" data-act="row" data-v="${best.lig}">${esc(ligOf(best).id)}</button><span class="num mut"> · ${fmt(best.vina)} kcal/mol</span></dd>` : '')
        + `</dl><div class="pv-res" aria-label="Residues of pocket ${p.rank}">${chips || '<span class="mut small">no residues listed</span>'}</div></td></tr>`;
    }).join('');
    return `<div role="tabpanel" aria-label="Pockets"><div class="scroll-x"><table class="dt"><thead><tr><th>Pocket</th><th class="r">Score</th><th class="r">Prob.</th>`
      + `<th class="r">Residues</th>` + (plddt ? `<th class="r" title="AlphaFold confidence of the pocket residues (mean pLDDT, 0–100)">pLDDT</th>` : '')
      + `<th>Docked</th><th class="r wrap" title="Ligands listed in Top hits whose highest-affinity pose is in this pocket">Highest-affinity poses</th></tr></thead><tbody>${rows}</tbody></table></div>`
      + `<p class="pv-note">P2Rank pockets of ${esc(protName(P))}; ${P.docked.length} of them were docked. Score and Prob.: P2Rank.${plddt ? ' pLDDT: AlphaFold confidence of the pocket residues.' : ''}`
      + ` Highest-affinity poses: ligands listed in Top hits whose highest-affinity pose`
      + ` is in that pocket. Click a pocket to show it in 3D and list its residues; a residue to look at it.</p></div>`;
  }
  function update(force) {
    const k = poseKey();
    if (k !== lastPoseKey) { lastPoseKey = k; S.hidden = []; }      // a new pose shows every interaction type again
    renderHead(); renderInfo(); renderOverlay(); renderChips(); renderTabs(); renderBody();
    send(force);
  }

  // ------------------------------------------------------------ input
  root.addEventListener('click', e => {
    const t = e.target.closest('[data-act]');
    if (!t || !root.contains(t) || t.disabled || t.tagName === 'SELECT' || t.tagName === 'INPUT') return;
    const a = t.dataset.act, v = t.dataset.v;
    if (a === 'tab') { S.tab = v; update(); }
    else if (a === 'row') { const r = S.prot.rows.find(x => x.lig === Number(v)); if (r) choose(r, null, null, 'best'); }
    else if (a === 'chip') chip(Number(v));
    else if (a === 'pk') choose(S.row, Number(v), 1, 'modes');
    else if (a === 'step') step(Number(v));
    else if (a === 'mode') choose(S.row, S.pk, Number(v), 'modes');
    else if (a === 'best') { if (S.view !== 'best') choose(S.row, null, null, 'best'); }
    else if (a === 'contact') {
      const [chain, seq] = v.split(':');
      if (S.focus3d.key === v) clearFocus3d();
      else { S.focusRes = { chain, seq: Number(seq) }; update(true); }
    }
    else if (a === 'type') { S.hidden = S.hidden.includes(v) ? S.hidden.filter(x => x !== v) : [...S.hidden, v]; update(); }
    else if (a === 'style') { S.pstyle = v; update(); }
    else if (a === 'reset') reset();
    else if (a === 'image') saveImage();
    else if (a === 'save-sdf' && S.row && S.row.sdf) saveText(S.row.sdf, `${pairStem()}_top.sdf`, 'chemical/x-mdl-sdfile');
    else if (a === 'save-pdbqt') { const P = pocketOf(S.row, S.pk); if (P) saveText(P.pdbqt, `${pairStem()}_pk${S.pk}_modes.pdbqt`, 'text/plain'); }
    else if (a === 'save-protein') saveText(S.prot.pdb, `PV_${S.prot.name}.pdb`, 'chemical/x-pdb');
    else if (a === 'save-residues') saveText(pocketsCsv(S.prot), `PV_${S.prot.name}_pockets.csv`, 'text/csv');
    else if (a === 'copy') copy(v);
    else if (a === 'pocket') {
      const k = Number(v);
      S.show = [k]; S.focusPocket = k; S.focusRes = null; S.openPocket = S.openPocket === k ? null : k; update();
    }
    else if (a === 'res') {
      const [chain, seq] = v.split(':'), on = S.focusRes && S.focusRes.chain === chain && S.focusRes.seq === Number(seq);
      S.focusRes = on ? null : { chain, seq: Number(seq) }; update(!on);
    }
    else if (a === 'theme') { S.theme = S.theme === 'dark' ? 'light' : 'dark'; S.themeFixed = true; document.documentElement.dataset.theme = S.theme; update(); }
    else if (a === 'shot-close') closeShot();
  });
  root.addEventListener('change', e => {
    const t = e.target, a = t.dataset && t.dataset.act;
    if (a === 'protein') setProtein(t.value);
    else if (a === 'allmodes') {
      S.allModes = t.checked;
      if (S.allModes && S.view === 'best' && S.row) choose(S.row, S.pk, S.mode, 'modes'); else update();
    }
    else if (a === 'img-scale') { S.imgScale = Number(t.value) || 2; update(); }
    else if (a === 'img-transparent') { S.imgTransparent = t.checked; update(); }
  });
  root.addEventListener('input', e => { if (e.target.id === 'pv-q') { S.q = e.target.value; renderBody(); } });
  root.addEventListener('toggle', e => { if (e.target.matches && e.target.matches('details.pv-legend')) S.legendOpen = e.target.open; }, true);
  // one panel of the protein's information open at a time
  root.addEventListener('toggle', e => { if (e.target.matches && e.target.matches('details.pv-pop') && e.target.open) root.querySelectorAll('details.pv-pop[open]').forEach(d => { if (d !== e.target) d.open = false; }); }, true);
  // a click outside the saved image closes it; a click or tap on the toast hides it (on a phone it covers the page's bottom)
  root.addEventListener('click', e => { if (e.target.id === 'pv-shot') closeShot(); else if (e.target.id === 'pv-toast') hideToast(); });
  // the arrow keys step through the modes, but not in a form field or the Python lines (there they scroll); Escape also works from the lines
  window.addEventListener('keydown', e => {
    const skip = e.key === 'Escape' ? 'input,select,textarea' : 'input,select,textarea,pre';
    if (e.target && e.target.closest && e.target.closest(skip)) return;
    if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && S.row) { e.preventDefault(); step(e.key === 'ArrowLeft' ? -1 : 1); }
    else if (e.key === 'Escape') {
      const pop = root.querySelector('details.pv-pop[open]');
      if (!$('pv-shot').hidden) closeShot(); else if (pop) pop.open = false; else if (S.focus3d.active) clearFocus3d();
    }
  });

  window.__pvState = () => ({
    protein: S.prot && S.prot.id, ligand: S.row ? ligOf(S.row).id : null, view: S.view, pocket: S.pk, mode: S.mode, allModes: S.allModes,
    show: S.show.slice(), surfaces: S.surfaces.slice(), tab: S.tab, hidden: S.hidden.slice(), style: S.pstyle, theme: S.theme,
    focus: { ...S.focus3d }, focusPocket: S.focusPocket, focusResidue: S.focusRes, drawn: S.drawn, status: S.status,
    contacts: S.contactsAll && S.contactsAll.key === poseKey() ? S.contactsAll.contacts.length : null, poseName: S.row ? poseName() : null,
    imageName: imageName(), proteins: D.proteins.map(p => p.id),
  });

  // ------------------------------------------------------------ start
  (async () => {
    try { [D, A] = await Promise.all([unpack('pv-data'), unpack('pv-assets')]); }
    catch (e) { root.innerHTML = `<p class="pv-loading">Cannot open this PocketVina view: ${esc(e && e.message || e)}</p>`; return; }
    const setting = document.documentElement.dataset.themeSetting;
    const dark = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
    S.theme = setting === 'dark' || setting === 'light' ? setting : dark && dark.matches ? 'dark' : 'light';
    S.themeFixed = setting === 'dark' || setting === 'light';
    document.documentElement.dataset.theme = S.theme;
    if (dark && dark.addEventListener) dark.addEventListener('change', () => { if (!S.themeFixed) { S.theme = dark.matches ? 'dark' : 'light'; document.documentElement.dataset.theme = S.theme; update(); } });
    const I = D.initial;
    S.prot = D.proteins.find(p => p.id === I.protein) || D.proteins[0];
    const row = I.ligand != null ? S.prot.rows.find(r => r.lig === I.ligand) : null;
    if (row) { S.row = row; S.view = I.view; S.pk = I.pocket; S.mode = I.mode; S.allModes = !!I.allModes; }
    S.show = (I.show || []).slice(); S.selShow = S.show.slice(); S.userShow = !!I.userShow; S.tab = row ? I.tab : 'top';
    lastPoseKey = poseKey();
    layout();
    update();
  })();
})();
