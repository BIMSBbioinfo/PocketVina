'use strict';
/* PocketVina: a copy of the PocketVina Atlas site's public/viewer/viewer.js (commit 40e8d59), adapted for a self-contained
 * page (pocketvina/viewer/README.md: every change is marked "PocketVina:"). The page sets window.__pvBridge before this file runs:
 * {post(m): a message to the page, structure(path): the protein's PDB text}; this file adds bridge.receive(m), which takes the
 * messages the site posts. Without a bridge it works as on the site. */
/* Runs inside viewer.html (isolated Mol*) of PocketVina Atlas. Messages accepted, from the same-origin parent only:
 *   {type:'show', spec}   draw a scene
 *   {type:'reset'}        clear a focus or a clicked line, then reset the camera
 *   {type:'clear-focus'}  clear a focus or a clicked line (round 7: the page's Esc)
 *   {type:'activity'}     the page is in use (round 12: idle work waits; the frame also listens to the page's own events)
 *   {type:'prefetch', protein}  round 12: the structure the next 'show' will name; fetched and read while Mol* starts
 * spec = {theme, protein:'structures/AF-…pdb', gene, pockets:[{rank,residues}], showPockets:[ranks], proteinStyle:'cartoon'|'surface'|'sticks' (round 13),
 *         focusPocket, focusResidue:{chain,seq}, hiddenInteractions:[types],
 *         ligand:{text, format:'pdbqt'|'sdf', model:0-based index, nModels, allModes, key}}
 *   gene (round 10): the protein's gene symbol, for the structure's name in Mol*'s hover box ("EGFR (AF-P00533-F1)"; optional)
 * Messages sent: viewer-ready, viewer-drawn, viewer-error, key (arrows), residues (names + pLDDT of the protein, once per protein),
 *   contacts (residues within 4 Å of the drawn pose and the interaction types Mol* found),
 *   focus {active, key: 'A:716' | 'ligand' | null, line: boolean} on every change of Mol*'s focus or of a clicked interaction line (round 7),
 *   key 'Escape' when Esc is pressed in this frame with nothing to clear (the page then leaves full screen),
 *   surfaces {shown: [pocket ranks]} after every drawing and every change of the pocket surfaces shown (round 8; the pocket chips).
 * A 'show' whose spec differs from the last drawing only in showPockets shows or hides surfaces without a new drawing (round 8).
 * Mol* is locked down: no panels or controls, no file drops, no context menu, and only whitelisted structure paths.
 *
 * Round 4, section 4: the drawn pose is merged into the protein as residue LIG (chain L), so Mol* computes protein-ligand
 * interactions; ligand, residues and modes are coloured by element with their own carbon colour.
 *
 * Round 12 (§2.1–2.4): no MVS and no data: URLs. The scene is built with Mol*'s plugin API and updated in place:
 *   protein   parsed and built once per protein (cartoon; one component per pocket, its surface built when first shown, the others
 *             at idle time);
 *   site      per pose: a small structure of the pose (chain L) and the protein residues within SITE_STRUCT_A Å, merged as before, so
 *             Mol* computes the protein-ligand interactions on it (component: the pose and the 5 Å residues, as before) and its focus
 *             on the pose has every residue around it; the pose is drawn from it;
 *   residues  sticks and labels of the residues near the pose: components of the protein structure (a click on one opens Mol*'s focus
 *             in the whole protein, as before);
 *   modes     "Show all modes": a structure per mode from the PDBQT text.
 * A drug, pocket or mode change replaces only the site, the residues, the modes and the camera; a theme change recolours in place.
 * Drawing is progressive: {type:'viewer-frame'} once the new scene's first frame (cartoon, pose) is on screen, then the
 * interactions, sticks and labels, then 'viewer-drawn'. Only the newest spec waiting is drawn (a burst of specs draws once).
 * Hooks: __pvDebug.timings {structureLoaded, firstFrame, drawn, switchMs, switches: [{kind, firstFrameMs, drawnMs}]} and
 * __pvDebug.counters {proteinParses, proteinBuilds, siteBuilds, surfaceBuilds}.
 *
 * Mol* names used here (agent B's small bundle provides exactly these; keep this list current):
 *   molstar.Viewer.create(elementId, options) -> Promise<{ plugin }>
 *   molstar.lib.plugin.StateTransforms: Data.RawData; Model.TrajectoryFromPDB (variant 'pdb' / 'pdbqt'), Model.ModelFromTrajectory,
 *     Model.StructureFromModel, Model.StructureComponent (type 'static' 'polymer' / 'bundle'); Representation.StructureRepresentation3D
 *   molstar.lib.structure: StructureElement.Bundle.fromLoci, StructureElement.Loci.extendToWholeResidues, Structure.Empty,
 *     Model.AsymIdOffset (round 13: the focus colours)
 *   molstar.lib.plugin.PluginStateObject.Molecule.Model.is (round 13)
 *   molstar.lib.loci: Loci.getBoundingSphere
 *   molstar.lib.text: getFontAtlas (optional: B's bundle; glyphs drawn ahead, see scheduleGlyphs)
 *   through the plugin: state.data (build: toRoot/to/apply/update/delete/commit, cells, tree.root, updateCellState),
 *     state.behaviors (build, cells; behaviour 'ms-plugin.create-structure-focus-representation' for the focus colours),
 *     builders.structure.representation.addRepresentation (the interactions), representation.structure (registry.get(type).getParams,
 *     themes.colorThemeRegistry, themes.sizeThemeRegistry), customStructureProperties.get('molstar_computed_interactions'),
 *     managers.structure.focus (setFromLoci, clear, behaviors.current), managers.structure.measurement (addDistance, addLabel),
 *     managers.camera (focusLoci, reset), managers.lociLabels (addProvider, providers), managers.interactivity, behaviors.interaction
 *     (click, hover), canvas3d (camera: state, project, getTargetDistance, getInvariantFocus, getSnapshot, changed, transition, viewport,
 *     projection, projectionView; requestCameraReset with a snapshot object or function; commit; identify; getLoci; setProps; props;
 *     webgl; interaction.hover)
 *   representation types cartoon, molecular-surface, ball-and-stick, interactions, label; colour themes uniform, element-symbol
 *   (carbon uniform / chain-id), interaction-type; size themes uniform, physical. No MVS, no eval. */
(() => {
  const POCKET_COLORS = ['#5c93e6', '#6a47b3', '#e65ce0', '#b34761', '#e69e5c'];
  // PocketVina: P2Rank ranks up to 127 (the site docked 5 pockets); the colours repeat from rank 6 on (pocketvina.viz.POCKET_COLORS)
  const MAX_POCKETS = 127, pocketColor = k => POCKET_COLORS[(k - 1) % POCKET_COLORS.length];
  const LIGAND_C = 0xE3B35A;                                   // ligand carbon (gold); protein carbon is grey
  const PROTEIN = { light: { cartoon: '#C9CFD6', carbon: 0xA9B1BB, bg: 0xf4f4f6 }, dark: { cartoon: '#8C96A3', carbon: 0x8C96A3, bg: 0x1c1c20 } };
  const MODE_DARK = [0x8A5A0B], MODE_LIGHT = [0xF3E2B3];       // "show all modes": carbon from dark (mode 1) to light (last mode)
  const CONTACT_A = 4.0, SITE_A = 5.0;                         // residues shown as sticks; residues in the interaction site
  const SITE_STRUCT_A = 8.0;                                   // round 12: protein residues in the site structure (Mol*'s focus on the pose: 5 Å)
  // Mol*'s orientation axes (bottom left) are off: on phones they showed next to the closed legend as a green bar
  const NO_AXES = { helper: { axes: { name: 'off', params: {} } } };
  const MAX_LABELS = 8, LABEL_SIZE = 1.3;                                        // labels only on residues with an interaction (round 7: all non-hydrophobic ones, hydrophobic-only up to 8 in all)
  const LABEL_Z = 10;          // labels drawn 10 Å towards the camera, in front of nearby helices and the pose (round 6, 8: "GLN 32" showed as "2")
  // Round 8 (§2.2): Mol* draws text in Å, so labels grew with every zoom step. Each label's size is set from the camera so that its text
  // is TEXT_PX CSS px high where Mol* draws it (the em of Mol*'s font: glyphs drawn at n px in a line of round(n · 1.58) px, so 256 / 405
  // of a text unit), updated when it is off by more than SIZE_STEP; up to FAR_ZOOM × the pair's default camera distance, beyond that
  // it shrinks with the distance again (the whole protein in view: the labels of one pocket would cover it).
  // Round 12: Mol*'s font quality 2 (was its default, 3): the same look at these sizes (shots in _work/round12/A/NOTES.md), a font
  // texture of 24 MB instead of 43 MB and glyphs drawn ~2× faster. Mol*'s atlas draws a glyph n = 64 × (quality + 1) px high in a line
  // of n + n/4 + n/3 px (3: 256 / 405 = 0.632, 2: 192 / 304 = 0.632)
  const FONT_QUALITY = 2, EM_N = 64 * (FONT_QUALITY + 1);
  const TEXT_PX = 14, EM = EM_N / Math.round(EM_N + EM_N / 4 + EM_N / 3), SIZE_STEP = 0.06, FAR_ZOOM = 3.5;
  const FOCUS_SIZE = 0.8, RES_SIZE = 0.5;                                        // a focused residue: thicker sticks, carbon stays grey (round 6, 6)
  const ACCENT = { light: 0x1f4e79, dark: 0x8cb4dc };                            // the site's accent: outline of a selection
  // Data root: a same-origin path given by the parent page (?root=/data/v1/), checked; default ../data/v1/.
  const Q = new URLSearchParams(location.search);
  const ROOT = Q.get('root');
  // PocketVina: the page's bridge (see the top); a page without a server (an iframe srcdoc document) has no data root
  const BRIDGE = window.__pvBridge || null;
  const DATA = BRIDGE ? null : ROOT && /^\/[A-Za-z0-9._~\/-]*\/$/.test(ROOT) && !ROOT.includes('..') ? new URL(ROOT, location.origin) : new URL('../data/v1/', location.href);
  const SAFE_STRUCTURE = /^structures\/AF-[A-Z0-9]+-F1-model_v6\.pdb$/;
  const MAX_LIGAND_CHARS = 8e6;
  const safeUrl = p => { if (typeof p !== 'string' || !SAFE_STRUCTURE.test(p)) throw new Error('rejected path'); return new URL(p, DATA).href; };
  // the page's theme from the start, so nothing light shows while Mol* loads in the dark theme
  document.documentElement.dataset.theme = Q.get('theme') === 'dark' ? 'dark' : 'light';

  // Mol* would open files dropped onto it: swallow all drag-and-drop and the context menu before Mol* sees them.
  for (const t of ['dragenter', 'dragover', 'dragleave', 'drop'])
    window.addEventListener(t, e => { e.preventDefault(); e.stopPropagation(); if (e.dataTransfer) e.dataTransfer.dropEffect = 'none'; }, true);
  window.addEventListener('contextmenu', e => e.preventDefault(), true);
  // after a click in 3D the focus is in this frame: pass the mode-stepping arrows to the page; Esc clears a focus or a clicked line
  // here (round 7), and with nothing to clear goes to the page (leave full screen)
  // PocketVina: the keys go to the page with post() (below: through the bridge when there is one)
  window.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); e.stopPropagation(); post({ type: 'key', key: e.key }); }
    else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); if (picked.focus || picked.line || picked.pending) clearPicked(); else post({ type: 'key', key: 'Escape' }); }
  }, true);
  // PocketVina: through the bridge when there is one
  const post = m => (BRIDGE ? BRIDGE.post(m) : parent.postMessage(m, location.origin));

  // pocket residues "A_793" -> residue keys "A:793" (the keys of this file)
  const residueKeys = ids => ids.map(r => { const [ch, n] = String(r).split('_'); return [ch, parseInt(n, 10)]; })
                                .filter(([ch, n]) => /^[A-Za-z0-9]{1,4}$/.test(ch) && Number.isFinite(n)).map(([ch, n]) => `${ch}:${n}`);
  const hexInt = s => parseInt(s.slice(1), 16);
  const mix = (a, b, t) => { const ch = s => [(a >> s) & 255, (b >> s) & 255]; return [16, 8, 0].reduce((v, s) => { const [x, y] = ch(s); return v | (Math.round(x + (y - x) * t) << s); }, 0); };

  // ---------------------------------------------------------------- protein: fetched once, parsed for contacts and names
  const proteins = new Map();
  // round 12 (B): a page may reuse the frame of another protein: 'residues' goes out again whenever a show names a protein other than
  // the last one sent, also from the cache
  // Round 12: also read ahead ({type:'prefetch'}, announce false): one load per path at a time; 'residues' only for a 'show' (a page
  // drops frame messages that come before its first 'show')
  let residuesSent = '';
  const proteinLoads = new Map();
  async function protein(path, announce = true) {
    let p = proteins.get(path);
    if (!p) {
      if (!proteinLoads.has(path)) proteinLoads.set(path, loadProtein(path).finally(() => proteinLoads.delete(path)));
      p = await proteinLoads.get(path);
    }
    if (announce && residuesSent !== path) { residuesSent = path; post({ type: 'residues', protein: path, residues: p.names }); }
    return p;
  }
  async function loadProtein(path) {
    // round 12 (B): viewer/structure-fetch.js fetches <url>.gz (gzip checked, unpacked with DecompressionStream), else the plain file
    // PocketVina: from the bridge (the page holds the text) when there is one
    const text = await (BRIDGE ? BRIDGE.structure(path) : window.fetchStructureText ? window.fetchStructureText(safeUrl(path))
      : fetch(safeUrl(path), { credentials: 'omit' }).then(r => { if (!r.ok) throw new Error(`structure: ${r.status}`); return r.text(); }));
    const lines = [], atoms = [], names = {};
    for (const l of text.split('\n')) {
      if (!l.startsWith('ATOM') && !l.startsWith('HETATM')) continue;
      lines.push(l);
      const a = { name: l.slice(12, 16).trim(), res: l.slice(17, 20).trim(), ch: l.slice(21, 22).trim() || 'A', seq: parseInt(l.slice(22, 26), 10),
        x: +l.slice(30, 38), y: +l.slice(38, 46), z: +l.slice(46, 54), b: +l.slice(60, 66), el: (l.slice(76, 78).trim() || l.slice(12, 14).trim()[0]).toUpperCase() };
      atoms.push(a);
      if (a.name === 'CA') names[`${a.ch}_${a.seq}`] = [a.res, a.b];
    }
    const p = { lines, atoms, names, serial: atoms.length + 1 };
    proteins.clear(); proteins.set(path, p);                     // one protein at a time
    return p;
  }

  // ---------------------------------------------------------------- ligand: PDBQT (one model) or SDF (V2000) -> atoms
  const AD = { A: 'C', C: 'C', N: 'N', NA: 'N', NS: 'N', OA: 'O', OS: 'O', O: 'O', SA: 'S', S: 'S', HD: 'H', HS: 'H', H: 'H', P: 'P', F: 'F', CL: 'Cl', Cl: 'Cl', BR: 'Br', Br: 'Br', I: 'I' };
  function ligandAtoms(L, model) {
    const out = [], bonds = [];
    if (L.format === 'pdbqt') {
      let m = -1, inModel = !/^MODEL/m.test(L.text);
      for (const l of L.text.split('\n')) {
        if (l.startsWith('MODEL')) { m++; inModel = m === model; continue; }
        if (l.startsWith('ENDMDL')) { if (inModel && m === model) break; continue; }
        if (!inModel || !(l.startsWith('ATOM') || l.startsWith('HETATM'))) continue;
        const t = l.slice(77, 79).trim();
        out.push({ x: +l.slice(30, 38), y: +l.slice(38, 46), z: +l.slice(46, 54), el: AD[t] || AD[t.toUpperCase()] || (t[0] || 'C') });
      }
    } else {
      const ls = L.text.split('\n'), n = parseInt(ls[3].slice(0, 3), 10), nb = parseInt(ls[3].slice(3, 6), 10);
      for (let i = 0; i < n; i++) { const l = ls[4 + i]; out.push({ x: +l.slice(0, 10), y: +l.slice(10, 20), z: +l.slice(20, 30), el: l.slice(31, 34).trim() }); }
      for (let i = 0; i < nb; i++) { const l = ls[4 + n + i]; bonds.push([parseInt(l.slice(0, 3), 10) - 1, parseInt(l.slice(3, 6), 10) - 1]); }
    }
    return { atoms: out.filter(a => Number.isFinite(a.x)), bonds };
  }
  const pad = (s, n) => String(s).padStart(n);
  function hetatm(serial, name, el, a) {
    const nm = name.length < 4 ? ' ' + name.padEnd(3) : name.slice(0, 4);
    return 'HETATM' + pad(serial, 5) + ' ' + nm + ' LIG L   1    ' + pad(a.x.toFixed(3), 8) + pad(a.y.toFixed(3), 8) + pad(a.z.toFixed(3), 8) + '  1.00  0.00          ' + pad(el.toUpperCase(), 2);
  }
  // residues within `cut` Å of any ligand heavy atom, nearest distance first
  function near(P, lig, cut) {
    const heavy = lig.filter(a => a.el !== 'H'); if (!heavy.length) return [];
    const lo = [0, 1, 2].map(k => Math.min(...heavy.map(a => [a.x, a.y, a.z][k])) - cut), hi = [0, 1, 2].map(k => Math.max(...heavy.map(a => [a.x, a.y, a.z][k])) + cut);
    const best = new Map(), c2 = cut * cut;
    for (const a of P.atoms) {
      if (a.el === 'H' || a.x < lo[0] || a.x > hi[0] || a.y < lo[1] || a.y > hi[1] || a.z < lo[2] || a.z > hi[2]) continue;
      for (const b of heavy) { const d = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2;
        if (d <= c2) { const k = `${a.ch}_${a.seq}`, v = best.get(k); if (!v || d < v.d2) best.set(k, { ch: a.ch, seq: a.seq, res: a.res, d2: d }); } }
    }
    return [...best.values()].sort((x, y) => x.d2 - y.d2).map(r => ({ chain: r.ch, seq: r.seq, name: r.res, dist: Math.round(Math.sqrt(r.d2) * 100) / 100 }));
  }

  // ---------------------------------------------------------------- the scene in Mol*'s state (round 12: no MVS)
  // Round 10 (§2): the structure's name in Mol*'s hover box. Mol* names a model after its entry ID (a PDB file's HEADER ID code, which
  // these files leave blank) or else the label of the data it was parsed from: every data cell here (protein, site, modes) is labelled
  // with the gene from the page's spec and the AlphaFold model, "EGFR (AF-P00533-F1)"; the model alone if the spec has no gene (the
  // other modes: the same name, the box adds "Model n"). Mol* puts the name into the box as HTML: letters, digits and . _ - only.
  // PocketVina: the page names the protein (spec.name, its id) and the pose (spec.ligand.name, its PV_ name): the site and modes of a
  // pose (pose = true) are named after the pose
  function structureName(spec, pose) {
    const own = pose ? spec.ligand && spec.ligand.name : spec.name;
    if (typeof own === 'string' && /^[A-Za-z0-9._-]{1,255}$/.test(own)) return own;
    const m = /^structures\/(AF-[A-Z0-9]+-F1)-model_v6\.pdb$/.exec(spec.protein), af = m ? m[1] : 'AlphaFold model';
    const gene = typeof spec.gene === 'string' && /^[A-Za-z0-9][A-Za-z0-9._-]{0,23}$/.test(spec.gene) ? spec.gene : '';
    return gene ? `${gene} (${af})` : af;
  }
  const ST = () => molstar.lib.plugin.StateTransforms;
  const SD = () => viewer.plugin.state.data, RB = () => viewer.plugin.builders.structure.representation;
  // test and measurement hooks (contract, round 12): cumulative counters; times as performance.now() of this frame (ms)
  const counters = { proteinParses: 0, proteinBuilds: 0, siteBuilds: 0, surfaceBuilds: 0 };
  // (and, for the breakdown of an open: scriptStart = this file running, pluginReady = Mol* ready, firstShow = the first 'show')
  const timings = { structureLoaded: null, firstFrame: null, drawn: null, switchMs: null, switches: [], scriptStart: Math.round(performance.now()), pluginReady: null, firstShow: null };
  function debug() { const D = window.__pvDebug = window.__pvDebug || {}; D.timings = timings; D.counters = counters; return D; }
  // what Mol*'s state holds now: the protein (path, refs, residue index, pocket components and surfaces), the site of the drawn pose,
  // the other modes, the residue components (sticks, labels)
  const scene = { protein: null, site: null, modes: null, residues: new Map() };
  const cellOf = ref => (ref ? SD().cells.get(ref) : null);
  const dataOf = ref => { const c = cellOf(ref); return c && c.obj ? c.obj.data : null; };
  // text -> data (labelled with the structure's name) -> trajectory -> model -> structure, on a state builder
  function applyStructure(b, text, label, variant, tag) {
    const data = b.toRoot().apply(ST().Data.RawData, { data: text, label }, { tags: [`${tag}-data`] });
    const struct = data.apply(ST().Model.TrajectoryFromPDB, { variant }).apply(ST().Model.ModelFromTrajectory, { modelIndex: 0 })
      .apply(ST().Model.StructureFromModel, { type: { name: 'model', params: {} } }, { tags: [tag] });
    return { data: data.ref, struct: struct.ref };
  }
  // residue "A:716" -> per unit the indices of its atoms in the unit (built once per structure): loci and bundles of residues
  function residueIndex(structure) {
    const m = new Map();
    for (const unit of structure.units) {
      if (!atomic(unit)) continue;
      const h = unit.model.atomicHierarchy, els = unit.elements;
      for (let i = 0; i < els.length; i++) {
        const el = els[i], k = `${h.chains.auth_asym_id.value(h.chainAtomSegments.index[el])}:${h.residues.auth_seq_id.value(h.residueAtomSegments.index[el])}`;
        let v = m.get(k); if (!v) m.set(k, v = new Map());
        let a = v.get(unit); if (!a) v.set(unit, a = []); a.push(i);
      }
    }
    return m;
  }
  function residuesLoci(structure, index, keys) {
    if (!structure || !index) return null;
    const per = new Map();
    for (const k of new Set(keys)) { const v = index.get(k); if (v) for (const [unit, idx] of v) { const a = per.get(unit); if (a) a.push(...idx); else per.set(unit, idx.slice()); } }
    const elements = [...per].map(([unit, a]) => ({ unit, indices: Int32Array.from(a.sort((x, y) => x - y)) }));
    return elements.length ? { kind: 'element-loci', structure, elements } : null;
  }
  const bundleOf = (structure, index, keys) => { const l = residuesLoci(structure, index, keys); return l ? molstar.lib.structure.StructureElement.Bundle.fromLoci(l) : null; };
  const component = (b, parent, bundle, label, tag) => b.to(parent).apply(ST().Model.StructureComponent, { type: { name: 'bundle', params: bundle }, nullIfEmpty: true, label }, { tags: [tag] });
  // a representation's parameters (what Mol*'s builder fills in: each parameter's default for the structure, then ours), applied on a
  // state builder; the parent may be new in the same builder (Mol*'s buildRepresentation skips such a parent)
  function reprParams(structure, p) {
    const R = viewer.plugin.representation.structure, ctx = { structure }, defaults = pd => { const o = {}; for (const [k, v] of Object.entries(pd)) o[k] = v.defaultValue; return o; };
    const type = R.registry.get(p.type), color = R.themes.colorThemeRegistry.get(p.color), size = R.themes.sizeThemeRegistry.get(p.size);
    return { type: { name: p.type, params: { ...defaults(type.getParams(R.themes, structure)), ...(p.typeParams || {}) } },
      colorTheme: { name: p.color, params: { ...defaults(color.getParams(ctx)), ...(p.colorParams || {}) } },
      sizeTheme: { name: p.size, params: { ...defaults(size.getParams(ctx)), ...(p.sizeParams || {}) } } };
  }
  function representation(b, parent, structure, p, tag, initialState) {
    return b.to(parent).apply(ST().Representation.StructureRepresentation3D, reprParams(structure, p), { tags: [tag], ...(initialState ? { state: initialState } : {}) }).ref;
  }
  // ball-and-stick as MVS drew it (type size 0.5, aspect 0.5; the stick size in a uniform size theme); `full`: heteroatoms at full colour,
  // else faded (round 7: contact-only residues)
  const sticks = (carbon, full, size, ignoreHydrogens, dark, alpha) => ({ type: 'ball-and-stick', typeParams: { sizeFactor: 0.5, sizeAspectRatio: 0.5, ignoreHydrogens, ...(alpha == null ? {} : { alpha }) },
    color: 'element-symbol', colorParams: (full ? elementTheme(carbon) : elementTheme(carbon, -1.5, dark ? -0.4 : 0.8)).params, size: 'uniform', sizeParams: { value: size } });
  // everything of the last protein goes (another protein in the same frame: the plugin stays)
  async function clearScene() {
    const S = SD(), root = S.tree.root.ref, b = S.build(); let n = 0;
    for (const [ref, c] of S.cells) if (ref !== root && c.transform.parent === root) { b.delete(ref); n++; }
    if (n) await b.commit();
    scene.protein = scene.site = scene.modes = null; scene.residues = new Map(); labelInfo = [];
  }
  // the protein: parsed and built once (cartoon, a component per pocket; their surfaces later, see buildSurfaces)
  async function buildProtein(spec, P) {
    await clearScene();
    const dark = spec.theme === 'dark', b = SD().build();
    const { data, struct } = applyStructure(b, P.lines.join('\n') + '\nEND\n', structureName(spec), 'pdb', 'pv-protein');
    const poly = b.to(struct).apply(ST().Model.StructureComponent, { type: { name: 'static', params: 'polymer' }, nullIfEmpty: true, label: 'Polymer' }, { tags: ['pv-polymer'] });
    await b.commit(); mark('Mol* parse');
    counters.proteinParses++;
    const s = dataOf(struct), index = residueIndex(s), pockets = new Map(), b2 = SD().build();
    const polyRepr = representation(b2, poly.ref, s, polymerStyle('cartoon', dark), 'pv-cartoon');
    for (const pk of (spec.pockets || [])) {
      const rank = Number(pk.rank), keys = residueKeys(pk.residues || []);
      if (!(rank >= 1 && rank <= MAX_POCKETS) || pockets.has(rank)) continue;          // PocketVina: was rank <= 5
      const bundle = bundleOf(s, index, keys);
      if (bundle) pockets.set(rank, { keys, comp: component(b2, struct, bundle, `Pocket ${rank}`, `pv-pocket:${rank}`).ref, surf: null });
    }
    await b2.commit(); mark('cartoon, pockets');
    counters.proteinBuilds++;
    scene.protein = { path: spec.protein, pocketsKey: JSON.stringify(spec.pockets || []), data, struct, index, pockets, labelSlots: [], stickSlots: [],
      polyRepr, style: `cartoon|${dark}` };
    timings.structureLoaded = Math.round(performance.now()); timings.firstFrame = timings.drawn = null;
  }
  // Round 13: the protein's style, picked with the buttons beside the legend (as Mol*'s representation choices): cartoon (default), a
  // molecular surface (translucent, so the pose and the residue sticks inside stay visible) or ball-and-stick (grey carbon, thinner
  // than the residue sticks). One representation (pv-cartoon) changes its type in place: no other part of the scene is rebuilt.
  const STYLES = ['cartoon', 'surface', 'sticks'];
  const styleOf = spec => (STYLES.includes(spec && spec.proteinStyle) ? spec.proteinStyle : 'cartoon');
  function polymerStyle(style, dark) {
    const C = PROTEIN[dark ? 'dark' : 'light'];
    if (style === 'surface') return { type: 'molecular-surface', typeParams: { alpha: 0.55, ignoreHydrogens: true }, color: 'uniform', colorParams: { value: hexInt(C.cartoon) },
      size: 'physical', sizeParams: { scale: 1 } };
    if (style === 'sticks') return sticks(C.carbon, true, 0.3, true, dark);
    return { type: 'cartoon', color: 'uniform', colorParams: { value: hexInt(C.cartoon) }, size: 'uniform', sizeParams: { value: 1 } };
  }
  // The camera and Mol*'s focus stay as they are (round 13b, the user: "do as for the modes"): Mol* resets its camera by itself when a
  // new render object reaches outside the scene's old bounds and the current view (a surface or all atoms reach past the cartoon), so
  // that check is off while the style changes and the scene is committed at once; the camera's depth range then follows the new scene.
  async function applyStyle(spec) {
    const prot = scene.protein, style = styleOf(spec), dark = spec.theme === 'dark', key = `${style}|${dark}`;
    if (!prot || !prot.polyRepr || prot.style === key || !cellOf(prot.polyRepr)) return false;
    const c3 = viewer.plugin.canvas3d, manual = !!(c3 && c3.props.camera.manualReset);
    if (c3 && !manual) c3.setProps({ camera: { manualReset: true } });
    try {
      await SD().build().to(prot.polyRepr).update(reprParams(dataOf(prot.struct), polymerStyle(style, dark))).commit();
      if (c3) c3.commit(true);
    } finally {
      if (c3 && !manual) {
        c3.setProps({ camera: { manualReset: false } });
        try { c3.camera.setState({ radiusMax: c3.boundingSphere.radius * (c3.props.sceneRadiusFactor || 1) }, 0); } catch (e) { /* the next scene change sets it */ }
      }
    }
    prot.style = key; debug().style = style;
    lastCam = ''; scheduleFit();                                     // the labels against the new solids
    return true;
  }
  // the pose: its atoms, the residues within 4 Å (sticks) and 5 Å (the interaction site), and the site structure's text: the protein
  // residues within SITE_STRUCT_A Å (their lines of the protein file) and the pose as residue LIG of chain L (round 4)
  function ligandOf(L, P) {
    if (!(L && typeof L.text === 'string' && L.text.length < MAX_LIGAND_CHARS && (L.format === 'pdbqt' || L.format === 'sdf'))) return null;
    const model = Math.max(0, Math.min(Number(L.model) || 0, (Number(L.nModels) || 1) - 1));
    const { atoms, bonds } = ligandAtoms(L, model);
    if (!atoms.length) return null;
    const counts = {}, het = atoms.map((a, i) => { counts[a.el] = (counts[a.el] || 0) + 1; return hetatm(P.serial + i, `${a.el.toUpperCase()}${counts[a.el]}`.slice(0, 4), a.el, a); });
    const conect = bonds.filter(([x, y]) => x >= 0 && y >= 0 && x < atoms.length && y < atoms.length).map(([x, y]) => 'CONECT' + pad(P.serial + x, 5) + pad(P.serial + y, 5));
    const keep = new Set(near(P, atoms, SITE_STRUCT_A).map(r => `${r.chain}_${r.seq}`)), lines = [];
    for (let i = 0; i < P.lines.length; i++) { const a = P.atoms[i]; if (keep.has(`${a.ch}_${a.seq}`)) lines.push(P.lines[i]); }
    return { model, atoms, text: [...lines, 'TER', ...het, ...conect, 'END', ''].join('\n'), contacts: near(P, atoms, CONTACT_A), site: near(P, atoms, SITE_A) };
  }
  // the site structure of the pose (replaced when the pose or the interaction types hidden in the legend change: Mol* computes the
  // interactions once per structure): the pose's sticks, and the component Mol* computes the interactions on (the pose and the 5 Å
  // residues, as before)
  async function syncSite(spec, lig, hidden) {
    const key = lig ? `${hidden.join(',')}|${lig.text}` : '';
    if ((scene.site ? scene.site.key : '') === key) return false;
    const b = SD().build();
    if (scene.site && cellOf(scene.site.data)) b.delete(scene.site.data);
    scene.site = null;
    if (!lig) { await b.commit(); return true; }
    const { data, struct } = applyStructure(b, lig.text, structureName(spec, true), 'pdb', 'pv-site');   // PocketVina: the pose's name
    await b.commit();
    const s = dataOf(struct), index = residueIndex(s), b2 = SD().build();
    const pose = bundleOf(s, index, ['L:1']), inter = bundleOf(s, index, ['L:1', ...lig.site.map(r => `${r.chain}:${r.seq}`)]);
    if (pose) representation(b2, component(b2, struct, pose, 'Pose', 'pv-lig').ref, s, sticks(LIGAND_C, true, 1, true), 'pv-lig-repr');
    if (inter) component(b2, struct, inter, 'Interaction site', 'pv-site-inter');
    await b2.commit();
    counters.siteBuilds++;
    scene.site = { key, data, struct, index };
    return true;
  }
  // "Show all modes": the other modes of the pocket, thin and translucent, carbon from dark (best) to light; a structure per mode
  async function syncModes(spec, lig) {
    const L = spec.ligand, on = !!(lig && L.allModes && L.format === 'pdbqt' && L.nModels > 1), key = on ? `${lig.model}|${L.nModels}|${L.text}` : '';
    if ((scene.modes ? scene.modes.key : '') === key) return;
    const b = SD().build();
    if (scene.modes && cellOf(scene.modes.data)) b.delete(scene.modes.data);
    scene.modes = null;
    if (on) {
      const n = L.nModels, data = b.toRoot().apply(ST().Data.RawData, { data: L.text, label: structureName(spec, true) }, { tags: ['pv-modes-data'] });   // PocketVina: the pose's name
      const traj = data.apply(ST().Model.TrajectoryFromPDB, { variant: 'pdbqt' });
      for (let i = 0; i < n; i++) {
        if (i === lig.model) continue;
        const s = traj.apply(ST().Model.ModelFromTrajectory, { modelIndex: i }).apply(ST().Model.StructureFromModel, { type: { name: 'model', params: {} } }, { tags: ['pv-modes'] });
        representation(b, s.ref, siteStructure() || molstar.lib.structure.Structure.Empty, sticks(mix(MODE_DARK[0], MODE_LIGHT[0], i / (n - 1)), true, i === 0 ? 0.9 : 0.55, true, false, i === 0 ? 0.8 : 0.5), `pv-mode-repr:${i}`);
      }
      scene.modes = { key, data: data.ref };
    }
    await b.commit();
  }
  // Round 12 (§2.3): pocket surfaces are built when first shown (the shown pocket first), the others at idle time (scheduleIdle).
  // POCKET_SURFACE: the surface's parameters besides its colour and opacity. Round 12: a molecular surface on a 0.7 Å grid instead of
  // Mol*'s automatic quality (EGFR pockets 1 / 3: 258 / 142 ms -> 164 / 98 ms; ~1 % of the pixels differ, by little; lower quality
  // and Gaussian surfaces looked different: the comparison is in _work/round12/A/NOTES.md)
  const POCKET_SURFACE = { type: 'molecular-surface', typeParams: { quality: 'custom', resolution: 0.7 } };
  const surfaceAlpha = lig => (lig ? 0.15 : 0.25);
  async function buildSurfaces(ranks, alpha, hidden) {
    const prot = scene.protein; if (!prot) return 0;
    const b = SD().build(); let n = 0;
    for (const k of ranks) {
      const pk = prot.pockets.get(k); if (!pk || pk.surf) continue;
      pk.surf = representation(b, pk.comp, dataOf(pk.comp) || dataOf(prot.struct), { type: POCKET_SURFACE.type, typeParams: { ...POCKET_SURFACE.typeParams, alpha }, color: 'uniform',
        colorParams: { value: hexInt(pocketColor(k)) }, size: 'physical', sizeParams: { scale: 1 } }, `pv-pocket-surface:${k}`, { isHidden: !!hidden(k) });   // PocketVina: pocketColor
      n++;
    }
    if (!n) return 0;
    await b.commit();
    counters.surfaceBuilds += n;
    quietSurfaces();
    return n;
  }
  // the surfaces' opacity: lighter with a pose (round 4)
  function setSurfaceAlpha(b, alpha) {
    let n = 0;
    for (const c of surfaceCells().values()) {
      if (c.transform.params.type.params.alpha === alpha) continue;
      b.to(c.transform.ref).update(old => ({ ...old, type: { ...old.type, params: { ...old.type.params, alpha } } })); n++;
    }
    return n;
  }
  // the first surface to build: the pocket picked in the Pockets tab if shown, else the first pocket shown (§2.3)
  const firstShown = spec => { const show = (spec.showPockets || []).map(Number); return show.includes(Number(spec.focusPocket)) ? Number(spec.focusPocket) : show.slice().sort((a, b) => a - b)[0]; };
  // Round 12: the camera after a drawing, as MVS set it (rounds 4–11): looking down -z with y up, onto the bounding sphere of the focus
  // target: a residue picked in a list + 6 Å, else a pocket picked in the Pockets tab (shown) + 6 Å, else the pose + 8 Å; without one,
  // onto everything shown
  function cameraTarget(spec, lig) {
    const prot = scene.protein, ps = prot && dataOf(prot.struct), fr = focusResidueOf(spec), Lc = molstar.lib.loci.Loci;
    const sphere = l => { const s = l && Lc.getBoundingSphere(l); return s && s.radius > 0 ? s : null; };
    let s = null;
    if (fr) s = { s: sphere(residuesLoci(ps, prot.index, [`${fr.chain}:${+fr.seq}`])), extent: 6 };
    else if (spec.focusPocket && (spec.showPockets || []).includes(Number(spec.focusPocket)) && prot.pockets.get(Number(spec.focusPocket)))
      s = { s: sphere(residuesLoci(ps, prot.index, prot.pockets.get(Number(spec.focusPocket)).keys)), extent: 6 };
    else if (lig && !spec.focusPocket && scene.site) s = { s: sphere(residuesLoci(dataOf(scene.site.struct), scene.site.index, ['L:1'])), extent: 8 };
    return s && s.s ? s : null;
  }
  function applyCamera(t) {
    const c3 = viewer.plugin.canvas3d; if (!c3) return;
    const snap = (sphere, extent) => { const r = Math.max(sphere.radius + extent, 0.01), d = c3.camera.getTargetDistance(r), c = sphere.center;
      return { target: [c[0], c[1], c[2]], position: [c[0], c[1], c[2] + d], up: [0, 1, 0], radius: r }; };
    c3.requestCameraReset({ durationMs: 0, snapshot: t ? snap(t.s, t.extent) : sc => snap(sc.boundingSphereVisible, 0) });
  }
  // Round 13: the camera on a switch keeps its orientation, as in Mol*. A new drug, pocket or mode moves it onto the new pose (the pose
  // + 8 Å, as the default view frames it; a mode's pose can lie anywhere in the 25 Å box); all modes, the legend's types or the theme
  // leave it where it is; with the ligand in focus it follows the pose as a click on the ligand does (Mol*'s focusLoci); with a residue
  // in focus it stays. The first drawing of a protein, no pose, a residue or pocket picked in a list (and Reset view) use the default
  // view as before (looking down -z).
  function switchCamera(kind, spec, lig, keepFocus) {
    if (kind === 'protein' || !lig || !scene.site || focusResidueOf(spec) || spec.focusPocket) { applyCamera(cameraTarget(spec, lig)); return; }
    if (keepFocus && keepFocus !== 'ligand') return;
    if (keepFocus !== 'ligand' && kind !== 'drug' && kind !== 'pocket' && kind !== 'mode') return;
    const loci = residuesLoci(dataOf(scene.site.struct), scene.site.index, ['L:1']), c3 = viewer.plugin.canvas3d;
    if (!loci || !c3) { applyCamera(cameraTarget(spec, lig)); return; }
    viewer.plugin.managers.camera.focusLoci(loci, keepFocus === 'ligand' ? { durationMs: 0 } : { minRadius: 0, extraRadius: 8, durationMs: 0 });
  }
  // Round 13: the focus a drawing keeps (switchCamera has placed the camera)
  function refocus(key) {
    let loci = null;
    if (key === 'ligand') loci = scene.site ? residuesLoci(dataOf(scene.site.struct), scene.site.index, ['L:1']) : null;
    else { const [ch, seq] = key.split(':'), S = proteinStructure(); loci = S ? residueLoci(S, ch, +seq) : null; }
    if (loci) viewer.plugin.managers.structure.focus.setFromLoci(loci);
  }
  // what was just built, on screen: Mol*'s scene committed now, then the next animation frame (Mol*'s render loop asked for it before
  // this callback, so Mol* has drawn when it runs); ≤ 250 ms (a tab in the background has no animation frames)
  function onScreen() {
    try { viewer.plugin.canvas3d.commit(true); } catch (e) { /* no canvas */ }
    return new Promise(r => { let done = false; const f = () => { if (!done) { done = true; r(); } }; requestAnimationFrame(f); setTimeout(f, 250); });
  }

  // ---------------------------------------------------------------- after loading: element colours, interactions, contacts
  const elementTheme = (carbon, sat = 0, light = 0) => ({ name: 'element-symbol', params: { carbonColor: { name: 'uniform', params: { value: carbon, saturation: 0, lightness: 0 } },
    saturation: sat, lightness: light, colors: { name: 'default', params: {} } } });
  // Round 12: a theme change recolours in place: the cartoon, the residue sticks (grey carbon; faded heteroatoms on contact-only residues)
  // and the labels (background, text); the pose, the modes and the pocket surfaces keep their colours in both themes
  async function recolor(spec) {
    const S = SD(), b = S.build(), dark = spec.theme === 'dark', C = PROTEIN[dark ? 'dark' : 'light'], same = (a, x) => JSON.stringify(a) === JSON.stringify(x);
    let k = 0;
    const set = (ref, old, colorTheme, typeParams) => {
      if (same(old.colorTheme, colorTheme) && (!typeParams || Object.entries(typeParams).every(([q, v]) => old.type.params[q] === v))) return;
      b.to(ref).update(o => ({ ...o, colorTheme, ...(typeParams ? { type: { ...o.type, params: { ...o.type.params, ...typeParams } } } : {}) })); k++;
    };
    for (const [ref, cell] of S.cells) {
      const tags = cell.transform.tags || [], p = cell.transform.params;
      if (!p || !p.colorTheme) continue;
      const r = tags.includes('pv-res-repr') && [...scene.residues.values()].find(sl => sl.repr === ref);
      if (r) set(ref, p, r.full ? elementTheme(C.carbon) : elementTheme(C.carbon, -1.5, dark ? -0.4 : 0.8));
      if (tags.includes('pv-label')) set(ref, p, { name: 'uniform', params: { ...p.colorTheme.params, value: dark ? 0xe8e8ec : 0x1b1b1f } }, { backgroundColor: dark ? 0x1c1c20 : 0xffffff });
    }
    if (k) await b.commit();
  }
  const TYPES = { 1: 'ionic', 2: 'cation–π', 3: 'π-stacking', 4: 'H-bond', 5: 'halogen bond', 6: 'hydrophobic', 7: 'metal', 8: 'weak H-bond', 9: 'water bridge' };
  // Mol*'s interactions of the site, per protein residue (inter-unit contacts: the pose is its own chain)
  function readInteractions(structure) {
    const box = structure.currentPropertyData && structure.currentPropertyData.molstar_computed_interactions;
    let I = box && box.data && box.data.value; if (I && I.value && !I.contacts) I = I.value;
    if (!I || !I.contacts || !I.unitsFeatures) return null;
    const out = new Map(), counts = {}, seen = new Set();
    const atomOf = (unit, feature) => {
      const F = I.unitsFeatures.get(unit.id); const el = unit.elements[F.members[F.offsets[feature]]];
      const h = unit.model.atomicHierarchy, ri = h.residueAtomSegments.index[el], ci = h.chainAtomSegments.index[el];
      return { chain: h.chains.auth_asym_id.value(ci), seq: h.residues.auth_seq_id.value(ri), name: h.atoms.label_comp_id.value(el) };   // residue names live in the atom table
    };
    for (let i = 0; i < I.contacts.edgeCount; i++) {
      const e = I.contacts.edges[i], uA = structure.unitMap.get(e.unitA), uB = structure.unitMap.get(e.unitB);
      if (!uA || !uB) continue;
      const a = atomOf(uA, e.indexA), b2 = atomOf(uB, e.indexB);
      const res = a.chain === 'L' ? b2 : b2.chain === 'L' ? a : null;
      if (!res || res.chain === 'L') continue;
      const key = `${res.chain}:${res.seq}`, t = TYPES[e.props.type] || 'other';
      if (!out.has(key)) out.set(key, { chain: res.chain, seq: res.seq, name: res.name, types: new Set(), n: 0 });
      out.get(key).types.add(t); out.get(key).n++;
      // the atom-level count of the legend's tooltip: each contact once (Mol* holds both directions) and only the drawn ones (flag 1 =
      // filtered, no line); the residues and their types above stay as they were (the Contacts list does not change, round 7)
      const id = [`${e.unitA}:${e.indexA}`, `${e.unitB}:${e.indexB}`].sort().join('|');
      if (e.props.flag !== 1 && !seen.has(id)) { seen.add(id); counts[t] = (counts[t] || 0) + 1; }
    }
    // round 6, 7: the legend counts residues per type, like the Contacts list; the atom-level count goes into its tooltip
    const residues = {};
    for (const r of out.values()) for (const t of r.types) residues[t] = (residues[t] || 0) + 1;
    out.counts = residues; out.atomCounts = counts;
    return out;
  }
  // Mol* turns hydrophobic and ionic contacts off by default; round 4 (4.2) wants them, with H-bonds, π-stacking, cation–π,
  // halogen bonds and metal coordination (on by default). The interactions property reads its settings from the root structure.
  // Types the legend can hide (spec.hiddenInteractions) are switched off before Mol* computes, so they are not drawn.
  const PROVIDER = { 'H-bond': 'hydrogen-bonds', hydrophobic: 'hydrophobic', 'π-stacking': 'pi-stacking', 'cation–π': 'cation-pi', 'halogen bond': 'halogen-bonds',
    ionic: 'ionic', metal: 'metal-coordination' };
  function enableInteractionTypes(structure, hidden) {
    const prov = viewer.plugin.customStructureProperties && viewer.plugin.customStructureProperties.get('molstar_computed_interactions');
    if (!prov) return false;
    const root = structure.root || structure, defs = prov.getParams(root), props = JSON.parse(JSON.stringify(prov.props(root)));
    for (const [type, name] of Object.entries(PROVIDER)) {
      if (hidden.includes(type)) { props.providers[name] = { name: 'off', params: {} }; continue; }
      const on = defs.providers.params[name].map('on'), params = {};
      for (const [k, d] of Object.entries(on.params)) params[k] = d.defaultValue;
      props.providers[name] = { name: 'on', params };
    }
    prov.set(root, props);
    return true;
  }
  const hiddenTypes = spec => (Array.isArray(spec.hiddenInteractions) ? spec.hiddenInteractions.filter(t => PROVIDER[t]) : []);
  async function interactions(spec, lig, P) {
    const site = scene.site ? [...SD().cells.values()].find(c => tagged(c, 'pv-site-inter') && c.transform.parent === scene.site.struct) : null;
    const hidden = hiddenTypes(spec);
    let found = null, source = 'distance';
    if (site && site.obj) {
      try {
        // once per site structure (a legend toggle builds a new one: syncSite)
        if (![...SD().cells.values()].some(c => tagged(c, 'pv-interactions') && c.transform.parent === site.transform.ref)) {
          try { enableInteractionTypes(site.obj.data, hidden); } catch (e) { window.__pvError = 'interaction types: ' + String(e && e.message || e); }
          // only the ligand <-> protein lines (inter-unit: the pose is its own chain); Mol*'s default also draws protein-protein contacts
          // inside the site (intra-unit), which the legend and the Contacts list do not count
          // ignoreHydrogens (round 7, §2): an H-bond line ends on its donor / acceptor heavy atom, not on a polar hydrogen of the pose,
          // which is not drawn (the sticks ignore hydrogens)
          await RB().addRepresentation(site, { type: 'interactions', typeParams: { visuals: ['inter-unit'], ignoreHydrogens: true } }, { tag: 'pv-interactions' });
        }
        found = readInteractions(site.obj.data); if (found) source = 'molstar';
        mark('interactions');
      } catch (e) { found = null; window.__pvError = String(e && e.message || e); }
    }
    try { lineHover(); await residueSticks(spec, lig, found); mark('sticks'); } catch (e) { window.__pvError = 'residue sticks: ' + String(e && e.message || e); }
    const list = lig.contacts.map(r => ({ ...r, types: found && found.get(`${r.chain}:${r.seq}`) ? [...found.get(`${r.chain}:${r.seq}`).types] : [] }));
    if (found) for (const [key, r] of found) if (!lig.contacts.some(c => `${c.chain}:${c.seq}` === key)) list.push({ chain: r.chain, seq: r.seq, name: r.name, dist: null, types: [...r.types] });
    // labels (round 7, §3): every residue with a non-hydrophobic interaction in priority order (ionic, cation–π, π-stacking, halogen
    // bond, H-bond), then hydrophobic-only ones up to MAX_LABELS in all, plus a residue picked in a list
    const lab = labelKeys(found);
    const fr = spec.focusResidue; if (fr) { const k = `${fr.chain}:${+fr.seq}`; if (!lab.includes(k)) lab.push(k); }
    await labels(spec, lab, P, lig, found);
    const elements = [...new Set(lig.atoms.map(a => a.el.length > 1 ? a.el[0].toUpperCase() + a.el.slice(1).toLowerCase() : a.el.toUpperCase()))].filter(e => e !== 'C' && e !== 'H').sort();
    const msg = { type: 'contacts', key: spec.ligand.key || '', source, contacts: list, elements, counts: found ? found.counts : {}, atomCounts: found ? found.atomCounts : {}, hidden, labelled: lab };
    window.__pvDebug = { contacts: msg.contacts, source, nInteractionResidues: found ? found.size : 0, counts: msg.counts, atomCounts: msg.atomCounts, elements, labelled: lab,
      labelFit: labelInfo.map(L => ({ key: L.key, hidden: L.hidden })), surfaces: surfacesNow() };
    debug(); lastContacts = msg;
    post(msg);
  }
  let lastContacts = null;                                    // round 12: posted again when a theme change keeps the drawing
  // Mol* label representations of the chosen residues, on a background. Size: uniform 1 × sizeFactor, the sizeFactor set from the camera
  // (round 8, §2.2: TEXT_PX on the screen; it was 1.3 Å whatever the zoom). The label representation has no textSize.
  // Round 12: labels are slots kept per protein (a component of the protein structure with one label representation each, tag
  // pv-label): a drawing moves them to its residues. A new label costs Mol* a texture upload of its font atlas (~43 MB at the font
  // quality used, per label); a moved one keeps its texture. Slots not needed are hidden.
  // Mol*'s text builder caches its font atlas (glyph images) under all the text's parameters, so a label whose geometry is built with
  // its own size or depth gets an atlas of its own (rounds 8–11: every label of every drawing, 43 MB each, never freed). Geometry is
  // therefore built with the same parameters for every label (LABEL_GEOMETRY: one atlas); each label's size, depth and the theme's
  // background follow in a parameter update (Mol*'s sizeFactor, offsets and background colour: no new geometry).
  const LABEL_GEOMETRY = { sizeFactor: 1, offsetZ: LABEL_Z, offsetX: 0, offsetY: 0, backgroundColor: 0xffffff, fontQuality: FONT_QUALITY };
  function labelParams(dark) {
    return { typeParams: { level: 'residue', background: true, backgroundOpacity: 0.85, backgroundMargin: 0.2, attachment: 'middle-center', ...LABEL_GEOMETRY },
      colorParams: { value: dark ? 0xe8e8ec : 0x1b1b1f } };
  }
  function hideLabelSlots(used = new Set()) {
    const S = SD(), slots = scene.protein ? scene.protein.labelSlots : [];
    for (const sl of slots) { const c = cellOf(sl.repr); if (c && !used.has(sl.repr) && !(c.state && c.state.isHidden)) S.updateCellState(sl.repr, { isHidden: true }); }
  }
  // label slots made ahead at idle time (hidden), so a first pose does not wait for their textures; on residues whose labels hold every
  // character any residue label of this protein needs (Mol* draws a glyph into the atlas when first used: later moves add none)
  const LABEL_SLOTS = 8;
  function prewarmKeys(P) {
    const text = k => { const [ch, seq] = k.split(':'); return `${(P.names[`${ch}_${seq}`] || ['RES'])[0]} ${seq}`; };
    const keys = Object.keys(P.names).map(k => k.replace('_', ':')), need = new Set(keys.flatMap(k => [...text(k)])), out = [];
    while (out.length < LABEL_SLOTS && need.size) {
      let best = null, gain = 0;
      for (const k of keys) { const g = new Set([...text(k)].filter(c => need.has(c))).size; if (g > gain) { gain = g; best = k; } }
      if (!best) break;
      out.push(best); for (const c of text(best)) need.delete(c);
    }
    for (const k of keys) { if (out.length >= LABEL_SLOTS) break; if (!out.includes(k)) out.push(k); }
    return out;
  }
  async function prewarmLabels() {                            // a slot per call (each uploads its font texture: ~10 ms on a P40)
    const prot = scene.protein, ps = prot && dataOf(prot.struct), P = proteins.get(prot && prot.path); if (!ps || !P || prot.labelSlots.length >= LABEL_SLOTS) return false;
    const keys = prot.slotKeys || (prot.slotKeys = prewarmKeys(P)), b = SD().build(), { typeParams, colorParams } = labelParams(document.documentElement.dataset.theme === 'dark');
    for (let i = prot.labelSlots.length, n = 0; i < LABEL_SLOTS && keys[i] && n < 1; i++, n++) {
      const comp = component(b, prot.struct, bundleOf(ps, prot.index, [keys[i]]), 'Label', 'pv-label-slot').ref;
      prot.labelSlots.push({ key: keys[i], comp, repr: representation(b, comp, ps, { type: 'label', typeParams, color: 'uniform', colorParams, size: 'uniform', sizeParams: { value: 1 } }, 'pv-label', { isHidden: true }) });
    }
    await b.commit();
    return true;
  }
  async function labels(spec, keys, P, lig, found) {
    labelInfo = []; lastCam = '';
    const prot = scene.protein, ps = prot && dataOf(prot.struct);
    if (!keys.length || !ps) { hideLabelSlots(); return; }
    const dark = spec.theme === 'dark', cam = viewer.plugin.canvas3d.camera, dpr = bufferScale(), b = SD().build(), made = [], slots = prot.labelSlots;
    // a slot already on one of these residues stays (no new geometry); the others move from residues not labelled now
    const live = slots.filter(sl => cellOf(sl.comp) && cellOf(sl.repr)), want = new Set(keys);
    const kept = new Map(live.filter(sl => want.has(sl.key)).map(sl => [sl.key, sl])), free = live.filter(sl => !want.has(sl.key)), used = new Set();
    for (const k of keys) {
      const bundle = bundleOf(ps, prot.index, [k]);
      if (!bundle) continue;
      // the residue picked in a list: the camera zooms onto it, so its label moves only 2 Å forward (10 Å would pass the near plane)
      const picked = !!spec.focusResidue && k === `${spec.focusResidue.chain}:${+spec.focusResidue.seq}`, offset = picked ? 2 : LABEL_Z;
      const [ch, seq] = k.split(':'), at = P.atoms.filter(a => a.ch === ch && a.seq === +seq && a.el !== 'H');
      if (!at.length) continue;
      const center = [0, 1, 2].map(i => at.reduce((s, a) => s + [a.x, a.y, a.z][i], 0) / at.length);
      const radius = Math.max(...at.map(a => Math.hypot(a.x - center[0], a.y - center[1], a.z - center[2])));
      // its first size: TEXT_PX where Mol* will draw it for the camera now (the fit corrects it once Mol*'s geometry is known)
      const size = sizeAt(cam, v3.add(center, v3.norm(v3.sub(cam.state.position, center)).map(x => x * (offset + 0.95 * radius))), EM, dpr);
      const { typeParams, colorParams } = labelParams(dark);
      let slot = kept.get(k);
      if (slot) kept.delete(k);
      else if ((slot = free.shift())) {
        b.to(slot.comp).update(old => ({ ...old, type: { name: 'bundle', params: bundle }, label: `Label ${k}` }));
        b.to(slot.repr).update(old => ({ ...old, type: { ...old.type, params: { ...old.type.params, ...typeParams } }, colorTheme: { ...old.colorTheme, params: { ...old.colorTheme.params, ...colorParams } } }));
      } else {
        const comp = component(b, prot.struct, bundle, `Label ${k}`, 'pv-label-slot').ref;
        slot = { comp, repr: representation(b, comp, ps, { type: 'label', typeParams, color: 'uniform', colorParams, size: 'uniform', sizeParams: { value: 1 } }, 'pv-label', { isHidden: true }) };
        slots.push(slot);
      }
      slot.key = k; used.add(slot.repr);
      // round 7 (§3): its interaction types (priority order), important = any type other than hydrophobic (never hidden), and the
      // label's shift in the screen plane (Å; Mol*'s offsetX / offsetY): `shift` is drawn, `want` is the last fit's choice.
      // Round 8: `size` (Å per text unit, Mol*'s sizeFactor) is drawn, `wantSize` goes with the fit's depth `wantZ`; emU: the em of
      // its text in text units
      const types = found && found.get(k) ? [...found.get(k).types].sort((a, b) => rankOf(a) - rankOf(b)) : [];
      made.push({ ref: slot.repr, key: k, chars: `${(P.names[`${ch}_${seq}`] || ['RES'])[0]} ${seq}`.length, hidden: false, picked, center, radius, offset,
        types, important: types.some(t => t !== 'hydrophobic'), shift: [0, 0], want: [0, 0], forced: false, pending: false, base: offset, wantZ: offset,
        size, wantSize: size, emU: EM });
    }
    await b.commit();
    hideLabelSlots(used);
    // each label's own first size and depth, unshifted, in the theme's colours (parameters only)
    const b2 = SD().build(), { colorParams } = labelParams(dark);
    for (const L of made) b2.to(L.ref).update(old => ({ ...old, type: { ...old.type, params: { ...old.type.params, sizeFactor: L.size, offsetZ: L.offset, offsetX: 0, offsetY: 0,
      backgroundColor: dark ? 0x1c1c20 : 0xffffff } }, colorTheme: { ...old.colorTheme, params: { ...old.colorTheme.params, ...colorParams } } }));
    if (made.length) await b2.commit();
    mark('labels built');
    labelInfo = made.filter(L => SD().cells.has(L.ref));
    labelInfo.sort((a, b) => b.picked - a.picked);           // the residue picked in a list first, then by priority (keys order)
    labelFront = frontAtoms(spec, P, lig);
    solids = solidParts(spec, P, lig, found); solidsKey = null;          // round 9: what can cover a label (covered)
    fitLabels(true); mark('labels placed');
    await settleLabels();                                     // the sizes of this camera applied before the drawing is reported
    mark('labels settled');
  }
  // Round 6, 8 / round 7, §3: a label is shown where all of it can be read: inside the window, over no label placed before it, and
  // with nothing drawn in front of its text. Mol* draws a label LABEL_Z Å (or more, see wantOffset) in front of its residue along the
  // line of sight; its box comes from Mol*'s text geometry at that point (labelBox); placeLabel shifts or hides it (see there).
  let labelInfo = [], lastCam = '', lastMoved = '', fitRetries = 0, fitCount = 0, fitTimes = [];          // round 9: placements so far, the last 50 placement times (ms)
  const v3 = { sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]], add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
    cross: (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
    norm: a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; } };
  function labelBox(cam, L, shift, offsetZ, size) {
    // Mol*'s text shader moves a label (offsetZ + 0.95 × the residue's radius) Å towards the camera along the line of sight: the same
    // place on the screen as the residue, nearer in depth; then offsetX / offsetY (Å, screen plane) move it right / up (round 7)
    const s = shift || L.shift || [0, 0], k = size === undefined ? L.size : size;
    // the label's frame (background) and its text (for the occlusion test), in text units around the anchor (× k = Å): from Mol*'s own
    // text geometry (round 7; it also sets the anchor and depth), else estimated (a unit high, ~0.62 of that per character)
    const g = L.geom || (L.geom = labelGeom(L)) || { bg: [-(L.chars * 0.62 + 0.4) / 2, (L.chars * 0.62 + 0.4) / 2, -0.7, 0.7], text: [-L.chars * 0.31, L.chars * 0.31, -0.5, 0.5] };
    const st = cam.state, ray = v3.norm(v3.sub(st.position, L.center)), right = v3.norm(v3.cross(v3.sub(st.target, st.position), st.up));
    const off = offsetZ === undefined ? L.offset : offsetZ;
    const pt = v3.add(L.center, ray.map(x => x * (off + 0.95 * L.radius))), o = [0, 0, 0, 0], q = [0, 0, 0, 0];
    cam.project(o, pt); cam.project(q, v3.add(pt, right));
    const px = Math.hypot(q[0] - o[0], q[1] - o[1]), vp = cam.viewport, cx = o[0] - vp.x + s[0] * px, cy = o[1] - vp.y + s[1] * px, u = k * px;
    const dir = v3.norm(v3.sub(st.target, st.position)), rel = v3.sub(pt, st.position);
    // front: not at or in front of the near clipping plane, where Mol* cuts the label off (depth 0; a little margin)
    return { l: cx + g.bg[0] * u, r: cx + g.bg[1] * u, b: cy + g.bg[2] * u, t: cy + g.bg[3] * u, w: vp.width, h: vp.height, front: o[2] > 0.05 && o[2] < 1,
             z: o[2], px, x0: vp.x, y0: vp.y, text: [cx + g.text[0] * u, cx + g.text[1] * u, cy + g.text[2] * u, cy + g.text[3] * u],
             dist: Math.hypot(rel[0], rel[1], rel[2]), depth: rel[0] * dir[0] + rel[1] * dir[1] + rel[2] * dir[2], shift: s };
  }
  // Mol*'s text geometry of a label: the first quad is its background (corners in text units: × size × sizeFactor = Å). `text` is
  // the area that must stay clear to read the label: the background just inside its edge (the glyphs lie (0.08 + 0.1) units in
  // from the sides and 0.08 from top and bottom: Mol*'s text builder, backgroundMargin 0.2).
  // Its anchor and depth are Mol*'s bounding sphere of the residue (centre, radius): they replace the estimate from the atoms.
  function labelGeom(L) {
    try {
      const ro = viewer.plugin.state.data.cells.get(L.ref).obj.data.repr.renderObjects[0], v = ro.values, m = v.aMapping.ref.value;
      const c = v.aPosition.ref.value, r = v.aDepth.ref.value[0];
      if (!(m.length >= 8)) return null;
      if (c.length >= 3 && [c[0], c[1], c[2], r].every(Number.isFinite)) { L.center = [c[0], c[1], c[2]]; L.radius = r; }
      const x0 = Math.min(m[0], m[4]), x1 = Math.max(m[0], m[4]), y0 = Math.min(m[1], m[3]), y1 = Math.max(m[1], m[3]);
      return x1 > x0 && y1 > y0 ? { bg: [x0, x1, y0, y1], text: [x0 + 0.05, x1 - 0.05, y0 + 0.04, y1 - 0.04] } : null;
    } catch (e) { return null; }
  }
  // drawing-buffer pixels per CSS pixel of the canvas
  function bufferScale() {
    const gl = viewer.plugin.canvas3d.webgl.gl, w = gl.canvas.getBoundingClientRect().width;
    return w > 0 ? gl.drawingBufferWidth / w : 1;
  }
  // the size (Å per text unit) that draws a text whose em is `emU` units TEXT_PX CSS px high at the point p, for the camera now; at most
  // the size that does so at FAR_ZOOM × the pair's default camera distance (see homeDist)
  function sizeAt(cam, p, emU, dpr) {
    const o = [0, 0, 0, 0], q = [0, 0, 0, 0], st = cam.state, right = v3.norm(v3.cross(v3.sub(st.target, st.position), st.up));
    cam.project(o, p); cam.project(q, v3.add(p, right));
    const px = Math.hypot(q[0] - o[0], q[1] - o[1]); if (!(px > 0)) return LABEL_SIZE;
    const far = homeDist(cam), p5 = cam.projection[5], depth = far ? FAR_ZOOM * far : 0;
    const min = depth && st.mode !== 'orthographic' && p5 > 0 ? cam.viewport.height / 2 * p5 / depth : 0;   // px per Å at that depth
    return Math.round(TEXT_PX * dpr / (emU * Math.max(px, min)) * 1000) / 1000;
  }
  // Mol*'s camera distance for the pair's view (its focus on the pose with 8 Å around it: Camera.targetDistance)
  let siteRadius = 0;
  function homeDist(cam) {
    if (!siteRadius) return 0;
    const vp = cam.viewport, aspect = vp.width / Math.max(1, vp.height), fov = cam.state.fov;
    return siteRadius / (vp.height < vp.width ? 1 : aspect) / Math.sin(fov / 2);
  }
  // a label's text height on the screen (CSS px) as drawn now
  const labelPx = (cam, L, dpr) => Math.round(L.emU * L.size * labelBox(cam, L, L.shift, L.offset, L.size).px / dpr * 10) / 10;
  // Characters of the label covered by something drawn in front of it. Round 9 (round-8 feedback §1.3): worked out on the CPU from
  // the drawing's atoms (solids, below), not with Mol*'s GPU picking (one pick per sample point: ~600 picks for a small camera move,
  // 100+ ms a placement on a laptop GPU). A solid sphere covers the characters whose columns it reaches into, where its front lies
  // in front of the label (along the line of sight: a label shifted in the screen plane keeps its depth). Labels themselves do not
  // count (their overlaps are tested on their boxes), nor do translucent pocket surfaces (Mol* did not pick them either). Mol* also
  // closes a cartoon cut by the near clipping plane with a flat cap drawn on that plane, in front of every label: the cartoon follows
  // the CA trace, so its points within 1.2 Å of the near plane mark those caps (capDiscs, 1.5 Å around each; projectSolids).
  let capDiscs = [];
  // one bit per character hit (a label has at most a dozen characters)
  function hitChars(l, r, bo, t, cw, chars, cx, cy, cr) {
    const nx = cx < l ? l : cx > r ? r : cx, ny = cy < bo ? bo : cy > t ? t : cy;
    if ((cx - nx) * (cx - nx) + (cy - ny) * (cy - ny) >= cr * cr) return 0;
    const k0 = Math.max(0, Math.floor((Math.max(l, cx - cr) - l) / cw)), k1 = Math.min(chars - 1, Math.floor((Math.min(r, cx + cr) - l) / cw));
    return k1 < k0 ? 0 : ((2 ** (k1 + 1)) - (2 ** k0));
  }
  function covered(cam, b, chars, full) {                     // full: count every character (the test hook), else stop at two
    const l = b.text[0], r = b.text[1], bo = b.text[2], t = b.text[3], cw = (r - l) / chars, lim = b.depth - 0.05, X = SV.x, Y = SV.y, R = SV.r, D = SV.d;
    let hit = 0;
    for (const q of capDiscs) hit |= hitChars(l, r, bo, t, cw, chars, q[0], q[1], q[2]);
    if (!full && (hit & (hit - 1))) return 2;
    for (let i = 0, n = SV.n; i < n; i++) {
      const sr = R[i]; if (D[i] >= lim || X[i] + sr <= l || X[i] - sr >= r || Y[i] + sr <= bo || Y[i] - sr >= t) continue;
      hit |= hitChars(l, r, bo, t, cw, chars, X[i], Y[i], sr);
      if (!full && (hit & (hit - 1))) return 2;               // two characters or more: placeLabel needs to know no more
    }
    let n = 0; for (; hit; hit &= hit - 1) n++;
    return n;
  }
  // The solid parts of the drawing as spheres (x, y, z, radius Å), a little larger than Mol* draws them (better to move a label than
  // to show it half hidden): the pose and the stick residues (heavy atoms, and points along their bonds), the cartoon (points along
  // the CA trace), the interaction lines, the other modes when shown (solids, set with the labels), and what a focus or a clicked line
  // adds (Mol*'s ball-and-stick of the residues around a focus, the clicked line's distance: solidsNow).
  const SOLID_R = { pose: 0.6, sticks: 0.6, thin: 0.4, focus: 0.55, mode: 0.45, cartoon: 1.6, line: 0.15 };
  let solids = new Float64Array(0), solidsAll = solids, solidsKey = '';
  function bondedSpheres(out, atoms, r) {
    for (let i = 0; i < atoms.length; i++) {
      const a = atoms[i]; out.push(a.x, a.y, a.z, r);
      for (let j = i + 1; j < atoms.length; j++) {
        const q = atoms[j], dx = q.x - a.x, dy = q.y - a.y, dz = q.z - a.z;
        if (dx * dx + dy * dy + dz * dz < 3.61) for (const f of [0.25, 0.5, 0.75]) out.push(a.x + dx * f, a.y + dy * f, a.z + dz * f, r * 0.75);
      }
    }
  }
  const heavyAtom = a => a.el !== 'H' && a.el !== 'h';
  function residueAtoms(P) {
    if (P.byRes) return P.byRes;
    const m = P.byRes = new Map();
    for (const a of P.atoms) if (heavyAtom(a)) { const k = `${a.ch}:${a.seq}`; let v = m.get(k); if (!v) m.set(k, v = []); v.push(a); }
    return m;
  }
  function solidParts(spec, P, lig, found) {
    const out = [], byRes = residueAtoms(P), thick = new Set(found ? found.keys() : []), fr = spec.focusResidue;
    if (fr) thick.add(`${fr.chain}:${+fr.seq}`);
    if (lig) {
      bondedSpheres(out, lig.atoms.filter(heavyAtom), SOLID_R.pose);
      for (const r of lig.contacts) { const k = `${r.chain}:${r.seq}`; if (!thick.has(k)) bondedSpheres(out, byRes.get(k) || [], SOLID_R.thin); }
      const L = spec.ligand;
      if (L.allModes && L.format === 'pdbqt' && L.nModels > 1) for (let i = 0; i < L.nModels; i++) if (i !== lig.model) bondedSpheres(out, ligandAtoms(L, i).atoms.filter(heavyAtom), SOLID_R.mode);
    }
    for (const k of thick) bondedSpheres(out, byRes.get(k) || [], SOLID_R.sticks);
    // the cartoon: the CA trace, a point half-way between neighbours (1.9 Å apart, 1.6 Å each: no gap)
    let prev = null;
    for (const a of P.atoms) {
      if (a.name !== 'CA') continue;
      if (prev && prev.ch === a.ch && a.seq - prev.seq === 1) out.push((prev.x + a.x) / 2, (prev.y + a.y) / 2, (prev.z + a.z) / 2, SOLID_R.cartoon);
      out.push(a.x, a.y, a.z, SOLID_R.cartoon); prev = a;
    }
    if (lig) for (const ln of drawnLines()) lineSpheres(out, ln.a, ln.b, SOLID_R.line);
    return Float64Array.from(out);
  }
  const drawnLines = () => { try { return window.__pvLines ? window.__pvLines() : []; } catch (e) { return []; } };
  function lineSpheres(out, a, b, r) {
    const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / (1.5 * r)));
    for (let i = 0; i <= n; i++) out.push(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, a[2] + (b[2] - a[2]) * i / n, r);
  }
  // the solids with what the focus and a clicked line add now
  function solidsNow() {
    const f = picked.focus, ln = picked.line, key = `${f ? f.key : ''}|${ln && ln.ends ? ln.key : ''}`;
    if (key === solidsKey) return solidsAll;
    solidsKey = key; solidsAll = solids;
    const extra = [];
    if (f && drawnAtoms) { const byRes = residueAtoms(drawnAtoms.P); for (const k of [f.key, ...f.surroundings]) if (k !== 'ligand') bondedSpheres(extra, byRes.get(k) || [], SOLID_R.focus); }
    if (ln && ln.ends) lineSpheres(extra, ln.ends[0], ln.ends[1], SOLID_R.line);
    if (extra.length) { solidsAll = new Float64Array(solids.length + extra.length); solidsAll.set(solids); solidsAll.set(extra, solids.length); }
    return solidsAll;
  }
  // Per placement the spheres are projected once (screen x, y and radius in buffer px relative to the view, the depth of their front
  // along the line of sight), keeping only those on the screen whose front lies in front of the deepest label (`deepest`): labels sit
  // in front of the site, so that is the part of the protein between them and the camera, a few hundred spheres at most.
  const SV = { x: new Float32Array(0), y: new Float32Array(0), r: new Float32Array(0), d: new Float32Array(0), n: 0 };
  function projectSolids(cam, deepest) {
    const A = solidsNow(), n = A.length / 4, st = cam.state, vp = cam.viewport, m = cam.projectionView, p0 = st.position;
    const dir = v3.norm(v3.sub(st.target, st.position)), p5 = cam.projection[5], persp = st.mode !== 'orthographic', half = vp.height / 2, near = cam.near;
    if (SV.x.length < n) { SV.x = new Float32Array(n); SV.y = new Float32Array(n); SV.r = new Float32Array(n); SV.d = new Float32Array(n); }
    let k = 0; capDiscs = [];
    const capR = 1.5 * (persp ? half * p5 / Math.max(near, 1e-3) : half * p5);
    for (let i = 0; i < n; i++) {
      const x = A[4 * i], y = A[4 * i + 1], z = A[4 * i + 2], R = A[4 * i + 3];
      const d = (x - p0[0]) * dir[0] + (y - p0[1]) * dir[1] + (z - p0[2]) * dir[2];
      if (R === SOLID_R.cartoon && d > 0 && Math.abs(d - near) <= 1.2) {   // the cartoon cut by the near plane: Mol*'s cap (see covered)
        const w = m[3] * x + m[7] * y + m[11] * z + m[15];
        if (w > 0) capDiscs.push([((m[0] * x + m[4] * y + m[8] * z + m[12]) / w + 1) * vp.width / 2, ((m[1] * x + m[5] * y + m[9] * z + m[13]) / w + 1) * half, capR]);
      }
      if (d - R >= deepest || d + R <= near) continue;                  // behind every label, or clipped away by the near plane
      const w = m[3] * x + m[7] * y + m[11] * z + m[15]; if (!(w > 0)) continue;
      const sx = ((m[0] * x + m[4] * y + m[8] * z + m[12]) / w + 1) * vp.width / 2, sy = ((m[1] * x + m[5] * y + m[9] * z + m[13]) / w + 1) * half;
      const sr = R * (persp ? half * p5 / Math.max(d, near) : half * p5);
      if (sx + sr < 0 || sx - sr > vp.width || sy + sr < 0 || sy - sr > vp.height) continue;
      SV.x[k] = sx; SV.y[k] = sy; SV.r[k] = sr; SV.d[k] = d - R; k++;
    }
    SV.n = k;
  }
  // test hook (round 9): for each label shown, the characters the CPU test finds covered at its place and those Mol*'s GPU picking
  // finds (the old test, sampled every ~3 px at three heights): the CPU test should see at least what picking sees
  let identifyCalls = 0;
  const identify = (c, page) => { identifyCalls++; if (window.__pvDebug) window.__pvDebug.identifyCalls = identifyCalls; return c.identify(page); };
  window.__pvCoverCheck = () => {
    const c = viewer.plugin.canvas3d, cam = c.camera, dpr = c.webgl.pixelRatio, H = c.webgl.gl.drawingBufferHeight, st = cam.state, dir = v3.norm(v3.sub(st.target, st.position));
    const own = new Set(labelInfo.map(L => { try { return viewer.plugin.state.data.cells.get(L.ref).obj.data.repr.renderObjects[0].id; } catch (e) { return -1; } }));
    projectSolids(cam, Infinity);
    return labelInfo.filter(L => !L.hidden).map(L => {
      const b = labelBox(cam, L), [l, r, bo, t] = b.text, nx = Math.max(2 * L.chars, Math.ceil((r - l) / (3 * dpr))), gpu = new Set();
      for (let i = 0; i < nx; i++) {
        const ci = Math.min(L.chars - 1, Math.floor((i + 0.5) / nx * L.chars)), x = b.x0 + l + (r - l) * (i + 0.5) / nx;
        for (const fy of [0.2, 0.5, 0.8]) {
          let p = null; try { p = c.identify([x / dpr, (H - (b.y0 + bo + (t - bo) * fy)) / dpr]); } catch (e) { p = null; }
          if (!p || !p.position || own.has(p.id.objectId)) continue;
          const d = v3.sub(p.position, st.position); if (d[0] * dir[0] + d[1] * dir[1] + d[2] * dir[2] < b.depth - 0.05) gpu.add(ci);
        }
      }
      return { key: L.key, cpu: covered(cam, b, L.chars, true), gpu: gpu.size, forced: L.forced };
    });
  };
  // Round 7, §3: a label is placed where all of it can be read (inside the window, over no label placed before it, nothing drawn in
  // front of its text). If its own place is taken it is shifted (Mol* label offsetX / offsetY, Å in the screen plane) to the nearest
  // free spot: up / down by a line, aside by half or a whole label, and inside the window at an edge. Labels are placed in priority
  // order (the residue picked in a list, then ionic, cation–π, π-stacking, halogen bond, H-bond, hydrophobic). A label of a residue
  // with any interaction other than hydrophobic (or the picked residue) is never hidden: without a free spot it is shown at the least
  // bad one (`forced`); a hydrophobic-only label is hidden then (hovering the residue still names it).
  const LABEL_EDGE = 2;
  const inWindow = b => b.front && b.l >= LABEL_EDGE && b.r <= b.w - LABEL_EDGE && b.b >= LABEL_EDGE && b.t <= b.h - LABEL_EDGE;
  const overlapArea = (b, q) => Math.max(0, Math.min(b.r, q.r) - Math.max(b.l, q.l)) * Math.max(0, Math.min(b.t, q.t) - Math.max(b.b, q.b));
  const sameShift = (a, b) => Math.abs(a[0] - b[0]) < 0.01 && Math.abs(a[1] - b[1]) < 0.01;
  // round 9: a label shifted in the screen plane by s (Å) is its unshifted box moved by s × px (no new projection)
  function shiftBox(b, s) {
    const dx = s[0] * b.px, dy = s[1] * b.px, x = b.text;
    return { ...b, l: b.l + dx, r: b.r + dx, b: b.b + dy, t: b.t + dy, text: [x[0] + dx, x[1] + dx, x[2] + dy, x[3] + dy], shift: s };
  }
  function shiftCandidates(b0) {
    const hA = (b0.t - b0.b) / b0.px * 1.1, wA = (b0.r - b0.l) / b0.px * 1.05, m = LABEL_EDGE + 1, out = [];
    // around its place: up / down by a line, aside by half or a whole label; at most one and a half labels aside and four lines up
    // or down (further away it would no longer point at its residue)
    for (const k of [0, 1, -1, 2, -2, 3, -3]) for (const f of [0, 0.55, -0.55, 1.1, -1.1]) out.push([f * wA, k * hA]);
    // at an edge: the shift that brings the label just inside the window, and from there up / down / aside (up to three labels in)
    const ex = b0.l < m ? (m - b0.l) / b0.px : b0.r > b0.w - m ? (b0.w - m - b0.r) / b0.px : 0;
    const ey = b0.b < m ? (m - b0.b) / b0.px : b0.t > b0.h - m ? (b0.h - m - b0.t) / b0.px : 0;
    if ((ex || ey) && Math.abs(ex) <= 3 * wA && Math.abs(ey) <= 6 * hA) for (const k of [0, 1, -1, 2, -2]) for (const f of [0, 0.55, -0.55]) out.push([ex + f * wA, ey + k * hA]);
    const cost = s => Math.hypot(s[0], s[1] * 1.15);
    return out.filter(s => (Math.abs(s[0]) <= 1.5 * wA + 0.01 && Math.abs(s[1]) <= 4 * hA + 0.01) || (Math.abs(s[0] - ex) <= 0.6 * wA && Math.abs(s[1] - ey) <= 2.1 * hA))
      .sort((a, b) => cost(a) - cost(b)).map(s => s.map(v => Math.round(v * 100) / 100));
  }
  // Depth: a label is drawn L.base Å (10; 2 for the residue picked in a list) in front of its residue, and further forward if needed
  // to lie in front of the whole site (the pose, the 5 Å residues and the residues of the pocket surfaces shown): a translucent
  // pocket surface between label and camera would tint the label (it cannot be picked, so covered() does not see it). Never past the
  // near clipping plane, where Mol* cuts the label off. Rounded to 0.5 Å, so a small camera move does not update the label.
  let labelFront = null;
  function frontAtoms(spec, P, lig) {
    const keys = new Set(lig ? lig.site.map(r => `${r.chain}_${r.seq}`) : []), shown = (spec.showPockets || []).map(Number), xyz = [];
    for (const pk of (spec.pockets || [])) if (shown.includes(Number(pk.rank))) for (const r of (pk.residues || [])) keys.add(String(r));
    for (const a of P.atoms) if (a.el !== 'H' && keys.has(`${a.ch}_${a.seq}`)) xyz.push(a.x, a.y, a.z);
    if (lig) for (const a of lig.atoms) if (a.el !== 'H') xyz.push(a.x, a.y, a.z);
    return Float64Array.from(xyz);
  }
  function frontDepth(cam) {
    const A = labelFront; if (!A || !A.length) return null;
    const st = cam.state, dir = v3.norm(v3.sub(st.target, st.position)), p = st.position; let m = Infinity;
    for (let i = 0; i < A.length; i += 3) { const d = (A[i] - p[0]) * dir[0] + (A[i + 1] - p[1]) * dir[1] + (A[i + 2] - p[2]) * dir[2]; if (d < m) m = d; }
    return m - 3.5;                                           // atom radius + the surface's probe, and a little more
  }
  // the nearest view depth a label may have: just behind the near clipping plane, at a projected depth (0..1) of `at` (round 8: it was
  // near / 0.9, which kept labels far behind the plane when the plane itself is far from the camera, e.g. zoomed out after a focus).
  // Residue labels: 0.07 (labelBox takes a label over 0.05 as drawn); a clicked line's labels, not shifted aside: 0.02
  function labelNear(cam, at = 0.07) {
    const n = cam.near, f = cam.far;
    if (!(f > n && n > 0)) return n / 0.9;
    return cam.state.mode === 'orthographic' ? n + at * (f - n) : f * n / (f - at * (f - n));
  }
  function wantOffset(cam, L, front) {
    const st = cam.state, dir = v3.norm(v3.sub(st.target, st.position)), ray = v3.norm(v3.sub(st.position, L.center));
    const cos = -(ray[0] * dir[0] + ray[1] * dir[1] + ray[2] * dir[2]), rel = v3.sub(L.center, st.position), dc = rel[0] * dir[0] + rel[1] * dir[1] + rel[2] * dir[2];
    if (!(cos > 0.2)) return L.base;
    let t = L.base + 0.95 * L.radius;                          // distance along the line of sight that Mol*'s text shader moves it
    if (front !== null && (dc - front) / cos > t) t = (dc - front) / cos;
    const cap = (dc - labelNear(cam)) / cos;                     // behind the near plane (labelBox wants a projected depth over 0.05)
    L.capped = t > cap && cap < L.base + 0.95 * L.radius;      // nearer than L.base Å in front of its residue: only the near plane allows
    if (t > cap) t = cap;
    return Math.max(0, Math.round((t - 0.95 * L.radius) * 2) / 2);
  }
  // Mol* skips a label whose bounding sphere (its residue's, padded by the text) lies outside the view, wherever offsetX / offsetY
  // would put the text: a label can only be shown while its residue is (nearly) in the window
  function drawable(cam, L) {
    const o = [0, 0, 0, 0], q = [0, 0, 0, 0], st = cam.state, vp = cam.viewport;
    const right = v3.norm(v3.cross(v3.sub(st.target, st.position), st.up)), r = Math.max(L.radius, 1.3 * L.wantSize * 1.9);
    cam.project(o, L.center); cam.project(q, v3.add(L.center, right.map(x => x * r)));
    const rp = Math.hypot(q[0] - o[0], q[1] - o[1]), x = o[0] - vp.x, y = o[1] - vp.y;
    return x > -rp && x < vp.width + rp && y > -rp && y < vp.height + rp;
  }
  function placeLabel(cam, L, placed) {
    if (!drawable(cam, L)) return L.picked || L.important ? { s: L.shift, b: null, forced: true, offscreen: true } : null;
    const b0 = L.box0 || labelBox(cam, L, [0, 0], L.wantZ, L.wantSize), cands = shiftCandidates(b0);
    // its own place first, then the shift it has now (no jumping while the view moves a little), then the nearest free one: free =
    // inside the window, over no label placed before, and nothing drawn in front of its text; for a label that is never hidden,
    // if every such spot has something in front of one character (a stick across a corner), the first of those
    const keep = L.picked || L.important;
    let near = null;
    for (const s of [[0, 0], L.shift, ...cands]) {
      const b = shiftBox(b0, s); if (!inWindow(b) || placed.some(q => overlapArea(b, q) > 0)) continue;
      const n = covered(cam, b, L.chars);
      if (n === 0) return { s: b.shift, b, forced: false };
      if (n === 1 && keep && !near) near = { s: b.shift, b, forced: false };
    }
    if (near) return near;
    if (!keep) return null;
    // no free spot: the least bad one (inside the window, the least overlap, near its residue)
    let best = null, score = Infinity;
    for (const s of cands) {
      const b = shiftBox(b0, s), area = (b.r - b.l) * (b.t - b.b) || 1;
      const v = (inWindow(b) ? 0 : 10) + 3 * placed.reduce((a, q) => a + overlapArea(b, q), 0) / area + 0.05 * Math.hypot(s[0], s[1]);
      if (v < score) { score = v; best = { s, b, forced: true }; }
    }
    return best || { s: [0, 0], b: b0, forced: true };
  }
  // The page draws its own controls over this frame (legend, full-screen and Done buttons, the "Tap to rotate" hint): labels keep
  // out of them. The frame is same-origin with the page; the boxes are those of the iframe element's siblings (one as large as the
  // view, the "Interact with the 3D view" overlay, counts by its visible children only), in drawing-buffer pixels, y up.
  // round 7 (merge of A and B): residue labels are placed around a clicked line's three labels like around the page's controls. Round 8:
  // their boxes where they are drawn (TEXT_PX high, the same on the screen at any depth), a few pixels around
  function lineLabelBoxes(cam) {
    const ln = picked.line, out = []; if (!ln || !ln.items) return out;
    for (const it of ln.items) { const b = lineBox(cam, it, it.wantZ, it.wantSize); if (b) out.push({ l: b.l - 3, r: b.r + 3, b: b.b - 3, t: b.t + 3 }); }
    return out;
  }
  function pageOverlays(cam) {
    let fe = null; try { fe = window.frameElement; } catch (e) { fe = null; }
    if (!fe || !fe.parentElement) return [];
    const fr = fe.getBoundingClientRect(), vp = cam.viewport, k = vp.width / Math.max(1, fr.width), out = [];
    const add = (el, depth) => {
      const r = el.getBoundingClientRect(), cs = el.ownerDocument.defaultView.getComputedStyle(el);
      if (r.width < 1 || r.height < 1 || cs.visibility === 'hidden' || cs.display === 'none' || Number(cs.opacity) === 0) return;
      if (r.width * r.height > 0.5 * fr.width * fr.height) { if (depth < 2) for (const ch of el.children) add(ch, depth + 1); return; }
      const l = (r.left - fr.left) * k, rr = (r.right - fr.left) * k, t = (fr.bottom - r.top) * k, b = (fr.bottom - r.bottom) * k;
      if (rr > 0 && l < vp.width && t > 0 && b < vp.height) out.push({ l: l - 2, r: rr + 2, b: b - 2, t: t + 2 });
    };
    try { for (const el of fe.parentElement.children) if (el !== fe) add(el, 0); } catch (e) { return []; }
    return out;
  }
  // Shifts are applied (a parameter update of the label representation) only once the camera has settled: the next fit that finds the
  // camera unchanged; while the view moves, labels are only shown or hidden. Round 8: depth and size go together at every fit, also while
  // the view moves (a label moved along its line of sight stays at its place on the screen, and its size keeps it TEXT_PX high): when the
  // depth changes or the size is off by more than SIZE_STEP (Mol*'s offsetZ, sizeFactor: no new geometry); one update at a time.
  // A clicked line's labels the same way.
  let commitBusy = null;
  const sizeOff = (a, b) => !(a > 0) || Math.abs(b / a - 1) > SIZE_STEP;
  function commitLabels(settled) {
    if (commitBusy || !viewer) return commitBusy;
    const S = viewer.plugin.state.data, b = S.build(); let n = 0;
    for (const L of labelInfo) {
      if (!S.cells.has(L.ref)) continue;
      const p = {};
      if (L.wantSize > 0 && (L.wantZ !== L.offset || sizeOff(L.size, L.wantSize))) { p.offsetZ = L.wantZ; p.sizeFactor = L.wantSize; L.offset = L.wantZ; L.size = L.wantSize; }
      if (settled && !L.hidden && !sameShift(L.want, L.shift)) { p.offsetX = L.want[0]; p.offsetY = L.want[1]; L.shift = L.want; }
      L.pending = !L.hidden && (!sameShift(L.want, L.shift) || L.wantZ !== L.offset || sizeOff(L.size, L.wantSize));
      if (Object.keys(p).length) { b.to(L.ref).update(old => ({ ...old, type: { ...old.type, params: { ...old.type.params, ...p } } })); n++; }
    }
    const ln = picked.line;
    if (ln && ln.items) for (const it of ln.items) {
      if (!it.ref || !S.cells.has(it.ref) || !(it.wantSize > 0) || (it.wantZ === it.offset && !sizeOff(it.size, it.wantSize))) continue;
      const q = { offsetZ: it.wantZ, textSize: it.wantSize };
      b.to(it.ref).update(old => ({ ...old, ...q })); it.offset = it.wantZ; it.size = it.wantSize; n++;
    }
    if (!n) return null;
    const t0 = performance.now(); if (window.__pvDebug) window.__pvDebug.labelCommit = { at: Math.round(t0), ms: null };
    commitBusy = b.commit().catch(e => { window.__pvError = 'label update: ' + String(e && e.message || e); })
      .finally(() => { commitBusy = null; if (window.__pvDebug && window.__pvDebug.labelCommit) window.__pvDebug.labelCommit.ms = Math.round(performance.now() - t0); });
    return commitBusy;
  }
  // everything the last fit chose, applied now (end of a drawing, a clicked line): waits for an update still running
  async function settleLabels() {
    for (let i = 0; i < 3; i++) { if (commitBusy) await commitBusy; const p = commitLabels(true); if (!p) return; await p; }
  }
  function fitLabels(force) {
    const c = viewer && viewer.plugin.canvas3d; if (!c || !(labelInfo.length || (picked.line && picked.line.items))) return;
    const cam = c.camera, vp = cam.viewport, dpr = bufferScale();
    // the key is the projection itself (what project() uses): it changes when Mol* has applied a camera move or new clipping planes,
    // not already when the camera's target state changes
    // (and the page's controls over the view: opening the legend re-fits)
    const page = pageOverlays(cam);
    const key = [...cam.projectionView, vp.width, vp.height, ...page.flatMap(o => [o.l, o.r, o.b, o.t])].map(x => Number(x).toPrecision(6)).join();
    // round 12: the camera moved but Mol* has not drawn the move yet (its projection follows at its next frame, which can come later than
    // the rest time when a frame is slow): placed again after the next rest (it used to be lost: labels kept the size of the last view)
    const moved = [...cam.state.position, ...cam.state.target, ...cam.state.up].map(x => Number(x).toPrecision(6)).join();
    if (!force && key === lastCam) {
      if (moved !== lastMoved && fitRetries < 10) { fitRetries++; try { c.requestDraw(); } catch (e) { /* drawn anyway */ } cameraMoved(); return; }
      commitLabels(true); return;
    }
    lastCam = key; lastMoved = moved; fitRetries = 0;
    const t0 = performance.now();
    fitLineLabels(cam, dpr);                                  // first: residue labels keep clear of them
    const placed = [...page, ...lineLabelBoxes(cam)], front = frontDepth(cam);   // the page's controls and a clicked line's labels are kept clear
    let deepest = -Infinity;
    for (const L of labelInfo) {
      if (!L.geom) L.geom = labelGeom(L);                    // Mol*'s anchor and depth first (wantOffset, drawable use them)
      L.wantZ = wantOffset(cam, L, front);
      L.wantSize = labelSize(cam, L, L.wantZ, dpr);           // round 8: its size at the depth the fit chooses
      L.box0 = labelBox(cam, L, [0, 0], L.wantZ, L.wantSize); deepest = Math.max(deepest, L.box0.depth);
    }
    projectSolids(cam, deepest);                              // also finds the near plane's caps (capDiscs)
    for (const L of labelInfo) {
      const p = placeLabel(cam, L, placed);
      if (p && p.b) placed.push(p.b);
      if (p) L.want = p.s;
      L.forced = !!(p && p.forced); L.offscreen = !!(p && p.offscreen);
      const hidden = !p;
      if (hidden !== L.hidden || force) { L.hidden = hidden; viewer.plugin.state.data.updateCellState(L.ref, { isHidden: hidden }); }
      L.pending = !L.hidden && (!sameShift(L.want, L.shift) || L.wantZ !== L.offset || sizeOff(L.size, L.wantSize));
    }
    setTimeout(settleLabels, 0);                              // round 9: placed only at rest: shifts, depth and size applied at once (Mol*'s update in its own task)
    fitCount++;
    if (window.__pvDebug) { window.__pvDebug.labelFit = labelInfo.map(L => ({ key: L.key, hidden: L.hidden, important: L.important, shift: L.shift, offsetZ: L.offset, forced: L.forced, size: L.size }));
      const ms = Math.round((performance.now() - t0) * 10) / 10; fitTimes.push(ms); if (fitTimes.length > 50) fitTimes.shift();
      window.__pvDebug.labelFitMs = ms; window.__pvDebug.labelFitTimes = fitTimes.slice(); window.__pvDebug.labelFits = fitCount; window.__pvDebug.identifyCalls = identifyCalls;
      window.__pvDebug.labelFitAt = Math.round(t0); }
  }
  // a residue label's size at depth offsetZ (Mol*'s point for its text: offsetZ + 0.95 × radius towards the camera)
  function labelSize(cam, L, offsetZ, dpr) {
    const ray = v3.norm(v3.sub(cam.state.position, L.center));
    return sizeAt(cam, v3.add(L.center, ray.map(x => x * (offsetZ + 0.95 * L.radius))), L.emU, dpr);
  }
  window.__pvLabelBox = L => labelBox(viewer.plugin.canvas3d.camera, L);         // for the acceptance test
  // round 8: `px` = the label's text height (em) in CSS px as drawn now
  window.__pvLabels = () => {
    const c = viewer && viewer.plugin.canvas3d;
    if (c) { const dpr = bufferScale(); for (const L of labelInfo) { try { L.px = labelPx(c.camera, L, dpr); } catch (e) { L.px = null; } } }
    return labelInfo;
  };

  // ---------------------------------------------------------------- round 8 (§2.1): a clicked line's labels in front
  // A clicked line has three labels (Mol*'s measurement labels: its two ends, the distance; picked.line.items). Each is drawn in front
  // of every atom whose drawing could reach into its box: atoms that project within 3 Å of the box and lie in front of the label's
  // anchor (also atoms up to 3.5 Å in front of the near clipping plane: cut off, their drawing still reaches behind it), by 3.5 Å (a
  // stick, the cartoon around the backbone, a pocket surface); never past the near plane (labelNear). The atoms: every heavy atom of
  // the drawing (protein, pose, other modes shown).
  let sceneAtoms = null;
  function lineGeom(it) {
    try {
      const cell = viewer.plugin.state.data.cells.get(it.ref), ro = cell.obj.data.repr.renderObjects.find(o => 'tFont' in o.values);
      const v = ro.values, m = v.aMapping.ref.value, c = v.aPosition.ref.value;
      if (!(m.length >= 8 && c.length >= 3)) return null;
      const x0 = Math.min(m[0], m[4]), x1 = Math.max(m[0], m[4]), y0 = Math.min(m[1], m[3]), y1 = Math.max(m[1], m[3]);
      if (!(x1 > x0 && y1 > y0)) return null;
      // Mol*'s text builder: the background is (0.8 + 0.8 × margin) × scale units high (scale: 1 for a label, larger for the distance
      // of big groups); the em of the text is EM × scale
      it.anchor = [c[0], c[1], c[2]]; it.depthA = v.aDepth.ref.value[0] || 0; it.emU = EM * (y1 - y0) / (0.8 * (1 + it.margin));
      return { bg: [x0, x1, y0, y1] };
    } catch (e) { return null; }
  }
  function lineBox(cam, it, offsetZ, size) {
    const g = it.unit || (it.unit = lineGeom(it)); if (!g || !(size > 0)) return null;
    const st = cam.state, ray = v3.norm(v3.sub(st.position, it.anchor)), right = v3.norm(v3.cross(v3.sub(st.target, st.position), st.up));
    const pt = v3.add(it.anchor, ray.map(x => x * (offsetZ + 0.95 * it.depthA))), o = [0, 0, 0, 0], q = [0, 0, 0, 0];
    cam.project(o, pt); cam.project(q, v3.add(pt, right));
    const px = Math.hypot(q[0] - o[0], q[1] - o[1]), vp = cam.viewport, cx = o[0] - vp.x, cy = o[1] - vp.y, u = size * px;
    const dir = v3.norm(v3.sub(st.target, st.position)), rel = v3.sub(pt, st.position);
    return { l: cx + g.bg[0] * u, r: cx + g.bg[1] * u, b: cy + g.bg[2] * u, t: cy + g.bg[3] * u, px, z: o[2], x0: vp.x, y0: vp.y,
             depth: rel[0] * dir[0] + rel[1] * dir[1] + rel[2] * dir[2] };
  }
  function fitLineLabels(cam, dpr) {
    const ln = picked.line; if (!ln || !ln.items || !sceneAtoms) return;
    const st = cam.state, dir = v3.norm(v3.sub(st.target, st.position)), p0 = st.position, vp = cam.viewport, near = cam.near;
    const p5 = st.mode !== 'orthographic' ? cam.projection[5] : 0, A = sceneAtoms, o = [0, 0, 0, 0];
    let pts = null;                                           // the atoms in front of the near plane: screen x, y, depth, 3 Å in px
    for (const it of ln.items) {
      if (!it.unit) it.unit = lineGeom(it); if (!it.unit) continue;
      const ray = v3.norm(v3.sub(p0, it.anchor)), cos = -(ray[0] * dir[0] + ray[1] * dir[1] + ray[2] * dir[2]);
      const rel = v3.sub(it.anchor, p0), za = rel[0] * dir[0] + rel[1] * dir[1] + rel[2] * dir[2];
      // its box on the screen once TEXT_PX high (the same at any depth along its line of sight)
      const z0 = Math.max(0, it.offset), s0 = sizeAt(cam, v3.add(it.anchor, ray.map(x => x * (z0 + 0.95 * it.depthA))), it.emU, dpr), b = lineBox(cam, it, z0, s0);
      if (!b || !(cos > 0.2)) { it.wantZ = it.offset; it.wantSize = b ? s0 : it.size; continue; }   // seen almost side-on: only its size
      if (!pts) {
        pts = [];
        const m = cam.projectionView;                         // round 9: Mol*'s projection by hand (no array per atom)
        for (let i = 0; i < A.length; i += 3) {
          const x = A[i], y = A[i + 1], z = A[i + 2], d = (x - p0[0]) * dir[0] + (y - p0[1]) * dir[1] + (z - p0[2]) * dir[2];
          if (d <= Math.max(0.5, near - 3.5)) continue;       // cut off (its drawing may reach behind the plane)
          const w = m[3] * x + m[7] * y + m[11] * z + m[15]; if (!(w > 0)) continue;
          pts.push(((m[0] * x + m[4] * y + m[8] * z + m[12]) / w + 1) * vp.width / 2, ((m[1] * x + m[5] * y + m[9] * z + m[13]) / w + 1) * vp.height / 2,
            d, p5 ? 3 * vp.height / 2 * p5 / d : 3 * b.px);
        }
      }
      let front = za - 3.5;                                   // at least clear of its own atom
      for (let i = 0; i < pts.length; i += 4) {
        const d = pts[i + 2], r = pts[i + 3]; if (d >= za || d - 3.5 >= front) continue;
        if (pts[i] > b.l - r && pts[i] < b.r + r && pts[i + 1] > b.b - r && pts[i + 1] < b.t + r) front = d - 3.5;
      }
      const zl = Math.max(front, labelNear(cam, 0.02));
      it.wantZ = Math.max(0, Math.round(((za - zl) / cos - 0.95 * it.depthA) * 4) / 4);
      it.wantSize = sizeAt(cam, v3.add(it.anchor, ray.map(x => x * (it.wantZ + 0.95 * it.depthA))), it.emU, dpr);
    }
  }
  // test hook (contract, round 8): the clicked line's labels as drawn now: text height (em) and box in CSS px of this frame (left, top)
  window.__pvLineLabels = () => {
    const ln = picked.line, c = viewer && viewer.plugin.canvas3d; if (!ln || !ln.items || !c) return [];
    const cam = c.camera, gl = c.webgl.gl, R = gl.canvas.getBoundingClientRect(), kx = R.width / gl.drawingBufferWidth, ky = R.height / gl.drawingBufferHeight;
    const S = viewer.plugin.state.data, r1 = x => Math.round(x * 10) / 10;
    return ln.items.map(it => {
      const b = it.ref ? lineBox(cam, it, it.offset, it.size) : null, cell = it.ref ? S.cells.get(it.ref) : null;
      return { role: it.role, text: it.text, px: b ? r1(it.emU * it.size * b.px * kx) : null,
               box: b ? { x: r1(R.left + (b.x0 + b.l) * kx), y: r1(R.top + (gl.drawingBufferHeight - b.y0 - b.t) * ky), w: r1((b.r - b.l) * kx), h: r1((b.t - b.b) * ky) } : null,
               hidden: !cell || !!(cell.state && cell.state.isHidden), offsetZ: it.offset, depth: b ? r1(b.depth) : null };
    });
  };

  // ---------------------------------------------------------------- round 7 (B): interacting residues, the lines' hover text, label order
  const PRIORITY = ['ionic', 'cation–π', 'π-stacking', 'halogen bond', 'H-bond'];          // then the other types; hydrophobic last
  const rankOf = t => t === 'hydrophobic' ? 99 : PRIORITY.includes(t) ? PRIORITY.indexOf(t) : PRIORITY.length;
  const INTERACT_SIZE = 1.0;                                     // sticks of a residue with an interaction: the size of the pose's
  // the residues to label: every one with a non-hydrophobic interaction (best type first, then the most involved), then
  // hydrophobic-only ones (the most involved first) up to MAX_LABELS labels in all
  function labelKeys(found) {
    if (!found) return [];
    const rs = [...found.values()].map(r => ({ key: `${r.chain}:${r.seq}`, n: r.n, nt: r.types.size, best: Math.min(...[...r.types].map(rankOf)) }))
      .sort((a, b) => a.best - b.best || b.n - a.n || b.nt - a.nt);
    const important = rs.filter(r => r.best < 99), hydrophobic = rs.filter(r => r.best >= 99);
    return [...important, ...hydrophobic.slice(0, Math.max(0, MAX_LABELS - important.length))].map(r => r.key);
  }
  // §2 (round 7): the residues near the pose as sticks: every residue within 4 Å thin with faded heteroatoms, the residue picked in a
  // list thicker (FOCUS_SIZE), every residue with an interaction drawn like the pose (size ≥ INTERACT_SIZE, heteroatoms at full colour,
  // protein-grey carbon), so both ends of every line sit on a visible atom; without a pose only the residue picked in a list.
  // Round 12: sticks are slots kept per protein (a component of the protein structure with a ball-and-stick each, tags pv-res and
  // pv-res-repr; a click on one opens Mol*'s focus in the whole protein): a drawing keeps the slots already on its residues, moves free
  // ones (Mol* updates their geometry, no new render objects), makes more if needed and hides the rest. Built once Mol* has found the
  // interactions (rounds 7–11 updated MVS's sticks then). window.__pvResidueKey(ref): the residue a shown slot draws (tests).
  async function residueSticks(spec, lig, found) {
    const prot = scene.protein, ps = prot && dataOf(prot.struct); if (!ps) return;
    const dark = spec.theme === 'dark', C = PROTEIN[dark ? 'dark' : 'light'], fr = focusResidueOf(spec), fk = fr ? `${fr.chain}:${+fr.seq}` : null;
    const inSite = new Set(lig ? lig.site.map(r => `${r.chain}:${r.seq}`) : []), want = new Map();
    if (lig) for (const r of lig.contacts) want.set(`${r.chain}:${r.seq}`, { size: RES_SIZE, full: false, ignoreH: true });
    if (fk) { const w = want.get(fk); if (w) w.size = FOCUS_SIZE; else want.set(fk, { size: FOCUS_SIZE, full: false, ignoreH: inSite.has(fk) }); }
    if (found) for (const k of found.keys()) { const w = want.get(k); if (w) { w.full = true; w.size = Math.max(INTERACT_SIZE, w.size); } else want.set(k, { size: INTERACT_SIZE, full: true, ignoreH: true }); }
    const slots = prot.stickSlots, live = slots.filter(sl => cellOf(sl.comp) && cellOf(sl.repr));
    const kept = new Map(live.filter(sl => want.has(sl.key)).map(sl => [sl.key, sl])), free = live.filter(sl => !want.has(sl.key));
    const b = SD().build(), next = new Map();
    for (const [k, w] of want) {
      const bundle = bundleOf(ps, prot.index, [k]); if (!bundle) continue;
      const p = sticks(C.carbon, w.full, w.size, w.ignoreH, dark);
      let slot = kept.get(k);
      if (slot) kept.delete(k);
      else if ((slot = free.shift())) b.to(slot.comp).update(old => ({ ...old, type: { name: 'bundle', params: bundle }, label: `Residue ${k}` }));
      else {
        const comp = component(b, prot.struct, bundle, `Residue ${k}`, 'pv-res').ref;
        slot = { comp, repr: representation(b, comp, ps, p, 'pv-res-repr'), made: true };
        slots.push(slot);
      }
      if (!slot.made) b.to(slot.repr).update(old => ({ ...old, type: { ...old.type, params: { ...old.type.params, ignoreHydrogens: w.ignoreH } },
        colorTheme: { name: p.color, params: p.colorParams }, sizeTheme: { name: p.size, params: p.sizeParams } }));
      slot.made = false; slot.key = k; slot.full = w.full; next.set(k, slot);
    }
    await b.commit();
    const S = SD(), used = new Set([...next.values()].map(sl => sl.repr));
    for (const sl of slots) { const c = cellOf(sl.repr), hide = !used.has(sl.repr); if (c && !!(c.state && c.state.isHidden) !== hide) S.updateCellState(sl.repr, { isHidden: hide }); }
    scene.residues = next;
  }
  // stick slots made ahead at idle time (hidden, on the protein's first residues), so a first pose moves them instead of making them
  const STICK_SLOTS = 16;
  async function prewarmSticks() {                            // four slots per call
    const prot = scene.protein, ps = prot && dataOf(prot.struct); if (!ps || prot.stickSlots.length >= STICK_SLOTS) return;
    const keys = [...prot.index.keys()], dark = document.documentElement.dataset.theme === 'dark', C = PROTEIN[dark ? 'dark' : 'light'], b = SD().build();
    for (let n = 0; n < 4 && prot.stickSlots.length < STICK_SLOTS; n++) {
      const k = keys[prot.stickSlots.length], comp = component(b, prot.struct, bundleOf(ps, prot.index, [k]), `Residue ${k}`, 'pv-res').ref;
      prot.stickSlots.push({ comp, repr: representation(b, comp, ps, sticks(C.carbon, false, RES_SIZE, true, dark), 'pv-res-repr', { isHidden: true }), key: null });
    }
    await b.commit();
  }
  window.__pvResidueKey = ref => { for (const [k, sl] of scene.residues) if (sl.comp === ref || sl.repr === ref) return k; return null; };
  // the sticks and labels of the last drawing are hidden (a new one moves them to its residues)
  async function removeResidues() {
    const S = SD();
    for (const sl of (scene.protein ? scene.protein.stickSlots : [])) { const c = cellOf(sl.repr); if (c && !(c.state && c.state.isHidden)) S.updateCellState(sl.repr, { isHidden: true }); }
    scene.residues = new Map(); labelInfo = []; lastCam = '';
    hideLabelSlots();
  }
  // Mol*'s hover box names a residue of the 5 Å site as MVS's tooltip did ("LYS716"; rounds 4–11): its own line under Mol*'s
  const tipNames = new Map();
  const TIP_LABELS = { label: loci => {
    try {
      const e = loci && loci.kind === 'element-loci' && loci.elements[0], bd = loci && loci.kind === 'bond-loci' && loci.bonds[0];
      const unit = e ? e.unit : bd ? bd.aUnit : null; if (!unit || !atomic(unit)) return undefined;
      const root = (e ? loci.structure : bd.aStructure || loci.structure).root, prot = scene.protein && dataOf(scene.protein.struct), site = scene.site && dataOf(scene.site.struct);
      if (!((prot && root === prot.root) || (site && root === site.root))) return undefined;
      const r = residueOf(unit, unit.elements[e ? firstIndex(e.indices) : bd.aIndex]);
      return r.chain === 'L' ? undefined : tipNames.get(`${r.chain}:${r.seq}`);
    } catch (x) { return undefined; }
  }, priority: -1 };

  // the interactions Mol* computed on a structure (the site component)
  function computedInteractions(structure) {
    const box = structure && structure.currentPropertyData && structure.currentPropertyData.molstar_computed_interactions;
    let I = box && box.data && box.data.value; if (I && I.value && !I.contacts) I = I.value;
    return I && I.contacts && I.unitsFeatures ? I : null;
  }
  function siteInteractions() {
    for (const c of viewer.plugin.state.data.cells.values()) {
      if (!(c.transform.tags || []).includes('pv-site-inter') || !c.obj || !c.obj.data) continue;
      const I = computedInteractions(c.obj.data); return I ? { structure: c.obj.data, I } : null;
    }
    return null;
  }
  // one end of a line: Mol*'s feature (an atom, or the atoms of a ring / charged group) with its centre, where the line is drawn
  function featureEnd(I, unit, f) {
    const F = I.unitsFeatures.get(unit.id), h = unit.model.atomicHierarchy, atoms = [];
    for (let k = F.offsets[f]; k < F.offsets[f + 1]; k++) atoms.push(h.atoms.label_atom_id.value(unit.elements[F.members[k]]));
    const el = unit.elements[F.members[F.offsets[f]]], ri = h.residueAtomSegments.index[el], ci = h.chainAtomSegments.index[el];
    const m = unit.conformation.operator.matrix, x = F.x[f], y = F.y[f], z = F.z[f];
    const pos = [m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]];
    return { chain: h.chains.auth_asym_id.value(ci), seq: h.residues.auth_seq_id.value(ri), name: h.atoms.label_comp_id.value(el), atoms, pos };
  }
  // a ligand–protein line: type, residue, hover text "LYS716 · cation–π · 4.2 Å" (distance between the two feature centres)
  function describeLine(I, unitA, indexA, unitB, indexB, type) {
    const A = featureEnd(I, unitA, indexA), B = featureEnd(I, unitB, indexB);
    const [p, l] = A.chain === 'L' ? [B, A] : B.chain === 'L' ? [A, B] : [null, null];
    if (!p || p.chain === 'L') return null;
    const t = TYPES[type] || 'other', d = Math.hypot(p.pos[0] - l.pos[0], p.pos[1] - l.pos[1], p.pos[2] - l.pos[2]), residue = `${p.name}${p.seq}`;
    return { type: t, key: `${p.chain}:${p.seq}`, residue, label: `${residue} · ${t} · ${d.toFixed(1)} Å`, distance: Math.round(d * 10) / 10,
             a: p.pos, b: l.pos, residueAtoms: p.atoms, ligandAtoms: l.atoms };
  }
  // the line under the pointer (Mol*'s interaction loci: a data loci tagged 'interactions', elements = unit / feature pairs)
  const isLineLoci = l => !!l && l.kind === 'data-loci' && l.tag === 'interactions';
  function lineOfLoci(loci) {
    if (!isLineLoci(loci) || !loci.elements.length || !loci.data || !loci.data.interactions) return null;
    const e = loci.elements[0], I = loci.data.interactions;
    if (!e.unitA || !e.unitB || e.unitA === e.unitB) return null;                      // intra-unit lines are not drawn
    const k = I.contacts.getEdgeIndex(e.indexA, e.unitA.id, e.indexB, e.unitB.id);
    return k < 0 ? null : describeLine(I, e.unitA, e.indexA, e.unitB, e.indexB, I.contacts.edges[k].props.type);
  }
  // §2: hover text of a line. Mol*'s hover box (.msp-highlight-info) lists the label of every loci label provider; its default
  // provider would add "Hydrogen Bond</br>…" for the same line: every other provider is wrapped to say nothing for interaction loci.
  // The box shows a one-line label as HTML: the text is escaped.
  const escHtml = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const LINE_LABELS = { label: loci => { const d = lineOfLoci(loci); return d ? escHtml(d.label) : undefined; }, priority: 1000 };
  let lineHoverPicking = false;
  function lineHover() {
    const P = viewer.plugin, M = P.managers.lociLabels, c = P.canvas3d;
    if (!M.providers.includes(LINE_LABELS)) M.addProvider(LINE_LABELS);
    for (const p of M.providers) {
      if (p === LINE_LABELS || p.pvQuietOnLines) continue;
      const f = p.label; p.label = (loci, repr) => isLineLoci(loci) ? undefined : f.call(p, loci, repr); p.pvQuietOnLines = true;
    }
    // Mol* picks what is under the pointer asynchronously (WebGL2 fences); where a fence does not complete within its few frames
    // (software rendering, e.g. SwiftShader in a browser without a GPU or in the headless test browser) every hover comes back empty.
    // For the interaction lines a synchronous pick at the same point fills in (Mol* does the same for a click).
    if (lineHoverPicking || !c) return;
    lineHoverPicking = true;
    c.interaction.hover.subscribe(e => {
      if (!e || !e.page || buttonsDown || (e.current && e.current.loci && e.current.loci.kind !== 'empty-loci')) return;   // round 9: not while dragging
      let hit = null;
      try { const p = identify(c, e.page); hit = p ? { pick: p, at: c.getLoci(p.id) } : null; } catch (x) { hit = null; }
      if (hit && hit.at && isLineLoci(hit.at.loci)) P.behaviors.interaction.hover.next({ ...e, current: hit.at, position: hit.pick.position });
    });
  }
  // test hooks (contract, round 7): every drawn ligand–protein line (Mol* draws no line for a contact it flagged as filtered), and a
  // 3D point in CSS pixels of this frame
  window.__pvLines = () => {
    const x = viewer && siteInteractions(); if (!x) return [];
    const { structure, I } = x, out = [], seen = new Set();
    for (let i = 0; i < I.contacts.edgeCount; i++) {
      const e = I.contacts.edges[i]; if (e.props.flag === 1) continue;
      // Mol*'s inter-unit graph holds every contact in both directions (A->B and B->A): one line each
      const id = [`${e.unitA}:${e.indexA}`, `${e.unitB}:${e.indexB}`].sort().join('|'); if (seen.has(id)) continue; seen.add(id);
      const uA = structure.unitMap.get(e.unitA), uB = structure.unitMap.get(e.unitB); if (!uA || !uB) continue;
      const d = describeLine(I, uA, e.indexA, uB, e.indexB, e.props.type); if (d) out.push(d);
    }
    return out;
  };
  window.__pvToScreen = p => {
    const c = viewer.plugin.canvas3d, o = [0, 0, 0, 0], gl = c.webgl.gl, r = gl.canvas.getBoundingClientRect();
    c.camera.project(o, p);
    return { x: r.left + o[0] * r.width / gl.drawingBufferWidth, y: r.top + (gl.drawingBufferHeight - o[1]) * r.height / gl.drawingBufferHeight };
  };
  // Round 9 (round-8 feedback §1.1–1.2): labels are placed only while the view is at rest: CAMERA_REST ms after Mol*'s camera last
  // changed (its view or projection: a drag, the wheel, a transition, a resize; camera.changed fires on such frames only) and with no
  // pointer button held. Never while it moves, and nothing runs while the viewer is idle: no timer, and no subscription to Mol*'s
  // drawing (a placement's own update draws again). Changes that move no camera (a surface shown, a clicked line, a focus, the
  // page's legend opened) ask for a placement soon (scheduleFit), unless the view is moving.
  const CAMERA_REST = 200;
  let fitTimer = 0, restTimer = 0, buttonsDown = false;
  const scheduleFit = () => { if (!fitTimer && !restTimer && !buttonsDown) fitTimer = setTimeout(() => { fitTimer = 0; if (!restTimer && !buttonsDown) fitLabels(false); }, 80); };
  // round 12: from a camera change until the next placement the labels are pending (__pvLabels: not yet placed for this view)
  function cameraMoved() { clearTimeout(restTimer); restTimer = setTimeout(cameraRested, CAMERA_REST); for (const L of labelInfo) L.pending = true; }
  function cameraRested() {
    restTimer = 0; if (buttonsDown) return;
    // a camera transition still running (its frames can be far apart with slow rendering) is not rest: look again later
    const tr = viewer && viewer.plugin.canvas3d && viewer.plugin.canvas3d.camera.transition;
    if (tr && tr.inTransition) { cameraMoved(); return; }
    fitLabels(false);
  }
  // round 12 (B): the frame moves between the pages' viewer boxes (one persistent frame): the box watched follows it (watchView, 'show')
  let watched = null, watcher = null;
  function watchBox() {
    try {
      const p = window.frameElement && window.frameElement.parentElement; if (!p || p === watched) return;
      (watcher || (watcher = new MutationObserver(scheduleFit))).disconnect(); watched = p; watcher.observe(p, { subtree: true, childList: true, attributes: true });
    } catch (e) { /* no page around */ }
  }
  function watchView(plugin) {
    plugin.canvas3d.camera.changed.subscribe(cameraMoved);
    addEventListener('pointerdown', e => { buttonsDown = e.buttons !== 0; }, true);
    const up = () => { if (buttonsDown) { buttonsDown = false; cameraMoved(); } };
    addEventListener('pointerup', up, true); addEventListener('pointercancel', up, true); addEventListener('blur', up);
    // the page's own controls over the view (the legend opened, the hint): labels keep clear of them (pageOverlays)
    watchBox();
  }

  // Round 6, 6: a selection or focus never recolours atoms (Mol*'s colour marker is off); it shows as an outline in the accent colour,
  // and the focused residue as thicker sticks. Also: background, no orientation axes.
  // Round 7: Mol*'s select colour (green, unused while the marker is off) is the accent too, so no green is left in the canvas settings;
  // faded interaction lines (alpha LINE_FADE) stay pickable (Mol* picks nothing below 0.5 by default; pocket surfaces, 0.15-0.25, still not).
  // Round 9: Mol* 5 reads the outline settings from the canvas props' top-level `marking` (the copy under postprocessing that rounds 6–8
  // set is not used: the selection outline was Mol*'s default green). And no ghost edges (round-8 feedback §2): Mol* draws the
  // outline of the hidden parts of a highlighted thing faintly over whatever hides them, labels included (a residue inside a pocket
  // surface: its whole outline); the outline of what can be seen stays.
  function canvasProps(dark) {
    const c = viewer && viewer.plugin.canvas3d; if (!c) return;
    const acc = ACCENT[dark ? 'dark' : 'light'];
    const marking = { ...(c.props.marking || c.props.postprocessing.marking), enabled: true, selectEdgeColor: acc, selectEdgeStrength: 2, ghostEdgeStrength: 0 };
    c.setProps({ renderer: { backgroundColor: PROTEIN[dark ? 'dark' : 'light'].bg, colorMarker: false, selectColor: acc, pickingAlphaThreshold: 0.3 }, camera: NO_AXES,
      ...(c.props.marking ? { marking } : { postprocessing: { marking } }) });
  }

  // ---------------------------------------------------------------- round 7, 1: Mol*'s focus, without green
  // A click on the ligand or a residue opens Mol*'s focus again (behaviour representation-focus-loci, Mol*'s default bindings): the target
  // and every residue within 5 Å as ball-and-stick (behaviour create-structure-focus-representation), the camera centred on it. Its carbons
  // are coloured by chain as in Mol*, but with the palette [protein grey, ligand gold] in Mol*'s chain order (chain L is the pose), and
  // without Mol*'s own interaction lines (components target + surroundings; ours stay). The same focus opens for spec.focusResidue
  // (a Contacts chip). Cleared by a click on empty space (Mol*), Esc, {type:'clear-focus'} and {type:'reset'}.
  const FOCUS_BEHAVIOUR = 'ms-plugin.create-structure-focus-representation';
  const picked = { focus: null, line: null, pending: null, drawing: false, sent: '', seq: 0, theme: '', alpha0: null };
  const tagged = (cell, tag) => (cell.transform.tags || []).includes(tag);
  const f64 = new Float64Array(1), i32 = new Int32Array(f64.buffer);
  // an element set of a loci: an interval (Mol* packs [start, end) into one number) or a sorted array of unit indices
  const firstIndex = ix => { if (typeof ix === 'number') { f64[0] = ix; return i32[0]; } return ix[0]; };
  const atomic = unit => unit.model && unit.model.atomicHierarchy;
  function residueOf(unit, el) {
    const h = unit.model.atomicHierarchy, ri = h.residueAtomSegments.index[el];
    return { chain: h.chains.auth_asym_id.value(h.chainAtomSegments.index[el]), seq: h.residues.auth_seq_id.value(ri), name: h.atoms.label_comp_id.value(el),
      atom: h.atoms.label_atom_id.value(el) };
  }
  // round 12: the protein structure (cartoon, residue sticks) and the site structure of the pose (pose, interaction lines)
  const proteinStructure = () => (scene.protein ? dataOf(scene.protein.struct) : null);
  const siteStructure = () => (scene.site ? dataOf(scene.site.struct) : null);
  const focusResidueOf = spec => (spec.focusResidue && /^[A-Za-z0-9]{1,4}$/.test(spec.focusResidue.chain) && Number.isFinite(+spec.focusResidue.seq) ? spec.focusResidue : null);
  // 'ligand' for the pose (chain L) and the other modes, else the residue 'A:716' (of the protein or of the site structure)
  function lociKey(loci) {
    const e = loci && loci.elements && loci.elements[0];
    if (!e || !atomic(e.unit)) return null;
    const r = residueOf(e.unit, e.unit.elements[firstIndex(e.indices)]), root = loci.structure.root, prot = proteinStructure(), site = siteStructure();
    return r.chain === 'L' || !((prot && root === prot.root) || (site && root === site.root)) ? 'ligand' : `${r.chain}:${r.seq}`;
  }
  function residueLoci(structure, chain, seq) {
    if (structure && scene.protein && structure === proteinStructure()) return residuesLoci(structure, scene.protein.index, [`${chain}:${seq}`]);
    const elements = [];
    for (const unit of structure.units) {
      if (!atomic(unit)) continue;
      const idx = [];
      for (let i = 0; i < unit.elements.length; i++) { const r = residueOf(unit, unit.elements[i]); if (r.chain === chain && r.seq === seq) idx.push(i); }
      if (idx.length) elements.push({ unit, indices: Int32Array.from(idx) });
    }
    return elements.length ? { kind: 'element-loci', structure, elements } : null;
  }
  // element colours; carbon by chain: the pose (chain L) gold, protein residues grey. Mol*'s chain-id theme numbers the chains of every
  // model loaded in the plugin in one series (its structure-info behaviour gives each new model an asym-id offset after the models
  // already there), so the colour list is laid out over those numbers. Round 13: since round 12 the pose's site is a second model after
  // the protein, and a list [grey, gold] by the site's own chain order gave its protein residues the ligand's gold.
  function focusTheme(dark) {
    const grey = PROTEIN[dark ? 'dark' : 'light'].carbon, colors = [], Model = molstar.lib.structure.Model, isModel = molstar.lib.plugin.PluginStateObject.Molecule.Model.is;
    for (const c of SD().cells.values()) {
      const m = c.obj && isModel(c.obj) ? c.obj.data : null, off = m && Model.AsymIdOffset.get(m).value;
      if (!m || !off || !m.properties.structAsymMap) continue;
      const ids = []; m.properties.structAsymMap.forEach(a => { if (!ids.includes(a.auth_id)) ids.push(a.auth_id); });
      ids.forEach((id, i) => { colors[off.auth + i] = id === 'L' ? LIGAND_C : grey; });
    }
    for (let i = 0; i < Math.max(2, colors.length); i++) if (colors[i] == null) colors[i] = grey;
    return { name: 'element-symbol', params: { carbonColor: { name: 'chain-id', params: { asymId: 'auth', palette: { name: 'colors', params: { list: { kind: 'set', colors } } } } },
      saturation: 0, lightness: 0, colors: { name: 'default', params: {} } } };
  }
  async function configureFocus(dark) {
    const plugin = viewer.plugin, cell = [...plugin.state.behaviors.cells.values()].find(c => c.transform.transformer.id === FOCUS_BEHAVIOUR);
    if (!cell) return;
    const th = focusTheme(dark), key = JSON.stringify(th);
    if (key === picked.theme) return;
    const p = cell.transform.params;
    await plugin.state.behaviors.build().to(cell.transform.ref).update({ ...p, expandRadius: SITE_A, components: ['target', 'surroundings'], ignoreHydrogens: true,
      targetParams: { ...p.targetParams, colorTheme: th }, surroundingsParams: { ...p.surroundingsParams, colorTheme: th } }).commit();
    picked.theme = key;
  }
  // the frame's state for the page and the tests: __pvDebug.focus / .line and the message 'focus' (not while drawing; once at its end)
  function publish() {
    const D = debug();
    D.focus = picked.focus ? { key: picked.focus.key, surroundings: picked.focus.surroundings } : null;
    D.line = picked.line ? picked.line.info : null;
    if (picked.drawing) return;
    const msg = { type: 'focus', active: !!(picked.focus || picked.line), key: picked.focus ? picked.focus.key : null, line: !!picked.line };
    const k = JSON.stringify(msg);
    if (k !== picked.sent) { picked.sent = k; post(msg); }
  }
  // Mol*'s focus changed (a click, a chip, a clear): published at once. The surroundings are computed from the drawn coordinates the
  // way Mol* selects them (whole residues with an atom within 5 Å of the target), not read back from Mol*'s selection cell: waiting
  // for that cell raced with the label updates (merge of rounds 7 A and B)
  let drawnAtoms = null;
  function focusSurroundings(key) {
    if (!drawnAtoms || !key) return [];
    const { P, lig } = drawnAtoms;
    const target = key === 'ligand' ? (lig ? lig.atoms : []) : P.atoms.filter(a => `${a.ch}:${a.seq}` === key);
    return near(P, target, SITE_A).map(r => `${r.chain}:${r.seq}`).sort((x, y) => { const [cx, sx] = x.split(':'), [cy, sy] = y.split(':'); return cx < cy ? -1 : cx > cy ? 1 : sx - sy; });
  }
  function watchFocus(plugin) {
    plugin.managers.structure.focus.behaviors.current.subscribe(f => {
      ++picked.seq; picked.pending = null;
      if (!f || !f.loci) { picked.focus = null; publish(); lastCam = ''; scheduleFit(); return; }
      const key = lociKey(f.loci), site = siteStructure(), prot = proteinStructure();
      // round 12: a residue of the site structure (Mol*'s focus residues around the pose are drawn from it) is focused in the whole
      // protein, so Mol*'s 5 Å around it are all there, as before (the site holds only the residues within SITE_STRUCT_A Å of the pose)
      if (key && key !== 'ligand' && site && prot && f.loci.structure.root === site.root) {
        const [ch, seq] = key.split(':'), l = residueLoci(prot, ch, +seq);
        if (l) setTimeout(() => plugin.managers.structure.focus.setFromLoci(l), 0);
      }
      picked.focus = key ? { key, surroundings: focusSurroundings(key) } : null;
      publish();
      lastCam = ''; scheduleFit();                                   // round 9: its ball-and-stick residues can cover labels
    });
  }
  // a Contacts chip (or a residue in the Pockets tab): the same focus on that residue after the draw, the camera on it
  async function focusFromSpec(spec, camera = true) {
    const fr = focusResidueOf(spec), S = proteinStructure();
    if (!fr || !S) return;
    const loci = residueLoci(S, fr.chain, +fr.seq);
    if (!loci) return;
    viewer.plugin.managers.structure.focus.setFromLoci(loci);
    if (camera) viewer.plugin.managers.camera.focusLoci(loci, { extraRadius: 6 });       // round 12: not after a theme change (the camera stays)
    if (picked.pending) await picked.pending;
  }
  // clear Mol*'s focus and a clicked line (Esc, clear-focus, reset, a new drawing)
  async function clearPicked() {
    picked.seq++; picked.pending = null;
    try { viewer.plugin.managers.structure.focus.clear(); } catch (e) { /* nothing focused */ }
    picked.focus = null;
    await queueLine(() => clearLine());
    publish();
  }

  // ---------------------------------------------------------------- round 7, 1b: a clicked interaction line
  // A click on one of our interaction lines (Mol*'s loci: data-loci tagged 'interactions'): Mol* drops its focus; we add a distance
  // between the line's two features (Mol*'s measurement, accent colour, thicker than the lines) with the text "4.2 Å" (the distance of
  // Mol*'s feature centres, as drawn), a label at each end ("LYS716 · NZ", "ligand · N3" or "ligand · aromatic ring" at a ring centre),
  // fade the other lines (alpha LINE_FADE) and move the camera to the two ends. All cells tagged 'pv-line'. A click anywhere else, Esc,
  // clear-focus and reset remove it.
  const LINE_TAG = 'pv-line', LINE_FADE = 0.35;
  let lineQueue = Promise.resolve();
  const queueLine = f => { lineQueue = lineQueue.then(f).catch(e => { window.__pvError = 'line: ' + String(e && e.message || e); }); return lineQueue; };
  function featureSide(I, unit, index) {
    const F = I.unitsFeatures.get(unit.id), members = [];
    for (let k = F.offsets[index]; k < F.offsets[index + 1]; k++) members.push(F.members[k]);
    members.sort((a, b) => a - b);
    const atoms = members.map(m => residueOf(unit, unit.elements[m])), r = atoms[0];
    const p = [F.x[index], F.y[index], F.z[index]];
    const op = unit.conformation && unit.conformation.operator;
    if (op && !op.isIdentity && op.matrix) { const M = op.matrix, [x, y, z] = p; p[0] = M[0] * x + M[4] * y + M[8] * z + M[12]; p[1] = M[1] * x + M[5] * y + M[9] * z + M[13]; p[2] = M[2] * x + M[6] * y + M[10] * z + M[14]; }
    // Mol*'s feature type 3 is an aromatic ring (its centre); other features name their atoms (named in lineOf, nearest first)
    const ring = F.types && F.types[index] === 3 && members.length > 2;
    const at = members.map((m, i) => { const el = unit.elements[m], c = unit.conformation; return { name: atoms[i].atom, pos: [c.x(el), c.y(el), c.z(el)] }; });
    return { unit, members, ligand: r.chain === 'L', residue: `${r.name}${r.seq}`, key: `${r.chain}:${r.seq}`, ring, at, pos: p };
  }
  // the atoms of a feature by distance to the other end: the nearest is the one named in __pvDebug.line, the label lists them all
  function nameSide(s, other) {
    if (s.ring) { s.atom = s.atoms = 'aromatic ring'; return; }
    const d = a => Math.hypot(a.pos[0] - other[0], a.pos[1] - other[1], a.pos[2] - other[2]);
    const names = s.at.slice().sort((a, b) => d(a) - d(b)).map(a => a.name);
    s.atom = names[0]; s.atoms = names.join('/');
  }
  function lineOf(loci) {
    const e = loci.elements && loci.elements[0], I = loci.data && loci.data.interactions;
    if (!e || !I) return null;
    const A = featureSide(I, e.unitA, e.indexA), B = featureSide(I, e.unitB, e.indexB);
    if (A.ligand === B.ligand) return null;                       // ligand-protein lines only (all we draw)
    const prot = A.ligand ? B : A, lig = A.ligand ? A : B, structure = siteStructure() || loci.data.structure;
    nameSide(prot, lig.pos); nameSide(lig, prot.pos);
    let type = 'other';
    if (e.unitA !== e.unitB) { const i = I.contacts.getEdgeIndex(e.indexA, e.unitA.id, e.indexB, e.unitB.id); if (i >= 0) type = TYPES[I.contacts.edges[i].props.type] || 'other'; }
    const d = Math.hypot(prot.pos[0] - lig.pos[0], prot.pos[1] - lig.pos[1], prot.pos[2] - lig.pos[2]);
    // the features' atoms as a loci on the whole structure: Mol*'s measurement turns a loci into a bundle of unit indices, which must
    // index the parent structure's units (the lines live on the 5 Å site, whose protein unit is a subset)
    const lociOf = s => {
      const pu = structure.unitMap.get(s.unit.id) && structure.unitMap.get(s.unit.id).invariantId === s.unit.invariantId ? structure.unitMap.get(s.unit.id)
        : structure.units.find(u => u.invariantId === s.unit.invariantId);
      if (!pu) return null;
      const idx = s.members.map(m => { const el = s.unit.elements[m]; let lo = 0, hi = pu.elements.length - 1;
        while (lo <= hi) { const mid = (lo + hi) >> 1; if (pu.elements[mid] < el) lo = mid + 1; else if (pu.elements[mid] > el) hi = mid - 1; else return mid; } return -1; });
      return idx.includes(-1) ? null : { kind: 'element-loci', structure, elements: [{ unit: pu, indices: Int32Array.from(idx) }] };
    };
    const protLoci = lociOf(prot), ligLoci = lociOf(lig);
    if (!protLoci || !ligLoci) return null;
    return { prot, lig, protLoci, ligLoci, distance: Math.round(d * 10) / 10,
      info: { type, residue: prot.residue, residueAtom: prot.atom, ligand: lig.atom, distance: Math.round(d * 10) / 10 } };
  }
  const linesCell = () => [...viewer.plugin.state.data.cells.values()].find(c => tagged(c, 'pv-interactions'));
  async function fadeLines(on) {
    const c = linesCell(); if (!c) return;
    const alpha = c.transform.params.type.params.alpha;
    if (on && picked.alpha0 == null) picked.alpha0 = alpha == null ? 1 : alpha;
    const want = on ? Math.min(LINE_FADE, picked.alpha0) : (picked.alpha0 == null ? alpha : picked.alpha0);
    if (!on) picked.alpha0 = null;
    if (want === alpha) return;
    await viewer.plugin.state.data.build().to(c.transform.ref).update(old => ({ ...old, type: { ...old.type, params: { ...old.type.params, alpha: want } } })).commit();
  }
  // where the labels go: each end label on the side away from the other end, the distance beside the middle (screen directions now)
  function attachments(a, b) {
    const cam = viewer.plugin.canvas3d.camera, pa = [0, 0, 0, 0], pb = [0, 0, 0, 0];
    cam.project(pa, a); cam.project(pb, b);
    const dx = pa[0] - pb[0], dy = pa[1] - pb[1], flat = Math.abs(dx) >= Math.abs(dy);   // projected y grows upwards
    const away = s => (flat ? (s * dx > 0 ? 'middle-left' : 'middle-right') : (s * dy > 0 ? 'bottom-center' : 'top-center'));
    return { prot: away(1), lig: away(-1), dist: flat ? 'top-center' : 'middle-left' };
  }
  // the camera to the two ends: centred on the middle, turned (least turn) so the line lies across the view, both ends in it
  function lineCamera(a, b) {
    const c = viewer.plugin.canvas3d, st = c.camera.state, dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2], len = p => Math.hypot(p[0], p[1], p[2]);
    const u = v3.norm(v3.sub(b, a)), v0 = v3.norm(v3.sub(st.target, st.position));
    let v = v3.sub(v0, u.map(x => x * dot(v0, u)));
    v = v3.norm(len(v) < 1e-3 ? v3.cross(u, st.up) : v);
    let up = v3.sub(st.up, v.map(x => x * dot(st.up, v)));
    up = v3.norm(len(up) < 1e-3 ? v3.cross(v, u) : up);
    return c.camera.getInvariantFocus(a.map((x, i) => (x + b[i]) / 2), len(v3.sub(b, a)) / 2 + 6, up, v);
  }
  async function showLine(loci) {
    const L = lineOf(loci);
    if (!L) return clearLine();
    // Round 9, the camera: Mol* carries a camera request out at its next scene update, and its click behaviour has just asked to
    // centre on the click. That request is replaced at once by "stay" (carried out now: canvas3d.commit), and the camera turns to the
    // line once its measurement and labels are built, carried out by the update that draws them (canvas3d.commit now, or Mol*'s next
    // one). The label placements used to update the scene while the view moved and so carried a late request out; they no longer do.
    const P = viewer.plugin, c3 = P.canvas3d, snapshot = lineCamera(L.prot.pos, L.lig.pos);
    c3.requestCameraReset({ snapshot: c3.camera.getSnapshot(), durationMs: 0 }); c3.commit(true);
    await clearLine(true);
    const M = P.managers.structure.measurement, dark = document.documentElement.dataset.theme === 'dark';
    const acc = ACCENT[dark ? 'dark' : 'light'], at = attachments(L.prot.pos, L.lig.pos);
    // round 8: their first size makes them TEXT_PX high for the camera the line turns to (5 Å in front of their point); the fit then
    // sets depth and size from where Mol* draws them (fitLineLabels)
    const cam = P.canvas3d.camera, dpr = bufferScale(), vd = v3.norm(v3.sub(snapshot.target, snapshot.position)), p5 = cam.projection[5];
    const first = p => { const r = v3.sub(p, snapshot.position), d = Math.max(1, r[0] * vd[0] + r[1] * vd[1] + r[2] * vd[2] - 5);
      return p5 > 0 && cam.state.mode !== 'orthographic' ? Math.round(TEXT_PX * dpr / (EM * cam.viewport.height / 2 * p5 / d) * 1000) / 1000 : 0.7; };
    const mid = L.prot.pos.map((x, i) => (x + L.lig.pos[i]) / 2);
    const text = (attachment, size) => ({ textColor: acc, textSize: size, sizeFactor: 1, borderWidth: 0, background: true, backgroundColor: dark ? 0x1c1c20 : 0xffffff,
      backgroundOpacity: 0.85, backgroundMargin: 0.25, attachment, offsetZ: 5, scaleByRadius: false, fontQuality: FONT_QUALITY });
    const tags = kind => ({ selectionTags: [LINE_TAG], reprTags: [LINE_TAG, `${LINE_TAG}:${kind}`] });
    const items = [{ role: 'residue', text: `${L.prot.residue} · ${L.prot.atoms}`, size: first(L.prot.pos) }, { role: 'ligand', text: `ligand · ${L.lig.atoms}`, size: first(L.lig.pos) },
      { role: 'distance', text: `${L.distance.toFixed(1)} Å`, size: first(mid) }].map(it => ({ ...it, offset: 5, margin: 0.25, emU: EM, wantZ: 5, wantSize: it.size }));
    const [res, lig, dist] = items;
    await M.addDistance(L.protLoci, L.ligLoci, { customText: dist.text, lineParams: { linesColor: acc, linesSize: 0.18, dashLength: 0.2 },
      labelParams: text(at.dist, dist.size), ...tags('distance') });
    // the end labels a little away from their atom (Mol*'s tether: a short pointer from the label to the atom), so they do not hide it
    const end = (attachment, size) => ({ ...text(attachment, size), tether: true, tetherLength: 0.7, tetherBaseWidth: 0.35 });
    await M.addLabel(L.protLoci, { labelParams: { ...end(at.prot, res.size), customText: res.text }, ...tags('residue') });
    await M.addLabel(L.ligLoci, { labelParams: { ...end(at.lig, lig.size), customText: lig.text }, ...tags('ligand') });
    for (const c of viewer.plugin.state.data.cells.values()) for (const it of items) if (tagged(c, `${LINE_TAG}:${it.role}`) && c.obj && c.obj.data && c.obj.data.repr) it.ref = c.transform.ref;
    await fadeLines(true);
    c3.requestCameraReset({ snapshot, durationMs: 250 }); c3.commit(true);
    // the three labels of the line: placed in front, sized, and kept clear by the residue labels (fitLabels)
    picked.line = { info: L.info, key: L.prot.key, items, ends: [L.prot.pos, L.lig.pos] };
    publish();
    lastCam = ''; scheduleFit();
  }
  async function clearLine(keepFade) {
    const S = viewer.plugin.state.data, cells = [...S.cells.values()].filter(c => tagged(c, LINE_TAG));
    if (cells.length) { const b = S.build(); for (const c of cells) if (S.cells.has(c.transform.ref)) b.delete(c.transform.ref); await b.commit(); }
    if (!keepFade && picked.alpha0 != null) await fadeLines(false);
    if (picked.line) { picked.line = null; if (!keepFade) publish(); lastCam = ''; scheduleFit(); }
  }
  // clicks in 3D: Mol*'s behaviours handle the focus (an atom: focus; the same again or empty space: clear; a line: clear) and the camera
  function watchClicks(plugin) {
    plugin.behaviors.interaction.click.subscribe(({ current, button, page }) => {
      if (button !== 1) return;                                     // primary button only, like Mol*'s focus
      const loci = current && current.loci;
      if (isLineLoci(loci)) { queueLine(() => showLine(loci)); return; }
      const missed = !loci || loci.kind === 'empty-loci';
      const at = page && missed ? freshPick(plugin, page) : null;     // now: the camera may move after this click
      if (isLineLoci(at)) { queueLine(() => showLine(at)); return; }
      if (at && at.kind === 'element-loci') clickedAtom(plugin, at);
      if (picked.line) queueLine(() => clearLine());
    });
  }
  // Round 9: a click Mol* took for empty space (its pick not ready, see freshPick) that a fresh pick finds on an atom: what Mol*'s
  // click behaviours would have done instead of their clear and reset (the focus on that residue or the pose, the camera on it).
  // Before round 9 the label placement's own GPU picks kept Mol*'s pick buffer fresh, so with slow rendering such misses were rarer.
  function clickedAtom(plugin, loci) {
    try {
      const whole = molstar.lib.structure.StructureElement.Loci.extendToWholeResidues(loci);
      plugin.managers.structure.focus.setFromLoci(whole);
      plugin.managers.camera.focusLoci(whole);
    } catch (e) { window.__pvError = 'click: ' + String(e && e.message || e); }
  }
  // Mol*'s hover picks asynchronously, and until that read completes (a few frames; with software WebGL it times out) the pick
  // buffer is not ready: a click right after a move then picks nothing (Mol* takes it for empty space and resets the camera). For
  // the lines, pick again at once on fresh buffers: setting the pick padding marks them stale (Mol*'s PickHelper.update), so the
  // next synchronous pick renders and reads them, still with the camera of the click.
  function freshPick(plugin, page) {
    const c = plugin.canvas3d, pp = c.props.pickPadding;
    try { c.setProps({ pickPadding: pp + 1 }); c.setProps({ pickPadding: pp }); const p = identify(c, page); return p ? c.getLoci(p.id).loci : null; } catch (e) { return null; }
  }
  window.__pvPicked = () => ({ focus: picked.focus && picked.focus.key, line: picked.line && picked.line.key });   // for the labels (B): keys kept visible

  let viewer = null;
  const ready = molstar.Viewer.create('app', {
    layoutIsExpanded: false, layoutShowControls: false, layoutShowRemoteState: false, layoutShowSequence: false, layoutShowLog: false,
    layoutShowLeftPanel: false, collapseLeftPanel: true, collapseRightPanel: true,
    viewportShowControls: false, viewportShowExpand: false, viewportShowSelectionMode: false, viewportShowAnimation: false,
    viewportShowSettings: false, viewportShowScreenshotControls: false, viewportShowTrajectoryControls: false,
    viewportShowToggleFullscreen: false, viewportShowReset: false,
  }).then(v => {
    viewer = v;
    window.__pvPlugin = v.plugin;   // read-only handle for the acceptance test (this frame is same-origin with the page anyway)
    timings.pluginReady = Math.round(performance.now()); debug();
    try { canvasProps(document.documentElement.dataset.theme === 'dark'); } catch (e) { /* cosmetic */ }
    try { v.plugin.canvas3d && watchView(v.plugin); } catch (e) { window.__pvError = 'view: ' + String(e && e.message || e); }
    try { watchFocus(v.plugin); watchClicks(v.plugin); } catch (e) { window.__pvError = 'focus: ' + String(e && e.message || e); }
    try { v.plugin.managers.lociLabels.addProvider(TIP_LABELS); } catch (e) { window.__pvError = 'tooltips: ' + String(e && e.message || e); }
  }).then(() => post({ type: 'viewer-ready' }));

  // ---------------------------------------------------------------- round 8 (§1): the pocket surfaces shown now
  // showPockets says which pocket surfaces are shown (round 12: each built when first shown, see buildSurfaces). __pvDebug.surfaces and
  // the message {type:'surfaces', shown} (after every drawing and every change) give the pocket ranks whose surface is drawn and not
  // hidden: the page's pocket chips show that. A spec that differs from the last drawing only in showPockets shows or hides surfaces
  // (no new drawing: the camera, a focus and a clicked line stay).
  function surfaceCells() {
    const out = new Map();
    for (const c of viewer.plugin.state.data.cells.values()) {
      const t = (c.transform.tags || []).find(x => x.startsWith('pv-pocket-surface:')); if (t) out.set(Number(t.slice(18)), c);
    }
    return out;
  }
  // Round 9 (round-8 feedback §2): the pocket surfaces are context: no hover or selection outline on them (Mol*'s marker actions off
  // for their representations), so Mol*'s pink hover outline of a residue under a surface is no longer drawn over the labels
  function quietSurfaces() {
    for (const c of surfaceCells().values()) { try { c.obj.data.repr.setState({ markerActions: 0 }); } catch (e) { /* not drawn yet */ } }
  }
  function showSurfaces(show) {
    const S = viewer.plugin.state.data, want = new Set((show || []).map(Number));
    for (const [k, c] of surfaceCells()) { const hide = !want.has(k); if (!!(c.state && c.state.isHidden) !== hide) S.updateCellState(c.transform.ref, { isHidden: hide }); }
  }
  function surfacesNow() {
    const out = [];
    try {
      for (const [k, c] of surfaceCells()) {
        const r = c.obj && c.obj.data && c.obj.data.repr;
        if (!r || (c.state && c.state.isHidden) || (r.state && r.state.visible === false)) continue;
        if (r.renderObjects.some(o => o.values.drawCount && o.values.drawCount.ref.value > 0)) out.push(k);
      }
    } catch (e) { /* no drawing yet */ }
    return out.sort((a, b) => a - b);
  }
  function reportSurfaces() {
    const shown = surfacesNow();
    debug().surfaces = shown;
    post({ type: 'surfaces', shown });
  }
  let drawnKey = '', drawnShow = '', drawnSpec = null;          // round 12: the spec of the last completed drawing
  const specKey = spec => JSON.stringify({ ...spec, showPockets: null, proteinStyle: null });
  async function toggleSurfaces(spec) {
    try {
      const want = (spec.showPockets || []).map(Number);
      await buildSurfaces(want, surfaceAlpha(drawnAtoms && drawnAtoms.lig), k => !want.includes(k));   // a chip switched on before its surface was built
      showSurfaces(spec.showPockets); drawnShow = JSON.stringify(spec.showPockets || []); drawnSpec = { ...drawnSpec, showPockets: spec.showPockets };
      quietSurfaces();
      if (drawnAtoms) labelFront = frontAtoms(spec, drawnAtoms.P, drawnAtoms.lig);     // labels stay in front of the surfaces shown
      lastCam = ''; scheduleFit();
    } catch (e) { window.__pvError = 'surfaces: ' + String(e && e.message || e); }
    reportSurfaces();
    post({ type: 'viewer-drawn' });
  }
  // every heavy atom of the drawing (protein, pose, the other modes when shown): a clicked line's labels keep in front of them
  function drawingAtoms(P, lig, spec) {
    const xyz = [], add = a => { if (a.el !== 'H' && a.el !== 'h') xyz.push(a.x, a.y, a.z); };
    for (const a of P.atoms) add(a);
    if (lig) {
      for (const a of lig.atoms) add(a);
      const L = spec.ligand;
      if (L.allModes && L.format === 'pdbqt' && L.nModels > 1) for (let i = 0; i < L.nModels; i++) if (i !== lig.model) for (const a of ligandAtoms(L, i).atoms) add(a);
    }
    return Float64Array.from(xyz);
  }

  // ---------------------------------------------------------------- round 12: drawing in place
  // What changed since the last completed drawing (the contract's kinds; 'other': the same pose with another residue picked in a list,
  // interaction types hidden in the legend, or the same spec again)
  const keyParts = L => String(L.key || '').split(':');               // `${drug}:${view}:${pocket}:${mode}` (the page's pose key)
  function kindOf(prev, spec) {
    if (!prev || prev.protein !== spec.protein) return 'protein';
    const a = prev.ligand, b = spec.ligand;
    if (!a !== !b || (a && keyParts(a)[0] !== keyParts(b)[0])) return 'drug';
    if (a) {
      const ka = keyParts(a), kb = keyParts(b);
      if (ka[1] !== kb[1] || ka[2] !== kb[2]) return 'pocket';
      if (ka[3] !== kb[3] || a.model !== b.model || a.text !== b.text) return 'mode';
      if (!!a.allModes !== !!b.allModes) return 'modes';
    }
    return prev.theme !== spec.theme ? 'theme' : 'other';
  }
  function record(kind, since, frameAt, drawnAt) {
    const s = { kind, firstFrameMs: frameAt == null ? null : Math.round(frameAt - since), drawnMs: Math.round(drawnAt - since) };
    timings.switches.push(s); if (timings.switches.length > 100) timings.switches.shift();
    timings.switchMs = s.drawnMs;
    if (timings.drawn == null) timings.drawn = Math.round(drawnAt);
    debug();
  }
  // A drawing (returns false if a newer spec came in at one of its checkpoints: the newer one is drawn instead). Stages: the protein
  // (once), the pose's site, the modes, the camera; the first frame (viewer-frame); the first pocket surface shown, the interactions,
  // sticks and labels, the focus; viewer-drawn. A theme change alone recolours in place (the camera stays).
  let pendingSpec = null, pendingSince = 0, carrySince = 0, busy = false, complete = false;   // complete: the last drawing was not stopped
  const superseded = () => !!pendingSpec;
  let stages = [];                                                // the last drawing's stages (ms from its start): where the time goes
  const mark = name => stages.push([name, Math.round(performance.now() - stages.t0)]);
  async function draw(spec, since) {
    picked.drawing = true;                                          // round 7: the focus state is published once, at the end
    drawnKey = ''; stages = []; stages.t0 = performance.now();
    const prev = drawnSpec, kind = kindOf(prev, spec), dark = spec.theme === 'dark', wasComplete = complete;
    let frameAt = null;
    complete = false;
    try {
      try { canvasProps(dark); } catch (e) { /* cosmetic */ }
      document.documentElement.dataset.theme = dark ? 'dark' : 'light';
      const P = await protein(spec.protein); mark('fetched');
      if (!scene.protein || scene.protein.path !== spec.protein || scene.protein.pocketsKey !== JSON.stringify(spec.pockets || [])) await buildProtein(spec, P);
      mark('protein'); if (superseded()) return false;
      const lig = ligandOf(spec.ligand, P), fr = focusResidueOf(spec);
      drawnAtoms = { P, lig };
      // round 8: what a clicked line's labels keep in front of; Mol*'s focus radius of the pair's view (the pose + 8 Å: homeDist)
      sceneAtoms = drawingAtoms(P, lig, spec);
      const heavy = lig ? lig.atoms.filter(a => a.el !== 'H') : [], cen = [0, 1, 2].map(i => heavy.reduce((t, a) => t + [a.x, a.y, a.z][i], 0) / Math.max(1, heavy.length));
      siteRadius = heavy.length ? Math.max(...heavy.map(a => Math.hypot(a.x - cen[0], a.y - cen[1], a.z - cen[2]))) + 8 : 0;
      tipNames.clear();                                             // hover names of the 5 Å site and of a residue picked in a list
      if (lig) for (const r of [...lig.contacts, ...lig.site]) tipNames.set(`${r.chain}:${r.seq}`, `${r.name}${r.seq}`);
      if (fr && !tipNames.has(`${fr.chain}:${+fr.seq}`)) tipNames.set(`${fr.chain}:${+fr.seq}`, `${(P.names[`${fr.chain}_${+fr.seq}`] || ['RES'])[0]}${+fr.seq}`);
      // round 13: Mol*'s focus stays on a new pose of the same protein (the ligand: on the new pose; a residue: on that residue), as in
      // Mol*; a clicked line does not (its atoms change). Round 7 cleared every focus here.
      const keepFocus = kind !== 'protein' && !fr && picked.focus ? picked.focus.key : null;
      await clearPicked();
      const recolourOnly = wasComplete && kind === 'theme' && specKey({ ...prev, theme: '' }) === specKey({ ...spec, theme: '' });
      if (recolourOnly) { await recolor(spec); await applyStyle(spec); }
      else {
        await removeResidues(); mark('residues out');               // this pose's sticks and labels come with the interactions
        await syncSite(spec, lig, hiddenTypes(spec)); mark('site');
        await syncModes(spec, lig); mark('modes');
        await recolor(spec);                                        // after a theme change
        await applyStyle(spec);                                     // round 13: the protein in the style picked (a new protein: cartoon first)
        switchCamera(kind, spec, lig, keepFocus);
      }
      const b = SD().build(); if (setSurfaceAlpha(b, surfaceAlpha(lig))) await b.commit();
      showSurfaces(spec.showPockets);
      await onScreen();                                             // the first frame: cartoon, pose (and the surfaces built before)
      frameAt = performance.now(); mark('frame'); if (timings.firstFrame == null) timings.firstFrame = Math.round(frameAt);
      post({ type: 'viewer-frame' });
      if (superseded()) return false;
      const show = (spec.showPockets || []).map(Number), first = firstShown(spec);
      // PocketVina: every surface shown is built with the drawing, the first one first (was: the others at idle time, but the idle work
      // waits while residue labels are shown, i.e. whenever a pose is drawn; pocket="all" shows every docked pocket's surface)
      if (first) await buildSurfaces([first, ...show.filter(k => k !== first)], surfaceAlpha(lig), k => !show.includes(k));
      showSurfaces(spec.showPockets); quietSurfaces(); reportSurfaces(); mark('surface');
      try { canvasProps(dark); } catch (e) { /* cosmetic */ }
      try { await configureFocus(dark); } catch (e) { window.__pvError = 'focus params: ' + String(e && e.message || e); }
      if (recolourOnly) { if (lastContacts) post(lastContacts); else if (!lig) post({ type: 'contacts', key: '', source: 'none', contacts: [] }); }
      else if (lig) await interactions(spec, lig, P);
      else { await residueSticks(spec, null, null); labelInfo = []; lastContacts = null; post({ type: 'contacts', key: '', source: 'none', contacts: [] }); }
      mark('interactions, sticks, labels');
      try { await focusFromSpec(spec, !recolourOnly); } catch (e) { window.__pvError = 'focus: ' + String(e && e.message || e); }
      if (keepFocus) try { refocus(keepFocus); } catch (e) { window.__pvError = 'refocus: ' + String(e && e.message || e); }
      picked.drawing = false; publish();
      drawnKey = specKey(spec); drawnShow = JSON.stringify(spec.showPockets || []); drawnSpec = spec; complete = true;
      reportSurfaces();
      // drawn = on screen: Mol* renders what came last (and hands new label textures to the GPU) in its next frame, on this main thread
      await onScreen();
      mark('drawn'); debug().stages = stages.slice();
      record(kind, since, frameAt, performance.now());
      post({ type: 'viewer-drawn' });
      return true;
    } catch (e) {
      picked.drawing = false; publish(); drawnSpec = null;
      reportSurfaces();
      post({ type: 'viewer-error', message: String(e && e.message || e).slice(0, 200) });
      return true;
    } finally {
      if (picked.drawing) { picked.drawing = false; publish(); }  // stopped for a newer spec
    }
  }
  // Only the newest spec waiting is drawn (round 12: the page could post three for one click); a drawing stopped for a newer one passes
  // its start on, so a switch is timed from its first 'show'.
  async function pump() {
    busy = true;
    try {
      await ready;
      while (pendingSpec) {
        const spec = pendingSpec, since = carrySince ? Math.min(carrySince, pendingSince) : pendingSince;
        pendingSpec = null;
        if (drawnKey && specKey(spec) === drawnKey) {                // round 8, 13: surfaces or the protein's style only: no drawing
          const surf = JSON.stringify(spec.showPockets || []) !== drawnShow;
          let styled = false;
          try { styled = await applyStyle(spec); } catch (e) { window.__pvError = 'style: ' + String(e && e.message || e); }
          if (styled) drawnSpec = { ...drawnSpec, proteinStyle: spec.proteinStyle };
          if (surf) { await toggleSurfaces(spec); continue; }
          if (styled) { await onScreen(); post({ type: 'viewer-drawn' }); continue; }
        }
        carrySince = (await draw(spec, since)) ? 0 : since;
      }
    } finally { busy = false; }
    scheduleIdle(); scheduleGlyphs();
  }
  // Round 12: the glyphs of every label text drawn into Mol*'s font atlas ahead: residue labels ("LYS 716") and a clicked line's
  // ("LYS716 · NZ", "ligand · aromatic ring", "C12/N3", "4.2 Å"). Mol* draws a glyph the first time a text uses it (~4–9 ms each, ~4×
  // that at 4× CPU) and hands the atlas texture to the GPU again. B's bundle (one atlas for every text, no upload for glyphs a copy
  // already holds) exposes the atlas (molstar.lib.text.getFontAtlas): the glyphs are drawn into it on the CPU only, in idle slices while
  // time is left in them, never during a drawing or with the user at work in the view, and they stop when a 'show' comes; no geometry,
  // no upload. Without that export nothing is drawn ahead.
  const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ·Å./acdgilmnort';   // residue labels' characters first
  const FONT = { fontFamily: 'sans-serif', fontQuality: FONT_QUALITY, fontStyle: 'normal', fontVariant: 'normal', fontWeight: 'normal' };
  let glyphsDone = 0, glyphsQueued = false;
  function scheduleGlyphs(delay = 0) {
    const T = molstar.lib.text;
    if (glyphsQueued || glyphsDone >= GLYPHS.length || !T || typeof T.getFontAtlas !== 'function' || !window.requestIdleCallback) return;
    glyphsQueued = true;
    setTimeout(() => requestIdleCallback(deadline => {
      glyphsQueued = false;
      if (busy || pendingSpec || buttonsDown || performance.now() - Math.max(pointerAt, showAt) < 1000) { scheduleGlyphs(500); return; }
      try {
        const atlas = T.getFontAtlas(FONT);
        while (glyphsDone < GLYPHS.length && !pendingSpec && deadline.timeRemaining() > 10) atlas.get(GLYPHS[glyphsDone++]);
      } catch (e) { glyphsDone = GLYPHS.length; window.__pvError = 'glyphs: ' + String(e && e.message || e); return; }
      debug().glyphs = glyphsDone;
      scheduleGlyphs();
    }, { timeout: 2000 }), delay);
  }
  // Idle work, one job per idle period and never during a drawing (§2.3): the shown surfaces not built yet, the label slots (one per
  // job), then the surfaces of the other pockets. Only while no residue label is shown (the protein without a pose): a job holds Mol*'s
  // state queue, and a label placed after a camera move waits behind it (a surface takes ~0.1 s on a GPU, seconds in software WebGL;
  // a new label uploads its font texture). With a pose, a chip switched on builds its surface at once and a new pocket's surface comes
  // with its drawing. Jobs also wait for a drag, for label updates in flight and for 3 s after pointer activity in the frame (the user
  // at work in the view). The glyphs (scheduleGlyphs) are not a job: they touch no state.
  let idleQueued = false, pointerAt = 0, showAt = 0;
  // the user at work: in this frame, and on the page around it (same origin: a click in a table there runs on this main thread too)
  const busyNow = () => { pointerAt = performance.now(); }, ACTIVITY = ['pointermove', 'pointerdown', 'wheel', 'touchstart', 'keydown', 'scroll'];
  for (const t of ACTIVITY) addEventListener(t, busyNow, { capture: true, passive: true });
  try { if (parent !== window) for (const t of ACTIVITY) parent.addEventListener(t, busyNow, { capture: true, passive: true }); } catch (e) { /* no page around */ }
  // the pointer over the view: a job that adds to the scene would drop Mol*'s hover (its pick is redone only when the pointer moves)
  let pointerIn = false;
  document.documentElement.addEventListener('pointerenter', () => { pointerIn = true; });
  document.documentElement.addEventListener('pointerleave', () => { pointerIn = false; });
  function idleJob() {
    const prot = scene.protein; if (!prot) return null;
    if (labelInfo.length) return null;
    const show = new Set(JSON.parse(drawnShow || '[]').map(Number)), todo = [...prot.pockets.keys()].filter(k => !prot.pockets.get(k).surf).sort((a, b) => a - b);
    const alpha = surfaceAlpha(drawnAtoms && drawnAtoms.lig), surface = k => async () => { await buildSurfaces([k], alpha, x => !show.has(x)); if (show.has(k)) { reportSurfaces(); lastCam = ''; scheduleFit(); } };
    const shown = todo.find(k => show.has(k));
    if (shown) return Object.assign(surface(shown), { shown: true });     // drawn content: not held back by activity on the page
    if (prot.labelSlots.length < LABEL_SLOTS) return Object.assign(prewarmLabels, { quiet: 1500 });
    if (prot.stickSlots.length < STICK_SLOTS) return Object.assign(prewarmSticks, { quiet: 1500 });
    return todo.length ? Object.assign(surface(todo[0]), { quiet: 5000 }) : null;
  }
  function scheduleIdle(delay = 0) {
    if (idleQueued || !idleJob()) return;
    idleQueued = true;
    const run = () => { idleQueued = false; idleWork(); };
    setTimeout(() => { if (window.requestIdleCallback) window.requestIdleCallback(run, { timeout: 250 }); else setTimeout(run, 50); }, delay);
  }
  async function idleWork() {
    if (busy || pendingSpec) return;                              // pump() schedules again when its drawing is done
    // a drag, labels still being placed, or the pointer at work in the view: later (Mol*'s camera also reports a change when the scene's
    // bounds change, so its rest timer is no guide here)
    const job = idleJob(); if (!job) return;
    // optional jobs not with the pointer over the view, and only after a quiet time without a 'show' or any activity: a label slot (its
    // texture goes to the GPU in the next frame: ~10 ms on a GPU) after 1.5 s, a hidden surface (~0.1–0.2 s, not splittable) after 5 s
    const quiet = performance.now() - Math.max(pointerAt, showAt);
    if (buttonsDown || commitBusy || labelInfo.some(L => L.pending) || (!job.shown && (pointerIn || quiet < job.quiet))) { scheduleIdle(job.shown ? 300 : 500); return; }
    busy = true; const t0 = performance.now();
    try { await job(); } catch (e) { window.__pvError = 'idle: ' + String(e && e.message || e); }
    finally { busy = false; }
    const D = debug(); (D.idleJobs = D.idleJobs || []).push([job.name || 'surface', Math.round(performance.now() - t0)]); if (D.idleJobs.length > 60) D.idleJobs.shift();
    if (pendingSpec) pump(); else scheduleIdle();
  }
  // PocketVina: the message handler is receive(m), called by the bridge (BRIDGE.receive) or for a message posted by the parent page
  function receive(m) {
    if (!m || typeof m !== 'object') return;
    if (m.type === 'show' && m.spec && typeof m.spec === 'object') {
      watchBox(); showAt = performance.now(); if (timings.firstShow == null) timings.firstShow = Math.round(showAt);
      if (!pendingSpec) pendingSince = performance.now();
      pendingSpec = m.spec;
      if (!busy) pump();
    }
    else if (m.type === 'reset') ready.then(() => clearPicked()).then(() => viewer.plugin.managers.camera.reset());
    else if (m.type === 'clear-focus') ready.then(() => clearPicked());
    else if (m.type === 'activity') pointerAt = performance.now();             // round 12: the page in use (idle work waits)
    // round 12: the protein of the coming 'show', as soon as this frame runs: fetched and read while Mol* starts (cached by protein())
    else if (m.type === 'prefetch' && typeof m.protein === 'string' && SAFE_STRUCTURE.test(m.protein)) protein(m.protein, false).catch(() => { /* the 'show' reports it */ });
  }
  // PocketVina: the site's listener hands the message to receive()
  window.addEventListener('message', ev => {
    if (ev.origin !== location.origin || ev.source !== parent) return;           // same-origin parent only
    receive(ev.data);
  });
  if (BRIDGE) BRIDGE.receive = receive;                                          // PocketVina: the bridge's messages
})();
