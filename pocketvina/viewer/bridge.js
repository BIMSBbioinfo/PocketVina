'use strict';
/* PocketVina: the 3D frame's side of the notebook bridge (README.md). Runs first in the frame that workspace.js builds (an iframe
 * srcdoc document: this file, the Mol* bundle, viewer.js), in place of the site's viewer.html and structure-fetch.js.
 * window.__pvBridge, read by viewer.js:
 *   post(m)          a message of viewer.js to the page
 *   structure(path)  the protein's PDB text (a Promise), which the page sends ahead of its 'show'
 *   receive(m)       set by viewer.js: takes the page's messages ('show', 'reset', 'clear-focus')
 * Messages go both ways by postMessage with the target origin '*' and are taken from the parent window only: a srcdoc frame of
 * a sandboxed page has an opaque origin, with which the site's same-origin check cannot work. Messages of the page:
 * {pv: 'page', m}, of this frame: {pv: 'frame', m}. The page's own messages here: {type: 'structure', path, text} and
 * {type: 'image', id, scale, transparent}, answered by {type: 'image', id, url, width, height, scale} or {type: 'image', id, error}:
 * a PNG of exactly the view (camera, surfaces, labels, theme) at `scale` × the frame's size (Mol*'s screenshot helper, its image
 * pass), smaller if the GPU cannot render that size. */
(() => {
  const texts = new Map(), waiting = new Map(), early = [];
  let receive = null;
  const send = m => parent.postMessage({ pv: 'frame', m }, '*');
  const bridge = window.__pvBridge = {
    post: send,
    structure(path) {
      if (texts.has(path)) return Promise.resolve(texts.get(path));
      return new Promise(resolve => {
        if (!waiting.has(path)) waiting.set(path, []);
        waiting.get(path).push(resolve);
        send({ type: 'need-structure', path });
      });
    },
    // viewer.js sets it as it ends; the page's messages that came before are handed over then
    get receive() { return receive; },
    set receive(f) { receive = f; while (receive && early.length) receive(early.shift()); },
  };
  async function image(opt) {
    const plugin = window.__pvPlugin, helper = plugin && plugin.helpers.viewportScreenshot, c3 = plugin && plugin.canvas3d;
    if (!helper || !c3) throw new Error('the 3D view is not ready');
    const r = c3.webgl.gl.canvas.getBoundingClientRect(), scale = Math.max(1, Number(opt.scale) || 2);
    const max = Math.floor(Math.min(c3.webgl.maxRenderbufferSize, c3.webgl.maxTextureSize) / 2) || 4096;
    const fit = Math.min(1, max / Math.max(r.width * scale, r.height * scale));
    const width = Math.max(1, Math.round(r.width * scale * fit)), height = Math.max(1, Math.round(r.height * scale * fit));
    const before = helper.values;
    helper.behaviors.values.next({ ...before, resolution: { name: 'custom', params: { width, height } }, transparent: !!opt.transparent });
    try {
      const url = await helper.getImageDataUri();
      if (typeof url !== 'string' || !url.startsWith('data:image/png')) throw new Error('no image');
      return { url, width, height, scale: Math.round(scale * fit * 100) / 100 };
    } finally {
      helper.behaviors.values.next(before);
    }
  }
  window.addEventListener('message', ev => {
    if (ev.source !== parent || !ev.data || ev.data.pv !== 'page') return;
    const m = ev.data.m;
    if (!m || typeof m !== 'object') return;
    if (m.type === 'structure' && typeof m.path === 'string' && typeof m.text === 'string') {
      texts.clear(); texts.set(m.path, m.text);                     // one protein at a time, as viewer.js keeps it
      for (const resolve of waiting.get(m.path) || []) resolve(m.text);
      waiting.delete(m.path);
    } else if (m.type === 'image') {
      image(m).then(x => send({ type: 'image', id: m.id, ...x }), e => send({ type: 'image', id: m.id, error: String(e && e.message || e) }));
    } else if (receive) receive(m);
    else early.push(m);
  });
  // after the Mol* bundle and viewer.js have run
  document.addEventListener('DOMContentLoaded', () => send({ type: 'bridge-ready' }));
})();
