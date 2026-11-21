// DOM API contracts for the real inline app script; no browser or Mol* emulation.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const template = fs.readFileSync(path.join(__dirname, '../app/templates/index.html'), 'utf8');
const scripts = [...template.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)]
    .filter((match) => !/\bsrc\s*=/.test(match[1]) && match[2].includes('function updateProteinList()'));
assert.equal(scripts.length, 1, 'Find exactly one inline application script');

class Element {
    constructor(tagName) {
        this.tagName = tagName;
        this.children = [];
        this.listeners = {};
        this.dataset = {};
        this.style = {};
        this.className = '';
        this.text = '';
    }
    set innerHTML(value) { throw new Error('Dynamic markup is forbidden in this DOM contract'); }
    set textContent(value) { this.replaceChildren(); this.text = String(value); }
    get textContent() { return this.text + this.children.map((child) => child.textContent).join(''); }
    appendChild(child) {
        if (child.parentElement) {
            const siblings = child.parentElement.children;
            siblings.splice(siblings.indexOf(child), 1);
        }
        child.parentElement = this;
        this.children.push(child);
        return child;
    }
    prepend(child) { child.parentElement = this; this.children.unshift(child); }
    replaceChildren(...children) {
        this.children.forEach((child) => { child.parentElement = null; });
        this.children = [];
        this.text = '';
        children.forEach((child) => this.appendChild(child));
    }
    addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
    dispatch(type, event = {}) {
        for (const listener of this.listeners[type] || []) listener.call(this, { target: this, ...event });
    }
    setAttribute(name, value) {
        assert.ok(!/^on/i.test(name), 'No executable event attributes');
        this[name] = String(value);
    }
    contains(node) {
        return node === this || this.children.some((child) => child.contains(node));
    }
    closest(selector) {
        assert.equal(selector, 'button[data-protein-url][data-ligand-url]');
        if (this.tagName === 'button' && 'proteinUrl' in this.dataset && 'ligandUrl' in this.dataset) return this;
        return this.parentElement ? this.parentElement.closest(selector) : null;
    }
    querySelectorAll(selector) {
        const tags = selector.split(',').map((tag) => tag.trim());
        return this.children.flatMap((child) => [
            ...(tags.includes(child.tagName) ? [child] : []), ...child.querySelectorAll(selector),
        ]);
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    getElementsByTagName(tag) { return this.querySelectorAll(tag); }
}

function harness() {
    const elements = new Map();
    const document = new Element('document');
    document.createElement = (tag) => new Element(tag);
    document.getElementById = (id) => {
        if (!elements.has(id)) elements.set(id, new Element('div'));
        return elements.get(id);
    };
    document.querySelectorAll = (selector) => {
        assert.equal(selector, '.sort-icon');
        return [...elements.values()].filter((node) => node.className.includes('sort-icon'));
    };
    const context = vm.createContext({
        document, URL, window: { location: { origin: 'https://example.test' } },
        console: { log() {}, warn() {}, error() {} },
    });
    // Register, but never fire, unrelated DOMContentLoaded setup. No external scripts run.
    vm.runInContext(scripts[0][2], context, { filename: 'app-inline.js' });
    return { context, document, run: (source) => vm.runInContext(source, context) };
}

function descendants(node, tag) {
    return node.children.flatMap((child) => [
        ...(child.tagName === tag ? [child] : []), ...descendants(child, tag),
    ]);
}

// Synthetic fixtures remain data and are never logged.
const specialName = `synthetic <img src=x onerror='synthetic()'> & "quoted".pdb`;
const names = [specialName, `punctuation ' " < > & \\ + # ?.pdb`, 'normal.pdb'];

test('shared upload response names and stored filenames pass through the real renderer', () => {
    // Standalone runs use the same synthetic shape; CI also feeds the actual Flask response.
    // A configured but missing response file fails, rather than silently using the fixture.
    const payload = process.env.POCKETVINA_DOM_PAYLOAD
        ? JSON.parse(fs.readFileSync(process.env.POCKETVINA_DOM_PAYLOAD, 'utf8'))
        : {
            proteins: [specialName, 'normal.pdb'],
            protein_info: [{ filename: 'stored_synthetic.pdb' }, { filename: 'stored_normal.pdb' }],
        };
    assert.equal(payload.proteins.length, 2);
    assert.equal(payload.proteins[0], specialName);
    assert.equal(payload.proteins[1], 'normal.pdb');
    const h = harness();
    const calls = [];
    h.context.payload = payload;
    h.context.capture = (...args) => calls.push(args);
    h.run('uploadedProteins = payload.proteins; proteinInfo = payload.protein_info; view3D = capture; updateProteinList();');
    const list = h.document.getElementById('protein-list');
    assert.deepEqual(descendants(list, 'span').map((node) => node.textContent), payload.proteins);
    descendants(list, 'button').forEach((button) => button.dispatch('click'));
    assert.deepEqual(calls, payload.protein_info.map((info) => [info.filename, 'protein']));
});

test('upload list preserves names as text and replaces stale rows', () => {
    const h = harness();
    h.context.names = names;
    h.run('uploadedProteins = names; updateProteinList(); updateProteinList();');
    const list = h.document.getElementById('protein-list');
    assert.equal(list.children.length, names.length);
    assert.deepEqual(descendants(list, 'span').map((node) => node.textContent), names);
    assert.deepEqual(descendants(list, 'small').map((node) => node.textContent), ['#1', '#2', '#3']);
    assert.equal(descendants(list, 'img').length, 0);
    assert.equal(list.children[0].className.includes('protein-item'), true);
});

test('upload click uses the stored filename closure and supports the fallback name', () => {
    const h = harness();
    const calls = [];
    const stored = `stored ' " < > & \\ ?.pdb`;
    h.context.names = names;
    h.context.info = [{ filename: stored }];
    h.context.capture = (...args) => calls.push(args);
    h.run('view3D = capture; uploadedProteins = names; proteinInfo = info; updateProteinList();');
    const buttons = descendants(h.document.getElementById('protein-list'), 'button');
    h.run('proteinInfo = [];');
    buttons.forEach((button) => button.dispatch('click'));
    assert.deepEqual(calls, [[stored, 'protein'], [names[1], 'protein'], [names[2], 'protein']]);
    assert.ok(buttons.every((button) => button.title === 'View in 3D' && !button.onclick));
});

test('empty upload state replaces prior rows with the muted message', () => {
    const h = harness();
    h.context.names = names;
    h.run('uploadedProteins = names; updateProteinList(); uploadedProteins = []; updateProteinList();');
    const list = h.document.getElementById('protein-list');
    assert.equal(list.children.length, 1);
    assert.equal(list.children[0].tagName, 'em');
    assert.equal(list.children[0].className, 'text-muted');
    assert.equal(list.textContent, 'No proteins uploaded yet...');
});

test('viewer error title, filename and message remain exact text', () => {
    const h = harness();
    h.context.values = names;
    h.run('currentFileName = values[1]; showViewerError(values[0], values[2]);');
    const viewer = h.document.getElementById('viewer3d');
    assert.equal(descendants(viewer, 'h5')[0].textContent, names[0]);
    assert.equal(descendants(viewer, 'p')[0].textContent, names[1]);
    assert.equal(descendants(viewer, 'small')[0].textContent, names[2]);
    assert.equal(descendants(viewer, 'img').length, 0);
    h.run('showViewerError(values[2], values[0]);');
    assert.equal(viewer.children.length, 1);
    assert.equal(descendants(viewer, 'small')[0].textContent, names[0]);
});

test('delegated preview clicks preserve encoded local routes and nested paths', () => {
    const h = harness();
    const calls = [];
    h.context.capture = (url) => calls.push(url);
    h.run('openMolstarOverlay = capture;');
    const container = h.document.getElementById('results-table');
    const proteinPath = `/synthetic run/${specialName}`;
    const ligandPath = `/synthetic run/${names[1]}.pdbqt`;
    for (const proteinUrl of [
        `/view_processed_file?${new URLSearchParams({ path: proteinPath })}`,
        `/view_protein/${encodeURIComponent(specialName)}`,
    ]) {
        // Replacement rows still work without registering more listeners.
        container.replaceChildren();
        const button = container.appendChild(new Element('button'));
        button.dataset = {
            proteinUrl,
            ligandUrl: `/view_result_file?${new URLSearchParams({ path: ligandPath })}`,
        };
        const icon = button.appendChild(new Element('i'));
        container.dispatch('click', { target: icon });
        const preview = new URL(calls.at(-1));
        assert.equal(preview.origin, 'https://example.test');
        assert.equal(preview.pathname, '/molstar_preview');
        assert.deepEqual([...preview.searchParams.keys()], ['protein-url', 'ligand-url']);
        assert.equal(preview.searchParams.get('protein-url'), new URL(proteinUrl, preview.origin).href);
        const ligand = new URL(preview.searchParams.get('ligand-url'));
        assert.equal(ligand.pathname, '/view_result_file');
        assert.equal(ligand.searchParams.get('path'), ligandPath);
        assert.equal(ligand.hash, '');
    }
    container.dispatch('click');
    const outside = new Element('button');
    outside.dataset = { proteinUrl: '/view_protein/outside.pdb', ligandUrl: '/view_result_file?path=outside' };
    container.dispatch('click', { target: outside });
    assert.equal(calls.length, 2);
});

test('real table sorting toggles numeric and text order and keeps preview bindings', () => {
    const h = harness();
    const table = h.document.getElementById('results-table');
    const tbody = table.appendChild(new Element('tbody'));
    for (const [score, label] of [[-6, 'protein10'], [-8, 'protein2'], [-7, 'protein1']]) {
        const row = tbody.appendChild(new Element('tr'));
        for (const value of [score, label]) {
            const cell = row.appendChild(new Element('td'));
            cell.textContent = value;
        }
        const cell = row.appendChild(new Element('td'));
        const button = cell.appendChild(new Element('button'));
        button.dataset = { proteinUrl: `/view_protein/${label}.pdb`, ligandUrl: '/view_result_file?path=pose' };
    }
    const order = (column) => tbody.children.map((row) => row.children[column].textContent);
    h.run('sortTable(0);');
    assert.deepEqual(order(0), ['-8', '-7', '-6']);
    assert.equal(h.document.getElementById('sort-icon-0').className, 'fas fa-sort-up sort-icon');
    h.run('sortTable(0);');
    assert.deepEqual(order(0), ['-6', '-7', '-8']);
    h.run('sortTable(1);');
    assert.deepEqual(order(1), ['protein1', 'protein2', 'protein10']);
    assert.equal(tbody.children[0].children[1].style.backgroundColor, 'rgba(255, 193, 7, 0.1)');
    const calls = [];
    h.context.capture = (url) => calls.push(url);
    h.run('openMolstarOverlay = capture;');
    table.dispatch('click', { target: descendants(tbody.children[0], 'button')[0] });
    assert.equal(new URL(calls[0]).searchParams.get('protein-url'), 'https://example.test/view_protein/protein1.pdb');
});
