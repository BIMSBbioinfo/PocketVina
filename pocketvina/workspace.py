"""The PocketVina Atlas website's protein page for a results directory: ``Results.show()``, ``PocketVinaGPU.show()``
and ``Selection.show()``.

One self-contained HTML document: the 3D view of the Atlas site (its Mol* viewer, ``pocketvina/viewer/viewer.js``,
in a frame of the page) with the controls around it (pocket chips, legend, protein style, Reset view, Save image)
beside the tabs Top hits, Selected pair and Pockets, rebuilt in plain JavaScript (``pocketvina/viewer/workspace.js``).
Every pose the page can show is in it (the PDBQT output of every docked pocket of each listed ligand, its best pose as
SDF), with the protein structures, the P2Rank pockets and a 2D drawing of each ligand, so it works offline, in a
saved notebook and on Colab. The data (``workspace_data``) and the viewer's code are gzip-compressed and base64-encoded
in the page; the browser unpacks them (DecompressionStream). PoseBusters results are shown only where the results
directory has them (PoseBusters is optional: ``pocketvina bust`` stores them, ``Selection.bust`` runs it in memory).
See ``pocketvina/viewer/README.md`` for the files and the changes to the site's code.

``Atlas.show`` and ``Selection.show`` of a pair of the PocketVina Atlas build the same page from the Atlas
(``pocketvina.atlas``), in the same data format with a few more fields, which a page of results does not have: the
protein's information (``info``), the pockets' pLDDT, the drugs' identifiers (``ids``) and the text above Top hits
(``topNote``: they are the site's Top hits, the drugs with a PoseBusters-valid pose); each listed pair's PoseBusters
result (``pb``, of its highest-affinity pose) is the one the Atlas stores.

In Jupyter, Colab or VSCode the page is shown as an iframe (``srcdoc``) below the cell, with a line saying to run the
cell again where the iframe is not drawn (``NotebookView``); elsewhere it is written to an HTML file whose path is
returned.
"""
import base64
import functools
import gzip
import html
import json
import logging
import numbers
import os
import re
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from . import pdbqt
from .io.results import Results, short_floats
from .selection import Selection, name_part

logger = logging.getLogger(__name__)

VIEWER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viewer")
MOLSTAR_DIR = os.path.join(VIEWER_DIR, "molstar-pv-5.13.1")
# the PocketVina Atlas website; molstar="url" loads its copy of the Mol* bundle (the same file as MOLSTAR_DIR's)
ATLAS_URL = "https://bimsbstatic.mdc-berlin.de/akalin/asarigun/pocketvina_atlas/"
MOLSTAR_URL = ATLAS_URL + "vendor/molstar-pv-5.13.1/molstar-pv.js"
MOLSTAR_SOURCES = ("inline", "url")
THEMES = ("auto", "light", "dark")
MAX_LIGANDS = 100  # ligands per protein in the page (Top hits), best first
HEIGHT = 760  # px of the notebook iframe
# a page above this size (characters) gets a warning: every protein and up to max_ligands ligands each are in it, and
# a browser or a notebook (which also saves it) becomes slow with a page of many proteins
LARGE_PAGE = 30_000_000
# the notebook iframe: its own origin (it cannot reach the notebook page), scripts, downloads (Save image, the pose
# files), which a sandboxed frame may start only with allow-downloads, and links that open in a new tab (an Atlas
# page links to UniProt, AlphaFold DB, ChEMBL and Open Targets), which load there without the sandbox
SANDBOX = "allow-scripts allow-downloads allow-popups allow-popups-to-escape-sandbox"
# the features the page and its 3D frame may use: Mol* asks whether WebXR is there (a frame not allowed it logs a
# permissions policy error), and full screen
FRAME_ALLOW = "xr-spatial-tracking *; fullscreen *"
# what a notebook shows where the iframe is not drawn: a saved notebook opened untrusted (Jupyter then removes the
# iframe; a cell run again is trusted) or a viewer that shows only text
NOT_SHOWN = "Run this cell again to see the PocketVina 3D view."
SVG_SIZE = (224, 160)  # px of the ligand's 2D drawing
# what may not occur inside the page's <script> / <style> elements (the HTML parser would end them there, or read
# "<!--" in a script as the start of an escaped section)
_RAW_TEXT_END = {"js": re.compile(r"</script|<!--", re.IGNORECASE), "css": re.compile(r"</style", re.IGNORECASE)}


def _read(*parts) -> str:
    with open(os.path.join(VIEWER_DIR, *parts), encoding="utf-8") as f:
        return f.read()


def _raw_text(text: str, what: str) -> str:
    """``text`` (a file of pocketvina/viewer: ``what`` ends in .js or .css) checked to be safe inside a <script> /
    <style> element of the page and of the 3D frame's page."""
    m = _RAW_TEXT_END[what.rsplit(".", 1)[-1]].search(text)
    if m:
        raise ValueError(f"{what} contains {m.group(0)!r}, which would end its <script> or <style> element")
    return text


def _pack(obj) -> str:
    """JSON, gzip-compressed (no timestamp: the same data gives the same page), base64. Bytes of the stored files that
    are not UTF-8 (read with surrogateescape, Results.pose_block) become "?"."""
    data = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8", "replace")
    return base64.b64encode(gzip.compress(data, compresslevel=9, mtime=0)).decode("ascii")


def unpack(text: str):
    """The object ``_pack`` packed (for tests and for reading a saved page)."""
    return json.loads(gzip.decompress(base64.b64decode(text)).decode("utf-8"))


@functools.lru_cache(maxsize=None)
def _assets(molstar: str) -> str:
    """The packed code of the 3D frame: Mol* (its text, or its URL with ``molstar="url"``), viewer.js, the bridge and
    their styles. Read and compressed once per process."""
    assets = {
        "molstarCss": _raw_text(_read("molstar-pv-5.13.1", "molstar-pv.css"), "molstar-pv.css"),
        "viewerCss": _raw_text(_read("viewer.css"), "viewer.css"),
        "bridgeJs": _raw_text(_read("bridge.js"), "bridge.js"),
        "viewerJs": _raw_text(_read("viewer.js"), "viewer.js"),
    }
    if molstar == "inline":
        assets["molstarJs"] = _raw_text(_read("molstar-pv-5.13.1", "molstar-pv.js"), "molstar-pv.js")
    else:
        assets["molstarUrl"] = MOLSTAR_URL
    return _pack(assets)


# --- data ----------------------------------------------------------------------------------------------------------

def _num(x, digits: Optional[int] = None):
    """A JSON number (None for NaN / None)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    x = float(x)
    return round(x, digits) if digits is not None else x


def ligand_svg(smiles: Optional[str]) -> Optional[str]:
    """2D drawing of a molecule as SVG text (RDKit), or None if the SMILES does not parse."""
    from rdkit import Chem, rdBase
    from rdkit.Chem.Draw import rdMolDraw2D

    if not smiles:
        return None
    block = rdBase.BlockLogs()  # no parser messages; RDKit's log settings as they were afterwards
    try:
        mol = Chem.MolFromSmiles(smiles)
    finally:
        del block
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(*SVG_SIZE)
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index("<svg"):] if "<svg" in svg else None


def _residues(text) -> List[str]:
    return text.split() if isinstance(text, str) else []


def _pockets(res: Results, pid: str, docked: Sequence[int]) -> List[dict]:
    """Every P2Rank pocket of the protein: rank, score, probability, centre, residues ("A_123"), docked or not."""
    df = res.pockets(pid).sort_values("pocket")
    return [{"rank": int(r.pocket), "score": _num(r.score, 2), "probability": _num(r.probability, 3),
             "center": [_num(r.center_x, 2), _num(r.center_y, 2), _num(r.center_z, 2)],
             "residues": _residues(r.residue_ids), "docked": int(r.pocket) in docked}
            for r in df.itertuples(index=False)]


def _pocket_poses(res: Results, pid: str, i: int, pockets: Sequence[int]) -> List[dict]:
    """The engine output (PDBQT text, all modes) of ligand ``i`` in each pocket with a pose, and its modes' scores."""
    out = []
    for k in pockets:
        text = res.pose_block(pid, i, k)
        modes = pdbqt.parse_modes(text.encode("utf-8", "surrogateescape"))
        if not modes:
            continue
        out.append({"rank": int(k), "pdbqt": text,
                    "modes": [[_num(m.affinity), _num(m.rmsd_lb), _num(m.rmsd_ub)] for m in modes]})
    return out


def _initial(sel: Optional[Selection], pid: str, docked: List[int]) -> dict:
    """The page's first state: the protein alone (Top hits, every docked pocket's surface), or a selected pair."""
    if sel is None:
        return {"protein": pid, "ligand": None, "view": "best", "pocket": None, "mode": None, "allModes": False,
                "show": list(docked), "userShow": False, "tab": "top"}
    poses = sel.poses
    if sel.pocket is None and sel.mode == 1 or sel.pocket == "all" and sel.mode == 1:
        view, k, m = "best", sel.best_pocket, sel.best_mode
    else:
        at_best = poses[poses["pocket"] == sel.best_pocket]
        first = (at_best if len(at_best) else poses).iloc[0]
        k = int(first["pocket"])
        m = (sel.best_mode if k == sel.best_pocket else 1) if sel.mode == "all" else int(first["mode"])
        view = "modes"
    show = sorted(sel.docked_pockets) if sel.pocket == "all" else [k]
    return {"protein": pid, "ligand": int(sel.ligand_idx), "view": view, "pocket": int(k), "mode": int(m),
            "allModes": sel.mode == "all", "show": show, "userShow": sel.pocket == "all", "tab": "pair"}


_ROW_COLUMNS = ["protein_id", "ligand_idx", "vina", "best_pocket", "best_mode", "rank", "protein_rank", "n_proteins"]
_DOCKED = "best_pocket >= 1 AND NOT isnan(vina)"  # a pair with a pose: the rows of best_scores()


def _listed(res: Results, pids: Sequence[str], max_ligands: Optional[int], selection: Optional[Selection]):
    """What the page lists, read with DuckDB from the Parquet parts without loading every pair, pocket run or ligand
    (a run may have millions): ``(rows, n, docked, with_pose, ligands)``.

    ``rows`` (DataFrame, ``_ROW_COLUMNS``): per protein of ``pids`` its ``max_ligands`` best pairs and the selected one,
    in the order of ``best_scores()`` (Vina score, then ligand_idx), with ``rank`` (on the protein) and
    ``protein_rank`` of ``n_proteins`` (the protein among the run's proteins with a pose of that ligand, best first;
    ties by protein id). ``n``: protein -> ligands with a pose in it; ``docked``: protein -> its docked pockets;
    ``with_pose``: (protein, ligand_idx) -> the pockets with a pose of that pair; ``ligands``: ligand_idx ->
    (ligand_id, SMILES)."""
    import duckdb

    from .io.results import _READ_PARQUET, _sql_list

    pids = list(dict.fromkeys(pids))

    def parts(name: str, protein=None) -> Optional[str]:
        files = res._part_files(name, protein)
        return f"read_parquet({_sql_list(files)}, {_READ_PARQUET})" if files else None

    rows, n, docked, with_pose, ligands = pd.DataFrame(columns=_ROW_COLUMNS), {}, {}, {}, {}
    runs, lig = parts("pocket_runs", pids), parts("ligands")
    limit = "" if max_ligands is None else f" LIMIT {int(max_ligands)}"
    con = duckdb.connect()
    try:
        if runs is not None:
            for pid, k in con.execute(f"SELECT DISTINCT protein_id, pocket FROM {runs} ORDER BY protein_id, pocket"
                                      ).fetchall():
                docked.setdefault(pid, []).append(int(k))
        tops = []
        for pid in pids:
            pairs = parts("pairs", pid)
            if pairs is None:
                continue
            docked_pairs = f"FROM {pairs} WHERE {_DOCKED}"
            n[pid] = con.execute(f"SELECT count(*) {docked_pairs}").fetchall()[0][0]
            # the best max_ligands (ORDER BY ... LIMIT keeps only those in memory), then the selected pair beyond them
            top = con.execute(f"SELECT ligand_idx, vina, best_pocket, best_mode {docked_pairs} "
                              f"ORDER BY vina, ligand_idx{limit}").df()
            top["rank"] = np.arange(1, len(top) + 1)
            top.insert(0, "protein_id", pid)
            tops.append(top)
            i = None if selection is None or selection.protein != pid else int(selection.ligand_idx)
            if i is not None and i not in set(top["ligand_idx"]):
                sel = con.execute(f"SELECT ligand_idx, vina, best_pocket, best_mode {docked_pairs} AND ligand_idx = {i}"
                                  ).df()
                if len(sel):
                    vina = float(sel["vina"].iloc[0])  # a float32 score: exact as a float
                    sel["rank"] = 1 + con.execute(f"SELECT count(*) {docked_pairs} AND (vina < ? OR vina = ? AND "
                                                  f"ligand_idx < {i})", [vina, vina]).fetchall()[0][0]
                    sel.insert(0, "protein_id", pid)
                    tops.append(sel)
        tops = [t for t in tops if len(t)]
        if tops:
            rows = pd.concat(tops, ignore_index=True)
        if len(rows):
            listed = rows[["protein_id", "ligand_idx"]]
            con.register("pv_listed", listed)
            # Across proteins: the listed ligands' pairs in every protein of the run
            across = con.execute(f"""
                SELECT a.protein_id, a.ligand_idx, a.protein_rank, a.n_proteins FROM (
                    SELECT protein_id, ligand_idx,
                           row_number() OVER (PARTITION BY ligand_idx ORDER BY vina, protein_id) AS protein_rank,
                           count(*) OVER (PARTITION BY ligand_idx) AS n_proteins
                    FROM {parts("pairs")}
                    WHERE {_DOCKED} AND ligand_idx IN (SELECT ligand_idx FROM pv_listed)) AS a
                JOIN pv_listed AS l ON a.protein_id = l.protein_id AND a.ligand_idx = l.ligand_idx""").df()
            rows = rows.merge(across, how="left", on=["protein_id", "ligand_idx"], sort=False)
            if runs is not None:
                for pid, i, k in con.execute(f"""
                        SELECT r.protein_id, r.ligand_idx, r.pocket FROM {runs} AS r
                        JOIN pv_listed AS l ON r.protein_id = l.protein_id AND r.ligand_idx = l.ligand_idx
                        WHERE r.status = 'ok' AND r.n_modes > 0 AND r."length" > 0
                        ORDER BY r.protein_id, r.ligand_idx, r.pocket""").fetchall():
                    with_pose.setdefault((pid, int(i)), []).append(int(k))
            if lig is not None:
                ligands = {int(i): (lid, smiles, docked, key) for i, lid, smiles, docked, key in con.execute(
                    f"SELECT ligand_idx, ligand_id, smiles_input, smiles_docked, inchikey FROM {lig} "
                    "WHERE ligand_idx IN (SELECT ligand_idx FROM pv_listed)").fetchall()}
    finally:
        con.close()
    return rows[_ROW_COLUMNS], n, docked, with_pose, ligands


def _posebusters(res: Results, pids: Sequence[str]) -> dict:
    """The PoseBusters results the run holds (``bust.bust_results``; none if it was not run, as PoseBusters is
    optional): {(protein, ligand_idx, pocket, mode): {status, valid, passed, total, failed}}."""
    out = {}
    for pid in pids:
        df = res.posebusters(pid, columns=["ligand_idx", "pocket", "mode", "pb_status", "pb_valid", "pb_mask",
                                           "pb_na_mask", "failed_checks"])
        if df.empty:
            continue
        n = len(res.posebusters_checks(pid))
        for r in df.itertuples(index=False):
            answered = n - bin(int(r.pb_na_mask)).count("1")
            out[(pid, int(r.ligand_idx), int(r.pocket), int(r.mode))] = {
                "status": r.pb_status, "valid": None if pd.isna(r.pb_valid) else bool(r.pb_valid),
                "passed": bin(int(r.pb_mask)).count("1"), "total": max(answered, 0),
                "failed": r.failed_checks.split(",") if isinstance(r.failed_checks, str) and r.failed_checks else []}
    return out


def workspace_data(res: Results, proteins: Optional[Sequence[str]] = None, *, max_ligands: Optional[int] = MAX_LIGANDS,
                   selection: Optional[Selection] = None) -> dict:
    """What the page shows (a JSON-ready dict): per protein its PDB text, P2Rank pockets, docked pockets and the
    ``max_ligands`` best ligands (best Vina score first; the selected one too) with their poses in every docked
    pocket (PDBQT, all modes), their best pose as SDF, their Vina score, rank, ligand efficiency and rank among the
    proteins; the ligands' ids, SMILES and 2D drawings; the first state (``_initial``). ``proteins``: ids (None: all
    proteins of the results). Only the listed pairs are read (``_listed``)."""
    from .bust import _protein_pdb

    pids = list(proteins) if proteins is not None else res.proteins()
    if not pids:
        raise ValueError(f"no docked protein in {res.path} yet")
    rows, n_of, docked_of, with_pose, lig = _listed(res, pids, max_ligands, selection)
    checks = _posebusters(res, pids)
    rows["ligand_efficiency"] = res._ligand_efficiency(rows)
    rows["vina"] = short_floats(rows["vina"])
    by_protein = {pid: g for pid, g in rows.groupby("protein_id", sort=False)}
    used = set()
    out = []
    for pid in pids:
        shown = by_protein.get(pid, rows.iloc[:0])
        docked = docked_of.get(pid, [])
        items = []
        for r in shown.itertuples(index=False):
            i = int(r.ligand_idx)
            used.add(i)
            items.append({
                "lig": i, "vina": _num(r.vina), "pocket": int(r.best_pocket), "mode": int(r.best_mode),
                "le": _num(r.ligand_efficiency, 3), "rank": int(r.rank),
                "across": [int(r.protein_rank), int(r.n_proteins)] if r.n_proteins > 1 else None,
                "sdf": res.top_sdf(pid, i), "pockets": _pocket_poses(res, pid, i, with_pose.get((pid, i), [])),
                "pb": checks.get((pid, i, int(r.best_pocket), int(r.best_mode)))})
        out.append({"id": pid, "name": name_part(pid), "key": f"structures/{pid}.pdb", "pdb": _protein_pdb(res, pid),
                    "pockets": _pockets(res, pid, docked), "docked": docked, "n": int(n_of.get(pid, 0)),
                    "rows": items})
    ligands = {}
    for i in sorted(used):
        # smiles_input: as the user gave it; smiles_docked: the fragment docked (e.g. without a salt's counter-ion)
        lid, smiles, docked, inchikey = lig[i]
        ligands[str(i)] = {"id": lid, "smiles": smiles, "docked": docked, "inchikey": inchikey,
                           "svg": ligand_svg(docked or smiles), "name": name_part(lid)}
    first = selection.protein if selection is not None else pids[0]
    docked0 = next(p["docked"] for p in out if p["id"] == first)
    return {"format": 1, "proteins": out, "ligands": ligands, "maxLigands": max_ligands,
            "initial": _initial(selection, first, docked0)}


# --- the page ------------------------------------------------------------------------------------------------------

def workspace_html(data: dict, *, molstar: str = "inline", theme: str = "auto", title: str = "PocketVina") -> str:
    """The self-contained HTML page of ``workspace_data``."""
    if molstar not in MOLSTAR_SOURCES:
        raise ValueError(f"molstar must be one of {MOLSTAR_SOURCES}, got {molstar!r}")
    if theme not in THEMES:
        raise ValueError(f"theme must be one of {THEMES}, got {theme!r}")
    css = _raw_text(_read("workspace.css"), "workspace.css")
    js = _raw_text(_read("workspace.js"), "workspace.js")
    return (
        "<!DOCTYPE html>\n"
        f'<html lang="en" data-theme="light" data-theme-setting="{theme}">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>{html.escape(title)}</title>\n<style>\n{css}</style>\n</head>\n<body>\n"
        '<div id="pv-root" class="pv-ws"><p class="pv-loading">Loading the PocketVina view&hellip;</p></div>\n'
        f'<script type="application/octet-stream" id="pv-assets">{_assets(molstar)}</script>\n'
        f'<script type="application/octet-stream" id="pv-data">{_pack(data)}</script>\n'
        f"<script>\n{js}</script>\n</body>\n</html>\n")


def page_payload(page: str, which: str = "pv-data"):
    """The object packed in a page's ``<script id=...>`` element (``pv-data`` or ``pv-assets``)."""
    m = re.search(rf'<script type="application/octet-stream" id="{re.escape(which)}">([^<]*)</script>', page)
    if m is None:
        raise ValueError(f"no {which} in the page")
    return unpack(m.group(1))


def iframe(page: str, height: int = HEIGHT) -> str:
    """The page as an <iframe srcdoc> element for a notebook output. Its content, ``NOT_SHOWN``, is not drawn by a
    browser, but stays where an HTML sanitizer removes the iframe (Jupyter, for a notebook opened untrusted)."""
    return (f'<iframe srcdoc="{html.escape(page, quote=True)}" sandbox="{SANDBOX}" title="PocketVina" '
            f'style="width: 100%; height: {int(height)}px; border: 0;" allow="{FRAME_ALLOW}">'
            f"{html.escape(NOT_SHOWN)}</iframe>")


class NotebookView:
    """The page as a notebook output (``IPython.display.display``): the iframe (text/html) and, for where it is not
    drawn, the line ``NOT_SHOWN`` (text/plain)."""

    def __init__(self, page: str, height: int = HEIGHT):
        self.page, self.height = page, height

    def _repr_html_(self) -> str:
        return iframe(self.page, self.height)

    def __repr__(self) -> str:
        return NOT_SHOWN


def in_notebook() -> bool:
    """True in a Jupyter kernel (JupyterLab, Notebook, VSCode, Colab), False in a script or a terminal IPython."""
    shell = getattr(sys.modules.get("IPython"), "get_ipython", lambda: None)()
    if shell is None:
        return False
    return "google.colab" in sys.modules or any(c.__name__ == "ZMQInteractiveShell" for c in type(shell).__mro__)


def _write(path: str, text: str) -> str:
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def _check_int(name: str, value, allow_none: bool = False) -> Optional[int]:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a whole number >= 1{' or None' if allow_none else ''}, got {value!r}")
    return int(value)


def _protein_list(res: Results, protein) -> List[str]:
    from .selection import _check_protein

    if protein is None:
        pids = res.proteins()
        if not pids:
            raise ValueError(f"no docked protein in {res.path} yet")
        return pids
    if isinstance(protein, str):
        return [_check_protein(res, protein)]
    pids = [_check_protein(res, p) for p in protein]
    if not pids:
        raise ValueError("protein is an empty list: give a protein id, a list of them, or None for all")
    return list(dict.fromkeys(pids))


def _check_options(molstar: str, theme: str, height, max_ligands):
    """The page options of a show(), checked: (height, max_ligands)."""
    if molstar not in MOLSTAR_SOURCES:
        raise ValueError(f"molstar must be one of {MOLSTAR_SOURCES}, got {molstar!r}")
    if theme not in THEMES:
        raise ValueError(f"theme must be one of {THEMES}, got {theme!r}")
    return _check_int("height", height), _check_int("max_ligands", max_ligands, allow_none=True)


def _output(page: str, name: str, path, height: int) -> Optional[str]:
    """Show a page: below the cell in a notebook (returns None), else written to ``path`` (default: ``<name>.html`` in
    the working folder) whose path is returned. ``path`` given: always written there."""
    if path is None and in_notebook():
        from IPython.display import display

        display(NotebookView(page, height))
        return None
    target = _write(path if path is not None else os.path.join(os.getcwd(), f"{name}.html"), page)
    if path is None:
        print(f"PocketVina view written to {target}: open it in a web browser.")
    return target


def show(res: Results, protein=None, *, selection: Optional[Selection] = None, path=None, molstar: str = "inline",
         theme: str = "auto", height: int = HEIGHT, max_ligands: Optional[int] = MAX_LIGANDS) -> Optional[str]:
    """Build the page for ``protein`` (an id, a list of ids, or None: every protein, with a protein menu) and show
    it: below the cell in a notebook (returns None), else written to ``path`` (default: ``PV_<protein>.html``, or
    ``<selection name>.html``, in the working folder) whose path is returned. ``path`` given: always written there."""
    height, max_ligands = _check_options(molstar, theme, height, max_ligands)
    pids = _protein_list(res, protein)
    data = workspace_data(res, pids, max_ligands=max_ligands, selection=selection)
    for p in data["proteins"]:
        if max_ligands is not None and p["n"] > max_ligands:
            logger.info("%s: the page lists the best %d of %d ligands (max_ligands=%d)", p["id"], max_ligands, p["n"],
                        max_ligands)
    if selection is not None:
        name, title = selection.name, f"PocketVina: {selection.protein} x {selection.ligand}"
    else:
        name = f"PV_{name_part(pids[0])}" if len(pids) == 1 else "PV_results"
        title = f"PocketVina: {', '.join(pids[:3])}{', ...' if len(pids) > 3 else ''}"
    page = workspace_html(data, molstar=molstar, theme=theme, title=title)
    if len(page) > LARGE_PAGE:
        logger.warning("the PocketVina view is %.0f MB (%d protein%s, up to %s ligands each): a browser or notebook "
                       "may be slow with it; show(protein=...) with fewer proteins or a smaller max_ligands make it "
                       "smaller",
                       len(page) / 1e6, len(pids), "" if len(pids) == 1 else "s",
                       "all" if max_ligands is None else max_ligands)
    return _output(page, name, path, height)


__all__ = ["show", "workspace_data", "workspace_html", "iframe", "NotebookView", "in_notebook", "ligand_svg",
           "page_payload", "unpack", "MAX_LIGANDS"]
