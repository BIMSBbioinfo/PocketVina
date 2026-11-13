"""pocketvina.workspace: the PocketVina Atlas page of a results directory (Results.show, PocketVinaGPU.show,
Selection.show), without a browser: the data the page carries (every pose of every docked pocket and mode, PDBQT text,
the best pose's SDF, no PoseBusters), the first state a selection sets, the HTML page and its packed parts, the
notebook iframe or the HTML file, and the vendored viewer files (pocketvina/viewer). The page itself is checked in a
headless browser outside the test suite.

Results used: the real run of tests/data imported with io.importer (see test_selection.py), and a copy of it with a
second protein (test_io_results.add_receptor_copy: the same poses, so the same scores on both proteins).
"""
import base64
import gzip
import hashlib
import html
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pocketvina import PocketVinaConfig, PocketVinaGPU, pdbqt, workspace
from pocketvina.bust import _protein_pdb
from pocketvina.io.importer import import_legacy
from pocketvina.io.results import Results, short_floats
from pocketvina.selection import name_part
from test_io_results import PID2, add_receptor_copy, write_pb_part

PID = "AF-Q96LA5-F1-model_v4"
PID3 = "AF-P99999-F1-model_v4"
VIEWER = Path(workspace.VIEWER_DIR)
# the vendored files of the PocketVina Atlas site (commit 40e8d59), unchanged
SITE_SHA256 = {
    "upstream/viewer.js": "dd2b67c5491faa31bb2de7ab3de5aa330038ccd83e84c8c85f784d2639455cfb",
    "viewer.css": "5a44832cc006dd838b0d413319a8146dfd9544655973ba782ae086c80831a11b",
    "molstar-pv-5.13.1/molstar-pv.js": "1c1389a53dda332a08ef5fb93a7fa5c069802f9cf6fe608eb643ebfdfb689ed8",
    "molstar-pv-5.13.1/molstar-pv.css": "90288e1f534f90167919b371dbea85ae2587f3313ea7b754af2ca0b92dda9b7b",
    "molstar-pv-5.13.1/LICENSE": "aa42900c6ea7bfcf22d25a4c369025e01a526c66501fd9193f9a8aecee7a0753",
    "molstar-pv-5.13.1/THIRD-PARTY-LICENSES.txt": "7056a20d21c4d9983bbf98770984a6ed656658d4a14c5e817218e95c34e9790f",
    "molstar-pv-5.13.1/SOURCES.txt": "5ef56005042eb07fd2fc167ec41c26db7bafec8a59940bf5b3f9a2bf713e5902",
}


@pytest.fixture(scope="module")
def results_dir(real_run, tmp_path_factory) -> str:
    return import_legacy(str(real_run.root), str(tmp_path_factory.mktemp("workspace") / "results"),
                         str(real_run.smiles_csv))


@pytest.fixture(scope="module")
def res(results_dir):
    with Results(results_dir) as r:
        yield r


@pytest.fixture(scope="module")
def two(real_run, tmp_path_factory):
    """Two proteins with the same poses; PoseBusters results for the first one."""
    base = tmp_path_factory.mktemp("two")
    legacy = base / "legacy"
    shutil.copytree(real_run.root, legacy)
    add_receptor_copy(legacy, PID2)
    path = import_legacy(str(legacy), str(base / "results"), str(real_run.smiles_csv))
    write_pb_part(path, PID, [(0, "molecule2", 2, 1, "valid", True), (1, "molecule5", 1, 1, "invalid", False)])
    with Results(path) as r:
        yield r


def set_scores(results_path, pid: str, scores: dict) -> None:
    """Rewrite the Vina scores in a protein's pairs part: {ligand_idx: score, or None for no pose}."""
    part = Path(results_path, "tables", "pairs", f"{pid}.parquet")
    table = pq.read_table(part)
    cols = {c: table.column(c).to_numpy().copy() for c in ("vina", "best_pocket", "best_mode")}
    for i, vina in scores.items():
        at = table.column("ligand_idx").to_numpy() == i
        cols["vina"][at] = np.nan if vina is None else vina
        if vina is None:
            cols["best_pocket"][at] = cols["best_mode"][at] = -1
    for c, values in cols.items():
        table = table.set_column(table.schema.get_field_index(c), table.schema.field(c), pa.array(values))
    pq.write_table(table, part)


@pytest.fixture(scope="module")
def three(real_run, tmp_path_factory):
    """Three proteins with other scores (the same poses): molecule2 -6.3 (PID), -7.1 (PID2), -5.2 (PID3); molecule5
    -5.2 (PID), no pose (PID2), -5.2 (PID3, tied with molecule2 there and with PID); molecule10 no pose."""
    base = tmp_path_factory.mktemp("three")
    legacy = base / "legacy"
    shutil.copytree(real_run.root, legacy)
    add_receptor_copy(legacy, PID2)
    add_receptor_copy(legacy, PID3)
    path = import_legacy(str(legacy), str(base / "results"), str(real_run.smiles_csv))
    v5 = pq.read_table(Path(path, "tables", "pairs", f"{PID}.parquet")).to_pandas().set_index("ligand_idx").vina[1]
    set_scores(path, PID2, {0: -7.1, 1: None})
    set_scores(path, PID3, {0: v5, 1: v5})
    with Results(path) as r:
        yield r


def old_listing(res, pids, max_ligands, selection=None) -> dict:
    """What workspace_data listed per protein (n, docked pockets, rows) as it computed it before it used DuckDB, with
    every pair of the run in memory."""
    pairs = res.pairs(columns=["protein_id", "ligand_idx", "vina", "best_pocket", "best_mode", "ligand_efficiency"])
    pairs = pairs[(pairs["best_pocket"] >= 1) & pairs["vina"].notna()].copy()
    pairs["vina"] = short_floats(pairs["vina"])
    pairs = pairs.sort_values(["vina", "protein_id", "ligand_idx"], kind="stable")
    pairs["protein_rank"] = pairs.groupby("ligand_idx").cumcount() + 1
    pairs["n_proteins"] = pairs.groupby("ligand_idx")["protein_id"].transform("size")
    out = {}
    for pid in pids:
        rows = pairs[pairs["protein_id"] == pid].reset_index(drop=True)
        rows["rank"] = np.arange(1, len(rows) + 1)
        shown = rows if max_ligands is None else rows.head(max_ligands)
        if selection is not None and selection.protein == pid and selection.ligand_idx not in set(shown["ligand_idx"]):
            shown = pd.concat([shown, rows[rows["ligand_idx"] == selection.ligand_idx]])
        runs = res.pocket_runs(pid, columns=["ligand_idx", "pocket", "status", "n_modes", "length"])
        ok = runs[(runs["status"] == "ok") & (runs["n_modes"] > 0) & (runs["length"] > 0)]
        with_pose = {i: sorted(int(k) for k in g["pocket"]) for i, g in ok.groupby("ligand_idx")}
        out[pid] = (len(rows), sorted({int(k) for k in runs["pocket"]}), [
            (int(r.ligand_idx), float(r.vina), int(r.best_pocket), int(r.best_mode), round(r.ligand_efficiency, 3),
             int(r.rank), [int(r.protein_rank), int(r.n_proteins)] if r.n_proteins > 1 else None,
             with_pose.get(int(r.ligand_idx), [])) for r in shown.itertuples(index=False)])
    return out


def listing(data) -> dict:
    return {p["id"]: (p["n"], p["docked"], [(r["lig"], r["vina"], r["pocket"], r["mode"], r["le"], r["rank"],
                                            r["across"], [k["rank"] for k in r["pockets"]]) for r in p["rows"]])
            for p in data["proteins"]}


def keys_of(obj) -> set:
    """Every key of every dict in a JSON-like object."""
    if isinstance(obj, dict):
        return set(obj) | {k for v in obj.values() for k in keys_of(v)}
    if isinstance(obj, list):
        return {k for v in obj for k in keys_of(v)}
    return set()


# --- the data ------------------------------------------------------------------------------------------------------

def test_data_holds_every_pose(res):
    """Per docked ligand: its best pose (SDF as stored), and for every pocket with a pose the engine output (PDBQT, all
    modes) with each mode's scores; the protein as PoseBusters reads it (PDB), every P2Rank pocket, the ligands' 2D
    drawings. molecule10 has no pose and is not listed."""
    data = workspace.workspace_data(res)
    assert data["format"] == 1 and data["maxLigands"] == workspace.MAX_LIGANDS
    (prot,) = data["proteins"]
    assert prot["id"] == PID and prot["name"] == name_part(PID) and prot["key"] == f"structures/{PID}.pdb"
    assert prot["pdb"] == _protein_pdb(res, PID) and prot["pdb"].count("ATOM") > 100
    assert prot["docked"] == [1, 2] and prot["n"] == 2
    pockets = res.pockets(PID).sort_values("pocket")
    assert [p["rank"] for p in prot["pockets"]] == list(range(1, 23)) == list(pockets["pocket"])
    assert [p["docked"] for p in prot["pockets"]] == [k in (1, 2) for k in range(1, 23)]
    first = prot["pockets"][0]
    assert first["residues"] == pockets["residue_ids"].iloc[0].split() and first["residues"][:2] == ["A_147", "A_150"]
    assert first["center"] == [round(float(pockets[c].iloc[0]), 2) for c in ("center_x", "center_y", "center_z")]
    assert first["score"] == round(float(pockets["score"].iloc[0]), 2)

    pairs = res.pairs(PID).set_index("ligand_id")
    assert [data["ligands"][str(r["lig"])]["id"] for r in prot["rows"]] == ["molecule2", "molecule5"]
    for rank, row in enumerate(prot["rows"], 1):
        lid = data["ligands"][str(row["lig"])]["id"]
        p = pairs.loc[lid]
        assert (row["rank"], row["pocket"], row["mode"]) == (rank, int(p.best_pocket), int(p.best_mode))
        assert row["vina"] == short_floats([p.vina])[0] and row["le"] == round(float(p.ligand_efficiency), 3)
        assert row["across"] is None  # one protein
        assert row["sdf"] == res.top_sdf(PID, lid)
        assert [k["rank"] for k in row["pockets"]] == [1, 2]
        for k in row["pockets"]:
            assert k["pdbqt"] == res.pose_block(PID, lid, k["rank"])
            modes = pdbqt.parse_modes(k["pdbqt"].encode())
            assert len(k["modes"]) == len(modes) == 9
            assert k["modes"] == [[m.affinity, m.rmsd_lb, m.rmsd_ub] for m in modes]
    for lig in data["ligands"].values():
        assert lig["svg"].startswith("<svg") and lig["svg"].rstrip().endswith("</svg>") and lig["smiles"]
        assert lig["name"] == name_part(lig["id"])
    assert data["initial"] == {"protein": PID, "ligand": None, "view": "best", "pocket": None, "mode": None,
                               "allModes": False, "show": [1, 2], "userShow": False, "tab": "top"}
    json.dumps(data, allow_nan=False)  # plain JSON


def test_data_of_several_proteins_with_posebusters_only_where_checked(two):
    """Every protein of the run; each ligand's rank among the proteins (Across proteins); max_ligands best ligands per
    protein, and the selected one also beyond them; the PoseBusters result of a pair's best pose only where the run
    has one (here one protein was checked; PoseBusters is optional), None elsewhere; the ligands' identifiers."""
    data = workspace.workspace_data(two)
    assert [p["id"] for p in data["proteins"]] == [PID2, PID]  # Results.proteins(): sorted
    across = {(p["id"], data["ligands"][str(r["lig"])]["id"]): r["across"] for p in data["proteins"] for r in p["rows"]}
    # the same scores on both proteins: ties by protein id
    assert across == {(PID2, "molecule2"): [1, 2], (PID, "molecule2"): [2, 2], (PID2, "molecule5"): [1, 2],
                      (PID, "molecule5"): [2, 2]}
    stored = two.posebusters(columns=["protein_id", "ligand_id", "pocket", "mode", "pb_status", "failed_checks"])
    checked = {(r.protein_id, r.ligand_id, int(r.pocket), int(r.mode)): r for r in stored.itertuples(index=False)}
    assert checked  # the fixture holds checks of one protein
    for p in data["proteins"]:
        for r in p["rows"]:
            lid = data["ligands"][str(r["lig"])]["id"]
            ref = checked.get((p["id"], lid, r["pocket"], r["mode"]))
            if ref is None:
                assert r["pb"] is None
            else:
                pb = r["pb"]
                assert set(pb) == {"status", "valid", "passed", "total", "failed"} and pb["status"] == ref.pb_status
                assert 0 <= pb["passed"] <= pb["total"] and len(pb["failed"]) == (len(ref.failed_checks.split(","))
                                                                                  if ref.failed_checks else 0)
    assert any(r["pb"] for p in data["proteins"] for r in p["rows"])
    for lig in data["ligands"].values():
        assert set(lig) == {"id", "smiles", "docked", "inchikey", "svg", "name"} and lig["inchikey"] and lig["docked"]

    one = workspace.workspace_data(two, [PID], max_ligands=1)
    (prot,) = one["proteins"]
    assert [r["rank"] for r in prot["rows"]] == [1] and prot["n"] == 2 and one["maxLigands"] == 1
    sel = two.select(PID, "molecule5")
    extra = workspace.workspace_data(two, [PID], max_ligands=1, selection=sel)
    assert [(r["lig"], r["rank"]) for r in extra["proteins"][0]["rows"]] == [(0, 1), (1, 2)]
    assert set(extra["ligands"]) == {"0", "1"}


def test_data_has_the_sas_points_where_the_results_have_them(two, monkeypatch):
    """A protein's SAS points (Results.sas) are in its data as "sas", without the protein's id; a protein without
    them has no "sas" (these imported results have none; the docking run's: test_pipeline)."""
    assert all("sas" not in p for p in workspace.workspace_data(two)["proteins"])
    points = {"protein": PID, "pockets": {"1": [[1.25, -2.5, 3.75, 0.5]]},
              "other_pockets": {"3": [[1.2, 2.5, 3.8, 0.3]]}, "rest": [[0.1, 0.2, 0.3, 0.01]]}
    monkeypatch.setattr(Results, "sas", lambda self, protein: points if protein == PID else None)
    data = workspace.workspace_data(two)
    by_id = {p["id"]: p for p in data["proteins"]}
    assert by_id[PID]["sas"] == {k: points[k] for k in ("pockets", "other_pockets", "rest")}
    assert "sas" not in by_id[PID2]
    assert workspace.page_payload(workspace.workspace_html(data)) == data  # packed with the rest


def test_listed_pairs_as_before_without_reading_every_pair(three, monkeypatch):
    """The pairs each protein lists, with the same numbers (rank on the protein, Across proteins, ligands with a pose,
    Vina score, pocket, mode, ligand efficiency, pockets with a pose) as when workspace_data read every pair of the run
    into memory; now only the listed pairs are read (DuckDB over the Parquet parts: a run may have millions)."""
    pids = three.proteins()
    assert pids == [PID2, PID3, PID]
    sel = three.select(PID3, "molecule5")
    cases = [(pids, 100, None), (pids, None, None), (pids, 1, None), ([PID3], 1, sel), ([PID, PID2], 1, None),
             ([PID3, PID3], 2, None)]
    want = [old_listing(three, *case) for case in cases]
    # ties on a protein (by ligand) and across proteins (by protein id), pairs without a pose, a selected pair beyond
    # max_ligands
    assert {(pid, r[0]): (r[5], r[6]) for pid, (_, _, rows) in want[0].items() for r in rows} == {
        (PID2, 0): (1, [1, 3]), (PID, 0): (1, [2, 3]), (PID3, 0): (1, [3, 3]), (PID3, 1): (2, [1, 2]),
        (PID, 1): (2, [2, 2])}
    assert [(n, len(rows)) for n, _, rows in want[0].values()] == [(1, 1), (2, 2), (2, 2)]
    assert [r[0] for r in want[3][PID3][2]] == [0, 1]
    lig = three.ligands(columns=["ligand_idx", "ligand_id", "smiles_input"]).set_index("ligand_idx")
    table = Results.table

    def table_read(self, name, protein=None, columns=None):
        # only the heavy atoms (ligand efficiency) and ids of every ligand, read once (cached by Results)
        assert name not in ("pairs", "pocket_runs") and (name != "ligands" or
                                                         set(columns) <= {"ligand_idx", "ligand_id", "n_heavy"}), name
        return table(self, name, protein, columns)

    monkeypatch.setattr(Results, "table", table_read)
    for (proteins, max_ligands, selection), expected in zip(cases, want):
        data = workspace.workspace_data(three, proteins, max_ligands=max_ligands, selection=selection)
        assert listing(data) == expected
        used = sorted({r["lig"] for p in data["proteins"] for r in p["rows"]})
        assert {k: (v["id"], v["smiles"]) for k, v in data["ligands"].items()} == {
            str(i): (lig.ligand_id[i], lig.smiles_input[i]) for i in used}
    with pytest.raises(AssertionError, match="pairs"):
        old_listing(three, pids, 1)


def test_listed_pairs_of_a_protein_without_poses(three, tmp_path):
    """A protein none of whose pairs has a pose: no rows, 0 ligands, its docked pockets; the others as before."""
    shutil.copytree(three.path, tmp_path / "results")
    set_scores(tmp_path / "results", PID3, {0: None, 1: None})
    with Results(tmp_path / "results") as r:
        want = old_listing(r, [PID3, PID], 1)
        assert want[PID3] == (0, [1, 2], [])
        assert listing(workspace.workspace_data(r, [PID3, PID], max_ligands=1)) == want


@pytest.mark.parametrize("pocket, mode, initial", [
    (None, 1, dict(view="best", pocket=2, mode=1, allModes=False, show=[2], userShow=False)),
    (None, 4, dict(view="modes", pocket=2, mode=4, allModes=False, show=[2], userShow=False)),
    (1, 3, dict(view="modes", pocket=1, mode=3, allModes=False, show=[1], userShow=False)),
    (1, "all", dict(view="modes", pocket=1, mode=1, allModes=True, show=[1], userShow=False)),
    ("all", 1, dict(view="best", pocket=2, mode=1, allModes=False, show=[1, 2], userShow=True)),
    ("all", 5, dict(view="modes", pocket=2, mode=5, allModes=False, show=[1, 2], userShow=True)),
    ("all", "all", dict(view="modes", pocket=2, mode=1, allModes=True, show=[1, 2], userShow=True)),
])
def test_first_state_of_a_selection(res, pocket, mode, initial):
    """select() sets only the page's first state: the pose (pocket, mode; "all" pockets: the best one), all modes,
    the surfaces (pocket="all": every docked pocket's), the Selected pair tab."""
    data = workspace.workspace_data(res, selection=res.select(PID, "molecule2", pocket=pocket, mode=mode))
    assert data["initial"] == dict(protein=PID, ligand=0, tab="pair", **initial)


def test_ligand_svg():
    from rdkit import rdBase

    before = rdBase.LogStatus()
    svg = workspace.ligand_svg("c1ccccc1O")
    assert svg.startswith("<svg") and "<?xml" not in svg and "width='224px'" in svg
    assert workspace.ligand_svg("not a smiles") is None and workspace.ligand_svg(None) is None
    assert rdBase.LogStatus() == before  # RDKit's log settings as they were (not its debug and info messages on)


def test_pack_bytes_that_are_not_utf8():
    text = b"REMARK x\xe9y\n".decode("utf-8", "surrogateescape")  # as Results.pose_block reads such a file
    assert workspace.unpack(workspace._pack({"pdbqt": text})) == {"pdbqt": "REMARK x?y\n"}


# --- the page ------------------------------------------------------------------------------------------------------

def test_page_carries_data_and_viewer(res):
    data = workspace.workspace_data(res)
    page = workspace.workspace_html(data, title="PocketVina <test>")
    assert page.startswith("<!DOCTYPE html>") and "<title>PocketVina &lt;test&gt;</title>" in page
    assert 'data-theme-setting="auto"' in page
    assert workspace.page_payload(page) == json.loads(json.dumps(data))
    assets = workspace.page_payload(page, "pv-assets")
    files = {"molstarJs": "molstar-pv-5.13.1/molstar-pv.js", "molstarCss": "molstar-pv-5.13.1/molstar-pv.css",
             "viewerJs": "viewer.js", "viewerCss": "viewer.css", "bridgeJs": "bridge.js"}
    assert set(assets) == set(files)
    for key, name in files.items():
        assert assets[key] == (VIEWER / name).read_text(encoding="utf-8"), name
    assert (VIEWER / "workspace.js").read_text(encoding="utf-8") in page
    assert (VIEWER / "workspace.css").read_text(encoding="utf-8") in page
    # the page needs nothing from elsewhere: no src/href to a file or URL outside it
    assert not re.search(r'(src|href)="(?!data:)', page.split('<script type="application/octet-stream"')[0])
    # packed: gzip of the JSON, base64
    m = re.search(r'id="pv-data">([^<]*)<', page)
    assert json.loads(gzip.decompress(base64.b64decode(m.group(1)))) == json.loads(json.dumps(data))
    assert len(page) < 2_000_000  # the Mol* bundle (2.5 MB) packed: about 1 MB

    small = workspace.workspace_html(data, molstar="url", theme="dark")
    assert 'data-theme-setting="dark"' in small and len(small) < 400_000
    assert workspace.page_payload(small, "pv-assets")["molstarUrl"] == workspace.MOLSTAR_URL
    assert "molstarJs" not in workspace.page_payload(small, "pv-assets")
    with pytest.raises(ValueError, match="molstar must be one of"):
        workspace.workspace_html(data, molstar="cdn")
    with pytest.raises(ValueError, match="theme must be one of"):
        workspace.workspace_html(data, theme="blue")


def test_page_script_names_and_no_posebusters():
    """The page's own code: files and images named with PV_ (lib/keys.ts), the pose named so in Mol*'s hover box,
    and no Atlas wording in the code (ChEMBL, Open Targets: an Atlas page brings those texts in its data). PoseBusters
    shows only what the data holds."""
    js = (VIEWER / "workspace.js").read_text(encoding="utf-8")
    for part in ("const pairStem = () => `PV_${S.prot.name}_${ligOf(S.row).name}`;",
                 "const poseName = () => `${pairStem()}_pk${S.pk}_m${pad2(S.mode)}`;",
                 "`${pairStem()}_top.sdf`", "`${pairStem()}_pk${S.pk}_modes.pdbqt`", "`${poseName()}.png`",
                 "key, name };"):
        assert part in js, part
    code = js.split("*/", 1)[1]  # after the header comment
    assert not re.search(r"chembl|open targets", code, re.IGNORECASE)
    assert "if (!pb) return '';" in code  # no PoseBusters row (or placeholder) without a result
    viewer = (VIEWER / "viewer.js").read_text(encoding="utf-8")
    assert "const own = pose ? spec.ligand && spec.ligand.name : spec.name;" in viewer
    # the longest pose name (ids of 100 characters, pocket 127, mode 99) is accepted as the name in Mol*'s hover box
    longest = f"PV_{'x' * 100}_{'y' * 100}_pk127_m99"
    assert int(re.search(r"/\^\[A-Za-z0-9\._-\]\{1,(\d+)\}\$/\.test\(own\)", viewer).group(1)) >= len(longest)
    # the ligand's 2D drawing (SVG from the user's SMILES) is an image, never HTML of the page
    assert 'src="data:image/svg+xml;charset=utf-8,${encodeURIComponent(L.svg)}"' in js
    assert not re.search(r"\$\{L\.svg(?! \?)", js)


def test_page_theme_from_the_first_paint(res, tmp_path):
    """The theme asked for is the page's from its first paint (data-theme), before its script runs; "auto" (the
    default) starts light and the script then takes the system's theme (data-theme-setting). Nothing else in the page
    depends on the theme."""
    js = (VIEWER / "workspace.js").read_text(encoding="utf-8")
    assert "const setting = document.documentElement.dataset.themeSetting;" in js
    pages = {}
    for theme, first in (("auto", "light"), ("light", "light"), ("dark", "dark")):
        page = Path(res.show(path=tmp_path / f"{theme}.html", theme=theme)).read_text(encoding="utf-8")
        assert page.startswith(f'<!DOCTYPE html>\n<html lang="en" data-theme="{first}" data-theme-setting="{theme}">\n')
        pages[theme] = page.split("\n", 2)[2]
    assert pages["auto"] == pages["light"] == pages["dark"]
    default = Path(res.show(path=tmp_path / "default.html")).read_text(encoding="utf-8")
    assert default == (tmp_path / "auto.html").read_text(encoding="utf-8")


def test_download_hint_names_no_notebook():
    """After a download the page says what to do if nothing was downloaded, in words that fit any frame that blocks
    downloads (a notebook, a website's sandboxed frame) and any page (with or without a pair in Python): use the
    Python lines of the Download tab. The page's text never says "notebook"."""
    js = (VIEWER / "workspace.js").read_text(encoding="utf-8")
    assert ("toast(`Saved ${name}. If nothing was downloaded, downloads are blocked here: use the Python lines in the "
            "Download tab.`);") in js
    assert "Downloads are blocked here" not in js  # 0.3.1's text, also shown where the download worked
    assert "pair.export() writes" not in js  # the hint named pair.export() also on pages with no pair in Python
    assert "export_protein" not in js
    assert "Nothing downloaded? Right-click the image &gt; <i>Save image as…</i>" in js
    text = re.sub(r"(?m)(^|\s)//.*$", "", js.split("*/", 1)[1])  # without the comments
    assert not re.search(r"notebook", text, re.IGNORECASE)


def test_download_tab_python_lines(res):
    """The Download tab ends with the Python lines that save its files where downloads are blocked: the shown pair
    selected by the ids the page shows, each a Python string ("<ligand>" before a ligand is picked), and saved with
    the protein. A results page: pocketvina.select with the protein and ligand ids, which select its pairs; an Atlas
    page: atlas.select with the gene and the drug's name (test_atlas_view). The page's copy button (as beside the
    ligand's ids) copies both lines as shown."""
    js = (VIEWER / "workspace.js").read_text(encoding="utf-8")
    for part in ("const pyStr = s => JSON.stringify(String(s));",  # a JSON string is a valid Python string
                 "const pyProtein = P => (/[;,/]/.test(protName(P)) ? P.id : protName(P));",
                 "const pythonLines = (P, r) => `pair = ${P.info ? 'atlas' : 'pocketvina'}.select("
                 "protein=${pyStr(pyProtein(P))}, ligand=${pyStr(r ? ligOf(r).id : '<ligand>')})`",
                 r"+ '\npair.export(protein=True)   # the poses and the protein';",
                 "py = pythonLines(P, r);",
                 '<div class="pv-py"><p class="small mut">In Python:${copyButton(\'the Python lines\', py)}</p>'
                 '<pre class="mono" data-python tabindex="0" role="region" aria-label="Python lines">${esc(py)}</pre>'
                 '</div>',
                 # the one copy button and copy of the page: data-v is the text copied, "Copied." the toast
                 'const copyButton = (label, v) => `<button type="button" class="pv-copy" data-act="copy" '
                 'data-v="${esc(v)}" title="Copy ${esc(label)}" aria-label="Copy ${esc(label)}">${ICON.copy}'
                 '</button>`;',
                 "else if (a === 'copy') copy(v);",
                 "try { await navigator.clipboard.writeText(text); toast('Copied.'); return; }",
                 # with the focus in the lines the arrow keys scroll them (elsewhere they step the modes)
                 "const skip = e.key === 'Escape' ? 'input,select,textarea' : 'input,select,textarea,pre';"):
        assert part in js, part
    assert "const protName = P => (P.info && P.info.title) || P.id;" in js  # results pages: the protein id
    css = (VIEWER / "workspace.css").read_text(encoding="utf-8")  # one visual line each; a long one scrolls in its box
    assert ".pv-py pre { margin: 2px 0 0; font-size: 12px; white-space: pre; overflow-x: auto; }" in css
    data = workspace.workspace_data(res, max_ligands=None)
    assert all("info" not in P for P in data["proteins"])
    pairs = [(P["id"], data["ligands"][str(r["lig"])]["id"], r["lig"]) for P in data["proteins"] for r in P["rows"]]
    assert len(pairs) == 2
    for protein, ligand, i in pairs:
        sel = res.select(protein=protein, ligand=ligand)
        assert (sel.protein, sel.ligand_idx) == (protein, i)


def test_python_lines_take_the_focus():
    """A click or tap into the Download tab's Python lines focuses their box (tabindex 0, a scroll region named
    "Python lines"), so there the arrow keys scroll it and do not step the modes. In 0.3.2 a click left the focus on
    the page, and the arrows stepped the modes and switched to Selected pair. Escape still closes things from the box.
    The box keeps the focus when the tab is drawn again (once the 3D view has drawn), but only while the page has the
    focus: in Firefox, focus() took it from a notebook cell outside the page's frame. The panel leaves room for the
    box's focus ring (the page's own :focus-visible outline) at its left edge and below the lines."""
    js = (VIEWER / "workspace.js").read_text(encoding="utf-8")
    assert '<pre class="mono" data-python tabindex="0" role="region" aria-label="Python lines">${esc(py)}</pre>' in js
    assert js.count("<pre") == 1
    for part in ("const skip = e.key === 'Escape' ? 'input,select,textarea' : 'input,select,textarea,pre';",
                 "if (e.target && e.target.closest && e.target.closest(skip)) return;",
                 "if ((e.key === 'ArrowLeft' || e.key === 'ArrowRight') && S.row) { e.preventDefault(); "
                 "step(e.key === 'ArrowLeft' ? -1 : 1); }",
                 "const a = document.hasFocus() ? document.activeElement : null, "
                 "py = a && a.matches('[data-python]') ? a : null, x = py ? py.scrollLeft : 0;",
                 "const now = py && !py.isConnected ? body.querySelector('[data-python]') : null;",
                 "if (now) { now.focus({ preventScroll: true }); now.scrollLeft = x; }"):
        assert part in js, part
    assert "closest('input,select,textarea,pre')) return;" not in js  # Escape did nothing from the box
    css = (VIEWER / "workspace.css").read_text(encoding="utf-8")
    assert ":focus-visible { outline: 2px solid var(--acc); outline-offset: 2px; }" in css
    assert ".pv-body { flex: 1; min-height: 0; overflow-y: auto; padding: 0 4px; margin-left: -4px; }" in css
    assert ".pv-py { margin-top: 12px; padding-bottom: 4px; }" in css


def test_tab_row_wraps_in_a_narrow_panel():
    """The 4 tabs need about 325 px; where the side panel is narrower (a 320 px window, or 801 to about 1020 px) the
    row wraps instead of making the page wider than the window. Where it fits it is one row, as before."""
    css = (VIEWER / "workspace.css").read_text(encoding="utf-8")
    assert (".pv-tabs { display: flex; flex-wrap: wrap; gap: 0 16px; border-bottom: 1px solid var(--line); "
            "margin-bottom: 12px; }") in css
    js = (VIEWER / "workspace.js").read_text(encoding="utf-8")
    assert '<div class="pv-tabs" role="tablist" aria-label="Panels" id="pv-tabs"></div>' in js
    assert ("[['top', 'Top hits'], ['pair', 'Selected pair'], ['pockets', 'Pockets'], ['download', 'Download']]"
            in js)
    assert ('`<button type="button" role="tab" data-act="tab" data-v="${k}" aria-selected="${S.tab === k}">${l}'
            "</button>`") in js


def test_toast_hides_on_a_tap():
    """A click or tap on the toast hides it at once (on a phone it covers the Download tab's Python lines) and stops
    its timer; untouched it hides after 6 s. It is a role="status" element and never takes the focus."""
    js = (VIEWER / "workspace.js").read_text(encoding="utf-8")
    for part in ("toastTimer = setTimeout(hideToast, 6000);",
                 "function hideToast() { clearTimeout(toastTimer); "
                 "document.getElementById('pv-toast').hidden = true; }",
                 # the click listener the saved image's close shares: a click outside the image still closes it
                 "root.addEventListener('click', e => { if (e.target.id === 'pv-shot') closeShot(); "
                 "else if (e.target.id === 'pv-toast') hideToast(); });",
                 '<div class="pv-toast" id="pv-toast" role="status" hidden></div>'):
        assert part in js, part
    css = (VIEWER / "workspace.css").read_text(encoding="utf-8")
    assert re.search(r"\.pv-toast \{[^}]*cursor: pointer; \}", css)


def test_sas_toggle_draws_p2ranks_points():
    """The SAS toggle (as on the PocketVina Atlas site): first in the Pockets row, off at first, aria-pressed; greyed
    out without points (older results, pockets from a CSV), and it says so when they cannot be read or drawn. Switched
    on, the protein's points (its "sas" field) are read once into flat lists and go to the 3D view with each spec; the
    legend shows their colour scale. viewer.js draws them as the site does: two structures (the points of every pocket,
    the rest), spheres of 0.48 and 0.24 A, coloured by P2Rank's score from green (0) to red (0.7), no hover or pick,
    and the camera stays where it is."""
    js = (VIEWER / "workspace.js").read_text(encoding="utf-8")
    for part in ("sasOn: false,",
                 "const sasStatus = P => (!sasSize(P) ? 'none' : sasErr.has(P.id) ? 'error' : 'ready');",
                 "const sasTip = sas === 'none' ? 'No SAS points for this protein' : sas === 'error' ? "
                 "'The SAS points could not be loaded'",
                 ": `${sasOn ? 'Hide' : 'Show'} P2Rank's solvent accessible surface (SAS) points in 3D`;",
                 # first in the row, before the chips
                 "`<span class=\"pv-label\">Pockets</span><button type=\"button\" class=\"tag\" data-act=\"sas\" "
                 "aria-pressed=\"${sasOn}\" data-sas=\"${sas}\"`",
                 "+ `${sas === 'none' ? ' disabled' : ''} title=\"${esc(sasTip)}\">SAS</button>`",
                 "else if (a === 'sas') toggleSas();",
                 # in the spec only the protein's key (lastSpec stays small); the points go with each message
                 "sas: sasShown() ? P.key : null,",
                 "toFrame({ type: 'show', spec: s.sas ? { ...s, sas: { protein: s.sas, ...sasPoints(S.prot) } } "
                 ": s });",
                 "sasLists.set(P.id, { pockets, others: flat(...Object.values(s.other_pockets || {})), "
                 "rest: flat(s.rest) });",
                 "case 'sas': { const P = D.proteins.find(p => p.key === m.protein); if (P && m.error) { "
                 "sasErr.add(P.id);",
                 # the legend: the colour scale, no numbers
                 "const SAS_SWATCH = ['#00FF00', '#55AA00', '#AA5500', '#FF0000'];",
                 "SAS points<span class=\"mut\">· P2Rank score</span></li>"):
        assert part in js, part
    assert js.index('data-act="sas"') < js.index('<div class="pv-chips-list"')
    viewer = (VIEWER / "viewer.js").read_text(encoding="utf-8")
    for part in ("const SAS_MAX = 200000, SAS_RADIUS = { pockets: 0.48, rest: 0.24 }, "
                 "SAS_SCALE = [0xff0000, 0x00ff00], SAS_SCORE_MAX = 0.7;",
                 "typeParams: { visuals: ['element-sphere'], sizeFactor: 1, ignoreHydrogens: false, "
                 "...(part === 'pockets' ? { alpha: 0.9 } : {}) },",
                 "color: 'uncertainty', colorParams: { domain: [0, SAS_SCORE_MAX], list: { kind: 'interpolate', "
                 "colors: SAS_SCALE } },",
                 "size: 'uniform', sizeParams: { value: SAS_RADIUS[part] } }, `pv-sas-points:${part}`);",
                 "c.obj.data.repr.setState({ markerActions: 0, pickable: false });",
                 # a point a PDB line cannot hold is left out; every pocket rank, the other pockets' points with them
                 "if (Math.abs(a[i]) < 1000 && Math.abs(a[i + 1]) < 1000 && Math.abs(a[i + 2]) < 1000)",
                 "r >= 1 && r <= MAX_POCKETS && okNumbers(v)) keep(pockets, v); }",
                 "if (okNumbers(x.others || [])) keep(pockets, x.others || []);",
                 "if (total / 4 > SAS_MAX) throw new Error(",
                 # the camera stays: no reset while the points come or go, then its depth range follows the scene
                 "try { await rebuildSas(spec, key, cells); if (c3) c3.commit(true); } finally {",
                 "const specKey = spec => JSON.stringify({ ...spec, showPockets: null, proteinStyle: null, sas: null, "
                 "hideLigand: null });"):
        assert part in viewer, part
    sync = viewer[viewer.index("async function syncSas(spec)"):viewer.index("async function rebuildSas(")]
    assert "c3.setProps({ camera: { manualReset: true } });" in sync and "radiusMax: c3.boundingSphere.radius" in sync
    css = (VIEWER / "workspace.css").read_text(encoding="utf-8")
    assert ".pv-sw-sas i { display: inline-block; width: 6px; height: 6px; }" in css


def test_bar_shows_or_hides_the_protein_and_the_ligand():
    """Under the legend, at the very bottom of the 3D view, one bar in the legend's style replaces the protein's
    style buttons: Protein (an eye), the protein's style as a small menu that opens upwards (Cartoon, Surface, Sticks,
    a tick on the current one; its label dimmed while the protein is hidden, and a pick shows the protein again), and
    with a pose Ligand (an eye; its interaction lines, residue sticks and labels go with it). Both are on at first, and
    a hidden ligand stays hidden on another mode, pocket or ligand. The menu works from the keyboard (Escape, the up and
    down arrows; the left and right arrows do not step the modes while it is open) and closes when the focus leaves."""
    js = (VIEWER / "workspace.js").read_text(encoding="utf-8")
    for part in ("protHidden: false, ligHidden: false,",
                 "proteinStyle: S.protHidden ? 'none' : S.pstyle,",
                 "hideLigand: !!(S.ligHidden && S.row) };",
                 '`<div class="pv-bar" role="group" aria-label="Shown in 3D" data-shown-bar>`',
                 '`<button type="button" data-act="protein-shown" aria-pressed="${prot}" title="${prot ? \'Hide\' : '
                 '\'Show\'} the protein in 3D">${prot ? ICON.eye : ICON.eyeOff}Protein</button>`',
                 '`<details class="pv-menu" data-style-menu><summary aria-label="Protein style: ${cur[1]}" '
                 'title="The protein\'s style in 3D" data-style-current="${cur[0]}"${prot ? \'\' : \' data-dim\'}>`',
                 '<div class="pv-menu-list" role="menu" aria-label="Protein style">',
                 '`<button type="button" role="menuitemradio" aria-checked="${S.pstyle === v}" data-act="style" '
                 'data-v="${v}" title="${esc(tip)}">${ICON.check}${label}</button>`',
                 "+ (S.row ? `<span class=\"pv-bar-sep\" aria-hidden=\"true\"></span><button type=\"button\" "
                 "data-act=\"ligand-shown\" aria-pressed=\"${lig}\"`",
                 "+ ` title=\"${lig ? 'Hide' : 'Show'} the ligand in 3D (with its interaction lines, residue sticks "
                 "and labels)\">${lig ? ICON.eye : ICON.eyeOff}Ligand</button>` : '')",
                 "S.pstyle = v; S.protHidden = false; update();",
                 "else if (a === 'protein-shown') { S.protHidden = !S.protHidden; update(); }",
                 "else if (a === 'ligand-shown') { S.ligHidden = !S.ligHidden; update(); }",
                 # the menu's keys only while it is open; closed, the keys go on to the page
                 "if (!menu || !menu.open) return;",
                 "if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closeMenu(menu, true); }",
                 "else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') e.stopPropagation();",
                 "items[i < 0 ? (down ? 0 : n - 1) : (i + (down ? 1 : n - 1)) % n].focus();",
                 "root.addEventListener('focusout', e => { const m = openMenu(); if (m && e.relatedTarget && "
                 "!m.contains(e.relatedTarget)) m.open = false; });",
                 "window.addEventListener('blur', () => { const m = openMenu(); if (m) m.open = false; });",
                 # the legend shows what is drawn; legend and bar each in a slot; the bar keeps the focus when drawn
                 "const ligShown = !!S.row && !S.ligHidden, lg = ligShown && S.contacts",
                 "+ (!S.protHidden || ligShown ? `<li data-carbon=\"protein\">",
                 '<div class="pv-slot" id="pv-legend-slot"></div><div class="pv-slot" id="pv-bar-slot"></div>',
                 "setKeep($('pv-bar-slot'), bar);",
                 "const a = document.hasFocus() ? document.activeElement : null, v = a && a.dataset ? a.dataset.v "
                 ": null;"):
        assert part in js, part
    assert "pv-styles" not in js and 'aria-label="Protein style in 3D"' not in js  # one style control, not two
    assert js.index('<div class="pv-slot" id="pv-legend-slot">') < js.index('id="pv-bar-slot"')  # Tab: legend, then bar
    css = (VIEWER / "workspace.css").read_text(encoding="utf-8")
    assert "pv-styles" not in css
    for part in ("display: flex; flex-direction: column;\n  justify-content: flex-end;",  # the bar at the very bottom
                 ".pv-slot { display: contents; }",
                 ".pv-menu > summary[data-dim] { opacity: .6; }",  # the label only: the open list stays opaque
                 ".pv-menu-list { position: absolute; left: 0; bottom: calc(100% + 4px);",  # opens upwards
                 ".pv-menu-list button[aria-checked=false] svg { visibility: hidden; }",
                 ".pv-bar > button[aria-pressed=false] { color: var(--mut); }",
                 "@container (max-width: 680px) {"):
        assert part in css, part
    viewer = (VIEWER / "viewer.js").read_text(encoding="utf-8")
    for part in ("const STYLES = ['cartoon', 'surface', 'sticks', 'none'];",
                 "if (style === 'none') { if (!hidden) S.updateCellState(prot.polyRepr, { isHidden: true }); }",
                 "const LIGAND_TAG = t => t === 'pv-lig-repr' || t === 'pv-interactions' || t === 'pv-line' || "
                 "t.startsWith('pv-mode-repr:');",
                 "ligandHidden = !!spec.hideLigand;",  # a drawing (a new mode, pocket or ligand) keeps it hidden
                 "if (ligandHidden) { hidePoseCells(true); await removeResidues(); }",
                 "{ tag: 'pv-interactions', initialState: { isHidden: ligandHidden } });",
                 # made hidden (no frame shows them); kept ones hidden or shown again; also with a theme change
                 "'pv-lig-repr', ligandHidden ? { isHidden: true } : null);",
                 "`pv-mode-repr:${i}`,\n          ligandHidden ? { isHidden: true } : null);",
                 "hidePoseCells(ligandHidden);",
                 "if (recolourOnly) { await recolor(spec); await applyStyle(spec); await setLigandHidden(spec); }",
                 # 0.4.1: Ligand off in one spec with another change drops Mol*'s focus, as Ligand off alone does
                 "const hiding = !!spec.hideLigand && !ligandHidden;",
                 "const keepFocus = kind !== 'protein' && !fr && picked.focus && !hiding ? picked.focus.key : null;"):
        assert part in viewer, part
    draw = viewer[viewer.index("async function draw(spec, since)"):]
    # `hiding` reads the last drawing's ligandHidden, before either branch of this drawing changes it
    order = ("const hiding = !!spec.hideLigand && !ligandHidden;",
             "if (recolourOnly) { await recolor(spec); await applyStyle(spec); await setLigandHidden(spec); }",
             "ligandHidden = !!spec.hideLigand;", "await syncSite(spec, lig, hiddenTypes(spec));",
             "hidePoseCells(ligandHidden);")
    assert [draw.index(part) for part in order] == sorted(draw.index(part) for part in order)


def test_raw_text_check():
    assert workspace._raw_text("a < b && c > d", "x.js") == "a < b && c > d"
    for text, what in (("x = '</script>'", "x.js"), ("<!-- y", "x.js"), ("a{} </style>", "x.css")):
        with pytest.raises(ValueError, match="would end its"):
            workspace._raw_text(text, what)


def test_iframe():
    tag = workspace.iframe('<p class="x">a & "b"</p>', height=500)
    assert tag.startswith('<iframe srcdoc="&lt;p class=&quot;x&quot;&gt;a &amp; &quot;b&quot;&lt;/p&gt;"')
    assert f'sandbox="{workspace.SANDBOX}"' in tag and "height: 500px" in tag
    assert 'allow="xr-spatial-tracking *; fullscreen *"' in tag
    assert {"allow-scripts", "allow-downloads"} <= set(workspace.SANDBOX.split())
    assert "allow-same-origin" not in workspace.SANDBOX


def test_notebook_view_has_a_line_for_where_the_iframe_is_not_drawn():
    """The notebook output: the iframe (text/html) first; a saved notebook opened untrusted (Jupyter's HTML sanitizer
    removes iframes and keeps their text) or a viewer of text only shows one line saying to run the cell again."""
    view = workspace.NotebookView('<p class="x">page</p>', height=500)
    tag = view._repr_html_()
    assert tag == workspace.iframe('<p class="x">page</p>', 500) and tag.startswith("<iframe srcdoc=")
    assert repr(view) == workspace.NOT_SHOWN == "Run this cell again to see the PocketVina 3D view."
    # the line is the iframe's content (a browser does not draw it), so it stays when the iframe element is removed
    assert tag.endswith(f">{workspace.NOT_SHOWN}</iframe>")
    assert re.sub(r"<iframe [^>]*>|</iframe>", "", tag) == workspace.NOT_SHOWN
    formatters = pytest.importorskip("IPython.core.formatters")
    data, _ = formatters.DisplayFormatter().format(view)
    assert data == {"text/html": tag, "text/plain": workspace.NOT_SHOWN}


def test_in_notebook(monkeypatch):
    assert not workspace.in_notebook()

    def shell_of(name):
        cls = type(name, (), {})
        sub = type("ColabShell", (cls,), {})
        return types.SimpleNamespace(get_ipython=lambda: sub())

    monkeypatch.setitem(sys.modules, "IPython", shell_of("ZMQInteractiveShell"))
    assert workspace.in_notebook()
    monkeypatch.setitem(sys.modules, "IPython", shell_of("TerminalInteractiveShell"))
    assert not workspace.in_notebook()
    monkeypatch.setitem(sys.modules, "IPython", types.SimpleNamespace(get_ipython=lambda: None))
    assert not workspace.in_notebook()


# --- show ----------------------------------------------------------------------------------------------------------

def test_show_writes_a_file_outside_a_notebook(res, results_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    path = res.show()
    assert path == str(tmp_path / f"PV_{PID}.html") and os.listdir(tmp_path) == [f"PV_{PID}.html"]
    assert "open it in a web browser" in capsys.readouterr().out
    page = Path(path).read_text(encoding="utf-8")
    assert workspace.page_payload(page) == json.loads(json.dumps(workspace.workspace_data(res)))
    sel = res.select(PID, "molecule5", pocket=2, mode=3)
    path = sel.show()
    assert path == str(tmp_path / f"PV_{PID}_molecule5_pk2_m03.html")
    assert workspace.page_payload(Path(path).read_text(encoding="utf-8"))["initial"]["pocket"] == 2
    # a path: written there (also in a notebook)
    monkeypatch.setattr(workspace, "in_notebook", lambda: True)
    assert res.show(path=tmp_path / "sub" / "x.html", molstar="url") == str(tmp_path / "sub" / "x.html")
    assert len((tmp_path / "sub" / "x.html").read_text(encoding="utf-8")) < 400_000


def test_selection_show_lists_as_results_show(res, tmp_path):
    """pair.show() lists as many ligands as Results.show: MAX_LIGANDS by default (a pair of the PocketVina Atlas has
    its own default), the number given, or all with None; the pair itself always."""
    sel = res.select(PID, "molecule5")
    for kwargs, want in (({}, workspace.MAX_LIGANDS), (dict(max_ligands=1), 1), (dict(max_ligands=None), None)):
        data = workspace.page_payload(Path(sel.show(path=tmp_path / "x.html", **kwargs)).read_text(encoding="utf-8"))
        assert data["maxLigands"] == want
        assert data == json.loads(json.dumps(workspace.workspace_data(res, [PID], max_ligands=want, selection=sel)))
        assert sel.ligand_idx in [r["lig"] for r in data["proteins"][0]["rows"]]


def test_show_warns_about_a_large_page(res, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(workspace, "LARGE_PAGE", 1000)
    res.show(path=tmp_path / "big.html", max_ligands=50)
    assert "(1 protein, up to 50 ligands each): a browser or notebook may be slow with it" in caplog.text


def test_show_in_a_notebook(res, results_dir, tmp_path, monkeypatch):
    """In a notebook: displayed as an iframe (srcdoc, sandboxed, downloads allowed), nothing written, None returned;
    also from PocketVinaGPU (a new session on the results folder) and from a Selection."""
    shown = []
    display = types.ModuleType("IPython.display")
    display.display = shown.append
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.setattr(workspace, "in_notebook", lambda: True)
    monkeypatch.chdir(tmp_path)
    config = PocketVinaConfig(protein_paths=str(tmp_path / "p.txt"), smiles_file=str(tmp_path / "l.csv"),
                              output_dir=results_dir, output_format="atlas")
    pv = PocketVinaGPU(config)
    assert pv.show(height=640) is None
    assert pv.select(PID, "molecule2", pocket="all").show(theme="dark") is None
    assert os.listdir(tmp_path) == []
    assert all(isinstance(v, workspace.NotebookView) for v in shown)
    tag, tag2 = (v._repr_html_() for v in shown)
    assert tag.startswith('<iframe srcdoc="') and "height: 640px" in tag
    page = html.unescape(re.match(r'<iframe srcdoc="([^"]*)"', tag).group(1))
    assert workspace.page_payload(page)["initial"]["tab"] == "top"
    page2 = html.unescape(re.match(r'<iframe srcdoc="([^"]*)"', tag2).group(1))
    assert '<html lang="en" data-theme="dark" data-theme-setting="dark">' in page2  # dark from the first paint
    assert workspace.page_payload(page2)["initial"]["show"] == [1, 2]


def test_show_arguments(two, tmp_path):
    assert workspace.page_payload(Path(two.show(PID2, path=tmp_path / "a.html")).read_text())["proteins"][0]["id"] \
        == PID2
    both = workspace.page_payload(Path(two.show([PID, PID2], path=tmp_path / "b.html")).read_text())
    assert [p["id"] for p in both["proteins"]] == [PID, PID2] and both["initial"]["protein"] == PID
    with pytest.raises(KeyError, match="no protein 'nope'"):
        two.show("nope", path=tmp_path / "c.html")
    for kw, match in ((dict(protein=[]), "empty list"), (dict(max_ligands=0), "max_ligands must be"),
                      (dict(height=True), "height must be"), (dict(molstar="x"), "molstar must be"),
                      (dict(theme="x"), "theme must be")):
        with pytest.raises(ValueError, match=match):
            two.show(path=tmp_path / "d.html", **kw)
    assert not (tmp_path / "c.html").exists() and not (tmp_path / "d.html").exists()
    every = workspace.page_payload(Path(two.show(path=tmp_path / "e.html", max_ligands=None)).read_text())
    assert every["maxLigands"] is None and all(len(p["rows"]) == 2 for p in every["proteins"])


# --- the vendored viewer -------------------------------------------------------------------------------------------

def test_vendored_files_are_the_sites():
    for name, digest in SITE_SHA256.items():
        assert hashlib.sha256((VIEWER / name).read_bytes()).hexdigest() == digest, name
    # the bundle's SRI hash on the site (SRI.txt there)
    sri = base64.b64encode(hashlib.sha384((VIEWER / "molstar-pv-5.13.1" / "molstar-pv.js").read_bytes()).digest())
    assert sri.decode() == "vmB7fQOzkJu//DyRxjJc0ZAfHokDD5OKrLOp7km/BLrm48PCYUgJr/sCr3Nmph+i"


def test_viewer_changes_are_marked():
    """Each change to the site's viewer.js is marked "PocketVina:" (README.md lists them): every changed block of
    lines has such a comment in it or right above it."""
    import difflib

    old = (VIEWER / "upstream" / "viewer.js").read_text(encoding="utf-8").splitlines()
    new = (VIEWER / "viewer.js").read_text(encoding="utf-8").splitlines()
    blocks = [op for op in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes() if op[0] != "equal"]
    assert 8 <= len(blocks) <= 30
    for tag, i1, i2, j1, j2 in blocks:
        near = new[max(0, j1 - 3):j2 + 1]
        assert any("PocketVina:" in line for line in near), (tag, old[i1:i2], new[j1:j2])
    readme = (VIEWER / "README.md").read_text(encoding="utf-8")
    assert "40e8d59" in readme and "diff -u upstream/viewer.js viewer.js" in readme


def test_page_text_says_highest_affinity():
    """The page's text has the Atlas website's words: "highest-affinity", never "best" before pose, mode, score, Vina,
    pocket, drug or a number (comments and names such as view 'best' are not text)."""
    js = (VIEWER / "workspace.js").read_text(encoding="utf-8")
    text = re.sub(r"(?m)(^|\s)//.*$", "", js)
    # spaces only: "best" before a dot or a quote is code (best.vina, view 'best'), not text
    assert not re.search(r"best\s+(poses?|modes?|scores?|vina|pockets?|drugs?|\d|\$\{)", text, re.IGNORECASE)
    assert ">Highest-affinity poses</th>" in text and "the pair's highest-affinity pose" in text
