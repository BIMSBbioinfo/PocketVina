"""``Atlas.select`` and ``Atlas.show`` on the fixture release ``tests/data/atlas/v1.1``: the same ``Selection`` as
``Results.select`` and the same page as ``Results.show`` (pocketvina.workspace), with the fields only an Atlas page
has (protein information, pocket pLDDT, identifier rows, stored PoseBusters results). The page is also built over
HTTP (the ``atlas_server`` fixture of conftest); the page itself is checked in a headless browser outside the suite.

Only fixture identifiers are written here (UniProt accessions and PVL IDs); everything else comes from the data at
run time. Expected poses come from ``posefile.Reader`` on the fixture's pose files.
"""
import gzip
import html
import io
import os
import re
import sys
import types
from pathlib import Path

import pytest
from rdkit import Chem

from pocketvina import pdbqt, workspace
from pocketvina.atlas import MAX_LIGANDS, NON_DRUG, TOP_HITS, Atlas, _across, _pb
from pocketvina.io import posefile
from pocketvina.selection import Selection, pose_name

ABL1 = "P00519"
EGFR = "P00533"
PTGS2 = "P35354"
PAIR = (ABL1, "PVL00129")     # pf 0, stored PoseBusters result
REDOCK = (ABL1, "PVL02203")   # pf 1 (poses_redock), no stored PoseBusters result
NO_TOP = (EGFR, "PVL02131")   # no stored top pose (top_len 0)
OTHER = (ABL1, "PVL00958")    # an entry not docked as the drug; the last row of ABL1
NO_ROW = (PTGS2, "PVL00257")  # no row of this pair


@pytest.fixture(scope="module")
def atlas(atlas_dir) -> Atlas:
    return Atlas(atlas_dir)


def stored(atlas, atlas_dir, protein, ligand):
    """(row, {pocket: engine output bytes}, top pose bytes or None) of a pair, read with posefile.Reader."""
    pair = atlas._pair(protein, ligand)
    with posefile.Reader(atlas_dir / pair.ref["file"]) as reader:
        blocks = {k: reader.modes(pair.index, k) for k in range(1, reader.K + 1)}
        top = reader.top(pair.index)
    return pair.row, {k: b for k, b in blocks.items() if b}, top


def payload(path):
    return workspace.page_payload(Path(path).read_text(encoding="utf-8"))


def top_hits(atlas, protein):
    """(the protein's rows best first, its rows of drugs, its Top hits): as the Atlas site's protein page lists them,
    the drugs whose highest-affinity pose is PoseBusters-valid (pb 0)."""
    raw = atlas._protein_json(protein)
    rows = [dict(zip(raw["columns"], r)) for r in raw["rows"]]
    drugs = [r for r in rows if not r["flags"] & NON_DRUG]
    return rows, drugs, [r for r in drugs if r["pb"] == 0]


# --- select --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pocket, mode", [(None, 1), (2, 1), ("all", 1), (None, 3), (None, "all"), (2, "all"),
                                          ("all", 3), ("all", "all")])
@pytest.mark.parametrize("which", [PAIR, REDOCK, NO_TOP])
def test_select_poses(atlas, atlas_dir, which, pocket, mode):
    """The poses, scores and models picked by the pocket / mode rules of Results.select, from the release's pose
    files (pf 0 and pf 1), with the pair's best pose by default."""
    row, blocks, top = stored(atlas, atlas_dir, *which)
    sel = atlas.select(*which, pocket=pocket, mode=mode)
    assert isinstance(sel, Selection) and sel.results_dir is None
    assert (sel.protein, sel.ligand, sel.ligand_idx) == (which[0], which[1], row["cidx"])
    assert (sel.best_pocket, sel.best_mode, sel.vina) == (row["pocket"], row["mode"], row["vina"])
    assert (sel.pocket, sel.mode) == (pocket, mode) and sel.docked_pockets == sorted(blocks)
    assert sel.smiles and sel.smiles_docked
    records = {k: pdbqt.parse_modes(b) for k, b in blocks.items()}
    pockets = [row["pocket"]] if pocket is None else sorted(blocks) if pocket == "all" else [pocket]
    want = []
    for k in pockets:
        if mode == "all":
            want += [(k, r.mode) for r in records[k]]
        elif mode == 1:
            want.append((k, row["mode"]) if k == row["pocket"] else (k, min(records[k], key=lambda r: (r.affinity,
                                                                                                    r.mode)).mode))
        elif mode <= len(records[k]):
            want.append((k, mode))
    assert list(zip(sel.poses["pocket"], sel.poses["mode"])) == want
    for p in sel._poses:
        record = records[p.pocket][p.mode - 1]
        assert p.model == pdbqt.model(blocks[p.pocket], p.mode) and p.vina == pytest.approx(record.affinity)
    has_best = (row["pocket"], row["mode"]) in want
    assert sel._top_sdf == (top.decode() if top and has_best else None)


def test_select_names(atlas):
    cases = {(None, 1): "top", ("all", 1): "pocketall_top", ("all", "all"): "pocketall_modes",
             ("all", 3): "pocketall_m03", (None, "all"): f"pk{atlas._pair(*PAIR).row['pocket']}_modes",
             (2, 3): "pk2_m03"}
    for (pocket, mode), end in cases.items():
        assert atlas.select(*PAIR, pocket=pocket, mode=mode).name == f"{pose_name(*PAIR)}_{end}"
    # any identifier of the protein and of the drug: the selection is named by UniProt accession and PVL ID
    drug = atlas.drug(PAIR[1])
    assert atlas.select("ABL1", drug.catalogue_numbers[0]).name == f"PV_{ABL1}_{PAIR[1]}_top"


def test_select_errors(atlas, monkeypatch):
    for kwargs, err, match in [
            (dict(pocket=9), ValueError, rf"pocket 9 has no poses of ligand '{PAIR[1]}' in protein '{ABL1}'.*"
                                         r"pockets with poses: 1, 2, 3, 4, 5"),
            (dict(pocket=2, mode=99), ValueError, "mode 99 is out of range: pocket 2 has modes 1 to"),
            (dict(pocket="all", mode=99), ValueError, "no pocket has that many modes"),
            (dict(pocket=0), ValueError, "pocket must be a pocket number"),
            (dict(mode="best"), ValueError, "mode must be a mode number")]:
        with pytest.raises(err, match=match):
            atlas.select(*PAIR, **kwargs)
    with pytest.raises(ValueError, match=f"no result for {NO_ROW[1]}"):
        atlas.select(*NO_ROW)
    with pytest.raises(KeyError, match="no protein"):
        atlas.select("NOSUCHGENE", PAIR[1])
    with pytest.raises(KeyError, match="no drug"):
        atlas.select(ABL1, "nosuchdrug")
    # a row whose pose file record has no output in any pocket
    monkeypatch.setattr("pocketvina.atlas.pose_records", lambda source, records, map=map: [([None] * 5, None)])
    with pytest.raises(ValueError, match=f"ligand '{PAIR[1]}' has no docked pose in protein '{ABL1}'"):
        atlas.select(*PAIR)


def test_select_summary(atlas):
    sel = atlas.select(*PAIR, pocket="all", mode="all")
    first = repr(sel).splitlines()[0]
    assert first.startswith(f"Selection {pose_name(*PAIR)}: protein {ABL1}, ligand {PAIR[1]}, every mode in each "
                            "pocket")
    assert sel.smiles in repr(sel)
    page = sel._repr_html_()
    assert pose_name(*PAIR) in page and "pair.show(), pair.bust(), pair.export()" in page
    # the website's wording: "highest-affinity", never "best" before pose, mode, Vina, ... in what a reader sees
    for pocket in (None, sel.best_pocket, "all"):
        for mode in (1, 2, "all"):
            shown = atlas.select(*PAIR, pocket=pocket, mode=mode)
            for text in (repr(shown), shown._repr_html_()):
                assert "lowest Vina score (kcal/mol)" in text
                assert not re.search(r"best\W+(pose|mode|score|vina|pocket|drug)", text, re.IGNORECASE), text
            assert "highest-affinity pose" in shown._repr_html_()


# --- files ---------------------------------------------------------------------------------------------------------

def test_export(atlas, tmp_path, monkeypatch):
    """SDF: one record per pose with protein, ligand, pocket, mode and affinity as properties, the best pose at the
    stored top pose's coordinates; PDBQT: the models as stored, each with a REMARK."""
    monkeypatch.chdir(tmp_path)
    sel = atlas.select(*PAIR)
    path = sel.export()
    assert path == str(tmp_path / f"{pose_name(*PAIR)}_top.sdf")
    (mol,) = Chem.ForwardSDMolSupplier(io.BytesIO(Path(path).read_bytes()), removeHs=False)
    assert {k: mol.GetProp(k) for k in ("protein_id", "ligand_id", "pocket", "mode", "affinity")} == {
        "protein_id": ABL1, "ligand_id": PAIR[1], "pocket": str(sel.best_pocket), "mode": str(sel.best_mode),
        "affinity": str(sel.vina)}
    top = Chem.RemoveHs(Chem.MolFromMolBlock(sel._top_sdf, removeHs=False))
    assert sorted(map(tuple, Chem.RemoveHs(mol).GetConformer().GetPositions().round(2).tolist())) == \
        sorted(map(tuple, top.GetConformer().GetPositions().round(2).tolist()))
    every = atlas.select(*PAIR, pocket="all")
    path = every.export(tmp_path / "x.pdbqt", fmt="pdbqt")
    data = Path(path).read_bytes()
    models = pdbqt.parse_modes(data)
    assert len(models) == len(every._poses) == 5
    lines = data.decode().splitlines()
    assert [line for line in lines if line.startswith("REMARK POCKETVINA")] == [
        f"REMARK POCKETVINA protein {ABL1} ligand {PAIR[1]} pocket {p.pocket} mode {p.mode}" for p in every._poses]


def test_export_protein_is_the_alphafold_model(atlas, atlas_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sel = atlas.select(*REDOCK)
    path = sel.export_protein()
    assert path == str(tmp_path / f"PV_{ABL1}.pdb")
    af = atlas._protein_json(ABL1)["af_id"]
    model = gzip.decompress((atlas_dir / "structures" / f"{af}.pdb.gz").read_bytes()).decode()
    assert Path(path).read_text() == model and model.count("\nATOM") > 100


def test_export_with_the_protein(atlas, atlas_dir, tmp_path, monkeypatch):
    """export(protein=True) of an Atlas pair: the poses and the AlphaFold model, next to each other, as [poses,
    protein]; export(ligand=False, protein=True): the model only, as export_protein() writes it."""
    sel = atlas.select(*PAIR, pocket="all")
    af = atlas._protein_json(ABL1)["af_id"]
    model = gzip.decompress((atlas_dir / "structures" / f"{af}.pdb.gz").read_bytes()).decode()
    poses = Path(sel.export(tmp_path / "expected.sdf")).read_bytes()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    paths = sel.export(protein=True)
    assert paths == [str(work / f"{pose_name(*PAIR)}_pocketall_top.sdf"), str(work / f"PV_{ABL1}.pdb")]
    assert Path(paths[0]).read_bytes() == poses and Path(paths[1]).read_text() == model
    sub = work / "sub"
    sub.mkdir()
    assert sel.export(sub, fmt="pdbqt", protein=True) == [str(sub / f"{pose_name(*PAIR)}_pocketall_top.pdbqt"),
                                                          str(sub / f"PV_{ABL1}.pdb")]
    path = sel.export(tmp_path / "only" / "protein.pdb", ligand=False, protein=True)
    assert path == str(tmp_path / "only" / "protein.pdb") and os.listdir(tmp_path / "only") == ["protein.pdb"]
    assert Path(path).read_text() == model


def test_export_paths(atlas, tmp_path, monkeypatch):
    """As for a pair of results: a path without an extension is a folder, made if needed; a file whose extension is
    not its format's (.sdf or .pdbqt for the poses, .pdb for the protein) is an error before anything is written or
    the AlphaFold model read."""
    sel = atlas.select(*PAIR)
    monkeypatch.chdir(tmp_path)
    poses = f"{pose_name(*PAIR)}_top"
    assert sel.export("poses", protein=True) == [str(tmp_path / "poses" / f"{poses}.sdf"),
                                                 str(tmp_path / "poses" / f"PV_{ABL1}.pdb")]
    assert sel.export("modes", fmt="pdbqt") == str(tmp_path / "modes" / f"{poses}.pdbqt")
    assert sel.export_protein("protein") == str(tmp_path / "protein" / f"PV_{ABL1}.pdb")
    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert written == ["modes", f"modes/{poses}.pdbqt", "poses", f"poses/PV_{ABL1}.pdb", f"poses/{poses}.sdf",
                       "protein", f"protein/PV_{ABL1}.pdb"]
    monkeypatch.setattr(sel._source, "protein_pdb", lambda protein: pytest.fail("the protein was read"))
    with pytest.raises(ValueError, match=re.escape(f"{tmp_path / 'abl1.pdb'}: poses are written as SDF; use a .sdf "
                                                   "name, or fmt='pdbqt' for .pdbqt")):
        sel.export("abl1.pdb", protein=True)
    with pytest.raises(ValueError, match=re.escape(f"{tmp_path / 'x.sdf'}: the protein is written as PDB; use a .pdb "
                                                   "name")):
        sel.export_protein("x.sdf")
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == written


def test_bust_one_pose(atlas):
    """PoseBusters on the stored top pose against the AlphaFold model: the same verdict as the release's."""
    pytest.importorskip("posebusters")
    sel = atlas.select(*PAIR)
    df = sel.bust(workers=1)
    assert df[["pocket", "mode"]].values.tolist() == [[sel.best_pocket, sel.best_mode]]
    row = atlas._pair(*PAIR).row
    assert df["pb_status"].iloc[0] == atlas.meta["pb_status"][row["pb"]] == "valid"


# --- the page ------------------------------------------------------------------------------------------------------

def test_show_page_data(atlas, atlas_dir, tmp_path, monkeypatch, capsys):
    """atlas.show: the page of format 1 that Results.show writes (PV_<UniProt>.html), with the protein's AlphaFold
    model, its Top hits as the Atlas site lists them (the drugs with a PoseBusters-valid pose, best first, ranked among
    the protein's drugs) with their poses in every pocket and their top pose, and the Atlas fields: the protein's
    information, the pockets' pLDDT, the identifier rows, the stored PoseBusters result and the Top hits text."""
    monkeypatch.chdir(tmp_path)
    path = atlas.show("ABL1")
    assert path == str(tmp_path / f"PV_{ABL1}.html") and "open it in a web browser" in capsys.readouterr().out
    data = payload(path)
    assert data["format"] == 1 and data["maxLigands"] == MAX_LIGANDS == 25
    (prot,) = data["proteins"]
    raw = atlas._protein_json(ABL1)
    rows, drugs, hits = top_hits(atlas, ABL1)
    assert 0 < len(hits) < len(drugs) < len(rows)  # the fixture has rows of both kinds that Top hits leave out
    assert (prot["id"], prot["name"], prot["key"]) == (ABL1, ABL1, f"structures/{raw['af_id']}.pdb")
    assert prot["pdb"] == atlas._structure(raw) and prot["n"] == len(drugs) and prot["topNote"] == TOP_HITS
    assert [item["lig"] for item in prot["rows"]] == [row["cidx"] for row in hits]
    assert prot["docked"] == [1, 2, 3, 4, 5]
    assert [p["plddt"] for p in prot["pockets"]] == [round(p["plddt"], 1) for p in raw["pockets"]]
    assert data["initial"] == {"protein": ABL1, "ligand": None, "view": "best", "pocket": None, "mode": None,
                               "allModes": False, "show": [1, 2, 3, 4, 5], "userShow": False, "tab": "top"}
    checks = atlas.meta["pb_checks"]
    for item, row in zip(prot["rows"], hits):
        _, blocks, top = stored(atlas, atlas_dir, ABL1, row["pvl"])
        assert (item["lig"], item["rank"], item["vina"], item["pocket"], item["mode"]) == (
            row["cidx"], drugs.index(row) + 1, row["vina"], row["pocket"], row["mode"])
        assert item["sdf"] == (top.decode() if top else None)
        assert [k["rank"] for k in item["pockets"]] == sorted(blocks)
        for k in item["pockets"]:
            assert k["pdbqt"].encode() == blocks[k["rank"]]
            assert len(k["modes"]) == len(pdbqt.parse_modes(blocks[k["rank"]]))
        assert item["pb"] == {"status": "valid", "valid": True, "passed": len(checks), "total": len(checks),
                              "failed": []}
        ligand = data["ligands"][str(row["cidx"])]
        drug = atlas._drug_info(row["pvl"], row["cat"])
        assert ligand["id"] == (drug.name or row["pvl"]) and ligand["name"] == row["pvl"]
        assert ligand["svg"].startswith("<svg")
        ids = ligand["ids"]
        assert ids[0]["label"] == "Name" and ids[1] == {"label": "PVL ID", "value": row["pvl"], "tip": ids[1]["tip"]}
        assert all(isinstance(x["value"], str) and x["value"] for x in ids)
        assert [x["label"] for x in ids][1:] == [label[:1].upper() + label[1:] for label, _ in drug._rows()]
        assert all(x["href"].startswith("https://") for x in ids if "href" in x)
        assert "Docked as" not in [x["label"] for x in ids]
    pair_row = next(row for row in hits if row["pvl"] == PAIR[1])
    chembl = next(x for x in data["ligands"][str(pair_row["cidx"])]["ids"] if x["label"] == "ChEMBL ID")
    assert chembl["href"] == f"https://www.ebi.ac.uk/chembl/explore/compound/{chembl['value']}"
    info = prot["info"]
    assert info["title"] == raw["gene"] and info["name"] == raw["protein_name"]
    assert [(x["label"], x["href"]) for x in info["links"]] == [
        ("UniProt", f"https://www.uniprot.org/uniprotkb/{ABL1}"), ("AlphaFold", f"https://alphafold.ebi.ac.uk/entry/{ABL1}")]
    assert [x["label"] for x in info["facts"]] == ["ChEMBL class", "Model"]
    assert "mean pLDDT" in info["facts"][1]["value"]
    diseases = info["diseases"]
    assert len(diseases["items"]) == len(raw["open_targets"]["diseases"]) > 0
    ensembl = raw["open_targets"]["ensembl_id"]
    assert all(d["href"].startswith(f"https://platform.opentargets.org/evidence/{ensembl}/") for d in diseases["items"])
    assert diseases["all"]["href"] == f"https://platform.opentargets.org/target/{ensembl}/associations"


def test_show_lists_max_ligands(atlas, tmp_path):
    rows, drugs, hits = top_hits(atlas, ABL1)
    two = payload(atlas.show(ABL1, path=tmp_path / "a.html", max_ligands=2))["proteins"][0]
    assert [r["lig"] for r in two["rows"]] == [r["cidx"] for r in hits[:2]]
    assert two["topNote"] == f"{TOP_HITS} The top 2 of {len(hits)} on ABL1 (max_ligands=2)."
    every = payload(atlas.show(ABL1, path=tmp_path / "b.html", max_ligands=None))
    assert [r["lig"] for r in every["proteins"][0]["rows"]] == [r["cidx"] for r in hits]
    assert every["maxLigands"] is None and every["proteins"][0]["topNote"] == TOP_HITS
    for kwargs, match in ((dict(max_ligands=0), "max_ligands must be"), (dict(theme="x"), "theme must be"),
                          (dict(molstar="x"), "molstar must be"), (dict(height=True), "height must be")):
        with pytest.raises(ValueError, match=match):
            atlas.show(ABL1, path=tmp_path / "c.html", **kwargs)
    assert not (tmp_path / "c.html").exists()


def test_pair_show(atlas, tmp_path, monkeypatch):
    """pair.show(): the page of the pair's protein with the pair selected (the first state of Results pages), named
    after the selection; a pair that is not among Top hits (beyond max_ligands, no PoseBusters result, not a drug)
    is listed after them, with its rank among the protein's drugs (none if it is not a drug); 25 rows by default."""
    monkeypatch.chdir(tmp_path)
    rows, drugs, hits = top_hits(atlas, ABL1)
    sel = atlas.select(*REDOCK, pocket="all")
    path = sel.show()
    assert path == str(tmp_path / f"{sel.name}.html")
    data = payload(path)
    assert data["initial"] == workspace._initial(sel, ABL1, [1, 2, 3, 4, 5])
    assert data["initial"]["ligand"] == sel.ligand_idx and data["initial"]["tab"] == "pair"
    assert data["maxLigands"] == MAX_LIGANDS
    (redock,) = [r for r in data["proteins"][0]["rows"] if r["lig"] == sel.ligand_idx]
    row = next(r for r in drugs if r["pvl"] == REDOCK[1])
    assert row["pb"] is None and redock["pb"] is None  # pf 1: no stored PoseBusters result, no PoseBusters row
    assert redock == data["proteins"][0]["rows"][-1] and redock["rank"] == drugs.index(row) + 1
    other = atlas.select(*OTHER)
    data = payload(other.show(path=tmp_path / "x.html", max_ligands=2))
    listed = data["proteins"][0]["rows"]
    assert [r["lig"] for r in listed] == [r["cidx"] for r in hits[:2]] + [other.ligand_idx]
    assert listed[-1]["rank"] is None and data["initial"]["ligand"] == other.ligand_idx
    assert "Docked as" in [x["label"] for x in data["ligands"][str(other.ligand_idx)]["ids"]]


def test_pair_show_posebusters_failed(atlas, tmp_path):
    """A pair whose stored PoseBusters result is invalid: the failed checks are the bits of pb_mask that are 0, named
    by meta.json pb_checks."""
    checks = atlas.meta["pb_checks"]
    rows, drugs, hits = top_hits(atlas, ABL1)
    row = next(r for r in drugs if r["pb"] == 1 and r["pb_mask"] is not None)
    sel = atlas.select(ABL1, row["pvl"])
    pb = payload(sel.show(path=tmp_path / "x.html"))["proteins"][0]["rows"][-1]["pb"]
    failed = [name for i, name in enumerate(checks) if not int(row["pb_mask"]) >> i & 1]
    assert pb == {"status": "invalid", "valid": False, "passed": len(checks) - len(failed), "total": len(checks),
                  "failed": failed}
    assert 0 < len(failed) == row["n_failed"]


def test_pair_show_entry_run(atlas, atlas_dir, tmp_path):
    """A catalogue number's own run (salt_forms/): selected, it is listed with no rank and no "Across proteins"
    (the drug's page ranks the drug's runs), as on the site."""
    salt = atlas.source.json(f"salt_forms/{ABL1}.json")
    cat = salt["rows"][0][salt["columns"].index("cat")]
    sel = atlas.select(ABL1, cat)
    data = payload(sel.show(path=tmp_path / "x.html"))
    entry = data["proteins"][0]["rows"][-1]
    assert entry["lig"] == sel.ligand_idx == salt["rows"][0][salt["columns"].index("cidx")]
    assert entry["rank"] is None and entry["across"] is None


def test_download_tab_python_lines(atlas, tmp_path):
    """The Python lines of an Atlas page's Download tab (workspace.js pythonLines) name the protein by its gene
    (P.info.title) and the drug as the page names it (L.id): atlas.select takes both and selects the pair shown, for
    every Top hit of each protein and for a row listed after them (not docked as a drug). A title of several genes
    ("A; B") is not taken: there the lines name the protein's id (workspace.js pyProtein)."""
    pages = [atlas._page(protein, None) for protein in (ABL1, EGFR, PTGS2)]
    pages.append(payload(atlas.select(*OTHER).show(path=tmp_path / "other.html", max_ligands=1)))
    n = 0
    for data in pages:
        (prot,) = data["proteins"]
        assert prot["info"]["title"] != prot["id"]  # the gene, not the accession
        for item in prot["rows"]:
            ligand = data["ligands"][str(item["lig"])]
            assert ligand["id"] != ligand["name"]  # the drug's name, not its PVL ID
            sel = atlas.select(protein=prot["info"]["title"], ligand=ligand["id"])
            assert (sel.protein, sel.ligand_idx) == (prot["id"], item["lig"])
            n += 1
    assert n > len(pages)
    with pytest.raises(KeyError, match="no protein"):
        atlas.select(protein=f"{prot['info']['title']}; X1", ligand=ligand["id"])


def test_across_proteins():
    """The rank of the protein among the drug's best targets (compounds/<PVL>.json), as the site's pair panel says it:
    [rank, number of docked proteins]; outside them [None, number, number of best targets]."""
    page = {"columns": ["uniprot", "vina"], "rows": [["P2", -9.0], ["P1", -10.0]], "n_targets": 500}
    assert _across(page, "P1") == [1, 500] and _across(page, "P2") == [2, 500]
    assert _across(page, "P3") == [None, 500, 2]
    assert _across(dict(page, n_targets=2), "P3") is None and _across(dict(page, n_targets=1), "P1") is None
    assert _across({"columns": []}, "P1") is None


def test_show_in_a_notebook(atlas, tmp_path, monkeypatch):
    shown = []
    display = types.ModuleType("IPython.display")
    display.display = shown.append
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    monkeypatch.setattr(workspace, "in_notebook", lambda: True)
    monkeypatch.chdir(tmp_path)
    assert atlas.show(ABL1, height=640) is None and atlas.select(*PAIR).show(theme="dark") is None
    assert os.listdir(tmp_path) == [] and all(isinstance(v, workspace.NotebookView) for v in shown)
    tag = shown[1]._repr_html_()
    assert "allow-popups allow-popups-to-escape-sandbox" in tag
    page = html.unescape(re.match(r'<iframe srcdoc="([^"]*)"', tag).group(1))
    assert '<html lang="en" data-theme="dark" data-theme-setting="dark">' in page  # dark from the first paint
    assert workspace.page_payload(page)["initial"]["tab"] == "pair"


def test_show_over_http(atlas, atlas_server, tmp_path, monkeypatch):
    """The page read over HTTP (Range requests for the poses) is the page read from the folder. Ranges close together
    come in one request; a second page asks for nothing (the ranges are kept in memory, the files in the cache)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    remote = Atlas(atlas_server.url)
    over_http = payload(remote.show(ABL1, path=tmp_path / "a.html"))
    assert over_http == payload(atlas.show(ABL1, path=tmp_path / "b.html"))
    ranges = sum(1 for _, header in atlas_server.requests if header)
    assert 0 < ranges < 3 * len(over_http["proteins"][0]["rows"])  # one each before: table B, blocks, top pose
    asked = len(atlas_server.requests)
    assert payload(remote.show(ABL1, path=tmp_path / "a2.html")) == over_http
    assert len(atlas_server.requests) == asked
    sel = remote.select(*REDOCK)
    assert payload(sel.show(path=tmp_path / "c.html"))["initial"]["ligand"] == sel.ligand_idx


def test_pb_of_a_row():
    checks, statuses = ["a", "b", "c"], ["valid", "invalid", "timeout", "missing", "error"]
    assert _pb({"pb": None, "pb_mask": None}, checks, statuses) is None
    assert _pb({"pb": 0, "pb_mask": 7}, checks, statuses) == {"status": "valid", "valid": True, "passed": 3,
                                                              "total": 3, "failed": []}
    assert _pb({"pb": 1, "pb_mask": 5.0}, checks, statuses) == {"status": "invalid", "valid": False, "passed": 2,
                                                                "total": 3, "failed": ["b"]}
    assert _pb({"pb": 2, "pb_mask": None}, checks, statuses) == {"status": "timeout", "valid": None, "passed": 0,
                                                                 "total": 0, "failed": []}


def test_results_pages_have_no_atlas_fields(real_run, tmp_path):
    """A page of docking results has none of the Atlas fields (it looks as in 0.2); links of a page may open a new tab
    (the notebook frame's sandbox), but it keeps its own origin."""
    from pocketvina.io.importer import import_legacy
    from pocketvina.io.results import Results

    with Results(import_legacy(str(real_run.root), str(tmp_path / "results"), str(real_run.smiles_csv))) as res:
        data = workspace.workspace_data(res)
    assert "info" not in data["proteins"][0] and all("plddt" not in p for p in data["proteins"][0]["pockets"])
    assert "topNote" not in data["proteins"][0] and all(r["rank"] >= 1 for r in data["proteins"][0]["rows"])
    assert all("ids" not in lig for lig in data["ligands"].values())
    assert {"allow-popups", "allow-popups-to-escape-sandbox"} <= set(workspace.SANDBOX.split())
    assert "allow-same-origin" not in workspace.SANDBOX


def test_page_script_draws_the_atlas_fields_only_when_present():
    js = (Path(workspace.VIEWER_DIR) / "workspace.js").read_text(encoding="utf-8")
    for part in ("D.proteins.some(p => p.info) ?", "if (L.ids) {", "P.pockets.some(p => p.plddt != null)",
                 'target="_blank" rel="noopener noreferrer"', "/^https?:\\/\\//i.test("):
        assert part in js, part
    # the site's words on an Atlas page only: "of N drugs on", "for this compound", "not among the top"
    for part in ("${P.info ? ' drugs' : ''} on", "for this ${P.info ? 'compound' : 'ligand'}",
                 "r.across[0] == null ? `not among the top ${int(r.across[2])} of ${int(r.across[1])}`"):
        assert part in js, part
