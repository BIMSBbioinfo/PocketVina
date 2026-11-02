"""pocketvina.viz: Mol* (MolViewSpec) and py3Dmol views of docked poses, the HTML export and the results table.

No network and no notebook: the tests read the MolViewSpec state, the files a view carries and the 3Dmol.js
commands, not the drawing. The views are made from the fixture run in ``tests/data`` imported into the atlas layout
(its structure is a PDBQT receptor); copies of it with a PDB or an mmCIF structure cover the other inputs.
"""
import base64
import gzip
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from pocketvina import convert, pdbqt, viz
from pocketvina.io import tables
from pocketvina.io.importer import import_legacy
from pocketvina.io.results import Results

PID = "AF-Q96LA5-F1-model_v4"


def _importable(name: str) -> bool:
    try:
        __import__(name)
    except Exception:  # molviewspec fails with TypeError on Python < 3.10
        return False
    return True


HAS_MVS = _importable("molviewspec")
HAS_PY3DMOL = _importable("py3Dmol")
needs_mvs = pytest.mark.skipif(not HAS_MVS, reason="needs molviewspec (Python >= 3.10)")
needs_py3dmol = pytest.mark.skipif(not HAS_PY3DMOL, reason="needs py3Dmol")
BACKENDS = [pytest.param("molstar", marks=needs_mvs), pytest.param("py3dmol", marks=needs_py3dmol)]


# --- fixtures and helpers ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results_dir(real_run, tmp_path_factory) -> str:
    return import_legacy(str(real_run.root), str(tmp_path_factory.mktemp("viz") / "results"),
                         str(real_run.smiles_csv))


@pytest.fixture(scope="module")
def res(results_dir):
    with Results(results_dir) as r:
        yield r


@pytest.fixture(scope="module")
def best_view(res):
    """``show`` of molecule2's best pose (pocket 2, mode 1, -6.3) with Mol*."""
    if not HAS_MVS:
        pytest.skip("needs molviewspec (Python >= 3.10)")
    return viz.show(res, PID, "molecule2", backend="molstar")


def copy_results(results_dir, dst) -> str:
    shutil.copytree(results_dir, dst)
    return str(dst)


def with_structure(results_dir, dst, name: str, text: bytes) -> str:
    """A copy of the results whose stored structure is ``structures/<name>`` (gzipped) instead of the PDBQT."""
    path = copy_results(results_dir, dst)
    os.unlink(os.path.join(path, "structures", f"{PID}.pdbqt.gz"))
    with open(os.path.join(path, "structures", name), "wb") as f:
        f.write(gzip.compress(text))
    return path


def state_of(view) -> dict:
    return json.loads(view.state_json())


def walk(node, path=()):
    """(kind, params, ancestors' kinds) of every node of a MolViewSpec tree."""
    yield node["kind"], node.get("params", {}), path
    for child in node.get("children", []):
        yield from walk(child, path + (node["kind"],))


def nodes(tree, kind=None) -> list:
    return [(k, p, a) for k, p, a in walk(tree) if kind is None or k == kind]


def mvsx_of(page: str) -> dict:
    """``{file name: bytes}`` of the MVSX archive embedded in a ``molstar_html`` page."""
    m = re.search(r"var mvsData = (\".*?\");", page)
    assert m, "no embedded MolViewSpec data"
    value = json.loads(m.group(1))
    assert value.startswith("base64,")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(value[len("base64,"):]))) as z:
        return {name: z.read(name) for name in z.namelist()}


def atom_lines(text: str) -> list:
    return [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]


def py3dmol_models(page: str) -> list:
    """(call, text, format) of the addModel / addModelsAsFrames calls of a py3Dmol page, in order."""
    calls = re.findall(r"\.(addModel|addModelsAsFrames)\((\".*?(?<!\\)\"),\"(\w+)\"\)", page)
    return [(call, json.loads(text), fmt) for call, text, fmt in calls]


def pdb_models(text: str) -> list:
    """ATOM/HETATM lines of each MODEL of a PDB text (the whole text if it has no MODEL records)."""
    if "MODEL" not in text:
        return [atom_lines(text)]
    return [atom_lines(block) for block in text.split("ENDMDL")[:-1]]


def same_pdbqt_atoms(pdb_lines: list, model: str) -> bool:
    """The PDB lines hold every atom of the PDBQT model as HETATM records: same names and coordinates, elements of
    the AD types."""
    src = atom_lines(model)
    return len(pdb_lines) == len(src) and all(
        a.startswith("HETATM") and a[6:54] == b[6:54] and a[76:78].strip() == convert.AD_ELEMENTS[b.split()[-1]].upper()
        for a, b in zip(pdb_lines, src))


def sdf_mol(data: bytes):
    """The one record of an SDF (with its data fields), sanitized."""
    mols = list(Chem.ForwardSDMolSupplier(io.BytesIO(data), removeHs=False))
    assert len(mols) == 1 and mols[0] is not None
    return mols[0]


def heavy_xyz_sdf(data: bytes) -> np.ndarray:
    mol = sdf_mol(data)
    pos = mol.GetConformer().GetPositions()
    return pos[[a.GetAtomicNum() > 1 for a in mol.GetAtoms()]]


def heavy_xyz_pdbqt(text: str) -> np.ndarray:
    xyz = []
    for line in atom_lines(text.split("ENDMDL")[0]):
        if line.split()[-1] not in ("H", "HD", "HS"):
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.array(xyz)


def same_points(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.allclose(np.array(sorted(map(tuple, a.round(3)))),
                                              np.array(sorted(map(tuple, b.round(3)))), atol=1e-3)


def brute_force_residues(receptor_pdbqt: str, ligand_xyz: np.ndarray, radius: float) -> list:
    near = {}
    for line in atom_lines(receptor_pdbqt):
        if line.split()[-1] in ("H", "HD", "HS"):
            continue
        p = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if np.sqrt(((ligand_xyz - p) ** 2).sum(axis=1)).min() <= radius:
            near[(line[21], int(line[22:26]))] = line[17:20].strip()
    return [f"{c}:{n}{s}" for (c, s), n in sorted(near.items())]


# --- Mol* (MolViewSpec) --------------------------------------------------------------------------------------------

def test_molstar_scene(best_view, res):
    v = best_view
    assert isinstance(v, viz.MolstarView)
    pairs = res.pairs(PID).set_index("ligand_id")
    assert v.info["pocket"] == pairs.loc["molecule2", "best_pocket"] == 2
    assert v.info["mode"] == 1 and v.info["affinity"] == pytest.approx(-6.3)
    assert v.info["ligand_format"] == "sdf"
    tree = state_of(v)
    assert tree["kind"] == "single"
    root = tree["root"]
    assert [p["url"] for _, p, _ in nodes(root, "download")] == ["protein.pdb", "ligand.sdf"]
    assert [p["format"] for _, p, _ in nodes(root, "parse")] == ["pdb", "sdf"]
    assert nodes(root, "canvas")[0][1]["background_color"] == viz.BACKGROUND
    reps = [p for _, p, _ in nodes(root, "representation")]
    assert [r["type"] for r in reps] == ["cartoon", "ball_and_stick", "surface", "ball_and_stick"]
    assert reps[1]["size_factor"] == viz.RESIDUE_SIZE and reps[1]["ignore_hydrogens"] is True
    assert reps[3]["ignore_hydrogens"] is True
    # the residues within 5 A, one selector each
    comps = [p for _, p, _ in nodes(root, "component")]
    assert comps[0]["selector"] == "polymer"
    site = comps[1]["selector"]
    assert [f"{e['auth_asym_id']}:{e['auth_seq_id']}" for e in site] == \
        [r.split(":")[0] + ":" + re.sub(r"\D", "", r.split(":")[1]) for r in v.info["residues"]]
    # colours: grey cartoon, grey residue carbons, gold ligand carbons, element colours by type_symbol
    colors = [p for _, p, _ in nodes(root, "color")]
    assert {"color": viz.CARTOON_COLOR} in colors
    assert {"color": viz.RESIDUE_CARBON} in colors and {"color": viz.LIGAND_CARBON} in colors
    assert {"selector": [{"type_symbol": "O"}], "color": viz.ELEMENT_COLORS["O"]} in colors
    assert {"selector": [{"type_symbol": "N"}], "color": viz.ELEMENT_COLORS["N"]} in colors
    # focus, tooltip and label on the ligand
    focus = nodes(root, "focus")
    assert len(focus) == 1 and focus[0][1] == {"radius_extent": viz.FOCUS_EXTENT}
    assert focus[0][2][-3:] == ("parse", "structure", "component")
    tooltip = nodes(root, "tooltip")[0][1]["text"]
    assert tooltip == f"{PID} | molecule2 | pocket 2 | mode 1 | -6.3 kcal/mol"
    label = [p for _, p, _ in nodes(root, "primitive")]
    assert len(label) == 1 and label[0]["kind"] == "label"
    assert label[0]["text"] == "molecule2 | pocket 2 | mode 1 | -6.3 kcal/mol"
    assert tree["metadata"]["title"] == tooltip


def test_molstar_files(best_view, res, real_run):
    v = best_view
    assert sorted(v.data) == ["ligand.sdf", "protein.pdb"] == v.info["files"]
    assert all(isinstance(b, bytes) for b in v.data.values())
    # the ligand: the stored best-pose SDF, a valid molecule of the docked SMILES
    assert v.data["ligand.sdf"] == res.top_sdf(PID, "molecule2").encode()
    mol = sdf_mol(v.data["ligand.sdf"])
    smiles = res.ligands().set_index("ligand_id").loc["molecule2", "smiles_docked"]
    assert Chem.MolToInchiKey(mol) == Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    assert mol.GetProp("pocket") == "2" and mol.GetProp("mode") == "1"
    # the protein: the receptor PDBQT as PDB, atom for atom, with element columns
    receptor = real_run.receptor.read_text()
    pdb = v.data["protein.pdb"].decode()
    src, out = atom_lines(receptor), atom_lines(pdb)
    assert len(out) == len(src) > 1000
    assert all(a[:66] == b[:66].ljust(66) for a, b in zip(out, src))
    elements = {b.split()[-1]: a[76:78].strip() for a, b in zip(out, src)}
    assert {"C", "NA", "OA"} <= set(elements)
    assert all(elements[t] == convert.AD_ELEMENTS[t].upper() for t in elements), elements
    assert pdb.rstrip().endswith("END")
    assert Chem.MolFromPDBBlock(pdb, sanitize=False).GetNumAtoms() == len(src)


def test_residues_within_radius(best_view, real_run):
    xyz = heavy_xyz_sdf(best_view.data["ligand.sdf"])
    expected = brute_force_residues(real_run.receptor.read_text(), xyz, 5.0)
    assert best_view.info["residues"] == expected and len(expected) > 5


@needs_mvs
def test_radius_and_surface_options(res, real_run):
    v = viz.show(res, PID, "molecule2", backend="molstar", pocket_radius=8, surface=False)
    root = state_of(v)["root"]
    assert [p["type"] for _, p, _ in nodes(root, "representation")] == ["cartoon", "ball_and_stick",
                                                                       "ball_and_stick"]
    xyz = heavy_xyz_sdf(v.data["ligand.sdf"])
    assert v.info["residues"] == brute_force_residues(real_run.receptor.read_text(), xyz, 8.0)
    v0 = viz.show(res, PID, "molecule2", backend="molstar", pocket_radius=0)
    assert v0.info["residues"] == []
    assert [p["type"] for _, p, _ in nodes(state_of(v0)["root"], "representation")] == ["cartoon",
                                                                                       "ball_and_stick"]


@needs_mvs
def test_pocket_and_mode(res):
    v = viz.show(res, PID, "molecule5", pocket=1, mode=3, backend="molstar")
    modes = res.modes(PID)
    row = modes[(modes.ligand_idx == 1) & (modes.pocket == 1) & (modes["mode"] == 3)].iloc[0]
    assert (v.info["pocket"], v.info["mode"]) == (1, 3)
    assert v.info["affinity"] == pytest.approx(float(row.affinity), abs=1e-6)
    assert v.info["ligand_format"] == "sdf"
    mol = sdf_mol(v.data["ligand.sdf"])
    assert mol.GetProp("_Name") == "molecule5"
    assert (mol.GetProp("pocket"), mol.GetProp("mode"), mol.GetProp("protein_id")) == ("1", "3", PID)
    # the converted pose sits where the engine put mode 3
    assert same_points(heavy_xyz_sdf(v.data["ligand.sdf"]), heavy_xyz_pdbqt(res.pose_model(PID, 1, 1, 3)))
    assert "mode 3" in nodes(state_of(v)["root"], "tooltip")[0][1]["text"]
    # ligand_idx works as well as ligand_id
    assert viz.show(res, PID, 1, pocket=1, mode=3, backend="molstar").data == v.data


@needs_mvs
def test_unconverted_pose_is_shown_from_pdbqt(res, monkeypatch):
    monkeypatch.setattr(convert, "pose_to_sdf", lambda *a, **k: (None, "convert_failed"))
    v = viz.show(res, PID, "molecule2", pocket=1, mode=2, backend="molstar")
    assert sorted(v.data) == ["ligand.pdbqt", "protein.pdb"] and v.info["ligand_format"] == "pdbqt"
    model = res.pose_model(PID, "molecule2", 1, 2)
    assert v.data["ligand.pdbqt"].decode() == model
    root = state_of(v)["root"]
    assert [p["format"] for _, p, _ in nodes(root, "parse")] == ["pdb", "pdbqt"]
    assert [p["url"] for _, p, _ in nodes(root, "download")] == ["protein.pdb", "ligand.pdbqt"]
    assert v.info["affinity"] == pytest.approx(pdbqt.parse_modes(model.encode())[0].affinity)
    assert v.info["residues"]  # coordinates read from the PDBQT


@needs_mvs
def test_best_pose_without_stored_sdf(real_run, tmp_path):
    path = import_legacy(str(real_run.root), str(tmp_path / "results"))  # no SMILES: no section A SDF
    with Results(path) as r:
        assert r.top_sdf(PID, "molecule2") is None
        v = viz.show(r, PID, "molecule2", backend="molstar")
        assert v.data["ligand.pdbqt"].decode() == r.pose_model(PID, "molecule2", 2, 1)
        assert (v.info["pocket"], v.info["mode"], v.info["ligand_format"]) == (2, 1, "pdbqt")
        # other modes: no SMILES to convert with, the PDBQT model
        assert viz.show(r, PID, "molecule2", pocket=1, mode=4, backend="molstar").info["ligand_format"] == "pdbqt"


@needs_mvs
def test_show_modes_molstar(res):
    v = viz.show_modes(res, PID, "molecule2", 1, backend="molstar")
    tree = state_of(v)
    assert tree["kind"] == "multiple" and v.ui == "stories"
    shots = tree["snapshots"]
    assert [s["metadata"]["key"] for s in shots] == [f"mode{m}" for m in range(1, 10)]
    assert v.info["modes"] == list(range(1, 10)) and len(v.info["affinities"]) == 9
    assert sorted(v.data) == sorted(["protein.pdb"] + [f"mode{m}.sdf" for m in range(1, 10)])
    modes = res.modes(PID)
    expected = modes[(modes.ligand_idx == 0) & (modes.pocket == 1)].sort_values("mode")["affinity"].tolist()
    assert v.info["affinities"] == pytest.approx(expected, abs=1e-6)
    for m, shot in enumerate(shots, 1):
        assert [p["url"] for _, p, _ in nodes(shot["root"], "download")] == ["protein.pdb", f"mode{m}.sdf"]
        assert f"| **{m}** |" in shot["metadata"]["description"]
        assert f"mode {m} |" in shot["metadata"]["title"]
        mol = sdf_mol(v.data[f"mode{m}.sdf"])
        assert mol.GetProp("mode") == str(m)
        assert same_points(heavy_xyz_sdf(v.data[f"mode{m}.sdf"]), heavy_xyz_pdbqt(res.pose_model(PID, 0, 1, m)))
    page = v.to_html()
    assert "mvs-stories" in page
    assert mvsx_of(page)["mode9.sdf"] == v.data["mode9.sdf"]
    assert viz.show_modes(res, PID, "molecule2", 1, backend="molstar", ui="viewer").ui == "viewer"
    # pocket=None: the pocket of the best pose (molecule2: pocket 2)
    best = viz.show_modes(res, PID, "molecule2", None, backend="molstar")
    assert best.info["pocket"] == 2 and best.info["affinities"][0] == pytest.approx(-6.3)


@needs_mvs
def test_to_html_molstar(res, best_view, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(viz.MOLSTAR_VERSION_ENV, raising=False)
    page = viz.to_html(res, PID, "molecule2", backend="molstar")
    assert os.listdir(tmp_path) == []  # nothing written without a path
    assert "molstar.Viewer" in page and f"cdn.jsdelivr.net/npm/molstar@{viz.MOLSTAR_VERSION}/" in page
    files = mvsx_of(page)
    assert sorted(files) == ["index.mvsj", "ligand.sdf", "protein.pdb"]
    assert files["ligand.sdf"] == best_view.data["ligand.sdf"]
    assert files["protein.pdb"] == best_view.data["protein.pdb"]
    index = json.loads(files["index.mvsj"])
    assert index["root"] == state_of(best_view)["root"]
    out = tmp_path / "share" / "pose.html"
    assert viz.to_html(res, PID, "molecule2", path=out, backend="molstar", molstar_version="5.13.1") == \
        out.read_text(encoding="utf-8")
    assert "molstar@5.13.1" in out.read_text(encoding="utf-8")
    assert sorted(os.listdir(out.parent)) == ["pose.html"]


def molstar_urls(page: str) -> set:
    """The Mol* versions a page loads its JavaScript and CSS from."""
    return set(re.findall(r"cdn\.jsdelivr\.net/npm/molstar@([^/\"']+)/build/", page))


@needs_mvs
def test_molstar_version(res, monkeypatch, tmp_path):
    """Mol* is pinned to an exact version (one that reads the scenes of molviewspec 1.8), not molviewspec's
    "latest", in every view and page; ``molstar_version=`` overrides it, else ``$POCKETVINA_MOLSTAR_VERSION`` (also
    for ``pocketvina show``), else ``viz.MOLSTAR_VERSION``; values that are not an npm version or tag are refused."""
    import molviewspec
    from click.testing import CliRunner

    from pocketvina import cli

    monkeypatch.delenv(viz.MOLSTAR_VERSION_ENV, raising=False)
    assert re.fullmatch(r"\d+\.\d+\.\d+", viz.MOLSTAR_VERSION)
    v = viz.show(res, PID, "molecule2", backend="molstar")
    assert v.molstar_version == viz.MOLSTAR_VERSION and molstar_urls(v.to_html()) == {viz.MOLSTAR_VERSION}
    stories = viz.show_modes(res, PID, "molecule2", 1, backend="molstar")
    page = stories.to_html()
    assert "build/mvs-stories/mvs-stories.js" in page and molstar_urls(page) == {viz.MOLSTAR_VERSION}
    assert molstar_urls(viz.show_modes(res, PID, "molecule2", 1, backend="molstar", ui="viewer").to_html()) == \
        {viz.MOLSTAR_VERSION}
    calls = []
    monkeypatch.setattr(molviewspec, "molstar_notebook", lambda *a, **k: calls.append(k))
    v.display()
    assert calls[-1]["molstar_version"] == viz.MOLSTAR_VERSION
    assert molstar_urls(v._repr_html_()) == {viz.MOLSTAR_VERSION}

    monkeypatch.setenv(viz.MOLSTAR_VERSION_ENV, "5.13.1")
    assert molstar_urls(viz.to_html(res, PID, "molecule2", backend="molstar")) == {"5.13.1"}
    assert viz.show_modes(res, PID, "molecule2", None, backend="molstar").molstar_version == "5.13.1"
    out = tmp_path / "page.html"
    result = CliRunner().invoke(cli.cli, ["show", "--results", res.path, "--protein", PID, "--ligand", "molecule2",
                                          "--out", str(out), "--backend", "molstar"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert molstar_urls(out.read_text(encoding="utf-8")) == {"5.13.1"}
    assert molstar_urls(viz.to_html(res, PID, "molecule2", backend="molstar", molstar_version="latest")) == {"latest"}
    monkeypatch.setenv(viz.MOLSTAR_VERSION_ENV, "")  # empty: not set
    monkeypatch.setattr(viz, "MOLSTAR_VERSION", "5.6.1")  # the module default can be changed too
    assert viz.show(res, PID, "molecule2", backend="molstar").molstar_version == "5.6.1"

    for bad in ("", " 5.5.0", "5.5.0/../x", '5"><script>alert(1)</script>', "5.5.0\n", 5.5):
        with pytest.raises(ValueError, match="molstar_version must be a Mol"):
            viz.show(res, PID, "molecule2", backend="molstar", molstar_version=bad)
    monkeypatch.setenv(viz.MOLSTAR_VERSION_ENV, "not a version")
    with pytest.raises(ValueError, match=r"\$POCKETVINA_MOLSTAR_VERSION must be"):
        viz.to_html(res, PID, "molecule2", backend="molstar")
    if HAS_PY3DMOL:  # py3Dmol does not use it
        viz.show(res, PID, "molecule2", backend="py3dmol")


@needs_mvs
def test_molstar_view_display(best_view, monkeypatch):
    import molviewspec

    calls = []
    monkeypatch.setattr(molviewspec, "molstar_notebook", lambda *a, **k: calls.append((a, k)))
    best_view._ipython_display_()
    (args, kw), = calls
    assert args[0] is best_view.state and kw["data"] is best_view.data
    assert (kw["width"], kw["height"], kw["ui"]) == (950, 600, "viewer")
    assert kw["download_filename"] == f"{PID}_molecule2_pocket2_mode1"
    frame = best_view._repr_html_()
    assert frame.startswith("<iframe srcdoc=") and "width: 950px" in frame and "&quot;base64," in frame
    assert "molecule2" in repr(best_view) and "protein.pdb" in repr(best_view)


def structure_file(real_run, kind: str):
    """(file name in structures/, text) of the fixture receptor as a PDB or an mmCIF input structure."""
    pdb_text = viz.pdbqt_to_pdb(real_run.receptor.read_text())
    if kind == "pdb":
        return f"{PID}.pdb.gz", ("REMARK   1 a PDB file is passed on as it is\n" + pdb_text).encode()
    gemmi = pytest.importorskip("gemmi")
    st = gemmi.read_pdb_string(pdb_text)
    st.setup_entities()
    return f"{PID}.cif.gz", st.make_mmcif_document().as_string().encode()


@needs_mvs
@pytest.mark.parametrize("kind", ["pdb", "cif"])
def test_pdb_and_cif_structures(results_dir, tmp_path, real_run, best_view, kind):
    name, data = structure_file(real_run, kind)
    path = with_structure(results_dir, tmp_path / "results", name, data)
    v = viz.show(path, PID, "molecule2", backend="molstar")
    protein_file = "protein.pdb" if kind == "pdb" else "protein.cif"
    assert sorted(v.data) == sorted([protein_file, "ligand.sdf"])
    assert v.data[protein_file] == data  # passed on unchanged
    root = state_of(v)["root"]
    assert [p["format"] for _, p, _ in nodes(root, "parse")] == ["pdb" if kind == "pdb" else "mmcif", "sdf"]
    assert v.info["residues"] == best_view.info["residues"]


# --- py3Dmol -------------------------------------------------------------------------------------------------------

@needs_py3dmol
def test_py3dmol_view(res):
    import py3Dmol

    v = viz.show(res, PID, "molecule2", backend="py3dmol", width=640, height=480)
    assert isinstance(v, py3Dmol.view)
    assert v.pocketvina_info["pocket"] == 2 and v.pocketvina_info["ligand_format"] == "sdf"
    page = v.write_html()  # the HTML fragment the notebook gets
    assert json.dumps(res.top_sdf(PID, "molecule2")) in page
    assert "addModel(" in page and '"pdb"' in page and '"sdf"' in page and "640px" in page
    assert json.dumps({"cartoon": {"color": viz.CARTOON_COLOR}}) in page
    assert str(int(viz.LIGAND_CARBON[1:], 16)) in page and str(int(viz.RESIDUE_CARBON[1:], 16)) in page
    assert "addSurface(" in page and "zoomTo(" in page and "animate(" not in page
    assert f"{PID} | molecule2 | pocket 2 | mode 1 | -6.3 kcal/mol" in page
    # residues: one selection per chain, the same residues as Mol*
    resi = sorted({int(re.sub(r"\D", "", r.split(":")[1])) for r in v.pocketvina_info["residues"]})
    assert json.dumps({"model": 0, "resi": resi, "chain": "A"}) in page


@needs_py3dmol
def test_show_modes_py3dmol(res, monkeypatch):
    v = viz.show_modes(res, PID, "molecule2", 1, backend="py3dmol")
    page = v.write_html()
    assert "addModelsAsFrames(" in page and "animate(" in page
    frames = re.search(r"addModelsAsFrames\((\".*?(?<!\\)\"),\"sdf\"\)", page)
    assert frames and json.loads(frames.group(1)).count("$$$$") == 9
    assert v.pocketvina_info["modes"] == list(range(1, 10))
    # one mode that does not convert: all frames from the PDBQT models, as PDB (3Dmol.js reads a PDBQT model only up
    # to its ENDROOT record: it drew the root fragment and split the modes into 36 frames)
    real = convert.pose_to_sdf
    monkeypatch.setattr(convert, "pose_to_sdf",
                        lambda model, smi, props=None: (None, "convert_failed") if props["mode"] == 4
                        else real(model, smi, props))
    v = viz.show_modes(res, PID, "molecule2", 1, backend="py3dmol")
    assert v.pocketvina_info["ligand_formats"] == ["pdbqt"] * 9
    _, (call, text, fmt) = py3dmol_models(v.write_html())
    assert (call, fmt) == ("addModelsAsFrames", "pdb")
    assert not re.search(r"^(ROOT|ENDROOT|BRANCH|ENDBRANCH|TORSDOF)", text, re.M)
    frames = pdb_models(text)
    assert len(frames) == 9
    for m, lines in enumerate(frames, 1):
        assert same_pdbqt_atoms(lines, res.pose_model(PID, "molecule2", 1, m)), m


@needs_py3dmol
def test_py3dmol_pdbqt_pose_has_every_atom(real_run, tmp_path):
    path = import_legacy(str(real_run.root), str(tmp_path / "results"))  # no SMILES: the poses stay PDBQT
    with Results(path) as r:
        model = r.pose_model(PID, "molecule2", 2, 1)
        v = viz.show(r, PID, "molecule2", backend="py3dmol")
        assert v.pocketvina_info["ligand_format"] == "pdbqt"
        (_, _, protein_fmt), (call, text, fmt) = py3dmol_models(v.write_html())
        assert (protein_fmt, call, fmt) == ("pdb", "addModel", "pdb")
        assert "MODEL" not in text and "ROOT" not in text
        (lines,) = pdb_models(text)
        assert same_pdbqt_atoms(lines, model) and len(lines) > 20  # not just the root fragment
        one = viz.show_modes(r, PID, "molecule2", 2, backend="py3dmol")  # all 9 modes, none converts
        _, (call, text, fmt) = py3dmol_models(one.write_html())
        assert (call, fmt, len(pdb_models(text))) == ("addModelsAsFrames", "pdb", 9)


@needs_py3dmol
@pytest.mark.parametrize("kind", ["pdb", "cif"])
def test_pdb_and_cif_structures_py3dmol(results_dir, res, tmp_path, real_run, kind):
    # 3Dmol.js gets the stored structure unchanged, in its own format name ("cif", not MolViewSpec's "mmcif")
    name, data = structure_file(real_run, kind)
    path = with_structure(results_dir, tmp_path / "results", name, data)
    v = viz.show(path, PID, "molecule2", backend="py3dmol")
    (call, text, fmt), (_, _, ligand_fmt) = py3dmol_models(v.write_html())
    assert (call, fmt, ligand_fmt) == ("addModel", kind, "sdf")
    assert text == data.decode()
    expected = viz.show(res, PID, "molecule2", backend="py3dmol").pocketvina_info["residues"]
    assert v.pocketvina_info["residues"] == expected and expected


@needs_py3dmol
def test_to_html_py3dmol(res, tmp_path):
    out = tmp_path / "pose.html"
    page = viz.to_html(res, PID, "molecule5", pocket=1, path=out, backend="py3dmol")
    assert page == out.read_text(encoding="utf-8")
    assert page.startswith("<html>") and "3Dmol" in page and "height: 100vh" in page
    assert "molecule5 | pocket 1 | mode 1" in page


# --- backends and errors -------------------------------------------------------------------------------------------

def test_auto_backend(res, monkeypatch):
    if HAS_MVS:
        assert isinstance(viz.show(res, PID, "molecule2"), viz.MolstarView)
    monkeypatch.setitem(sys.modules, "molviewspec", None)  # import fails
    with pytest.raises(ImportError, match="pip install molviewspec"):
        viz.show(res, PID, "molecule2", backend="molstar")
    if HAS_PY3DMOL:
        v = viz.show(res, PID, "molecule2")
        assert type(v).__name__ == "view" and type(v).__module__ == "py3Dmol"
    monkeypatch.setitem(sys.modules, "py3Dmol", None)
    with pytest.raises(ImportError, match="molviewspec .* or py3Dmol"):
        viz.show(res, PID, "molecule2")
    with pytest.raises(ImportError, match="pip install py3Dmol"):
        viz.to_html(res, PID, "molecule2", backend="py3dmol")
    assert len(viz.table(res)) == 2  # the table needs neither


def test_import_without_viewers(results_dir):
    code = ("import sys; sys.modules['molviewspec'] = None; sys.modules['py3Dmol'] = None\n"
            "import pocketvina\n"
            "assert 'pocketvina.viz' not in sys.modules\n"
            "from pocketvina import viz\n"
            f"print(len(viz.table({results_dir!r})))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "2"


@pytest.mark.parametrize("backend", BACKENDS)
def test_errors(res, results_dir, tmp_path, backend):
    with pytest.raises(KeyError, match="protein 'nope' is not in"):
        viz.show(res, "nope", "molecule2", backend=backend)
    with pytest.raises(KeyError, match="no ligand 'nope'"):
        viz.show(results_dir, PID, "nope", backend=backend)
    with pytest.raises(KeyError, match="ligand_idx 7 out of range"):
        viz.show(res, PID, 7, backend=backend)
    with pytest.raises(KeyError, match="'molecule10' has no docked pose"):
        viz.show(res, PID, "molecule10", backend=backend)
    with pytest.raises(KeyError, match="no engine output"):
        viz.show(res, PID, "molecule10", pocket=1, backend=backend)
    with pytest.raises(KeyError, match="no pocket 3"):
        viz.show(res, PID, "molecule2", pocket=3, backend=backend)
    with pytest.raises(KeyError, match="has no mode 10 \\(modes 1..9\\)"):
        viz.show(res, PID, "molecule2", pocket=1, mode=10, backend=backend)
    with pytest.raises(KeyError, match="protein 'nope'"):
        viz.show_modes(res, "nope", "molecule2", 1, backend=backend)
    with pytest.raises(KeyError, match="'molecule10' has no docked pose"):
        viz.show_modes(res, PID, "molecule10", None, backend=backend)
    with pytest.raises(KeyError, match="no ligand 'nope'"):
        viz.to_html(res, PID, "nope", backend=backend)
    with pytest.raises(ValueError, match="mode needs a pocket"):
        viz.show(res, PID, "molecule2", mode=2, backend=backend)
    with pytest.raises(ValueError, match="pocket_radius"):
        viz.show(res, PID, "molecule2", pocket_radius=-1, backend=backend)
    with pytest.raises(ValueError, match="ui must be one of"):
        viz.show_modes(res, PID, "molecule2", 1, backend=backend, ui="other")
    with pytest.raises(TypeError, match="protein must be a protein_id string"):
        viz.show(res, 0, "molecule2", backend=backend)
    with pytest.raises(FileNotFoundError, match="MANIFEST.json"):
        viz.show(tmp_path, PID, "molecule2", backend=backend)
    assert sorted(os.listdir(tmp_path)) == []


def test_unknown_backend(res):
    with pytest.raises(ValueError, match="backend must be one of"):
        viz.show(res, PID, "molecule2", backend="nglview")
    with pytest.raises(ValueError, match="backend must be one of"):
        viz.to_html(res, PID, "molecule2", backend="")


# --- structures and residues ---------------------------------------------------------------------------------------

PDB_LINES = """\
ATOM      1  N   GLY A  52      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  GLY A  52      11.000  10.000  10.000  1.00  0.00           C
ATOM      3  H   GLY A  52       9.500  10.000  10.000  1.00  0.00           H
ATOM      4  CA  ALA A  52A     13.000  10.000  10.000  1.00  0.00           C
ATOM      5  CA  SER B   7      30.000  10.000  10.000  1.00  0.00           C
HETATM    6  O   HOH A 201      10.500  10.500  10.000  1.00  0.00           O
ATOM      7  CA  LEU    99      12.000  12.000  10.000  1.00  0.00
ENDMDL
ATOM      8  CA  GLY A  60      10.000  10.000  10.000  1.00  0.00           C
"""


def test_near_residues():
    atoms = viz._pdb_atoms(PDB_LINES)
    # hydrogens, waters and the second model are left out; no element column: from the atom name
    assert list(atoms.name) == ["GLY", "GLY", "ALA", "SER", "LEU"]
    assert list(atoms.element) == ["N", "C", "C", "C", "C"]
    near = viz.near_residues(atoms, np.array([[10.0, 11.0, 10.0]]), 3.5)
    assert [(r.chain, r.seq, r.icode, r.name) for r in near] == [("", 99, "", "LEU"), ("A", 52, "", "GLY"),
                                                                  ("A", 52, "A", "ALA")]
    assert [r.distance for r in near] == [2.24, 1.0, 3.16]  # closest atom of each residue
    assert [r.label for r in near] == ["LEU99", "A:GLY52", "A:ALA52A"]
    assert [r.name for r in viz.near_residues(atoms, np.array([[10.0, 11.0, 10.0]]), 3.0)] == ["LEU", "GLY"]
    assert viz.near_residues(atoms, np.empty((0, 3)), 5.0) == []
    assert viz.near_residues(atoms, np.array([[100.0, 0.0, 0.0]]), 5.0) == []


@needs_mvs
def test_residue_selectors():
    def expr(r):
        return viz._residue_expression(r).model_dump(exclude_none=True)

    assert expr(viz.Residue("A", 52, "", "GLY", 1.0)) == {"auth_asym_id": "A", "auth_seq_id": 52}
    assert expr(viz.Residue("", 99, "", "LEU", 1.0)) == {"auth_seq_id": 99}
    assert expr(viz.Residue("B", 52, "A", "ALA", 1.0)) == {"auth_asym_id": "B", "auth_seq_id": 52,
                                                          "pdbx_PDB_ins_code": "A"}
    # two-letter elements: as PDB files (CL) and SDF/mmCIF files (Cl) spell them
    assert [e.type_symbol for e in viz._type_symbols("Cl")] == ["CL", "Cl"]
    assert [e.type_symbol for e in viz._type_symbols("N")] == ["N"]


def test_pdbqt_to_pdb():
    pdbqt_text = ("ATOM      1  N   GLY A   1      1.118   1.446   5.798  0.00  0.00    +0.000 NA\n"
                  "ATOM      2 HN   GLY A   1      1.000   1.000   5.000  0.00  0.00    +0.000 HD\n"
                  "ATOM      3 CL1  LIG A   2      2.000   2.000   2.000  0.00  0.00    +0.000 Cl\n"
                  "ATOM      4  SG  CYS A   3      3.000   3.000   3.000  0.00  0.00    +0.000 SA\n"
                  "TER\nREMARK ignored\n")
    out = viz.pdbqt_to_pdb(pdbqt_text).splitlines()
    assert [line[76:78] for line in out[:4]] == [" N", " H", "CL", " S"]
    assert all(len(line) == 78 for line in out[:4])
    assert out[0][:66] == pdbqt_text.splitlines()[0][:66]
    assert out[4:] == ["TER", "END"]
    atoms = viz._pdb_atoms("\n".join(out))
    assert list(atoms.element) == ["N", "Cl", "S"]


def test_pdbqt_metals_keep_their_element():
    # OpenBabel writes elements outside the AutoDock types as their symbol: Hg is mercury (not a hidden hydrogen),
    # Pt platinum (not phosphorus), Cd cadmium (not carbon)
    pdbqt_text = "".join(
        f"ATOM   {n:4d} {name:<4} {res:>3} A {n:3d}      {10.0 + n:6.3f}  10.000  10.000  1.00  0.00    +0.000 {t}\n"
        for n, (name, res, t) in enumerate((("HG", "HG", "Hg"), ("PT", "PT", "Pt"), ("CD", "CD", "Cd"),
                                            ("AS", "CAC", "As"), ("ZN", "ZN", "Zn"), ("C1", "LIG", "A")), 1))
    out = viz.pdbqt_to_pdb(pdbqt_text)
    assert [line[76:78] for line in out.splitlines()[:6]] == ["HG", "PT", "CD", "AS", "ZN", " C"]
    atoms = viz._pdb_atoms(out)
    assert list(atoms.element) == ["Hg", "Pt", "Cd", "As", "Zn", "C"]
    near = viz.near_residues(atoms, np.array([[11.0, 10.0, 10.0]]), 0.5)
    assert [r.name for r in near] == ["HG"]  # the ion is a pocket residue
    elements, _ = viz._ligand_atoms(pdbqt_text.encode(), "pdbqt")
    assert elements == ["As", "C", "Cd", "Hg", "Pt", "Zn"]
    assert all(e in viz.ELEMENT_COLORS for e in elements if e != "C")  # carbon: the representation's colour


def test_cif_atoms():
    gemmi = pytest.importorskip("gemmi")
    st = gemmi.read_pdb_string(PDB_LINES.split("ENDMDL")[0])
    st.setup_entities()
    cif = st.make_mmcif_document().as_string()
    pdb_atoms, cif_atoms = viz._pdb_atoms(PDB_LINES), viz._cif_atoms(cif)
    for name in ("chain", "seq", "icode", "name", "element"):
        assert list(getattr(cif_atoms, name)) == list(getattr(pdb_atoms, name)), name
    assert np.allclose(cif_atoms.xyz, pdb_atoms.xyz)
    # quoted values and a second model
    text = """data_x
loop_
_atom_site.group_PDB
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
HETATM C "C1'" NAG A 5 ? 1.0 2.0 3.0 1
HETATM C 'C2"' NAG A 5 ? 1.5 2.0 3.0 1
ATOM C CA GLY A 6 B 4.0 5.0 6.0 1
ATOM C CA GLY A 6 ? 9.0 9.0 9.0 2
#
"""
    atoms = viz._cif_atoms(text)
    assert list(atoms.seq) == [5, 5, 6] and list(atoms.icode) == ["", "", "B"]
    assert atoms.xyz.tolist() == [[1.0, 2.0, 3.0], [1.5, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert len(viz._cif_atoms("data_x\n_cell.length_a 1\n").xyz) == 0


# --- table ---------------------------------------------------------------------------------------------------------

def test_table(res, results_dir):
    df = viz.table(results_dir)
    assert list(df.columns) == ["protein_id", "ligand_id", "smiles_docked", "vina", "best_pocket", "best_mode",
                                "ligand_efficiency"]
    assert df["ligand_id"].tolist() == ["molecule2", "molecule5"]  # molecule10 has no pose
    assert df["vina"].tolist() == pytest.approx([-6.3, -5.2])
    lig = res.ligands().set_index("ligand_id")
    assert df["smiles_docked"].tolist() == lig.loc[["molecule2", "molecule5"], "smiles_docked"].tolist()
    assert df["ligand_efficiency"].tolist() == pytest.approx([6.3 / 26, 5.2 / 34])
    assert df[["best_pocket", "best_mode"]].values.tolist() == [[2, 1], [1, 1]]
    assert viz.table(res, top=1)["ligand_id"].tolist() == ["molecule2"]
    assert viz.table(res, top=None, protein=PID).equals(df)
    assert viz.table(res, top=0).empty
    with pytest.raises(KeyError, match="protein 'nope'"):
        viz.table(res, protein="nope")
    with pytest.raises(ValueError, match="top must be"):
        viz.table(res, top=-1)


def test_table_computes_ligand_efficiency(res, monkeypatch):
    real = Results.pairs

    def pairs_without(self, protein=None, columns=None):
        return real(self, protein, columns).drop(columns=["ligand_efficiency", "pb_valid"], errors="ignore")

    monkeypatch.setattr(Results, "pairs", pairs_without)
    df = viz.table(res)
    assert df["ligand_efficiency"].tolist() == pytest.approx([6.3 / 26, 5.2 / 34])


@pytest.mark.skipif(not hasattr(tables, "POSEBUSTERS") or not hasattr(Results, "posebusters"),
                    reason="needs the PoseBusters table (Phase 2, job P)")
def test_table_with_posebusters(results_dir, tmp_path):
    path = copy_results(results_dir, tmp_path / "results")
    rows = []
    for idx, lid, k, status, valid in ((0, "molecule2", 2, "valid", True), (1, "molecule5", 1, "invalid", False),
                                      (1, "molecule5", 2, "valid", True)):  # not molecule5's best pocket
        rows.append({"protein_id": PID, "uniprot": None, "ligand_idx": idx, "ligand_id": lid, "pocket": k, "mode": 1,
                     "pb_config": "dock", "pb_version": "0", "pb_status": status, "pb_valid": valid, "pb_mask": 0,
                     "pb_na_mask": 0, "failed_checks": None if valid else "tetrahedral_chirality",
                     "n_heavy_docked": 26, "runtime_s": 1.0, "error": None})
    tables.write_table(rows, tables.POSEBUSTERS, os.path.join(path, "tables", "posebusters", f"{PID}.parquet"))
    df = viz.table(path)
    assert list(df.columns)[-2:] == ["pb_status", "pb_valid"]
    assert df["pb_status"].tolist() == ["valid", "invalid"]
    assert df["pb_valid"].tolist() == [True, False]
    assert isinstance(df, pd.DataFrame) and len(df) == 2


def test_no_files_in_package_dir():
    # the views keep everything in memory; nothing under pocketvina/ is written (also checked for the session)
    pkg = os.path.dirname(viz.__file__)
    assert not [n for n in os.listdir(pkg) if n.endswith((".html", ".mvsj", ".mvsx", ".sdf"))]

