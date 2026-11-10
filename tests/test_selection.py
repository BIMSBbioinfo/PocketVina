"""pocketvina.selection: select() on Results and PocketVinaGPU, and the Selection's summary, export and bust.

Results used: the real run of tests/data imported with io.importer (molecule2 and molecule5 docked into pockets 1 and
2 of a PDBQT receptor, 9 modes each; molecule10 without a pose; 22 P2Rank pockets). molecule2's best pose is pocket 2,
mode 1 (-6.3 kcal/mol). Expected poses come from the stored tables (modes, pairs) and the engine output
(Results.pose_block); PoseBusters results from PoseBusters' own path-based call (test_bust.path_based).
"""
import hashlib
import io
import os
import re
from collections import namedtuple
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from pocketvina import PocketVinaConfig, PocketVinaGPU, pdbqt
from pocketvina.io.importer import import_legacy
from pocketvina.io.results import Results, short_floats
from pocketvina.selection import Selection, name_part, pose_name, select

PID = "AF-Q96LA5-F1-model_v4"
LIG = "molecule2"
STEM = f"PV_{PID}_{LIG}"


@pytest.fixture(scope="module")
def results_dir(real_run, tmp_path_factory) -> str:
    return import_legacy(str(real_run.root), str(tmp_path_factory.mktemp("selection") / "results"),
                         str(real_run.smiles_csv))


@pytest.fixture(scope="module")
def res(results_dir):
    with Results(results_dir) as r:
        yield r


def digests(root) -> dict:
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def modes_of(res, ligand="molecule2"):
    """{(pocket, mode): (affinity, rmsd_lb, rmsd_ub)} of a ligand, as printed by the engine."""
    m = res.modes(PID)
    m = m[m["ligand_idx"] == res.ligand_idx(ligand)]
    return {(int(k), int(n)): tuple(short_floats([a, lb, ub])) for k, n, a, lb, ub in
            m[["pocket", "mode", "affinity", "rmsd_lb", "rmsd_ub"]].itertuples(index=False)}


# --- select --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pocket, mode, poses, name", [
    (None, 1, [(2, 1)], "top"),
    (None, 3, [(2, 3)], "pk2_m03"),
    (1, 1, [(1, 1)], "pk1_m01"),
    (2, 9, [(2, 9)], "pk2_m09"),
    (2, "all", [(2, m) for m in range(1, 10)], "pk2_modes"),
    ("all", 1, [(1, 1), (2, 1)], "pocketall_top"),
    ("all", 4, [(1, 4), (2, 4)], "pocketall_m04"),
    ("all", "all", [(k, m) for k in (1, 2) for m in range(1, 10)], "pocketall_modes"),
])
def test_select_poses_and_names(res, pocket, mode, poses, name):
    """Every argument combination: the poses picked (pocket, mode, the engine's scores, the model text) and the
    file name stem (Atlas IDs with PV_)."""
    sel = res.select(PID, LIG, pocket=pocket, mode=mode)
    assert isinstance(sel, Selection) and (sel.protein, sel.ligand, sel.ligand_idx) == (PID, LIG, 0)
    assert (sel.pocket, sel.mode) == (pocket, mode)
    assert (sel.best_pocket, sel.best_mode, sel.vina) == (2, 1, -6.3) and sel.docked_pockets == [1, 2]
    assert sel.smiles == res.ligands().set_index("ligand_id").loc[LIG, "smiles_input"]
    df = sel.poses
    assert list(df.columns) == ["pocket", "mode", "vina", "rmsd_lb", "rmsd_ub"]
    assert list(zip(df["pocket"], df["mode"])) == poses
    scores = modes_of(res)
    assert [tuple(r) for r in df[["vina", "rmsd_lb", "rmsd_ub"]].itertuples(index=False)] == [scores[p] for p in poses]
    for p, (k, m) in zip(sel._poses, poses):
        assert p.model.decode() == res.pose_model(PID, LIG, k, m)
    assert sel.name == f"{STEM}_{name}"


def test_select_ligand_by_index_and_from_pocketvinagpu(res, results_dir, tmp_path):
    assert res.select(PID, 0).poses.equals(res.select(PID, LIG).poses)
    # PocketVinaGPU of an atlas config, in a new session (the folder holds the results)
    config = PocketVinaConfig(protein_paths=str(tmp_path / "proteins.txt"), smiles_file=str(tmp_path / "ligands.csv"),
                              output_dir=results_dir, output_format="atlas")
    sel = PocketVinaGPU(config).select(protein=PID, ligand=LIG, pocket=1, mode=2)
    assert (sel.protein, sel.ligand, list(sel.poses["mode"]), sel.name) == (PID, LIG, [2], f"{STEM}_pk1_m02")
    # the files format has no results to select from
    files = PocketVinaGPU(PocketVinaConfig(protein_paths=str(tmp_path / "proteins.txt"),
                                           smiles_file=str(tmp_path / "ligands.csv"), output_dir=str(tmp_path / "out")))
    for call in (lambda: files.select(PID, LIG), lambda: files.show()):
        with pytest.raises(ValueError, match='set output_format="atlas"'):
            call()


@pytest.mark.parametrize("args, err, match", [
    (("nope", LIG), KeyError, rf"no protein 'nope' in .*; proteins in these results: '{PID}'"),
    ((PID, "nope"), KeyError, r"no ligand 'nope' in .*; ligands \(as in best_scores\(\)\): 'molecule2', 'molecule5', "
                              r"'molecule10'"),
    ((7, LIG), TypeError, "protein must be a protein id"),
    ((PID, "molecule10"), ValueError, r"ligand 'molecule10' has no docked pose in protein .* \(the docking gave no "
                                      r"pose in any pocket\)"),
    ((PID, LIG, 3), ValueError, r"pocket 3 has no poses of ligand 'molecule2' .*pockets with poses: 1, 2"),
    ((PID, LIG, 2, 10), ValueError, "mode 10 is out of range: pocket 2 has modes 1 to 9"),
    ((PID, LIG, None, 10), ValueError, "mode 10 is out of range: pocket 2 has modes 1 to 9"),
    ((PID, LIG, "all", 10), ValueError, r"no pocket has that many modes .*\(modes per pocket: 1: 9, 2: 9\)"),
    ((PID, LIG, 0), ValueError, "pocket must be a pocket number"),
    ((PID, LIG, "best"), ValueError, "pocket must be a pocket number"),
    ((PID, LIG, True), ValueError, "pocket must be a pocket number"),
    ((PID, LIG, 1.0), ValueError, "pocket must be a pocket number"),
    ((PID, LIG, None, 0), ValueError, "mode must be a mode number"),
    ((PID, LIG, None, "best"), ValueError, "mode must be a mode number"),
    ((PID, LIG, None, 1.5), ValueError, "mode must be a mode number"),
])
def test_select_errors(res, args, err, match):
    with pytest.raises(err, match=match):
        res.select(*args)


def test_select_a_ligand_that_could_not_be_prepared(res, monkeypatch):
    """No pose because the ligand's preparation failed (e.g. an invalid SMILES): the error says so."""
    i = res.ligand_idx("molecule10")
    real = Results.ligands

    def ligands(self, columns=None):
        df = real(self, columns)
        failed = df["ligand_idx"] == i
        for col, value in (("status", "invalid_smiles"), ("error", "SMILES syntax error")):
            if col in df:
                df.loc[failed, col] = value
        return df

    monkeypatch.setattr(Results, "ligands", ligands)
    with pytest.raises(ValueError, match=r"ligand 'molecule10' has no docked pose in protein .* \(it could not be "
                                         r"prepared for docking: invalid_smiles, SMILES syntax error\)"):
        res.select(PID, "molecule10")


def test_unknown_ids_list_a_few_valid_ones(res, monkeypatch):
    many = [f"protein{n}" for n in range(8)]
    monkeypatch.setattr(Results, "proteins", lambda self: many)
    with pytest.raises(KeyError, match=r"'protein0', 'protein1', 'protein2', 'protein3', 'protein4', \.\.\. \(8 in "
                                       r"all\)"):
        select(res, "nope", LIG)


def test_pose_names():
    assert pose_name("1fbl", "molecule1", "pk2", "m03") == "PV_1fbl_molecule1_pk2_m03"
    assert pose_name("1fbl", "imatinib mesylate (salt)/2", "top") == "PV_1fbl_imatinib_mesylate_salt_2_top"
    assert name_part("x" * 150) == "x" * 100 and name_part("...") == "x" and name_part("é") == "x"


# --- summary -------------------------------------------------------------------------------------------------------

def test_summary(res):
    sel = res.select(PID, LIG, pocket="all", mode="all")
    text = repr(sel)
    assert text.splitlines()[0] == (f"Selection PV_{PID}_{LIG}: protein {PID}, ligand {LIG}, every mode in each "
                                    "pocket: 18 poses")
    assert sel.smiles in text and "   1   " in text and "-5.6  1-9" in text and "2 *" in text and "-6.3  1-9" in text
    page = sel._repr_html_()
    for part in (f"PV_{PID}_{LIG}", sel.smiles, "<td style='text-align:right'>2 &#9733;</td>", "-6.3", "1-9",
                 "pair.show(), pair.bust(), pair.export()"):
        assert part in page, part
    one = res.select(PID, LIG, pocket=1, mode=3)
    assert "pocket 1, mode 3: 1 pose" in repr(one) and "  -4.6  3" in repr(one)
    assert "pocket 2, the highest-affinity pose: 1 pose" in repr(res.select(PID, LIG))
    assert "pocket 2, mode 3: 1 pose" in repr(res.select(PID, LIG, mode=3))


@pytest.mark.parametrize("pocket, mode", [("all", "all"), (None, 1), (1, 3)])
def test_summary_table_is_aligned(res, pocket, mode):
    """The numbers end under the end of the score header, and the modes start under "modes"."""
    header, *rows = repr(res.select(PID, LIG, pocket=pocket, mode=mode)).splitlines()[2:]
    assert header == "  pocket  lowest Vina score (kcal/mol)  modes"
    cut = header.index("modes")
    assert rows
    for row in rows:
        assert row[cut - 3].isdigit() and row[cut - 2:cut] == "  " and row[cut] != " ", row


@pytest.mark.parametrize("pocket", [None, 1, "all"])
@pytest.mark.parametrize("mode", [1, 3, "all"])
def test_summary_says_highest_affinity_not_best(res, pocket, mode):
    """What a reader sees says "highest-affinity" (as the Atlas website does), never "best" before pose, mode, score,
    Vina, pocket or drug. Field names such as best_pocket keep "best"."""
    sel = res.select(PID, LIG, pocket=pocket, mode=mode)
    for shown in (repr(sel), sel._repr_html_()):
        assert "lowest Vina score (kcal/mol)" in shown
        assert not re.search(r"best\W+(pose|mode|score|vina|pocket|drug)", shown, re.IGNORECASE), shown
    assert "the pocket of the pair's highest-affinity pose" in sel._repr_html_()


# --- export --------------------------------------------------------------------------------------------------------

def sdf_records(path) -> list:
    return list(Chem.ForwardSDMolSupplier(io.BytesIO(Path(path).read_bytes()), removeHs=False))


def test_export_sdf_default_name_and_properties(res, results_dir, tmp_path, monkeypatch):
    """One file in the working folder, named after the selection; one record per pose with the docking scores as
    properties; the best pose's record is the stored best-pose SDF (same molecule, same coordinates)."""
    before = digests(results_dir)
    monkeypatch.chdir(tmp_path)
    sel = res.select(PID, LIG)
    path = sel.export()
    assert path == str(tmp_path / f"{STEM}_top.sdf") and os.listdir(tmp_path) == [f"{STEM}_top.sdf"]
    (mol,) = sdf_records(path)
    assert mol.GetProp("_Name") == LIG
    assert {k: mol.GetProp(k) for k in ("protein_id", "ligand_id", "ligand_idx", "pocket", "mode", "affinity",
                                        "rmsd_lb", "rmsd_ub")} == {
        "protein_id": PID, "ligand_id": LIG, "ligand_idx": "0", "pocket": "2", "mode": "1", "affinity": "-6.3",
        "rmsd_lb": "0.0", "rmsd_ub": "0.0"}
    stored = Chem.MolFromMolBlock(res.top_sdf(PID, LIG), removeHs=False)
    assert Chem.MolToSmiles(mol) == Chem.MolToSmiles(stored)
    assert mol.GetConformer().GetPositions().round(3).tolist() == stored.GetConformer().GetPositions().round(3).tolist()

    every = res.select(PID, LIG, pocket="all", mode="all").export()
    mols = sdf_records(every)
    assert os.path.basename(every) == f"{STEM}_pocketall_modes.sdf" and len(mols) == 18
    scores = modes_of(res)
    assert [(int(m.GetProp("pocket")), int(m.GetProp("mode")), float(m.GetProp("affinity"))) for m in mols] == [
        (k, n, scores[(k, n)][0]) for k in (1, 2) for n in range(1, 10)]
    assert digests(results_dir) == before  # nothing written into the results


def test_export_pdbqt(res, tmp_path):
    """The engine's models, numbered 1, 2, ... in the file, each with a REMARK naming its pocket and mode."""
    sel = res.select(PID, LIG, pocket="all", mode=2)
    path = sel.export(tmp_path, fmt="pdbqt")
    assert path == str(tmp_path / f"{STEM}_pocketall_m02.pdbqt")
    data = Path(path).read_bytes()
    models = pdbqt.parse_modes(data)
    assert [m.mode for m in models] == [1, 2]
    assert [m.affinity for m in models] == [modes_of(res)[(k, 2)][0] for k in (1, 2)]
    for n, (m, k) in enumerate(zip(models, (1, 2)), 1):
        text = data[m.raw_offset:m.raw_offset + m.raw_length].decode()
        original = res.pose_model(PID, LIG, k, 2)
        lines = text.splitlines()
        assert lines[0] == f"MODEL {n:>8}"
        assert lines[1] == f"REMARK POCKETVINA protein {PID} ligand {LIG} pocket {k} mode 2"
        assert lines[2:] == original.splitlines()[1:]
    # a whole pocket: every mode, as the engine wrote them
    path = res.select(PID, LIG, pocket=1, mode="all").export(tmp_path / "modes.pdbqt", fmt="pdbqt")
    got = pdbqt.parse_modes(Path(path).read_bytes())
    assert [(m.mode, m.affinity) for m in got] == [(n, modes_of(res)[(1, n)][0]) for n in range(1, 10)]


def test_export_paths_and_errors(res, tmp_path):
    sel = res.select(PID, LIG, pocket=1, mode=3)
    assert sel.export(tmp_path / "sub" / "pose.sdf") == str(tmp_path / "sub" / "pose.sdf")
    assert sel.export(str(tmp_path)) == str(tmp_path / f"{STEM}_pk1_m03.sdf")
    with pytest.raises(ValueError, match=r"ends in \.pdbqt but fmt='sdf': pass fmt='pdbqt'"):
        sel.export(tmp_path / "pose.pdbqt")
    with pytest.raises(ValueError, match="fmt must be one of"):
        sel.export(tmp_path / "pose.mol2", fmt="mol2")
    assert not (tmp_path / "pose.pdbqt").exists() and not (tmp_path / "pose.mol2").exists()


def test_export_protein(res, results_dir, tmp_path, monkeypatch):
    """The protein the poses were docked into as PDB (the stored PDBQT input converted): PV_<protein>.pdb in the
    working folder by default, or the path / folder given; the results folder is not changed."""
    before = digests(results_dir)
    sel = res.select(PID, LIG)
    monkeypatch.chdir(tmp_path)
    path = sel.export_protein()
    assert path == str(tmp_path / f"PV_{PID}.pdb")
    text = Path(path).read_text()
    atoms = [line for line in text.splitlines() if line.startswith(("ATOM", "HETATM"))]
    assert atoms and not any(line[17:20] == "HOH" for line in atoms)
    assert sel.export_protein(tmp_path / "sub" / "r.pdb") == str(tmp_path / "sub" / "r.pdb")
    assert sel.export_protein(tmp_path / "sub") == str(tmp_path / "sub" / f"PV_{PID}.pdb")
    assert digests(results_dir) == before


def test_export_with_the_protein(res, results_dir, tmp_path, monkeypatch):
    """export(protein=True): the poses as export() writes them and the protein as export_protein() writes it, next to
    the poses; returns [poses, protein]. export(ligand=False, protein=True): only the protein, as export_protein()
    (the same path and text); returns its path. A call without both is an error, and nothing is written."""
    before = digests(results_dir)
    sel = res.select(PID, LIG)
    protein_text = Path(sel.export_protein(tmp_path / "expected.pdb")).read_text()
    poses_bytes = Path(sel.export(tmp_path / "expected.sdf")).read_bytes()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    paths = sel.export(protein=True)
    assert paths == [str(work / f"{STEM}_top.sdf"), str(work / f"PV_{PID}.pdb")]
    assert sorted(os.listdir(work)) == sorted(os.path.basename(p) for p in paths)
    assert Path(paths[0]).read_bytes() == poses_bytes and Path(paths[1]).read_text() == protein_text
    # a file or a folder for the poses: the protein goes next to them; fmt is only for the poses
    assert sel.export(work / "sub" / "pose.pdbqt", fmt="pdbqt", protein=True) == [
        str(work / "sub" / "pose.pdbqt"), str(work / "sub" / f"PV_{PID}.pdb")]
    assert sel.export(str(work / "sub"), protein=True) == [
        str(work / "sub" / f"{STEM}_top.sdf"), str(work / "sub" / f"PV_{PID}.pdb")]
    # the protein only: export_protein()'s path and text; no poses
    only = tmp_path / "only"
    only.mkdir()
    monkeypatch.chdir(only)
    path = sel.export(ligand=False, protein=True)
    assert path == str(only / f"PV_{PID}.pdb") and os.listdir(only) == [f"PV_{PID}.pdb"]
    assert Path(path).read_text() == protein_text
    assert sel.export(only / "r.pdb", ligand=False, protein=True) == str(only / "r.pdb")
    (only / "sub").mkdir()
    assert sel.export(only / "sub", ligand=False, protein=True) == str(only / "sub" / f"PV_{PID}.pdb")
    # the default call is unchanged: one path, a str
    assert sel.export(ligand=True, protein=False) == str(only / f"{STEM}_top.sdf")

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    with pytest.raises(ValueError, match=r"ligand=False and protein=False: nothing to write \(pair\.export\(\) "
                                         r"writes the poses, pair\.export\(protein=True\) the protein too"):
        sel.export(ligand=False)
    with pytest.raises(ValueError, match="protein must be True or False, got '1fbl'"):
        sel.export(protein="1fbl")
    with pytest.raises(ValueError, match="ligand must be True or False, got None"):
        sel.export(ligand=None, protein=True)
    with pytest.raises(ValueError, match=r"fmt must be one of \('sdf', 'pdbqt'\), got 'pdb' \(fmt is the format of "
                                         r"the poses; the protein is always PDB\)"):
        sel.export(ligand=False, protein=True, fmt="pdb")
    with pytest.raises(TypeError):
        sel.export(None, "sdf", True)  # protein= and ligand= are named
    with pytest.raises(ValueError, match=re.escape(f"PV_{PID}.pdb is the protein's file: give the poses another name, "
                                                   f"e.g. {STEM}_top.sdf")):
        sel.export(f"PV_{PID}.pdb", protein=True)  # the poses would be written over by the protein
    assert os.listdir(empty) == []
    assert digests(results_dir) == before


def test_export_into_a_new_folder(res, tmp_path, monkeypatch):
    """A path that ends in "/" is a folder, made if needed, as one that exists (it was a file named "new"): export(),
    export(protein=True), export(ligand=False, protein=True) and export_protein()."""
    sel = res.select(PID, LIG)
    monkeypatch.chdir(tmp_path)
    assert sel.export("new/") == str(tmp_path / "new" / f"{STEM}_top.sdf")
    assert sel.export("both/", fmt="pdbqt", protein=True) == [str(tmp_path / "both" / f"{STEM}_top.pdbqt"),
                                                            str(tmp_path / "both" / f"PV_{PID}.pdb")]
    assert sel.export("only/", ligand=False, protein=True) == str(tmp_path / "only" / f"PV_{PID}.pdb")
    assert sel.export_protein("protein/") == str(tmp_path / "protein" / f"PV_{PID}.pdb")
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        f"both/PV_{PID}.pdb", f"both/{STEM}_top.pdbqt", f"new/{STEM}_top.sdf", f"only/PV_{PID}.pdb",
        f"protein/PV_{PID}.pdb"]


def test_export_a_path_without_an_extension_is_a_folder(res, tmp_path, monkeypatch):
    """As the CLI's --out: a path without an extension is a folder, made if needed (it was a file named "poses"):
    export(), export(protein=True), export(ligand=False, protein=True) and export_protein(). "my.poses" has an
    extension; "results_v2" and ".poses" (a leading dot only) have none."""
    sel = res.select(PID, LIG)
    monkeypatch.chdir(tmp_path)
    assert sel.export("poses") == str(tmp_path / "poses" / f"{STEM}_top.sdf")
    assert sel.export("a/b/c", fmt="pdbqt") == str(tmp_path / "a" / "b" / "c" / f"{STEM}_top.pdbqt")
    assert sel.export(tmp_path / "both", protein=True) == [str(tmp_path / "both" / f"{STEM}_top.sdf"),
                                                           str(tmp_path / "both" / f"PV_{PID}.pdb")]
    assert sel.export("results_v2", ligand=False, protein=True) == str(tmp_path / "results_v2" / f"PV_{PID}.pdb")
    assert sel.export_protein("protein") == str(tmp_path / "protein" / f"PV_{PID}.pdb")
    assert sel.export(".poses") == str(tmp_path / ".poses" / f"{STEM}_top.sdf")
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()) == [
        f".poses/{STEM}_top.sdf", f"a/b/c/{STEM}_top.pdbqt", f"both/PV_{PID}.pdb", f"both/{STEM}_top.sdf",
        f"poses/{STEM}_top.sdf", f"protein/PV_{PID}.pdb", f"results_v2/PV_{PID}.pdb"]
    # a file of the right format, in any case: as before
    assert sel.export("P.SDF") == str(tmp_path / "P.SDF") and sel.export_protein("R.PDB") == str(tmp_path / "R.PDB")


def test_export_a_file_needs_the_extension_of_its_format(res, tmp_path, monkeypatch):
    """A file for the poses ends in .sdf (fmt="sdf") or .pdbqt (fmt="pdbqt"), the protein's in .pdb: any other
    extension is an error before anything is written (it was written: SDF poses into "abl1.pdb"); the protein is not
    read."""
    sel = res.select(PID, LIG)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sel._source, "protein_pdb", lambda protein: pytest.fail("the protein was read"))
    for args, kwargs in ((("abl1.pdb",), {}), (("abl1.pdb",), {"protein": True}), (("my.poses",), {}),
                         (("x.sdf.gz",), {}), (("run_2026.11",), {})):
        with pytest.raises(ValueError, match=re.escape(f"{tmp_path / args[0]}: poses are written as SDF; use a .sdf "
                                                       "name, or fmt='pdbqt' for .pdbqt, or a / at the end for a "
                                                       "folder")):
            sel.export(*args, **kwargs)
    with pytest.raises(ValueError, match=re.escape(f"{tmp_path / 'x.txt'}: poses are written as PDBQT; use a .pdbqt "
                                                   "name, or fmt='sdf' for .sdf")):
        sel.export("x.txt", fmt="pdbqt", protein=True)
    for call in (lambda: sel.export_protein("x.sdf"), lambda: sel.export_protein("x.cif"),
                 lambda: sel.export("x.sdf", ligand=False, protein=True)):
        with pytest.raises(ValueError, match=r"/x\.(sdf|cif): the protein is written as PDB; use a \.pdb name, or a "
                                             r"/ at the end for a folder$"):
            call()
    assert os.listdir(tmp_path) == []
    # a folder whose name has a dot: with a "/" at the end, as the message says
    assert sel.export("run_2026.11/") == str(tmp_path / "run_2026.11" / f"{STEM}_top.sdf")


def test_export_protein_with_no_path_into_a_folder_of_its_name(res, tmp_path, monkeypatch):
    """With no path, a folder named PV_<protein>.pdb in the working folder is written into, as in 0.3.0:
    export_protein() and export(ligand=False, protein=True) write PV_<protein>.pdb/PV_<protein>.pdb."""
    sel = res.select(PID, LIG)
    monkeypatch.chdir(tmp_path)
    (tmp_path / f"PV_{PID}.pdb").mkdir()
    inside = tmp_path / f"PV_{PID}.pdb" / f"PV_{PID}.pdb"
    assert sel.export_protein() == str(inside) and inside.read_text() == sel._source.protein_pdb(PID)
    inside.unlink()
    assert sel.export(ligand=False, protein=True) == str(inside) and inside.is_file()
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == [
        f"PV_{PID}.pdb", f"PV_{PID}.pdb/PV_{PID}.pdb"]


def test_export_into_a_folder_that_is_a_file(res, tmp_path, monkeypatch):
    """A path without an extension is a folder: a file of that name, e.g. "poses" written by 0.3.0, is an error that
    says so (it was overwritten), before anything is written or the protein read; the file is kept."""
    sel = res.select(PID, LIG)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "poses").write_text("old\n")
    monkeypatch.setattr(sel._source, "protein_pdb", lambda protein: pytest.fail("the protein was read"))
    for call, ext in ((lambda: sel.export("poses"), ".sdf"),
                      (lambda: sel.export(tmp_path / "poses", fmt="pdbqt", protein=True), ".pdbqt"),
                      (lambda: sel.export_protein("poses"), ".pdb"),
                      (lambda: sel.export("poses", ligand=False, protein=True), ".pdb")):
        with pytest.raises(ValueError, match=re.escape(f"{tmp_path / 'poses'} is a file, but a path without an "
                                                       f"extension is a folder: use e.g. poses{ext} for a file")):
            call()
    assert os.listdir(tmp_path) == ["poses"] and (tmp_path / "poses").read_text() == "old\n"


def test_export_leaves_out_poses_that_do_not_convert(res, tmp_path, monkeypatch, caplog):
    from pocketvina import convert

    real = convert.pose_to_sdf
    monkeypatch.setattr(convert, "pose_to_sdf", lambda model, smiles, props: (
        (None, "convert_failed") if props["pocket"] == 1 else real(model, smiles, props)))
    sel = res.select(PID, LIG, pocket="all")
    mols = sdf_records(sel.export(tmp_path))
    assert [m.GetProp("pocket") for m in mols] == ["2"] and "pocket 1 mode 1 does not convert" in caplog.text
    with pytest.raises(ValueError, match="no selected pose converts to SDF; use fmt='pdbqt'"):
        res.select(PID, LIG, pocket=1).export(tmp_path / "none.sdf")


# --- PoseBusters ---------------------------------------------------------------------------------------------------

def test_bust_in_memory(res, results_dir, tmp_path, monkeypatch):
    """PoseBusters on the selected poses: the checks of PoseBusters' own path-based call on the same poses (the stored
    SDF of the best pose, converted modes otherwise), nothing written, the stored results unchanged."""
    pytest.importorskip("posebusters")
    from pocketvina import bust
    from test_bust import path_based

    before = digests(results_dir)
    monkeypatch.chdir(tmp_path)
    sel = res.select(PID, LIG, pocket="all")
    df = sel.bust(workers=2)
    names = bust.check_names("dock")
    assert list(df.columns) == ["pocket", "mode", "vina", "pb_valid", "pb_status", "failed_checks", *names]
    assert df[["pocket", "mode", "vina"]].values.tolist() == [[1, 1, -5.6], [2, 1, -6.3]]
    assert str(df["pb_valid"].dtype) == "boolean" and all(str(df[n].dtype) == "boolean" for n in names)
    Row = namedtuple("Row", "ligand_idx ligand_id pocket mode")
    assert os.listdir(tmp_path) == []  # nothing written
    (tmp_path / "expected").mkdir()
    expected = path_based(res, PID, [Row(0, LIG, 1, 1), Row(0, LIG, 2, 1)], tmp_path / "expected")
    for (_, got), (_, want) in zip(df.iterrows(), expected.iterrows()):
        values = [bust._flag(want[n]) for n in names]
        assert [None if pd.isna(got[n]) else bool(got[n]) for n in names] == values
        failed = [n for n, v in zip(names, values) if v is False]
        assert got["pb_valid"] == (not failed) and got["pb_status"] == ("invalid" if failed else "valid")
        assert got["failed_checks"] == (", ".join(failed) or None)
    assert set(df["pb_status"]) == {"valid"}
    assert digests(results_dir) == before
    assert res.posebusters(PID).empty  # no PoseBusters table was written


def test_bust_rows_of_poses_not_checked(res, monkeypatch, caplog):
    pytest.importorskip("posebusters")
    from pocketvina import bust

    def check_pose(pb, names, payload, protein, tmp_dir):
        if payload[0] == "pdbqt":
            return dict(status="convert_failed", error="the pose does not convert")
        raise RuntimeError("boom")

    monkeypatch.setattr(bust, "START_METHOD", "fork")
    monkeypatch.setattr(bust, "_check_pose", check_pose)
    df = res.select(PID, LIG, pocket="all").bust(workers=1)
    assert df[["pocket", "mode", "pb_status"]].values.tolist() == [[1, 1, "convert_failed"], [2, 1, "error"]]
    assert df["pb_valid"].isna().all() and df["failed_checks"].isna().all()
    assert "pocket 2 mode 1: not checked (error): RuntimeError: boom" in caplog.text
    for bad in (0, True, 1.5):
        with pytest.raises(ValueError, match="workers must be"):
            res.select(PID, LIG).bust(workers=bad)


def test_selection_outlives_the_results_object(results_dir, tmp_path):
    """A Selection holds its poses: export works after the Results it came from is closed (as PocketVinaGPU.select
    closes it); show and bust open the results folder again."""
    with Results(results_dir) as r:
        sel = r.select(PID, LIG, pocket=2, mode="all")
    assert len(sdf_records(sel.export(tmp_path))) == 9 and sel.results_dir == results_dir
