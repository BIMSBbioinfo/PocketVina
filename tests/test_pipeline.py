"""pocketvina.pipeline (the "atlas" output format) and the "files" format of ``PocketVinaGPU.execute_docking``, with the
fake engine (tests/fake_engine.py).

P2Rank never runs here (except in ``test_real_p2rank``, which needs Java and ``POCKETVINA_P2RANK_TESTS=1``): pockets
come from the fixture predictions CSV through a stubbed ``pipeline.predict_pockets`` or ``P2RankProcessor.run_p2rank``.
OpenBabel needs 15-25 s to turn each example protein into a receptor PDBQT, so most tests convert only its first
atoms (``cropped_receptor``); ``test_receptor_pdbqt_matches_obabel_cli`` checks the conversion itself.
"""
import dataclasses
import gzip
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

import fake_engine as fe
from pocketvina import PocketVinaConfig, PocketVinaGPU, gpu, pipeline, workspace
from pocketvina import main as pv_main
from pocketvina.io import sas, tables
from pocketvina.io.posefile import Reader
from pocketvina.io.results import Results
from pocketvina.io.tables import read_table
from pocketvina.io.writer import ResultsDir
from pocketvina.ligands import find_obabel
from pocketvina.main import P2RankConfig, P2RankProcessor, PocketConfig, check_ligand_names, read_protein_list

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"
REPO_ROOT = TESTS_DIR.parent
PACKAGE_DIR = REPO_ROOT / "pocketvina"
POCKETS_CSV = DATA_DIR / "legacy_run" / "proteins" / "pockets" / "AF-Q96LA5-F1-model_v4.csv"
# P2Rank's points file of example/proteins/1fbl.pdb (ranks 1-4 and 0), where P2Rank writes it in its output folder
POINTS_FILE = DATA_DIR / "p2rank" / "visualizations" / "data" / "1fbl.pdb_points.pdb.gz"
FIXTURE_RECEPTOR = DATA_DIR / "legacy_run" / "proteins" / "structures" / "AF-Q96LA5-F1-model_v4.pdbqt"
EXAMPLE_PROTEINS = ("1fbl", "2W83")
FEW_LIGANDS = ("molecule1", "molecule2", "molecule3", "molecule5")  # quick to embed
CROP_ATOMS = 200
SEED = 42
LEGACY_CSV_COLUMNS = ["protein", "molecule_name", "affinity (kcal/mol)", "dist from rmsd l.b.",
                      "dist from rmsd u.b.", "mode", "pocket rank"]
PER_PROTEIN = ("manifest/{p}.json", "poses/{p}.poses", "structures/{p}.{ext}.gz", "tables/proteins/{p}.parquet",
               "tables/pockets/{p}.parquet", "tables/pocket_runs/{p}.parquet", "tables/modes/{p}.parquet",
               "tables/pairs/{p}.parquet")


# --- stubs (also used by the crash driver, in a subprocess) ---------------------------------------------------------

def write_pockets(dst, n_pockets=None):
    """The fixture P2Rank predictions (22 pockets), or only its first ``n_pockets`` rows, at ``dst``."""
    lines = POCKETS_CSV.read_text().splitlines(True)
    Path(dst).write_text("".join(lines if n_pockets is None else lines[:1 + n_pockets]))
    return str(dst)


def stub_predict_pockets(n_pockets=None, calls=None, points=None):
    """``pipeline.predict_pockets`` writing the fixture predictions, and with ``points`` (a P2Rank points file) that
    file where P2Rank writes it."""
    def predict_pockets(structure, work_dir, p2rank_config=None, *, while_running=None):
        os.makedirs(work_dir, exist_ok=True)
        if calls is not None:
            calls.append(str(structure))
        name = os.path.basename(structure)
        if points is not None:
            os.makedirs(os.path.join(work_dir, "visualizations", "data"))
            shutil.copy(points, os.path.join(work_dir, "visualizations", "data", f"{name}_points.pdb.gz"))
        return write_pockets(os.path.join(work_dir, f"{name}_predictions.csv"), n_pockets)
    return predict_pockets


def cropped_receptor(n_atoms=CROP_ATOMS, real=pipeline.receptor_pdbqt):
    """``receptor_pdbqt`` on the first ``n_atoms`` ATOM/HETATM records of a PDB file (other files unchanged)."""
    def receptor_pdbqt(structure, legacy=False):
        if not str(structure).endswith(".pdb"):
            return real(structure, legacy=legacy)
        with open(structure) as f:
            atoms = [line for line in f if line.startswith(("ATOM", "HETATM"))][:n_atoms]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, os.path.basename(structure))
            with open(path, "w") as f:
                f.writelines(atoms)
            return real(path, legacy=legacy)
    return receptor_pdbqt


def install_stubs(n_pockets=None):
    """Patch the pocket source and the receptor conversion, and find no GPU (one engine slot, as with
    ``fake_engine_env``), for a separate process; tests use ``stubs``."""
    pipeline.predict_pockets = stub_predict_pockets(n_pockets)
    pipeline.receptor_pdbqt = pv_main.receptor_pdbqt = cropped_receptor()
    gpu.list_gpus = lambda: []


@pytest.fixture
def stubs(monkeypatch):
    """Stub P2Rank (the fixture pockets) and crop the receptors (atlas pipeline and the "files" format's
    ``P2RankProcessor``); returns the list of structures P2Rank was asked for in atlas runs."""
    calls = []
    monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets(calls=calls))
    monkeypatch.setattr(pipeline, "receptor_pdbqt", cropped_receptor())
    monkeypatch.setattr(pv_main, "receptor_pdbqt", cropped_receptor())
    return calls


# --- inputs and checks ---------------------------------------------------------------------------------------------

def example_ligands(example_dir):
    return dict(row for row in (line.split(",", 1) for line in
                                (example_dir / "ligands.txt").read_text().splitlines()[1:] if line.strip()))


def make_inputs(tmp_path, example_dir, ligands=FEW_LIGANDS, proteins=None, extra_smiles=None):
    """(protein list, SMILES CSV) in ``tmp_path/inputs``; proteins default to the example PDB files."""
    inputs = tmp_path / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    if proteins is None:
        proteins = [example_dir / "proteins" / f"{p}.pdb" for p in EXAMPLE_PROTEINS]
    plist = inputs / "proteins.txt"
    plist.write_text("".join(f"{p}\n" for p in proteins))
    smiles = dict(example_ligands(example_dir), **(extra_smiles or {}))
    names = list(smiles) if ligands is None else list(ligands)
    csv_path = inputs / "ligands.csv"
    csv_path.write_text("molecule_name,smiles\n" + "".join(f"{n},{smiles[n]}\n" for n in names))
    return plist, csv_path


def atlas_config(plist, smiles, out, tmp_base, **kw):
    kw.setdefault("top_pockets", 2)
    # every fake GPU, one engine each, as before the defaults became the first GPU filled by memory
    # (test_default_slots_are_the_first_gpu_filled_by_memory checks those)
    kw.setdefault("gpus", "all")
    kw.setdefault("procs_per_gpu", 1)
    return PocketVinaConfig(protein_paths=str(plist), smiles_file=str(smiles), output_dir=str(out),
                            output_format="atlas", tmp_dir=str(tmp_base), **kw)


def results_files(root) -> set:
    root = Path(root)
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def expected_files(proteins: dict) -> set:
    """``proteins``: {protein_id: structure kind ("pdb", "pdbqt")}."""
    files = {"MANIFEST.json", "tables/ligands.parquet"}
    for pid, ext in proteins.items():
        files |= {f.format(p=pid, ext=ext) for f in PER_PROTEIN}
    return files


def engine_output(ligand_pdbqt, receptor_base, pocket, stem, seed=SEED, num_modes=20, energy_range=20.0) -> bytes:
    """What the fake engine writes for this ligand x pocket."""
    modes = fe.score_modes(receptor_base, pocket, stem, seed, num_modes, energy_range)
    return fe.render_pdbqt(ligand_pdbqt, modes).encode()


def within(path, root) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)


def calls_for(calls, protein_id):
    return [c for c in calls if os.path.basename(c["options"]["receptor"]) == f"{protein_id}.pdbqt"]


def check_protein(results, pid, ligands, n_pockets, K, docked, seed=SEED):
    """Tables and pose file of one protein docked with the fake engine, every prepared ligand into ``docked``."""
    rd = ResultsDir(results)
    assert rd.is_done(pid) and rd.verify(pid) == []
    tdir = Path(results) / "tables"
    prot = read_table(tdir / "proteins" / f"{pid}.parquet")
    assert len(prot) == 1 and prot.status[0] == "ok"
    assert (prot.n_pockets[0], prot.n_pockets_docked[0]) == (n_pockets, len(docked))
    assert len(prot.receptor_sha256[0]) == 32
    pockets = read_table(tdir / "pockets" / f"{pid}.parquet")
    assert list(pockets.pocket) == list(range(1, n_pockets + 1))
    assert set(pockets.size_x) == {20.0}

    ok = ligands[ligands.status == "ok"]
    runs = read_table(tdir / "pocket_runs" / f"{pid}.parquet")
    assert len(runs) == len(ligands) * len(docked)
    modes = read_table(tdir / "modes" / f"{pid}.parquet")
    pairs = read_table(tdir / "pairs" / f"{pid}.parquet").set_index("ligand_idx")
    with Reader(Path(results) / "poses" / f"{pid}.poses") as r:
        assert (r.N, r.K) == (len(ligands), K)
        for row in ok.itertuples():
            i, stem = row.ligand_idx, f"L{row.ligand_idx:07d}"
            best = None
            for k in range(1, K + 1):
                if k not in docked:
                    assert r.modes(i, k) is None
                    continue
                raw = r.modes(i, k)
                assert raw == engine_output(row.pdbqt, pid, k, stem, seed)
                scores = fe.score_modes(pid, k, stem, seed)
                run = runs[(runs.ligand_idx == i) & (runs.pocket == k)].iloc[0]
                assert (run.status, run.n_modes, run.seed) == ("ok", len(scores), seed)
                assert run.best_affinity == pytest.approx(scores[0][0])
                assert run.sha256 == hashlib.sha256(raw).digest()
                m = modes[(modes.ligand_idx == i) & (modes.pocket == k)]
                assert list(m["mode"]) == list(range(1, len(scores) + 1))
                assert list(m.affinity) == pytest.approx([s[0] for s in scores])
                for mode in m.itertuples():
                    text = raw[mode.raw_offset:mode.raw_offset + mode.raw_length]
                    assert text.startswith(f"MODEL {mode.mode}\n".encode()) and text.endswith(b"ENDMDL\n")
                if best is None or scores[0][0] < best[0]:
                    best = (scores[0][0], k)
            pair = pairs.loc[i]
            assert (pair.vina, pair.best_pocket, pair.best_mode) == (pytest.approx(best[0]), best[1], 1)
            assert pair.sdf_status == "ok"
            assert r.top_range(i) == (pair.top_offset, pair.top_length)
            mol = Chem.MolFromMolBlock(r.top(i).decode())
            assert mol.GetProp("_Name") == row.ligand_id
            assert mol.GetNumHeavyAtoms() == row.n_heavy
            if row.inchikey:  # connectivity only: the pose defines stereo that the SMILES may leave open
                assert Chem.MolToInchiKey(mol)[:14] == row.inchikey[:14]
        for row in ligands[ligands.status != "ok"].itertuples():
            assert r.top(row.ligand_idx) is None
            assert pairs.loc[row.ligand_idx].sdf_status == "no_pose"
    return runs, pairs


# --- atlas format --------------------------------------------------------------------------------------------------

def test_atlas_end_to_end_example_proteins(tmp_path, example_dir, fake_engine_env, stubs):
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=None)
    out, tmp_base = tmp_path / "results", tmp_path / "tmp"
    path = pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_base)))

    assert path == str(out)
    assert results_files(out) == expected_files({p: "pdb" for p in EXAMPLE_PROTEINS})
    assert os.listdir(tmp_base) == []  # the run's temp folder is gone
    assert [os.path.basename(s) for s in stubs] == ["1fbl.pdb", "2W83.pdb"]

    manifest = json.loads((out / "MANIFEST.json").read_text())
    assert manifest["format"] == "pocketvina-atlas" and manifest["n_ligands"] == 9
    assert manifest["params"]["seed"] == SEED and manifest["params"]["top_pockets"] == 2
    ligands = read_table(out / "tables" / "ligands.parquet")
    assert list(ligands.ligand_id) == [f"molecule{n}" for n in range(1, 10)]
    assert set(ligands.status) == {"ok"} and set(ligands.seed) == {SEED}

    calls = fake_engine_env.calls()
    assert len(calls) == 4  # 2 proteins x 2 pockets x one batch of 9 ligands
    for c in calls:
        assert c["returncode"] == 0 and c["options"]["seed"] == SEED
        assert within(c["cwd"], tmp_base) and not within(c["cwd"], PACKAGE_DIR)
        assert within(c["options"]["ligand_directory"], tmp_base)
        assert within(c["options"]["output_directory"], tmp_base)

    for p in EXAMPLE_PROTEINS:
        check_protein(out, p, ligands, n_pockets=22, K=2, docked=(1, 2))
        prot = read_table(out / "tables" / "proteins" / f"{p}.parquet").iloc[0]
        source = example_dir / "proteins" / f"{p}.pdb"
        assert prot.source_path == str(source) and prot.structure_file == f"structures/{p}.pdb.gz"
        assert prot.uniprot is None
        assert gzip.decompress((out / prot.structure_file).read_bytes()) == source.read_bytes()
    with Results(out) as res:
        assert res.sas("1fbl") is None  # P2Rank (the stub) wrote no points file


def test_atlas_keeps_p2rank_sas_points(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    """P2Rank's points file becomes sas/<protein_id>.json.gz (io.sas: the docked pockets 1 and 2 in "pockets", its
    pockets 3 and 4 in "other_pockets"), committed with the protein; Results.sas reads it. Pockets from a CSV have
    none."""
    monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets(points=POINTS_FILE))
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",))
    out = tmp_path / "results"
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    assert results_files(out) == (expected_files({p: "pdb" for p in EXAMPLE_PROTEINS})
                                  | {f"sas/{p}.json.gz" for p in EXAMPLE_PROTEINS})
    assert os.listdir(tmp_path / "tmp") == []
    points = sas.read_points(POINTS_FILE)
    with Results(out) as res:
        for p in EXAMPLE_PROTEINS:
            expected = sas.build(points, [1, 2], p)
            assert list(expected["pockets"]) == ["1", "2"] and list(expected["other_pockets"]) == ["3", "4"]
            assert (out / "sas" / f"{p}.json.gz").read_bytes() == sas.encode(expected, 6)  # gzip_level 6
            assert f"sas/{p}.json.gz" in res.manifest(p)["files"] and ResultsDir(out).verify(p) == []
            assert res.sas(p) == expected
        page = workspace.workspace_data(res)  # the page's data: the points without the protein's id
        assert [prot["sas"] for prot in page["proteins"]] == [
            {k: sas.build(points, [1, 2], p)[k] for k in sas.FIELDS} for p in res.proteins()]

    pockets_dir = tmp_path / "pockets"
    pockets_dir.mkdir()
    write_pockets(pockets_dir / "1fbl.csv")
    out = tmp_path / "with_csv"
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp",
                                                  pockets_dir=str(pockets_dir))))
    assert sorted(path.name for path in (out / "sas").iterdir()) == ["2W83.json.gz"]
    with Results(out) as res:
        assert res.sas("1fbl") is None and res.sas("2W83")["pockets"]


def test_stage_sas_without_points(tmp_path, monkeypatch, caplog):
    """_stage_sas: no points file gives no SAS file and one warning for the whole run; a points file with no points
    gives none either (a warning); one not gzipped is read as the gzipped one."""
    monkeypatch.setattr(pipeline, "_no_points_logged", False)
    out = tmp_path / "out"
    out.mkdir()
    csv_path = str(out / "1fbl.pdb_predictions.csv")
    stage = str(tmp_path / "stage")
    with caplog.at_level(logging.WARNING, logger="pocketvina.pipeline"):
        assert pipeline._stage_sas(csv_path, "1fbl", [1], stage, 6) is None
        assert pipeline._stage_sas(csv_path, "2W83", [1], stage, 6) is None
        assert caplog.text.count("P2Rank wrote no SAS points file") == 1 and "1fbl" in caplog.text
        caplog.clear()
        points = out / "visualizations" / "data" / "1fbl.pdb_points.pdb"
        points.parent.mkdir(parents=True)
        points.write_text("REMARK no points\n")
        assert pipeline._stage_sas(csv_path, "1fbl", [1], stage, 6) is None
        assert "has no points, none are kept" in caplog.text
        assert not os.path.exists(os.path.join(stage, "sas"))
    points.write_bytes(gzip.decompress(POINTS_FILE.read_bytes()))
    rel, staged = pipeline._stage_sas(csv_path, "1fbl", [1, 2], stage, 6)
    assert rel == "sas/1fbl.json.gz"
    with open(staged, "rb") as f:
        assert f.read() == sas.encode(sas.build(sas.read_points(POINTS_FILE), [1, 2], "1fbl"), 6)


def test_atlas_file_count_per_protein(tmp_path, example_dir, fake_engine_env, stubs):
    proteins = [example_dir / "proteins" / "1fbl.pdb", FIXTURE_RECEPTOR]
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"), proteins=proteins)
    out = tmp_path / "results"
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))

    files = results_files(out)
    pids = {"1fbl": "pdb", "AF-Q96LA5-F1-model_v4": "pdbqt"}
    assert files == expected_files(pids)
    for pid in pids:
        assert len([f for f in files if Path(f).name.startswith(pid + ".")]) == 8


def test_atlas_stages_everything_in_tmp_dir(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    """Every Parquet file (ligands.parquet included) is staged in config.tmp_dir, not in $TMPDIR."""
    for var in ("POCKETVINA_TMPDIR", "TMPDIR"):
        (tmp_path / var).mkdir()
        monkeypatch.setenv(var, str(tmp_path / var))
    staged = []
    real = tables.write_table

    def spy(rows, schema, path):
        staged.append(path)
        return real(rows, schema, path)

    monkeypatch.setattr(tables, "write_table", spy)
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"),
                                proteins=[example_dir / "proteins" / "1fbl.pdb"])
    tmp_base = tmp_path / "tmp"
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, tmp_path / "results", tmp_base, top_pockets=1)))
    assert len(staged) == 6  # ligands + proteins, pockets, pocket_runs, modes, pairs
    assert all(within(p, tmp_base) for p in staged), staged
    assert os.listdir(tmp_base) == []


def test_atlas_pdbqt_receptor_statuses_and_batches(tmp_path, example_dir, fake_engine_env, stubs, data_dir,
                                                   monkeypatch):
    """Batches of 2; molecule10 (Si) is not prepared, molecule1 makes its engine call fail, molecule5 has no
    TORSDOF (the engine skips it); the receptor is a PDBQT file, used as it is."""
    names = ("molecule2", "molecule5", "molecule10", "molecule1")
    fixture_smiles = dict(r.split(",", 1) for r in (data_dir / "smiles.csv").read_text().splitlines()[1:])
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=names, proteins=[FIXTURE_RECEPTOR],
                                extra_smiles=fixture_smiles)
    real_prepare = pipeline.prepare_ligands

    def prepare(rows, **kw):
        records = real_prepare(rows, **kw)
        records[1].pdbqt = "".join(line for line in records[1].pdbqt.splitlines(True) if "TORSDOF" not in line)
        records[3].pdbqt = "REMARK FAIL\n" + records[3].pdbqt
        return records

    monkeypatch.setattr(pipeline, "prepare_ligands", prepare)
    out = tmp_path / "results"
    pv = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp", batch_size=2))
    pipeline.run_atlas(pv)

    pid = "AF-Q96LA5-F1-model_v4"
    assert pv.failed_proteins == {}
    calls = fake_engine_env.calls()
    assert len(calls) == 4 and sorted(c["returncode"] for c in calls) == [0, 0, 1, 1]
    ligands = read_table(out / "tables" / "ligands.parquet")
    assert list(ligands.status) == ["ok", "ok", "unsupported_element", "ok"]

    runs = read_table(out / "tables" / "pocket_runs" / f"{pid}.parquet").set_index(["ligand_idx", "pocket"])
    for k in (1, 2):
        assert runs.loc[(0, k)].status == "ok"
        assert runs.loc[(1, k)].status == "no_output" and "Missing TORSDOF" in runs.loc[(1, k)].error
        assert runs.loc[(2, k)].status == "skipped"
        assert runs.loc[(2, k)].error.startswith("ligand not prepared: unsupported_element")
        assert runs.loc[(3, k)].status == "engine_error" and "FAIL" in runs.loc[(3, k)].error
        assert runs.loc[(2, k)].n_modes == 0 and runs.loc[(2, k)].length == 0
    pairs = read_table(out / "tables" / "pairs" / f"{pid}.parquet").set_index("ligand_idx")
    assert list(pairs.sdf_status) == ["ok", "no_pose", "no_pose", "no_pose"]
    assert list(pairs.best_pocket) == [pairs.best_pocket[0], -1, -1, -1]

    prot = read_table(out / "tables" / "proteins" / f"{pid}.parquet").iloc[0]
    assert (prot.status, prot.uniprot, prot.structure_file) == ("ok", "Q96LA5", f"structures/{pid}.pdbqt.gz")
    assert prot.error == "2 of 4 engine runs failed"
    receptor = FIXTURE_RECEPTOR.read_bytes()
    assert gzip.decompress((out / prot.structure_file).read_bytes()) == receptor
    assert prot.receptor_sha256 == hashlib.sha256(receptor).digest()
    summary = json.loads((out / "manifest" / f"{pid}.json").read_text())["summary"]
    assert summary["runs_by_status"] == {"ok": 2, "no_output": 2, "skipped": 2, "engine_error": 2}
    assert summary["engine_runs"] == 4 and summary["engine_runs_failed"] == 2


def test_atlas_batch_of_ligands_the_engine_rejects(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    """A batch whose ligands all fail to parse makes the engine exit 1 ("No valid ligands"); its runs are
    "no_output" as when such a ligand shares a batch with valid ones, and the call is not an engine failure: a
    run of only such ligands is committed (a rerun would give the same), not left as a failed protein."""
    real_prepare = pipeline.prepare_ligands

    def no_torsdof(rows, **kw):
        records = real_prepare(rows, **kw)
        for rec in records:
            if rec.ligand_id in ("molecule3", "molecule5"):
                rec.pdbqt = "".join(line for line in rec.pdbqt.splitlines(True) if "TORSDOF" not in line)
        return records

    monkeypatch.setattr(pipeline, "prepare_ligands", no_torsdof)
    pid = "AF-Q96LA5-F1-model_v4"
    for name, ligands, statuses in (("mixed", ("molecule1", "molecule2", "molecule3"), ["ok", "ok", "no_output"]),
                                    ("only", ("molecule5",), ["no_output"])):
        plist, smiles = make_inputs(tmp_path / name, example_dir, ligands=ligands, proteins=[FIXTURE_RECEPTOR])
        out = tmp_path / name / "results"
        pv = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp", batch_size=2, top_pockets=1))
        pipeline.run_atlas(pv)
        assert pv.failed_proteins == {} and ResultsDir(out).done_proteins() == {pid}
        runs = read_table(out / "tables" / "pocket_runs" / f"{pid}.parquet")
        assert list(runs.status) == statuses
        assert "Missing TORSDOF" in runs.error.iloc[-1]
        prot = read_table(out / "tables" / "proteins" / f"{pid}.parquet").iloc[0]
        assert (prot.status, prot.error) == ("ok", None)
        summary = json.loads((out / "manifest" / f"{pid}.json").read_text())["summary"]
        assert summary["engine_runs_failed"] == 0
    assert sorted(c["returncode"] for c in fake_engine_env.calls()) == [0, 1, 1]


def test_atlas_pocket_slots(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    """top_pockets=None docks every pocket (K = number of pockets); top_pockets=N gives N slots, empty when the
    protein has fewer pockets; a protein without pockets is committed as "no_pockets"."""
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"), proteins=[FIXTURE_RECEPTOR])
    pid = "AF-Q96LA5-F1-model_v4"
    ligands = None
    for name, n_pockets, top, K, docked in (("all", 3, None, 3, (1, 2, 3)), ("top5", 3, 5, 5, (1, 2, 3)),
                                            ("top1", 3, 1, 1, (1,))):
        monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets(n_pockets))
        out = tmp_path / name
        pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp", top_pockets=top)))
        ligands = read_table(out / "tables" / "ligands.parquet")
        check_protein(out, pid, ligands, n_pockets=n_pockets, K=K, docked=docked)

    monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets(0))
    out = tmp_path / "none"
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    prot = read_table(out / "tables" / "proteins" / f"{pid}.parquet").iloc[0]
    assert (prot.status, prot.n_pockets, prot.n_pockets_docked) == ("no_pockets", 0, 0)
    assert len(read_table(out / "tables" / "pocket_runs" / f"{pid}.parquet")) == 0
    assert len(read_table(out / "tables" / "pairs" / f"{pid}.parquet")) == 0
    with Reader(out / "poses" / f"{pid}.poses") as r:
        assert (r.N, r.K) == (2, 2)
    assert results_files(out) == expected_files({pid: "pdbqt"})


def test_htvs_mode_docks_five_pockets(tmp_path, example_dir, fake_engine_env, stubs):
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[FIXTURE_RECEPTOR])
    out = tmp_path / "results"
    pv = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp", top_pockets=None), htvs_mode=True)
    assert pv.top_pockets == 5
    assert pv.execute_docking() == str(out)
    assert len(fake_engine_env.calls()) == 5
    assert json.loads((out / "MANIFEST.json").read_text())["params"]["top_pockets"] == 5
    assert results_files(out) == expected_files({"AF-Q96LA5-F1-model_v4": "pdbqt"})  # no log files, no proteins/


def test_atlas_rerun_skips_done_proteins_and_ligand_preparation(tmp_path, example_dir, fake_engine_env, stubs,
                                                               monkeypatch):
    plist, smiles = make_inputs(tmp_path, example_dir)
    out = tmp_path / "results"
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    before = {f: (out / f).read_bytes() for f in results_files(out)}
    n_calls = len(fake_engine_env.calls())

    def no_preparation(*args, **kwargs):
        raise AssertionError("ligands were prepared again")

    monkeypatch.setattr(pipeline, "prepare_ligands", no_preparation)
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    assert len(fake_engine_env.calls()) == n_calls
    assert {f: (out / f).read_bytes() for f in results_files(out)} == before
    assert os.listdir(tmp_path / "tmp") == []


def test_atlas_resume_keeps_stored_ligands(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    """When a new preparation differs from the stored ligands (an unseeded preparation, or another process that
    started the same run first), the run docks the stored ones."""
    plist, smiles = make_inputs(tmp_path, example_dir)
    out = tmp_path / "results"
    plist.write_text(f"{example_dir / 'proteins' / '1fbl.pdb'}\n")
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    stored = read_table(out / "tables" / "ligands.parquet")
    real_prepare, real_stored = pipeline.prepare_ligands, pipeline._stored_ligands
    prepared, lookups = [], []

    def other_conformers(rows, **kw):
        prepared.append(len(rows))
        return real_prepare(rows, **{**kw, "seed": 7})

    def not_there_yet(*args, **kw):  # the first look finds nothing, as if the run had not been started yet
        lookups.append(kw.get("wait", 0))
        return None if len(lookups) == 1 else real_stored(*args, **kw)

    monkeypatch.setattr(pipeline, "prepare_ligands", other_conformers)
    monkeypatch.setattr(pipeline, "_stored_ligands", not_there_yet)
    plist.write_text("".join(f"{example_dir / 'proteins' / p}.pdb\n" for p in EXAMPLE_PROTEINS))
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    assert prepared == [len(FEW_LIGANDS)] and lookups == [0, pipeline.STORED_LIGANDS_WAIT_S]
    pd.testing.assert_frame_equal(read_table(out / "tables" / "ligands.parquet"), stored)
    check_protein(out, "2W83", stored, n_pockets=22, K=2, docked=(1, 2))


def test_atlas_resume_rejects_other_settings(tmp_path, example_dir, fake_engine_env, stubs):
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[FIXTURE_RECEPTOR])
    out = tmp_path / "results"
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    with pytest.raises(ValueError, match="other params: seed"):
        pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp", seed=7)))
    other_plist, other_smiles = make_inputs(tmp_path / "other", example_dir, ligands=("molecule2",),
                                            proteins=[FIXTURE_RECEPTOR])
    with pytest.raises(ValueError, match="other ligands"):
        pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, other_smiles, out, tmp_path / "tmp")))
    assert os.listdir(tmp_path / "tmp") == []

    # force=True replaces the run definition; proteins already done stay done
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp", seed=7)), force=True)
    assert json.loads((out / "MANIFEST.json").read_text())["params"]["seed"] == 7
    assert len(fake_engine_env.calls()) == 2


def test_atlas_force_with_another_ligand_list_docks_again(tmp_path, example_dir, fake_engine_env, stubs):
    """force=True with another ligand list: the protein done before is docked again. It used to stay done with the
    old ligand_idx, so molecule1's pose was read as that of molecule2 (now ligand_idx 0)."""
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"),
                                proteins=[FIXTURE_RECEPTOR])
    out, pid = tmp_path / "results", "AF-Q96LA5-F1-model_v4"
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp", top_pockets=1)))
    n_calls = len(fake_engine_env.calls())
    _, other = make_inputs(tmp_path / "other", example_dir, ligands=("molecule2",), proteins=[FIXTURE_RECEPTOR])
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, other, out, tmp_path / "tmp", top_pockets=1)), force=True)
    assert len(fake_engine_env.calls()) == n_calls + 1
    ligands = read_table(out / "tables" / "ligands.parquet")
    assert ligands.ligand_id.tolist() == ["molecule2"]
    check_protein(out, pid, ligands, n_pockets=22, K=1, docked=(1,))


DRIVER = """\
import json, os, signal, sys, time
sys.path.insert(0, {tests_dir!r})
import test_pipeline
from pocketvina import PocketVinaConfig, PocketVinaGPU, pipeline
from pocketvina.io.writer import ResultsDir

args = json.loads(sys.argv[1])
test_pipeline.install_stubs()
event, at = args["kill"]
rd = ResultsDir(args["config"]["output_dir"])
count = [0]
run_engine = pipeline.run_engine

def kill_on(n):
    count[0] += 1
    if count[0] == n:
        os.kill(os.getpid(), signal.SIGKILL)

if event == "after_commit":
    commit = ResultsDir.commit
    def killing_commit(self, *a, **kw):
        manifest = commit(self, *a, **kw)
        kill_on(at)
        return manifest
    ResultsDir.commit = killing_commit
    calls = [0]
    def holding_run_engine(*a, **kw):  # the calls of the protein prepared ahead wait for the kill
        calls[0] += 1
        if calls[0] > args["hold_after"]:
            time.sleep(600)
        return run_engine(*a, **kw)
    pipeline.run_engine = holding_run_engine
else:
    def killing_run_engine(*a, **kw):
        if count[0] + 1 == at:  # the first protein's calls are back: let the main thread commit it
            deadline = time.monotonic() + 120
            while not rd.done_proteins() and time.monotonic() < deadline:
                time.sleep(0.05)
        kill_on(at)
        return run_engine(*a, **kw)
    pipeline.run_engine = killing_run_engine
pipeline.run_atlas(PocketVinaGPU(PocketVinaConfig(**args["config"])))
"""


def run_and_kill(tmp_path, config: dict, kill, hold_after=2):
    """Run ``config`` in a driver process that SIGKILLs itself at ``kill`` = ("after_commit", n): right after the
    n-th commit (engine calls after the first ``hold_after`` wait for it, so that none of the protein prepared ahead
    runs), or ("engine_call", n): when the n-th engine call starts (once a protein is committed)."""
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(tests_dir=str(TESTS_DIR)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))
    proc = subprocess.run([sys.executable, str(driver), json.dumps(dict(config=config, kill=kill,
                                                                         hold_after=hold_after))],
                          env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=600)
    assert proc.returncode == -signal.SIGKILL, proc.stdout[-3000:] + proc.stderr[-3000:]


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs SIGKILL")
@pytest.mark.parametrize("kill", [("after_commit", 1), ("engine_call", 4)], ids=["between_proteins", "in_protein"])
def test_atlas_resume_after_crash(tmp_path, example_dir, fake_engine_env, stubs, kill):
    """Kill the run (SIGKILL) right after the first protein is committed, or in the middle of the second one (its
    second engine call); a rerun docks only the second protein and gives the same results as a run without crash."""
    plist, smiles = make_inputs(tmp_path, example_dir)
    out, tmp_base = tmp_path / "results", tmp_path / "tmp"
    config = dict(protein_paths=str(plist), smiles_file=str(smiles), output_dir=str(out), output_format="atlas",
                  tmp_dir=str(tmp_base), top_pockets=2)
    run_and_kill(tmp_path, config, kill)

    first, second = EXAMPLE_PROTEINS
    rd = ResultsDir(out)
    assert rd.done_proteins() == {first}
    assert not any(second in f for f in results_files(out))  # nothing of a protein before its commit
    first_files = {f: (out / f).read_bytes() for f in results_files(out) if first in f}
    calls_before = len(fake_engine_env.calls())
    assert len(calls_for(fake_engine_env.calls(), first)) == 2

    pipeline.run_atlas(PocketVinaGPU(PocketVinaConfig(**config)))
    new_calls = fake_engine_env.calls()[calls_before:]
    assert len(new_calls) == 2 and calls_for(new_calls, second) == new_calls  # the first protein is not redone
    assert rd.done_proteins() == {first, second}
    assert {f: (out / f).read_bytes() for f in results_files(out) if first in f} == first_files
    assert results_files(out) == expected_files({p: "pdb" for p in EXAMPLE_PROTEINS})

    # same results as a run that was not interrupted
    ref = tmp_path / "reference"
    pipeline.run_atlas(PocketVinaGPU(PocketVinaConfig(**dict(config, output_dir=str(ref)))))
    for p in EXAMPLE_PROTEINS:
        assert (out / "poses" / f"{p}.poses").read_bytes() == (ref / "poses" / f"{p}.poses").read_bytes()
        for table in ("pairs", "modes", "proteins", "pockets"):
            pd.testing.assert_frame_equal(read_table(out / "tables" / table / f"{p}.parquet"),
                                          read_table(ref / "tables" / table / f"{p}.parquet"))
        runs = [read_table(d / "tables" / "pocket_runs" / f"{p}.parquet").drop(columns="runtime_s") for d in (out, ref)]
        pd.testing.assert_frame_equal(*runs)


SIGTERM_DRIVER = """\
import json, os, subprocess, sys
sys.path.insert(0, {tests_dir!r})
import test_pipeline
from pocketvina import PocketVinaConfig, PocketVinaGPU

args = json.loads(sys.argv[1])
test_pipeline.install_stubs()
popen = subprocess.Popen

class RecordingPopen(popen):  # tells the test the engine's pid
    def __init__(self, cmd, *a, **kw):
        super().__init__(cmd, *a, **kw)
        if cmd[0] == os.environ["POCKETVINA_ENGINE"]:
            with open(args["pid_file"] + ".tmp", "w") as f:
                f.write(str(self.pid))
            os.replace(args["pid_file"] + ".tmp", args["pid_file"])

subprocess.Popen = RecordingPopen
PocketVinaGPU(PocketVinaConfig(**args["config"])).execute_docking()
"""


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _running(pid) -> bool:
    """The process exists and is no zombie (a killed orphan stays one until init reaps it)."""
    if not os.path.isdir("/proc/self"):
        return _alive(pid)
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read()
    except OSError:
        return False
    return stat[stat.rindex(b")") + 2:][:1] != b"Z"


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX signals")
@pytest.mark.parametrize("output_format", ["atlas", "files"])
def test_sigterm_stops_the_engine_and_removes_temp_files(tmp_path, example_dir, fake_engine_env, monkeypatch,
                                                         output_format):
    """``kill <pid>`` (or a batch scheduler's SIGTERM) while the engine runs: the run still dies of SIGTERM, but the
    engine is not left running on the GPU and the temp folder is removed (both were left behind before)."""
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "60")
    _, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",))
    tmp_base, out = tmp_path / "tmp", tmp_path / "out"
    if output_format == "atlas":
        plist = tmp_path / "proteins.txt"
        plist.write_text(f"{FIXTURE_RECEPTOR}\n")
        config = dict(protein_paths=str(plist), output_format="atlas", top_pockets=1)
    else:
        db, protein_list = make_database(tmp_path)
        config = dict(database_path=str(db), protein_list=str(protein_list))
    config.update(smiles_file=str(smiles), output_dir=str(out), tmp_dir=str(tmp_base))
    driver, pid_file, log = tmp_path / "driver.py", tmp_path / "engine.pid", tmp_path / "driver.log"
    driver.write_text(SIGTERM_DRIVER.format(tests_dir=str(TESTS_DIR)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))
    engine_pid = None
    with open(log, "wb") as stderr:
        proc = subprocess.Popen([sys.executable, str(driver), json.dumps(dict(config=config, pid_file=str(pid_file)))],
                                env=env, cwd=str(tmp_path), stdout=subprocess.DEVNULL, stderr=stderr)
        try:
            deadline = time.monotonic() + 120
            while not pid_file.exists():
                assert proc.poll() is None and time.monotonic() < deadline, log.read_text()[-3000:]
                time.sleep(0.05)
            engine_pid = int(pid_file.read_text())
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=60)
        finally:
            proc.kill()
            if engine_pid is not None and _alive(engine_pid):
                os.kill(engine_pid, signal.SIGKILL)
                pytest.fail("the engine was left running")
    assert proc.returncode == -signal.SIGTERM, log.read_text()[-3000:]
    assert os.listdir(tmp_base) == []
    if output_format == "atlas":
        assert ResultsDir(out).done_proteins() == set()


def test_atlas_failed_engine_leaves_proteins_for_a_rerun(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"))
    broken = tmp_path / "broken_engine"
    broken.write_text("#!/bin/sh\necho 'clGetPlatformIDs failed' >&2\nexit 3\n")
    broken.chmod(0o755)
    if not os.access(broken, os.X_OK):
        pytest.skip(f"cannot execute {broken} (noexec temp dir?)")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(broken))
    out = tmp_path / "results"
    pv = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp"))
    assert pipeline.run_atlas(pv) == str(out)
    assert set(pv.failed_proteins) == set(EXAMPLE_PROTEINS)
    assert "clGetPlatformIDs failed" in pv.failed_proteins["1fbl"]
    assert ResultsDir(out).done_proteins() == set()
    assert results_files(out) == {"MANIFEST.json", "tables/ligands.parquet"}

    monkeypatch.setenv("POCKETVINA_ENGINE", str(fake_engine_env))
    pv = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp"))
    pipeline.run_atlas(pv)
    assert pv.failed_proteins == {}
    assert ResultsDir(out).done_proteins() == set(EXAMPLE_PROTEINS)


# --- PocketVinaGPU.best_scores, and the table at the end of execute_docking -----------------------------------------

def docking_output(capsys) -> str:
    """What execute_docking printed after the citation it starts with."""
    out = capsys.readouterr().out
    assert out.startswith(pv_main.CITATION + "\n"), out[:200]
    return out[len(pv_main.CITATION) + 1:]


def test_best_scores_after_docking_and_in_a_new_session(tmp_path, example_dir, fake_engine_env, stubs, capsys,
                                                        monkeypatch):
    """An "atlas" execute_docking() returns the results folder as before and then prints the best scores of its
    proteins (outside a notebook: all rows, or the best BEST_SCORES_SHOWN and how many there are); best_scores() and
    results() work right after it and from a new PocketVinaGPU of the same config. A rerun (all done) prints them
    again, of its proteins only."""
    monkeypatch.setattr(pv_main, "BEST_SCORES_SHOWN", 3)
    plist, smiles = make_inputs(tmp_path, example_dir)
    out = tmp_path / "results"
    pv = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp"))
    assert pv.execute_docking() == str(out)
    printed = docking_output(capsys)
    df = pv.best_scores()
    assert len(df) == len(EXAMPLE_PROTEINS) * len(FEW_LIGANDS) and df["vina"].is_monotonic_increasing
    assert set(df.protein_id) == set(EXAMPLE_PROTEINS) and set(df.ligand_id) == set(FEW_LIGANDS)
    assert printed == (f"Best scores:\n{df.head(3).to_string(index=False)}\n"
                       f"The top 3 of {len(df)} protein x ligand pairs; pocketvina.best_scores() returns all of "
                       "them.\n")

    again = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp"))
    pd.testing.assert_frame_equal(again.best_scores(), df)
    pd.testing.assert_frame_equal(again.best_scores(protein="1fbl", top=2),
                                  df[df.protein_id == "1fbl"].head(2).reset_index(drop=True))
    with again.results() as res:
        assert isinstance(res, Results) and res.proteins() == sorted(EXAMPLE_PROTEINS)

    monkeypatch.setattr(pv_main, "BEST_SCORES_SHOWN", len(FEW_LIGANDS))  # all rows of 1fbl: no line below them
    one = tmp_path / "inputs" / "one.txt"
    one.write_text(f"{example_dir / 'proteins' / '1fbl.pdb'}\n")
    n_calls = len(fake_engine_env.calls())
    PocketVinaGPU(atlas_config(one, smiles, out, tmp_path / "tmp")).execute_docking()
    assert len(fake_engine_env.calls()) == n_calls  # done before
    only = df[df.protein_id == "1fbl"].reset_index(drop=True)
    assert docking_output(capsys) == f"Best scores:\n{only.to_string(index=False)}\n"


def test_execute_docking_displays_the_best_scores_in_a_notebook(tmp_path, example_dir, fake_engine_env, stubs, capsys,
                                                                monkeypatch, caplog):
    """In IPython/Jupyter the best scores are shown with IPython.display.display (the whole table, or the best
    BEST_SCORES_SHOWN rows and a line saying how many there are); IPython imported without a shell running (a
    script): printed; a display that fails: a warning, the run's result unchanged."""
    shown = []
    ipython, display = types.ModuleType("IPython"), types.ModuleType("IPython.display")
    ipython.get_ipython, ipython.display, display.display = (lambda: object()), display, shown.append
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"))
    pv = PocketVinaGPU(atlas_config(plist, smiles, tmp_path / "results", tmp_path / "tmp"))
    assert pv.execute_docking() == str(tmp_path / "results")
    (df,) = shown
    pd.testing.assert_frame_equal(df, pv.best_scores())
    assert len(df) == 4 and docking_output(capsys) == ""

    ipython.get_ipython = lambda: None  # reruns (all done) show the table again
    assert pv.execute_docking() == str(tmp_path / "results")
    assert docking_output(capsys) == f"Best scores:\n{df.to_string(index=False)}\n" and len(shown) == 1

    ipython.get_ipython = lambda: object()
    monkeypatch.setattr(pv_main, "BEST_SCORES_SHOWN", 3)
    assert pv.execute_docking() == str(tmp_path / "results")
    pd.testing.assert_frame_equal(shown[1], pv.best_scores(top=3))
    assert docking_output(capsys) == ("The top 3 of 4 protein x ligand pairs; pocketvina.best_scores() returns all "
                                       "of them.\n")

    def broken_display(obj):
        raise RuntimeError("no display")

    ipython.get_ipython, display.display = (lambda: object()), broken_display
    assert pv.execute_docking() == str(tmp_path / "results")
    assert "cannot show the best scores: RuntimeError: no display" in caplog.text


def test_execute_docking_best_scores_never_fail_the_run(tmp_path, example_dir, fake_engine_env, stubs, capsys,
                                                       monkeypatch, caplog):
    """No protein docked: a line says so. A table that cannot be made: a warning, the run's result unchanged."""
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",))
    broken = tmp_path / "broken_engine"
    broken.write_text("#!/bin/sh\nexit 3\n")
    broken.chmod(0o755)
    if not os.access(broken, os.X_OK):
        pytest.skip(f"cannot execute {broken} (noexec temp dir?)")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(broken))
    out = tmp_path / "results"
    pv = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp"))
    assert pv.execute_docking() == str(out) and set(pv.failed_proteins) == set(EXAMPLE_PROTEINS)
    assert docking_output(capsys) == "No docked poses in this run.\n"

    def broken_table(self, protein, n):
        raise RuntimeError("no table")

    monkeypatch.setenv("POCKETVINA_ENGINE", str(fake_engine_env))
    monkeypatch.setattr(Results, "_best_scores_head", broken_table)
    assert pv.execute_docking() == str(out) and pv.failed_proteins == {}
    assert docking_output(capsys) == ""
    assert "cannot show the best scores: RuntimeError: no table" in caplog.text


def test_best_scores_need_atlas_results(tmp_path, example_dir, fake_engine_env, stubs):
    plist, smiles = make_inputs(tmp_path, example_dir)
    files = PocketVinaGPU(PocketVinaConfig(protein_paths=str(plist), smiles_file=str(smiles),
                                           output_dir=str(tmp_path / "out")))
    for call in (files.best_scores, files.results):
        with pytest.raises(ValueError, match='set output_format="atlas"'):
            call()
    pv = PocketVinaGPU(atlas_config(plist, smiles, tmp_path / "results", tmp_path / "tmp"))
    with pytest.raises(FileNotFoundError, match=r"yet \(no MANIFEST.json\): run execute_docking\(\) first"):
        pv.best_scores()
    assert fake_engine_env.calls() == []


def test_atlas_missing_engine_aborts_the_run(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",))
    monkeypatch.setenv("POCKETVINA_ENGINE", str(tmp_path / "no_such_engine"))
    out = tmp_path / "results"
    pv = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp"))
    pv.failed_proteins = {"from_an_earlier_run": "..."}
    with pytest.raises(pipeline.RunAborted, match="no_such_engine"):
        pipeline.run_atlas(pv)
    assert pv.failed_proteins == {}  # this run's, not the earlier one's
    assert ResultsDir(out).done_proteins() == set()
    assert os.listdir(tmp_path / "tmp") == []


def test_atlas_invalid_engine_settings_abort_the_run(tmp_path, example_dir, fake_engine_env, stubs):
    """Settings the engine config rejects are the same for every protein: the run stops at once."""
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",),
                                proteins=[FIXTURE_RECEPTOR, example_dir / "proteins" / "1fbl.pdb"])
    out = tmp_path / "results"
    config = atlas_config(plist, smiles, out, tmp_path / "tmp", pocket_config=PocketConfig(thread=500))
    with pytest.raises(pipeline.RunAborted, match="invalid engine settings: thread"):
        pipeline.run_atlas(config)
    assert ResultsDir(out).done_proteins() == set() and fake_engine_env.calls() == []
    assert len(stubs) == 1 and os.listdir(tmp_path / "tmp") == []


def test_run_atlas_takes_any_config_but_only_an_atlas_pocketvina(tmp_path, example_dir, fake_engine_env, stubs):
    """A config made for the "files" format is docked in the atlas layout (no P2Rank folders or logs in the results
    dir, num_modes checked against the atlas tables); a "files" PocketVinaGPU, which already set up that layout in its
    output_dir, is refused."""
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[FIXTURE_RECEPTOR])
    out = tmp_path / "results"
    config = PocketVinaConfig(protein_paths=str(plist), smiles_file=str(smiles), output_dir=str(out),
                              tmp_dir=str(tmp_path / "tmp"), top_pockets=1)
    assert pipeline.run_atlas(config) == str(out)
    assert config.output_format == "files"  # the caller's config is not changed
    assert results_files(out) == expected_files({"AF-Q96LA5-F1-model_v4": "pdbqt"})
    assert len(fake_engine_env.calls()) == 1
    with pytest.raises(ValueError, match="num_modes"):
        pipeline.run_atlas(dataclasses.replace(config, num_modes=200))
    with pytest.raises(ValueError, match='output_format="atlas"'):
        pipeline.run_atlas(PocketVinaGPU(dataclasses.replace(config, output_dir=str(tmp_path / "files"))))
    assert len(fake_engine_env.calls()) == 1


def test_atlas_missing_structure_is_skipped(tmp_path, example_dir, fake_engine_env, stubs):
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",),
                                proteins=[tmp_path / "missing.pdb", FIXTURE_RECEPTOR])
    pv = PocketVinaGPU(atlas_config(plist, smiles, tmp_path / "results", tmp_path / "tmp"))
    pipeline.run_atlas(pv)
    assert list(pv.failed_proteins) == ["missing"] and "structure not found" in pv.failed_proteins["missing"]
    assert ResultsDir(tmp_path / "results").done_proteins() == {"AF-Q96LA5-F1-model_v4"}


def test_atlas_input_errors(tmp_path, example_dir, fake_engine_env, stubs):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        shutil.copy(FIXTURE_RECEPTOR, d / "x.pdbqt")
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[a / "x.pdbqt", b / "x.pdbqt"])
    out = tmp_path / "results"
    with pytest.raises(ValueError, match="same protein_id"):
        pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    plist.write_text(f"{a / 'x.pdbqt'}\n")
    smiles.write_text("molecule_name,smiles\nm1,CCO\nm1,CCN\n")
    with pytest.raises(ValueError, match="duplicate names"):
        pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    smiles.write_text("molecule_name,smiles\n")
    with pytest.raises(ValueError, match="no ligands"):
        pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    smiles.write_text("molecule_name,smiles\nm1,CCO\n")
    plist.write_text("\n")
    with pytest.raises(ValueError, match="no proteins"):
        pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    assert not out.exists() and fake_engine_env.calls() == []


def test_atlas_no_ligand_prepared_registers_no_run(tmp_path, example_dir, fake_engine_env, stubs):
    """If no ligand can be prepared, the run stops before MANIFEST.json: otherwise every protein would be committed
    with nothing docked, and a resume would reuse the failed ligands."""
    plist, _ = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[FIXTURE_RECEPTOR])
    smiles = tmp_path / "bad.csv"
    smiles.write_text("molecule_name,smiles\nsilane,C[Si](C)(C)C\nnot_smiles,C1CC\n")
    out = tmp_path / "results"
    with pytest.raises(RuntimeError, match="none of the 2 ligands could be prepared"):
        pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp")))
    assert not (out / "MANIFEST.json").exists() and ResultsDir(out).done_proteins() == set()
    assert fake_engine_env.calls() == [] and stubs == [] and os.listdir(tmp_path / "tmp") == []


def test_atlas_resume_with_numpy_settings(tmp_path, example_dir, fake_engine_env, stubs):
    """Box sizes taken from numpy/pandas values: the run parameters must still be comparable on a resume."""
    import numpy as np

    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[FIXTURE_RECEPTOR])
    out = tmp_path / "results"
    pocket_config = PocketConfig(size_x=np.int64(20), size_y=np.float32(20), size_z=20, thread=np.int64(8000))
    for _ in range(2):
        pipeline.run_atlas(atlas_config(plist, smiles, out, tmp_path / "tmp", top_pockets=1,
                                        pocket_config=pocket_config))
    assert json.loads((out / "MANIFEST.json").read_text())["params"]["box_size"] == [20, 20, 20]
    assert len(fake_engine_env.calls()) == 1


def test_atlas_no_pockets_without_top_pockets(tmp_path, example_dir, fake_engine_env, monkeypatch):
    """No pockets and no top_pockets: a pose file with no pocket slots (K = 0)."""
    monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets(0))
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[FIXTURE_RECEPTOR])
    out = tmp_path / "results"
    pipeline.run_atlas(atlas_config(plist, smiles, out, tmp_path / "tmp", top_pockets=None))
    pid = "AF-Q96LA5-F1-model_v4"
    assert ResultsDir(out).verify(pid) == []
    prot = read_table(out / "tables" / "proteins" / f"{pid}.parquet").iloc[0]
    assert (prot.status, prot.n_pockets) == ("no_pockets", 0)
    with Reader(out / "poses" / f"{pid}.poses") as r:
        assert (r.N, r.K) == (1, 0) and r.top(0) is None
    assert fake_engine_env.calls() == []


def test_atlas_database_workflow(tmp_path, example_dir, fake_engine_env):
    """Receptors and pockets from a database folder: no P2Rank, no receptor conversion."""
    db = tmp_path / "db"
    structures, pockets = db / "AF_conservation" / "structures", db / "AF_conservation" / "pockets"
    structures.mkdir(parents=True)
    pockets.mkdir(parents=True)
    shutil.copy(FIXTURE_RECEPTOR, structures / "AF-Q96LA5-F1-model_v1_filtered.pdbqt")
    write_pockets(pockets / "AF-Q96LA5-F1-model_v1.pdb.gz_predictions.csv", 2)
    protein_list = tmp_path / "list.csv"
    protein_list.write_text("database,id\nUniProt,Q96LA5\n")
    _, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",))
    out = tmp_path / "results"
    config = PocketVinaConfig(database_path=str(db), protein_list=str(protein_list), smiles_file=str(smiles),
                              output_dir=str(out), output_format="atlas", tmp_dir=str(tmp_path / "tmp"))
    pipeline.run_atlas(config)
    pid = "AF-Q96LA5-F1-model_v1_filtered"
    prot = read_table(out / "tables" / "proteins" / f"{pid}.parquet").iloc[0]
    assert (prot.uniprot, prot.n_pockets, prot.n_pockets_docked) == ("Q96LA5", 2, 2)
    assert json.loads((out / "MANIFEST.json").read_text())["params"]["pocket_source"] == "database"
    assert results_files(out) == expected_files({pid: "pdbqt"})


# --- ligands per engine call, ligand order, engine timeout ---------------------------------------------------------

_OUTPUT_STEM_RE = re.compile(r"-(L\d+)_out\.pdbqt$")


def docked_idx(call) -> list:
    """ligand_idx of the ligands one fake engine call docked, in the order it docked them."""
    return [int(_OUTPUT_STEM_RE.search(p).group(1)[1:]) for p in call["outputs"]]


def test_atlas_ligands_per_call(tmp_path, example_dir, fake_engine_env, stubs, caplog, monkeypatch):
    """Chunks of DEFAULT_LIGANDS_PER_CALL (1000) ligands per engine call by default, in ligand_idx order; 0 or a value
    >= the number of ligands: all of a pocket in one call; ligands_per_call=N: chunks of N; a batch_size set by the
    user is honoured when ligands_per_call is not (as batch_size was); ligands_per_call wins over batch_size. The
    ligands per call are a run parameter (the real engine's poses depend on the chunks), stored as
    main.ligand_chunk_size: at most the number of ligands, so settings that give the same calls are the same run."""
    plist, smiles = make_inputs(tmp_path, example_dir, proteins=[FIXTURE_RECEPTOR])  # 4 ligands
    pid = "AF-Q96LA5-F1-model_v4"
    assert pv_main.DEFAULT_LIGANDS_PER_CALL == 1000
    n_seen = 0
    for name, kw, default, configured, stored, chunks in (
            ("default", {}, None, 1000, 4, [[0, 1, 2, 3]]),
            ("bounded", {}, 3, 3, 3, [[0, 1, 2], [3]]),  # a library larger than the default chunk
            ("zero", dict(ligands_per_call=0), None, 0, 4, [[0, 1, 2, 3]]),
            ("large", dict(ligands_per_call=5), None, 5, 4, [[0, 1, 2, 3]]),
            ("three", dict(ligands_per_call=3), None, 3, 3, [[0, 1, 2], [3]]),
            ("batch_size", dict(batch_size=2), None, 2, 2, [[0, 1], [2, 3]]),
            ("both", dict(batch_size=1, ligands_per_call=3), None, 3, 3, [[0, 1, 2], [3]]),
            ("both_zero", dict(batch_size=1, ligands_per_call=0), None, 0, 4, [[0, 1, 2, 3]])):
        out = tmp_path / name
        caplog.clear()
        with monkeypatch.context() as mp:
            if default is not None:
                mp.setattr(pv_main, "DEFAULT_LIGANDS_PER_CALL", default)
            config = atlas_config(plist, smiles, out, tmp_path / "tmp", **kw)
            assert pv_main.effective_ligands_per_call(config) == configured
            assert pv_main.ligand_chunk_size(config, 4) == stored
            pv = PocketVinaGPU(config)
            with caplog.at_level(logging.INFO, logger="pocketvina"):
                pipeline.run_atlas(pv)
        assert pv.failed_proteins == {}
        calls = fake_engine_env.calls()[n_seen:]
        n_seen += len(calls)
        assert sorted((c["options"]["pocket_rank"], docked_idx(c)) for c in calls) == sorted(
            (k, chunk) for k in (1, 2) for chunk in chunks), name
        assert json.loads((out / "MANIFEST.json").read_text())["params"]["ligands_per_call"] == stored
        ligands = read_table(out / "tables" / "ligands.parquet")
        check_protein(out, pid, ligands, n_pockets=22, K=2, docked=(1, 2))
        summary = json.loads((out / "manifest" / f"{pid}.json").read_text())["summary"]
        assert summary["engine_runs"] == 2 * len(chunks) and summary["engine_runs_failed"] == 0
        assert ("batch_size=1 is ignored: ligands_per_call=" in caplog.text) == name.startswith("both")
        n = len(chunks)
        assert f"4 ligands, at most {stored} per engine call: {n} engine call{'s' * (n > 1)} per pocket" in caplog.text

    # resuming with other chunks is refused; the same chunks set another way are the same run
    with pytest.raises(ValueError, match="other params: ligands_per_call"):
        pipeline.run_atlas(atlas_config(plist, smiles, tmp_path / "default", tmp_path / "tmp", ligands_per_call=2))
    with pytest.raises(ValueError, match="other params: ligands_per_call"):
        pipeline.run_atlas(atlas_config(plist, smiles, tmp_path / "bounded", tmp_path / "tmp"))  # 1000 again
    pipeline.run_atlas(atlas_config(plist, smiles, tmp_path / "batch_size", tmp_path / "tmp", ligands_per_call=2))
    for name, kw in (("default", dict(ligands_per_call=0)), ("default", dict(ligands_per_call=4)),
                     ("zero", {}), ("large", dict(batch_size=50)), ("bounded", dict(ligands_per_call=3))):
        pipeline.run_atlas(atlas_config(plist, smiles, tmp_path / name, tmp_path / "tmp", **kw))
    assert len(fake_engine_env.calls()) == n_seen  # already done
    assert os.listdir(tmp_path / "tmp") == []


def test_atlas_results_do_not_depend_on_the_ligand_file_order(tmp_path, example_dir, fake_engine_env, stubs,
                                                              monkeypatch):
    """With an engine whose ligands share one random number generator (the real one; here the fake engine's
    SHARED_RNG hook) the results are the same whatever order the ligand files are created in, since the engine docks
    them in file name order; other chunks (ligands_per_call) give other poses."""
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SHARED_RNG", "1")
    plist, smiles = make_inputs(tmp_path, example_dir, proteins=[FIXTURE_RECEPTOR])  # 4 ligands
    pid = "AF-Q96LA5-F1-model_v4"
    real_write = pipeline.write_ligand_dir

    def write_newest_first(records, folder):
        """The same files and return value, created in reverse order, next to entries the engine skips."""
        records = list(records)
        for rec in reversed(records):
            real_write([rec], folder)
        os.makedirs(os.path.join(folder, "Lfolder.pdbqt"))
        Path(folder, "notes.txt").write_text("not a ligand\n")
        return {f"L{r.ligand_idx:07d}": r.ligand_idx for r in records if r.status == "ok"}

    def results(out):
        tdir = out / "tables"
        runs = read_table(tdir / "pocket_runs" / f"{pid}.parquet").drop(columns=["runtime_s"])
        return ((out / "poses" / f"{pid}.poses").read_bytes(), read_table(tdir / "modes" / f"{pid}.parquet"),
                read_table(tdir / "pairs" / f"{pid}.parquet"), runs)

    pipeline.run_atlas(atlas_config(plist, smiles, tmp_path / "plain", tmp_path / "tmp", top_pockets=1))
    plain = results(tmp_path / "plain")
    monkeypatch.setattr(pipeline, "write_ligand_dir", write_newest_first)
    pipeline.run_atlas(atlas_config(plist, smiles, tmp_path / "reversed", tmp_path / "tmp", top_pockets=1))
    monkeypatch.setattr(pipeline, "write_ligand_dir", real_write)
    pipeline.run_atlas(atlas_config(plist, smiles, tmp_path / "single", tmp_path / "tmp", top_pockets=1,
                                    ligands_per_call=1))
    reordered, single = results(tmp_path / "reversed"), results(tmp_path / "single")

    assert reordered[0] == plain[0]
    for got, want in zip(reordered[1:], plain[1:]):
        pd.testing.assert_frame_equal(got, want)
    modes = plain[1]
    for i in range(4):  # ligand i is docked i-th in the call
        want = [m[0] for m in fe.score_modes(pid, 1, f"L{i:07d}", SEED, position=i)]
        assert list(modes[modes.ligand_idx == i].affinity) == pytest.approx(want)
    assert single[0] != plain[0]
    single_modes = single[1]
    assert list(single_modes[single_modes.ligand_idx == 3].affinity) == pytest.approx(
        [m[0] for m in fe.score_modes(pid, 1, "L0000003", SEED, position=0)])


def slow_pocket_engine(tmp_path, fake_engine_env, pocket):
    """An engine that docks like the fake engine and then, for ``pocket``, hangs for 60 s: its outputs are written,
    but any engine_timeout below that kills it (with the ``sleep`` it started, via its process group)."""
    exe = tmp_path / "slow_pocket_engine"
    exe.write_text(f'#!/bin/sh\n{shlex.quote(str(fake_engine_env))} "$@"\nrc=$?\n'
                   f'if grep -q "^pocket_rank = {pocket}$" "$2"; then sleep 60; fi\nexit $rc\n')
    exe.chmod(0o755)
    if not os.access(exe, os.X_OK):
        pytest.skip(f"cannot execute {exe} (noexec temp dir?)")
    return exe


def test_atlas_engine_timeout(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    """A call that takes longer than engine_timeout is killed; its ligands are engine_error runs with a clear message
    (its outputs, maybe cut short, are not used); a protein whose calls all time out is left for a rerun."""
    plist, smiles = make_inputs(tmp_path, example_dir, proteins=[FIXTURE_RECEPTOR])  # 4 ligands
    pid = "AF-Q96LA5-F1-model_v4"
    monkeypatch.setenv("POCKETVINA_ENGINE", str(slow_pocket_engine(tmp_path, fake_engine_env, pocket=2)))
    out = tmp_path / "results"
    t0 = time.monotonic()
    pv = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp", engine_timeout=2))
    pipeline.run_atlas(pv)
    assert time.monotonic() - t0 < 50
    assert pv.failed_proteins == {}
    assert len(fake_engine_env.calls()) == 2  # the pocket 2 call wrote its outputs, then hung
    message = ("engine timed out after 2 s (engine_timeout) and was killed; none of the 4 ligands of this call was "
               "docked")
    runs = read_table(out / "tables" / "pocket_runs" / f"{pid}.parquet")
    assert set(runs[runs.pocket == 1].status) == {"ok"}
    assert set(runs[runs.pocket == 2].status) == {"engine_error"} and set(runs[runs.pocket == 2].error) == {message}
    assert (runs[runs.pocket == 2].n_modes == 0).all()
    pairs = read_table(out / "tables" / "pairs" / f"{pid}.parquet")
    assert set(pairs.best_pocket) == {1}
    prot = read_table(out / "tables" / "proteins" / f"{pid}.parquet").iloc[0]
    assert (prot.status, prot.error) == ("ok", "1 of 2 engine runs failed")
    summary = json.loads((out / "manifest" / f"{pid}.json").read_text())["summary"]
    assert summary["engine_runs_failed"] == 1 and summary["runs_by_status"] == {"ok": 4, "engine_error": 4}
    assert "engine_timeout" not in json.loads((out / "MANIFEST.json").read_text())["params"]  # not a result setting

    # every call times out: nothing is committed, the reason is in failed_proteins
    monkeypatch.setenv("POCKETVINA_ENGINE", str(fake_engine_env))
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "60")
    out = tmp_path / "all_timed_out"
    pv = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp", top_pockets=1, engine_timeout=0.5))
    pipeline.run_atlas(pv)
    assert ResultsDir(out).done_proteins() == set()
    assert "all 1 engine runs failed, the last one: engine timed out after 0.5 s" in pv.failed_proteins[pid]
    assert os.listdir(tmp_path / "tmp") == []


# --- "files" format ------------------------------------------------------------------------------------------------

def test_files_format_combined_results(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    """The pre-v0.2 layout: batch folders, one output per ligand x pocket, combined_results.csv with the same columns.
    Relative paths in the protein list are relative to the list's folder, only this run's outputs are combined, and
    the engine runs in a temp dir with the run's seed."""
    inputs = tmp_path / "inputs"
    (inputs / "proteins").mkdir(parents=True)
    for p in EXAMPLE_PROTEINS:
        shutil.copy(example_dir / "proteins" / f"{p}.pdb", inputs / "proteins")
    _, smiles = make_inputs(tmp_path, example_dir)
    p2rank_calls = []

    def run_p2rank(self, pdb_file):
        p2rank_calls.append(pdb_file)
        base = os.path.splitext(os.path.basename(pdb_file))[0]
        write_pockets(os.path.join(self.pockets_dir, f"{base}.csv"), 3)
        return True

    monkeypatch.setattr(P2RankProcessor, "run_p2rank", run_p2rank)
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    expected = {}
    for n, p in enumerate(EXAMPLE_PROTEINS):
        (inputs / "proteins.txt").write_text(f"proteins/{p}.pdb\n")  # relative to inputs/
        config = PocketVinaConfig(protein_paths=os.path.relpath(inputs / "proteins.txt"), smiles_file=str(smiles),
                                  output_dir="out", top_pockets=2)
        assert config.output_format == "files" and config.output_dir == str(cwd / "out")
        assert config.protein_paths == str(inputs / "proteins.txt")
        PocketVinaGPU(config).execute_docking()

        df = pd.read_csv(cwd / "out" / "combined_results.csv", float_precision="round_trip")
        assert list(df.columns) == LEGACY_CSV_COLUMNS
        # the second run reuses the batch folder and must not combine the first protein's outputs again
        expected = {(f"{p}-pocket{k}", name, e, lb, ub, mode, k)
                    for name in FEW_LIGANDS for k in (1, 2)
                    for mode, (e, lb, ub) in enumerate(fe.score_modes(p, k, f"{name}_cleaned", SEED), 1)}
        assert set(df.itertuples(index=False, name=None)) == expected and len(df) == len(expected)
        assert p2rank_calls[n] == str(inputs / "proteins" / f"{p}.pdb")

    out = cwd / "out"
    assert sorted(os.listdir(out / "batch_1")) == sorted(f"{n}_cleaned.pdbqt" for n in FEW_LIGANDS)
    assert len(list((out / "batch_1_out").glob("*.json"))) == 2 * 2 * len(FEW_LIGANDS)  # both runs' outputs
    calls = fake_engine_env.calls()
    assert len(calls) == 4
    for c in calls:
        assert c["options"]["seed"] == SEED and c["returncode"] == 0
        assert not within(c["cwd"], PACKAGE_DIR) and not os.path.exists(c["cwd"])  # a temp dir, removed
        assert c["options"]["receptor"].startswith(str(out / "proteins" / "structures"))
    config_files = sorted((out / "configs").iterdir())
    assert len(config_files) == 4
    assert f"ligand_directory = {out / 'batch_1'}\n" in config_files[0].read_text()

    # a rerun whose engine fails must not report the outputs already on disk as its results
    broken = tmp_path / "broken_engine"
    broken.write_text("#!/bin/sh\nexit 1\n")
    broken.chmod(0o755)
    if not os.access(broken, os.X_OK):
        pytest.skip(f"cannot execute {broken} (noexec temp dir?)")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(broken))
    PocketVinaGPU(config).execute_docking()
    df = pd.read_csv(out / "combined_results.csv")
    assert list(df.columns) == LEGACY_CSV_COLUMNS and len(df) == 0
    assert json.loads((out / "combined_results.json").read_text()) == []


def test_files_format_engine_timeout(tmp_path, example_dir, fake_engine_env, monkeypatch, caplog):
    """engine_timeout applies to the "files" format too: a call that timed out is logged and its outputs (maybe cut
    short) are left out of combined_results.csv. ligands_per_call does not apply (a warning says so)."""
    db, protein_list = make_database(tmp_path)
    _, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"))
    monkeypatch.setenv("POCKETVINA_ENGINE", str(slow_pocket_engine(tmp_path, fake_engine_env, pocket=2)))
    out = tmp_path / "out"
    pv = PocketVinaGPU(PocketVinaConfig(database_path=str(db), protein_list=str(protein_list), smiles_file=str(smiles),
                                        output_dir=str(out), engine_timeout=2, ligands_per_call=1))
    assert pv.batch_size == 10
    pv.execute_docking()
    df = pd.read_csv(out / "combined_results.csv")
    assert set(df["pocket rank"]) == {1} and set(df.molecule_name) == {"molecule1", "molecule2"}
    assert len(fake_engine_env.calls()) == 2  # one batch of 2 ligands per pocket; pocket 2 wrote its outputs, then hung
    assert len(list((out / "batch_1_out").glob("*pocket2*.json"))) == 2
    assert "engine timed out after 2 s (engine_timeout) and was killed" in caplog.text
    assert 'ligands_per_call applies to output_format="atlas" only' in caplog.text


def test_files_format_rejects_colliding_ligand_names(tmp_path, example_dir, fake_engine_env, stubs):
    plist, _ = make_inputs(tmp_path, example_dir)
    smiles = tmp_path / "ligands.csv"
    for rows, message in (("a,CCO\na,CCN\n", "duplicate names"),
                          ("x.pdbqt1,CCO\nx.pdbqt2,CCN\n", "same engine output name"),
                          ("sub/x,CCO\n", "not file names")):
        smiles.write_text("molecule_name,smiles\n" + rows)
        out = tmp_path / "out"
        pv = PocketVinaGPU(PocketVinaConfig(protein_paths=str(plist), smiles_file=str(smiles), output_dir=str(out)))
        with pytest.raises(ValueError, match=message):
            pv.execute_docking()
        assert not list(out.glob("*.pdbqt")) and not list(out.glob("batch_*"))
    assert fake_engine_env.calls() == []


def test_files_format_ligand_that_cannot_be_written(tmp_path):
    """A ligand whose file cannot be created (name too long) is skipped; the ligands after it are still written."""
    smiles = tmp_path / "ligands.csv"
    smiles.write_text(f"molecule_name,smiles\na,CCO\n{'x' * 300},CCN\nb,CCC\n")
    plist = tmp_path / "proteins.txt"
    plist.write_text("x.pdb\n")
    out = tmp_path / "out"
    PocketVinaGPU(PocketVinaConfig(protein_paths=str(plist), smiles_file=str(smiles), output_dir=str(out))
                  ).smiles_to_pdbqt()
    assert sorted(out.glob("*.pdbqt")) == [out / "a_cleaned.pdbqt", out / "b_cleaned.pdbqt"]


def make_database(tmp_path):
    """(database folder, protein list) with the fixture receptor as UniProt Q96LA5 (2 pockets) + a missing entry."""
    db = tmp_path / "db"
    structures, pockets = db / "AF_conservation" / "structures", db / "AF_conservation" / "pockets"
    structures.mkdir(parents=True)
    pockets.mkdir(parents=True)
    shutil.copy(FIXTURE_RECEPTOR, structures / "AF-Q96LA5-F1-model_v1_filtered.pdbqt")
    write_pockets(pockets / "AF-Q96LA5-F1-model_v1.pdb.gz_predictions.csv", 2)
    protein_list = tmp_path / "list.csv"
    protein_list.write_text("database,id\nUniProt,Q96LA5\nUniProt,MISSING\n")
    return db, protein_list


def check_database_combined_csv(out, names):
    """combined_results.csv of the ``make_database`` protein docked with the fake engine (legacy protein names)."""
    df = pd.read_csv(out / "combined_results.csv", float_precision="round_trip")
    assert list(df.columns) == LEGACY_CSV_COLUMNS
    expected = {("AF-Q96LA5-F1-model_v1", name, e, lb, ub, mode, k) for name in names
                for k in (1, 2) for mode, (e, lb, ub) in enumerate(
                    fe.score_modes("AF-Q96LA5-F1-model_v1_filtered", k, f"{name}_cleaned", SEED), 1)}
    assert set(df.itertuples(index=False, name=None)) == expected and len(df) == len(expected)


def test_files_format_database_workflow(tmp_path, example_dir, fake_engine_env):
    """Receptor PDBQT and pockets CSV from a database folder (no P2Rank); legacy protein names in the CSV."""
    db, protein_list = make_database(tmp_path)
    receptor = db / "AF_conservation" / "structures" / "AF-Q96LA5-F1-model_v1_filtered.pdbqt"
    _, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"))
    out = tmp_path / "out"
    pv = PocketVinaGPU(PocketVinaConfig(database_path=str(db), protein_list=str(protein_list),
                                        smiles_file=str(smiles), output_dir=str(out)))
    proteins = pv.process_proteins()
    assert list(proteins) == ["UniProt_Q96LA5"] and proteins["UniProt_Q96LA5"]["pdbqt_path"] == str(receptor)
    assert [p["rank"] for p in proteins["UniProt_Q96LA5"]["pockets"]] == [1, 2]

    pv.execute_docking()
    check_database_combined_csv(out, ("molecule1", "molecule2"))
    assert len(fake_engine_env.calls()) == 2
    assert not (out / "proteins").exists()  # no P2Rank in the database workflow


NO_PYARROW = """\
import sys
for name in ("pyarrow", "pyarrow.parquet", "duckdb"):
    sys.modules[name] = None  # importing them now fails
try:
    import pandas  # noqa: F401
except ImportError:
    sys.exit(77)  # a pandas that needs pyarrow itself
from pocketvina import *  # the pre-v0.2 public names (failed while __all__ listed the lazy run_atlas)
from pocketvina import PocketVinaConfig, PocketVinaGPU
from pocketvina.main import P2RankConfig, P2RankProcessor

db, protein_list, smiles, out, pdb = sys.argv[1:]
PocketVinaGPU(PocketVinaConfig(database_path=db, protein_list=protein_list, smiles_file=smiles,
                               output_dir=out)).execute_docking()
print(P2RankProcessor(P2RankConfig(output_dir=out + "_p2rank")).process_pdb_file(pdb))
"""


def test_files_format_needs_no_pyarrow(tmp_path, example_dir, fake_engine_env):
    """The "files" format (the pre-v0.2 API, ``from pocketvina import *`` included) docks and converts receptors
    without pyarrow or duckdb, which setup.py does not install: all used to fail with ModuleNotFoundError. A
    subprocess, so the imports can be blocked."""
    db, protein_list = make_database(tmp_path)
    _, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"))
    out = tmp_path / "out"
    pdb = crop_pdb(example_dir / "proteins" / "1fbl.pdb", tmp_path / "r.pdb")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))
    proc = subprocess.run([sys.executable, "-c", NO_PYARROW, str(db), str(protein_list), str(smiles), str(out),
                           str(pdb)], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=600)
    if proc.returncode == 77:
        pytest.skip("this pandas cannot be imported without pyarrow")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    check_database_combined_csv(out, ("molecule1", "molecule2"))
    receptor = Path(proc.stdout.strip().splitlines()[-1])
    assert receptor == tmp_path / "out_p2rank" / "proteins" / "structures" / "r.pdbqt"
    assert receptor.read_text() == pipeline.receptor_pdbqt(pdb)


# --- pieces --------------------------------------------------------------------------------------------------------

def crop_pdb(src, dst, n_atoms=CROP_ATOMS):
    with open(src) as f:
        lines = [line for line in f if line.startswith(("ATOM", "HETATM"))][:n_atoms]
    with open(dst, "w") as f:
        f.writelines(lines + ["HETATM 9999  O   HOH A 999      1.000   1.000   1.000  1.00  0.00           O\n"])
    return dst


def test_receptor_pdbqt_matches_obabel_cli(tmp_path, example_dir):
    """Same text as ``obabel x.pdb -xr -O x.pdbqt`` (rigid) on the cleaned PDB, ATOM lines kept; ``legacy=True``: the
    pre-v0.2 ``obabel x.pdb -O x.pdbqt``, the same atom lines in another order."""
    obabel = find_obabel()
    if obabel is None:
        pytest.skip("no obabel executable")
    pdb = crop_pdb(example_dir / "proteins" / "1fbl.pdb", tmp_path / "r.pdb")
    cleaned = tmp_path / "r_cleaned.pdb"
    cleaned.write_text("".join(line for line in pdb.read_text().splitlines(True)
                               if line.startswith(("ATOM", "HETATM")) and "HOH" not in line))

    def obabel_atoms(*options):
        out = tmp_path / "r_temp.pdbqt"
        subprocess.run([obabel, str(cleaned), *options, "-O", str(out)], check=True, capture_output=True)
        return "".join(line for line in out.read_text().splitlines(True) if line.startswith("ATOM"))

    text = pipeline.receptor_pdbqt(pdb)
    assert text == obabel_atoms("-xr") and text.count("\n") == CROP_ATOMS
    legacy = pipeline.receptor_pdbqt(pdb, legacy=True)
    assert legacy == obabel_atoms()
    assert sorted(line[12:] for line in legacy.splitlines()) == sorted(line[12:] for line in text.splitlines())
    for flag, expected in ((False, text), (True, legacy)):  # also in the child process used while engines dock
        dest = tmp_path / f"child_{flag}.pdbqt"
        assert pipeline._receptor_pdbqt(str(pdb), str(dest), lambda: None, legacy=flag) == expected.encode()

    gz = tmp_path / "r.pdb.gz"
    gz.write_bytes(gzip.compress(pdb.read_bytes()))
    assert pipeline.receptor_pdbqt(gz) == text
    crlf = tmp_path / "crlf.pdb"
    crlf.write_bytes(pdb.read_bytes().replace(b"\n", b"\r\n"))
    assert pipeline.receptor_pdbqt(crlf) == text
    assert pipeline.receptor_pdbqt(FIXTURE_RECEPTOR) == FIXTURE_RECEPTOR.read_text()  # PDBQT: as it is


def test_receptor_pdbqt_from_mmcif(tmp_path, example_dir):
    from openbabel import pybel

    pdb = crop_pdb(example_dir / "proteins" / "1fbl.pdb", tmp_path / "r.pdb")
    cif = tmp_path / "r.cif"
    cif.write_text(pybel.readstring("pdb", pdb.read_text()).write("mmcif"))
    text = pipeline.receptor_pdbqt(cif)
    assert text.count("\n") == CROP_ATOMS and all(line.startswith("ATOM") for line in text.splitlines())


def test_pdb_for_p2rank_from_pdbqt():
    atoms = [line for line in FIXTURE_RECEPTOR.read_text().splitlines() if line.startswith("ATOM")]
    lines = pipeline._pdb_from_pdbqt(FIXTURE_RECEPTOR.read_text()).splitlines()
    assert lines[-1] == "END" and sorted(line[:66] for line in lines[:-1]) == sorted(a[:66] for a in atoms)
    elements = {a[:66]: {"OA": "O", "NA": "N", "SA": "S", "HD": "H", "A": "C"}.get(a[77:].strip(), a[77:].strip())
                for a in atoms}
    for line in lines[:-1]:
        assert len(line) == 78 and line[76:78].strip() == elements[line[:66]].upper()
    residues = [(line[21], int(line[22:26])) for line in lines[:-1]]
    starts = [r for n, r in enumerate(residues) if n == 0 or r != residues[n - 1]]
    assert len(starts) == len(set(residues))  # each residue in one piece (the PDBQT has them split up)
    assert len(set(residues)) < sum(1 for n, a in enumerate(atoms) if n == 0 or a[21:26] != atoms[n - 1][21:26])


def test_read_pockets_csv(tmp_path):
    pockets = pipeline.read_pockets_csv(POCKETS_CSV)
    assert [p.rank for p in pockets] == list(range(1, 23))
    p1 = pockets[0]
    assert (p1.name, p1.score, p1.probability, p1.center) == ("pocket1", 5.73, 0.278, (4.9008, 6.5637, -15.0131))
    assert p1.residue_ids.startswith("A_147 A_150")
    minimal = tmp_path / "minimal.csv"
    minimal.write_text("rank,center_x,center_y,center_z\n2,1,2,3\n1,4,5,6\n")
    assert [(p.rank, p.name, p.probability, p.residue_ids) for p in pipeline.read_pockets_csv(minimal)] == [
        (1, "pocket1", None, None), (2, "pocket2", None, None)]
    for body, message in (("rank,center_x\n1,0\n", "missing pocket columns"),
                          ("rank,center_x,center_y,center_z\n0,1,2,3\n", "invalid pocket rank"),
                          ("rank,center_x,center_y,center_z\n1,1,,3\n", "no valid center"),
                          ("rank,center_x,center_y,center_z\n1,1,2,3\n1,1,2,3\n", "duplicate pocket ranks")):
        minimal.write_text(body)
        with pytest.raises(ValueError, match=message):
            pipeline.read_pockets_csv(minimal)


def test_inputs_helpers(tmp_path, monkeypatch):
    assert pipeline.uniprot_from_id("AF-Q96LA5-F1-model_v4") == "Q96LA5"
    assert pipeline.uniprot_from_id("1fbl") is None
    plist = tmp_path / "lists" / "proteins.txt"
    plist.parent.mkdir()
    plist.write_text(f"a.pdb\n\n  ../b/c.pdb  \n{tmp_path / 'abs.pdb'}\n")
    assert read_protein_list(plist) == [str(tmp_path / "lists" / "a.pdb"), str(tmp_path / "b" / "c.pdb"),
                                        str(tmp_path / "abs.pdb")]
    # pre-v0.2 lists held paths relative to the working directory: still found when not next to the list
    (tmp_path / "cwd" / "lists").mkdir(parents=True)
    (tmp_path / "cwd" / "lists" / "old.pdb").write_text("")
    (tmp_path / "lists" / "both.pdb").write_text("")
    (tmp_path / "cwd" / "both.pdb").write_text("")
    plist.write_text("lists/old.pdb\nboth.pdb\nmissing.pdb\n")
    monkeypatch.chdir(tmp_path / "cwd")
    assert read_protein_list(plist) == [str(tmp_path / "cwd" / "lists" / "old.pdb"),
                                        str(tmp_path / "lists" / "both.pdb"), str(tmp_path / "lists" / "missing.pdb")]
    check_ligand_names(["a", "b"])
    check_ligand_names(["x.pdbqt1", "x.pdbqt2"], files=False)  # atlas ligand files are named by index
    with pytest.raises(ValueError, match="duplicate"):
        check_ligand_names(["a", "a"], files=False)


def test_scan_engine_log(tmp_path):
    log = tmp_path / "engine.log"
    log.write_bytes(b"\nUsing random seed: 7\nDocking ligand: /t/batch_1/L12345678.pdbqt\n"
                    b"Parse error on line 3 in file \"/t/batch_1/L12345678.pdbqt\": Unknown or inappropriate tag\n"
                    b"\rPerform docking|===   |\rPerform docking|======|\n"
                    b"Parse error in /t/batch_1/L0000002.pdbqt: Missing TORSDOF\n")
    messages, seed = pipeline._scan_engine_log(log, {"L12345678": 12345678, "L0000001": 1, "L0000002": 2})
    assert seed == 7
    assert messages == {"L12345678": 'Parse error on line 3 in file "/t/batch_1/L12345678.pdbqt": Unknown or '
                                     'inappropriate tag',
                        "L0000002": "Parse error in /t/batch_1/L0000002.pdbqt: Missing TORSDOF"}
    assert pipeline._scan_engine_log(tmp_path / "missing.log", {"L0000001": 1}) == ({}, None)


def test_config_fields_and_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = dict(protein_paths="p.txt", smiles_file="l.csv")
    cfg = PocketVinaConfig(**base)
    assert (cfg.output_format, cfg.top_pockets, cfg.seed, cfg.ligand_prep, cfg.gzip_level) == (
        "files", None, 42, "standard", 6)
    assert (cfg.batch_size, cfg.num_modes, cfg.energy_range, cfg.tmp_dir) == (None, None, None, None)
    assert (cfg.ligands_per_call, cfg.engine_timeout) == (None, None)
    assert pv_main.effective_ligands_per_call(cfg) == pv_main.DEFAULT_LIGANDS_PER_CALL == 1000
    assert pv_main.DEFAULT_BATCH_SIZE == 10  # what PocketVinaGPU uses for the "files" batches when batch_size is None
    for kw, want in ((dict(batch_size=5), 5), (dict(ligands_per_call=7), 7),
                     (dict(batch_size=5, ligands_per_call=7), 7), (dict(ligands_per_call=0), 0),
                     (dict(batch_size=5, ligands_per_call=0), 0)):
        assert pv_main.effective_ligands_per_call(PocketVinaConfig(**base, **kw)) == want
    # ligands per call of a library of n: all of them when 0 or not smaller; at least 1
    for kw, n, want in (({}, 4, 4), ({}, 999, 999), ({}, 1000, 1000), ({}, 1001, 1000), ({}, 250000, 1000),
                        (dict(ligands_per_call=0), 5000, 5000), (dict(ligands_per_call=7), 3, 3),
                        (dict(ligands_per_call=7), 8, 7), (dict(batch_size=5), 12, 5), ({}, 0, 1),
                        (dict(ligands_per_call=0), 0, 1)):
        assert pv_main.ligand_chunk_size(PocketVinaConfig(**base, **kw), n) == want, (kw, n)
    cfg = PocketVinaConfig(**base, engine_timeout=600, ligands_per_call=500)
    assert (cfg.engine_timeout, cfg.ligands_per_call) == (600.0, 500) and isinstance(cfg.engine_timeout, float)
    assert cfg.output_dir == str(tmp_path / "output") and cfg.smiles_file == str(tmp_path / "l.csv")
    assert PocketVinaConfig(**base, tmp_dir="t").tmp_dir == str(tmp_path / "t")
    for bad in (dict(output_format="sqlite"), dict(ligand_prep="meeko"), dict(top_pockets=0), dict(top_pockets=128),
                dict(seed=-1), dict(seed=1.5), dict(num_modes=0), dict(output_format="atlas", num_modes=128),
                dict(energy_range=-1), dict(energy_range=float("nan")), dict(gzip_level=10), dict(batch_size=0),
                dict(ligands_per_call=-1), dict(ligands_per_call=2.5), dict(ligands_per_call=True),
                dict(engine_timeout=0), dict(engine_timeout=-5), dict(engine_timeout=float("inf")),
                dict(engine_timeout=float("nan")), dict(engine_timeout="600"), dict(engine_timeout=True)):
        with pytest.raises(ValueError):
            PocketVinaConfig(**base, **bad)
    assert PocketVinaConfig(**base, num_modes=200).num_modes == 200  # only the atlas tables need <= 127
    assert PocketVinaConfig(**base, seed=None).seed is None


def test_predict_pockets_calls_while_running_and_kills_p2rank_on_errors(tmp_path, example_dir, monkeypatch):
    """A fake ``prank`` on PATH: ``while_running`` is called while it runs (the slot pool's results are recorded
    meanwhile) and its output is still read whole; if ``while_running`` raises, P2Rank is killed and reaped."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    prank = bin_dir / "prank"
    pid_file, args = tmp_path / "prank.pid", tmp_path / "prank.args"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.delenv("P2RANK_HOME", raising=False)  # it would come before PATH (pocketvina.p2rank.find_p2rank)
    structure = example_dir / "proteins" / "1fbl.pdb"

    def install(body):
        prank.write_text(f"#!/bin/sh\necho $$ > {shlex.quote(str(pid_file))}\necho \"$@\" > {shlex.quote(str(args))}\n"
                         "while [ $# -gt 0 ]; do case \"$1\" in -o) out=\"$2\"; shift;; -f) f=\"$2\"; shift;; esac; "
                         f"shift; done\n{body}\n")
        prank.chmod(0o755)
        if not os.access(prank, os.X_OK):
            pytest.skip(f"cannot execute {prank} (noexec temp dir?)")

    install(f"seq 1 3000\nsleep 1.6\ncp {shlex.quote(str(POCKETS_CSV))} \"$out/$(basename \"$f\")_predictions.csv\"\n"
            "echo prediction done")
    ticks = []
    csv_path = pipeline.predict_pockets(structure, tmp_path / "p2rank", while_running=lambda: ticks.append(1))
    assert len(ticks) >= 2 and len(pipeline.read_pockets_csv(csv_path)) == 22
    # with P2Rank's visualizations (its SAS points file), without a copy of the protein in them
    assert " -visualizations true " in args.read_text() and " -vis_copy_proteins false " in args.read_text()
    install("exec sleep 60")

    def stop():
        raise RuntimeError("stop (test)")

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="stop"):
        pipeline.predict_pockets(structure, tmp_path / "p2rank_stopped", while_running=stop)
    assert time.monotonic() - t0 < 10
    with pytest.raises(ProcessLookupError):  # killed and reaped
        os.kill(int(pid_file.read_text()), 0)

    # the real launcher runs Java as a child (no exec): it is killed too, also on Ctrl-C
    child_file = tmp_path / "java.pid"
    install(f"sleep 60 &\necho $! > {shlex.quote(str(child_file))}\nwait $!")

    def stop_once_java_runs():
        if child_file.exists() and child_file.read_text().strip():
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        pipeline.predict_pockets(structure, tmp_path / "p2rank_interrupted", while_running=stop_once_java_runs)
    java = int(child_file.read_text())
    deadline = time.monotonic() + 10
    while _running(java) and time.monotonic() < deadline:
        time.sleep(0.05)
    try:
        assert not _running(java), "P2Rank's child process was left running"
    finally:
        if _running(java):
            os.kill(java, signal.SIGKILL)


def test_receptor_pdbqt_in_a_child_process(tmp_path, example_dir, monkeypatch, caplog):
    """While engine calls are due (``while_running``), OpenBabel, which holds the GIL, converts the receptor in a
    child process: the same bytes, ``while_running`` is called meanwhile and other threads keep running. Its errors
    are raised as in this process; if ``while_running`` raises, the child is killed; a child that cannot run, a
    PDBQT input or no ``while_running`` convert in this process. No timing margins: the child ends only once another
    thread of this process has taken 20 steps while it ran and ``while_running`` was called (the file "go"); if this
    process were blocked meanwhile (OpenBabel holding its GIL), the child gives up waiting after 30 s, no "saw_go"."""
    pdb = crop_pdb(example_dir / "proteins" / "1fbl.pdb", tmp_path / "r.pdb", n_atoms=600)
    expected = pipeline.receptor_pdbqt(pdb).encode("utf-8")
    gate, code = tmp_path / "gate", pipeline._RECEPTOR_CODE
    gate.mkdir()
    monkeypatch.setattr(pipeline, "_RECEPTOR_CODE",
                        f"open({str(gate / 'started')!r}, 'w').close()\n{code}import os, time\n"
                        f"deadline = time.monotonic() + 30\n"
                        f"while not os.path.exists({str(gate / 'go')!r}) and time.monotonic() < deadline:\n"
                        f"    time.sleep(0.01)\n"
                        f"if os.path.exists({str(gate / 'go')!r}):\n"
                        f"    open({str(gate / 'saw_go')!r}, 'w').close()\n")
    steps, polls, stop = [], [], threading.Event()

    def tick():  # another thread: 20 steps while the child runs, then "go" (once while_running was called too)
        while not stop.is_set() and (len(steps) < 20 or not polls):
            if (gate / "started").exists():
                steps.append(1)
            time.sleep(0.01)
        (gate / "go").touch()

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    monkeypatch.setattr(pipeline, "P2RANK_POLL_S", 0.05)  # polled often: the test is quick
    try:
        got = pipeline._receptor_pdbqt(str(pdb), str(tmp_path / "child.pdbqt"), lambda: polls.append(1))
    finally:
        stop.set()
        ticker.join()
    monkeypatch.setattr(pipeline, "_RECEPTOR_CODE", code)
    assert got == expected and (tmp_path / "child.pdbqt").read_bytes() == expected
    assert polls and "converting it here" not in caplog.text
    assert (gate / "saw_go").exists() and len(steps) >= 20  # this process ran on while the child ran: no GIL held

    empty = tmp_path / "water.pdb"
    empty.write_text("HETATM 9999  O   HOH A 999      1.000   1.000   1.000  1.00  0.00           O\n")
    with pytest.raises(ValueError) as here:
        pipeline.receptor_pdbqt(str(empty))
    with pytest.raises(ValueError) as child:
        pipeline._receptor_pdbqt(str(empty), str(tmp_path / "empty.pdbqt"), lambda: None)
    assert str(child.value) == str(here.value) and "no ATOM/HETATM records" in str(here.value)

    pid_file = tmp_path / "child.pid"
    monkeypatch.setattr(pipeline, "_RECEPTOR_CODE",
                        f"import os, time\nopen({str(pid_file)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n")

    def stop_once_started():
        if pid_file.exists() and pid_file.read_text():
            raise KeyboardInterrupt

    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        pipeline._receptor_pdbqt(str(pdb), str(tmp_path / "killed.pdbqt"), stop_once_started)
    assert time.monotonic() - t0 < 30 and not _running(int(pid_file.read_text()))

    # a child killed by a signal (OpenBabel crashed, the OOM killer): converting the receptor here would take the
    # docking process down too, with the engine calls under way; the protein fails instead
    monkeypatch.setattr(pipeline, "_RECEPTOR_CODE", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
    with pytest.raises(RuntimeError, match="receptor conversion process was killed by SIGKILL"):
        pipeline._receptor_pdbqt(str(pdb), str(tmp_path / "crashed.pdbqt"), lambda: None)
    assert "converting it here" not in caplog.text

    monkeypatch.setattr(pipeline, "_RECEPTOR_CODE", "import sys; sys.exit(1)")  # e.g. pocketvina not importable
    assert pipeline._receptor_pdbqt(str(pdb), str(tmp_path / "fallback.pdbqt"), lambda: None) == expected
    assert caplog.text.count("converting it here") == 1
    assert pipeline._receptor_pdbqt(str(FIXTURE_RECEPTOR), str(tmp_path / "f.pdbqt"), lambda: None) == \
        FIXTURE_RECEPTOR.read_bytes()
    assert pipeline._receptor_pdbqt(str(pdb), str(tmp_path / "here.pdbqt")) == expected
    assert caplog.text.count("converting it here") == 1  # neither used a child process
    for exe in (None, ""):  # an embedded interpreter (sys.executable unknown): Popen raised TypeError for None
        monkeypatch.setattr(sys, "executable", exe)
        assert pipeline._receptor_pdbqt(str(pdb), str(tmp_path / "no_exe.pdbqt"), lambda: None) == expected
    assert caplog.text.count("converting it here") == 1


needs_p2rank =pytest.mark.skipif(os.environ.get("POCKETVINA_P2RANK_TESTS") != "1" or shutil.which("java") is None,
                                  reason="runs P2Rank (Java): set POCKETVINA_P2RANK_TESTS=1")


@needs_p2rank
def test_real_p2rank(tmp_path):
    """P2Rank on a PDBQT receptor (converted to PDB for it), in a temp folder. The fixture is cut to the region of
    two pockets; the conservation model finds nothing there without conservation data, the "alphafold" model does."""
    p2rank = P2RankConfig(threads=2, model="alphafold")
    csv_path = pipeline.predict_pockets(FIXTURE_RECEPTOR, tmp_path / "p2rank", p2rank)
    assert within(csv_path, tmp_path / "p2rank")
    pockets = pipeline.read_pockets_csv(csv_path)
    assert pockets and pockets[0].rank == 1 and pockets[0].residue_ids
    # its SAS points: the points of every pocket, and others; no copy of the protein next to them
    points_file = sas.points_file(csv_path)
    assert points_file and os.listdir(os.path.dirname(points_file)) == [os.path.basename(points_file)]
    points = sas.read_points(points_file)
    assert {p[0] for p in points} - {0} == {p.rank for p in pockets} and any(p[0] == 0 for p in points)


@needs_p2rank
def test_real_p2rank_and_receptor_end_to_end(tmp_path, example_dir, fake_engine_env):
    """Nothing stubbed but the engine: P2Rank and the full receptor conversion of an example protein (~40 s)."""
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"),
                                proteins=[example_dir / "proteins" / "1fbl.pdb"])
    out = tmp_path / "results"
    config = atlas_config(plist, smiles, out, tmp_path / "tmp", top_pockets=1,
                          p2rank_config=P2RankConfig(threads=2))
    pv = PocketVinaGPU(config)
    pipeline.run_atlas(pv)
    assert pv.failed_proteins == {}
    ligands = read_table(out / "tables" / "ligands.parquet")
    n_pockets = len(read_table(out / "tables" / "pockets" / "1fbl.parquet"))
    check_protein(out, "1fbl", ligands, n_pockets=n_pockets, K=1, docked=(1,))
    assert results_files(out) == expected_files({"1fbl": "pdb"}) | {"sas/1fbl.json.gz"}
    assert os.listdir(tmp_path / "tmp") == []
    with Results(out) as res:
        points = res.sas("1fbl")
    assert list(points["pockets"]) == ["1"] and len(points["other_pockets"]) == n_pockets - 1 and points["rest"]
