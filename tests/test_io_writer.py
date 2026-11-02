"""pocketvina.io.tables and pocketvina.io.writer: schemas, atomic moves, results directory, per-protein writer."""
import csv
import errno
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import fake_engine as fe
from pocketvina import pdbqt
from pocketvina.io import posefile, tables
from pocketvina.io import writer as wr
from pocketvina.io.tables import read_table, write_table
from pocketvina.io.writer import ProteinWriter, ResultsDir, atomic_place, protein_id_from_path, resolve_tmp_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
ATLAS_SCRIPTS = os.environ.get("POCKETVINA_ATLAS_SCRIPTS",
                               "/data/akalin/asarigun/pocketvina_atlas/_work/db_prototype/scripts")
ATLAS_POSEFILE = Path(ATLAS_SCRIPTS) / "posefile.py"
PID = "AF-Q96LA5-F1-model_v4"


@pytest.fixture(scope="module")
def atlas_pf():
    """The original Atlas pose-file module, imported by path (skipped where the Atlas folder is not available)."""
    if not os.path.isfile(ATLAS_POSEFILE):  # also False where /data is not readable
        pytest.skip(f"PocketVina Atlas posefile.py not found: {ATLAS_POSEFILE}")
    spec = importlib.util.spec_from_file_location("_atlas_posefile", ATLAS_POSEFILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ligands_df(names=("molecule2", "molecule5", "molecule10"), **extra):
    n = len(names)
    cols = dict(ligand_idx=list(range(n)), ligand_id=list(names), smiles_input=["C" * (q + 1) for q in range(n)],
                smiles_docked=[None] * n, prep_method="standard", seed=42, status="ok")
    cols.update(extra)
    return pd.DataFrame(cols)


def fake_output(affinities, ligand=b"ROOT\nATOM      1  C   UNK A   1       0.000   0.000   0.000  0.00  0.00    "
                                    b"+0.000 C \nENDROOT\nTORSDOF 0\n"):
    """Engine-formatted PDBQT with the given mode affinities (rmsd 0 for mode 1, then q, q+1)."""
    modes = [(a, 0.0 if q == 0 else float(q), 0.0 if q == 0 else q + 1.0) for q, a in enumerate(affinities)]
    return fe.render_pdbqt(ligand.decode(), modes).encode()


def fake_sdf(i, k, model):
    return b"fake sdf %d %d\n" % (i, k) + hashlib.sha256(model).hexdigest().encode() + b"\n$$$$\n", "ok"


def no_tmp_files(root):
    return [str(p) for p in Path(root).rglob("*") if ".tmp-" in p.name]


# --- tables --------------------------------------------------------------------------------------------------------

CONTRACT = {  # docs/output-format.md, "Tables"; "?" marks a nullable column
    "LIGANDS": "ligand_idx int32, ligand_id string, smiles_input string, smiles_docked string?, inchikey string?, "
               "n_heavy int16?, prep_method string, seed int64?, status string, error string?, pdbqt string?",
    "PROTEINS": "protein_id string, uniprot string?, source_path string, structure_file string, "
                "receptor_sha256 fixed_size_binary[32], n_pockets int16, n_pockets_docked int16, status string, "
                "error string?",
    "POCKETS": "protein_id string, pocket int16, name string, score float, probability float?, center_x float, "
               "center_y float, center_z float, size_x float, size_y float, size_z float, residue_ids string?",
    "POCKET_RUNS": "protein_id string, ligand_idx int32, ligand_id string, pocket int8, status string, n_modes int8, "
                   "best_affinity float?, offset uint64, length uint32, raw_length uint32, "
                   "sha256 fixed_size_binary[32]?, seed int64?, runtime_s float?, error string?",
    "MODES": "protein_id string, uniprot string?, ligand_idx int32, pocket int8, mode int8, affinity float, "
             "rmsd_lb float, rmsd_ub float, block_offset uint64, block_length uint32, raw_offset uint32, "
             "raw_length uint32",
    "PAIRS": "protein_id string, uniprot string?, ligand_idx int32, ligand_id string, vina float, best_pocket int8, "
             "best_mode int8, rmsd_lb float, rmsd_ub float, pb_status string?, top_offset uint64, top_length uint32, "
             "top_raw_length uint32, top_sha256 fixed_size_binary[32]?, sdf_status string",
}


@pytest.mark.parametrize("name", sorted(CONTRACT))
def test_schemas_match_contract(name):
    schema = getattr(tables, name)
    got = ", ".join(f"{f.name} {f.type}{'?' if f.nullable else ''}" for f in schema)
    assert got == CONTRACT[name]
    assert tables.SCHEMAS[name.lower()] is schema


def test_round_trip_rows_and_frames(tmp_path):
    rows = [dict(ligand_idx=0, ligand_id="a", smiles_input="CCO", prep_method="standard", status="ok", seed=42,
                 n_heavy=3, pdbqt="ROOT\nENDROOT\nTORSDOF 0\n"),
            dict(ligand_idx=1, ligand_id="b", smiles_input="[Si]", prep_method="standard", status="unsupported_element",
                 error="Si")]
    assert write_table(rows, tables.LIGANDS, tmp_path / "l1.parquet") == 2
    assert pq.read_schema(tmp_path / "l1.parquet").equals(tables.LIGANDS)
    df = read_table(tmp_path / "l1.parquet")
    assert list(df.columns) == tables.LIGANDS.names
    assert df.ligand_id.tolist() == ["a", "b"] and df.error.tolist() == [None, "Si"]
    assert df.n_heavy[0] == 3 and math.isnan(df.n_heavy[1]) and df.smiles_docked.isna().all()
    write_table(df, tables.LIGANDS, tmp_path / "l2.parquet")  # a DataFrame read back writes the same table
    assert pq.read_table(tmp_path / "l2.parquet").equals(pq.read_table(tmp_path / "l1.parquet"))
    write_table(pq.read_table(tmp_path / "l1.parquet"), tables.LIGANDS, tmp_path / "l3.parquet")
    assert pq.read_table(tmp_path / "l3.parquet").equals(pq.read_table(tmp_path / "l1.parquet"))


def test_nan_is_null_only_in_nullable_columns(tmp_path):
    nan = float("nan")
    run = dict(protein_id="P", ligand_idx=0, ligand_id="a", pocket=1, status="no_output", n_modes=0,
               best_affinity=nan, offset=0, length=0, raw_length=0, runtime_s=nan)
    write_table([run], tables.POCKET_RUNS, tmp_path / "r.parquet")
    t = pq.read_table(tmp_path / "r.parquet")
    assert t.column("best_affinity").null_count == 1 and t.column("runtime_s").null_count == 1
    pair = dict(protein_id="P", ligand_idx=0, ligand_id="a", vina=nan, best_pocket=-1, best_mode=-1, rmsd_lb=nan,
                rmsd_ub=nan, top_offset=0, top_length=0, top_raw_length=0, sdf_status="no_pose")
    write_table(pd.DataFrame([pair]), tables.PAIRS, tmp_path / "p.parquet")
    t = pq.read_table(tmp_path / "p.parquet")
    assert t.column("vina").null_count == 0 and math.isnan(t.column("vina")[0].as_py())


@pytest.mark.parametrize("change, message", [
    (dict(ligand_id=None), "'ligand_id' is required"),
    (dict(status=float("nan")), "status"),
    (dict(bogus=1), "not in the schema"),
    (dict(ligand_idx=2 ** 31), "ligand_idx"),
    (dict(n_heavy=1.5), "n_heavy"),
    (dict(seed="x"), "seed"),
])
def test_write_table_rejects(tmp_path, change, message):
    row = dict(ligand_idx=0, ligand_id="a", smiles_input="C", prep_method="standard", status="ok")
    row.update(change)
    with pytest.raises(ValueError, match=message):
        write_table([row], tables.LIGANDS, tmp_path / "x.parquet")
    with pytest.raises(ValueError, match=message):
        write_table(pd.DataFrame([row]), tables.LIGANDS, tmp_path / "x.parquet")
    assert not (tmp_path / "x.parquet").exists() and not no_tmp_files(tmp_path)


def test_write_table_binary_and_columns(tmp_path):
    run = dict(protein_id="P", ligand_idx=0, ligand_id="a", pocket=1, status="ok", n_modes=1, offset=10, length=5,
               raw_length=7, sha256=b"\1" * 31)
    with pytest.raises(ValueError, match="sha256"):
        write_table([run], tables.POCKET_RUNS, tmp_path / "r.parquet")
    run["sha256"] = b"\1" * 32
    cols = {k: [v, v] for k, v in run.items()}
    assert write_table(cols, tables.POCKET_RUNS, tmp_path / "r.parquet") == 2
    assert read_table(tmp_path / "r.parquet", columns=["sha256", "offset"]).values.tolist() == [[b"\1" * 32, 10]] * 2
    with pytest.raises(ValueError, match="different lengths"):
        write_table(dict(protein_id=["P"], ligand_idx=[0, 1]), tables.POCKET_RUNS, tmp_path / "x.parquet")


def test_write_table_atomic_and_zstd(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "dir" / "t.parquet"
    rows = [dict(protein_id="P", pocket=k, name=f"pocket{k}", score=1.0, center_x=0, center_y=0, center_z=0,
                 size_x=20, size_y=20, size_z=20) for k in (1, 2)]
    write_table(rows, tables.POCKETS, path)
    md = pq.ParquetFile(path).metadata
    assert all(md.row_group(0).column(c).compression == "ZSTD" for c in range(md.num_columns))
    before = path.read_bytes()

    def broken(table, where, **kw):
        Path(where).write_bytes(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(pq, "write_table", broken)
    with pytest.raises(OSError, match="disk full"):
        write_table(rows[:1], tables.POCKETS, path)
    assert path.read_bytes() == before and not no_tmp_files(tmp_path)


def test_read_table_globs_and_folders(tmp_path):
    for name, ks in (("b", [3]), ("a", [1, 2])):
        rows = [dict(protein_id=name, pocket=k, name=f"p{k}", score=k, center_x=0, center_y=0, center_z=0,
                     size_x=1, size_y=1, size_z=1) for k in ks]
        write_table(rows, tables.POCKETS, tmp_path / "pk" / f"{name}.parquet")
    (tmp_path / "pk" / "notes.txt").write_text("not parquet")
    df = read_table(tmp_path / "pk")
    assert df.protein_id.tolist() == ["a", "a", "b"] and df.pocket.tolist() == [1, 2, 3]
    assert read_table(str(tmp_path / "pk" / "*.parquet")).equals(df)
    assert list(read_table(tmp_path / "pk", columns=["pocket", "protein_id"]).columns) == ["pocket", "protein_id"]
    empty = read_table(str(tmp_path / "none" / "*.parquet"), schema=tables.MODES)
    assert list(empty.columns) == tables.MODES.names and len(empty) == 0
    assert list(read_table(tmp_path / "none", columns=["mode"], schema=tables.MODES).columns) == ["mode"]
    with pytest.raises(FileNotFoundError):
        read_table(str(tmp_path / "none" / "*.parquet"))
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "missing.parquet")


@pytest.mark.parametrize("folder", ["run:1", "s3:", "seed=42", "x%20y #1"])
def test_relative_paths_that_look_like_uris(tmp_path, monkeypatch, folder):
    """pyarrow parses a relative path to a file that does not exist yet as a URI ("run:1/t.parquet" -> scheme run)."""
    monkeypatch.chdir(tmp_path)
    rel = f"{folder}/pockets/t.parquet"
    rows = [dict(protein_id="P", pocket=1, name="p1", score=1.0, center_x=0, center_y=0, center_z=0, size_x=1,
                 size_y=1, size_z=1)]
    assert write_table(rows, tables.POCKETS, rel) == 1
    assert os.listdir(tmp_path / folder / "pockets") == ["t.parquet"]
    assert read_table(rel).name.tolist() == ["p1"]
    assert list(read_table(f"{folder}/pockets").columns) == tables.POCKETS.names  # no hive partition column
    assert read_table(f"{folder}/pockets/*.parquet").pocket.tolist() == [1]


# --- temp dirs, atomic moves, protein ids ---------------------------------------------------------------------------

@pytest.fixture
def clean_tmp_env(monkeypatch):
    for var in wr.TMP_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_resolve_tmp_dir_order(tmp_path, clean_tmp_env):
    mp = clean_tmp_env
    dirs = {name: tmp_path / name for name in ("explicit", "pv", "slurm", "tmpdir")}
    for d in list(dirs.values())[1:]:
        d.mkdir()
    mp.setenv("TMPDIR", str(dirs["tmpdir"]))
    assert Path(resolve_tmp_dir()).parent == dirs["tmpdir"]
    mp.setenv("SLURM_TMPDIR", str(dirs["slurm"]))
    assert Path(resolve_tmp_dir()).parent == dirs["slurm"]
    mp.setenv("POCKETVINA_TMPDIR", str(dirs["pv"]))
    a, b = resolve_tmp_dir(), resolve_tmp_dir()
    assert Path(a).parent == Path(b).parent == dirs["pv"] and a != b
    assert Path(a).name.startswith("pocketvina_") and os.listdir(a) == []
    e = resolve_tmp_dir(str(dirs["explicit"]))  # created if missing
    assert Path(e).parent == dirs["explicit"] and Path(e).is_dir()


def test_resolve_tmp_dir_skips_unusable(tmp_path, clean_tmp_env):
    clean_tmp_env.setenv("POCKETVINA_TMPDIR", str(tmp_path / "missing"))
    clean_tmp_env.setenv("SLURM_TMPDIR", "")
    (tmp_path / "ok").mkdir()
    clean_tmp_env.setenv("TMPDIR", str(tmp_path / "ok"))
    assert Path(resolve_tmp_dir()).parent == tmp_path / "ok"
    clean_tmp_env.delenv("TMPDIR")
    fallback = resolve_tmp_dir()
    os.rmdir(fallback)
    assert Path(fallback).parent == Path(tempfile.gettempdir())


def test_tmp_dirs_are_absolute(tmp_path, clean_tmp_env):
    """Relative temp folders give absolute paths (mkdtemp keeps them relative before Python 3.12), so engine
    configs and staged paths still work after a change of cwd."""
    clean_tmp_env.chdir(tmp_path)
    (tmp_path / "env").mkdir()
    clean_tmp_env.setenv("POCKETVINA_TMPDIR", "env")
    d = resolve_tmp_dir()
    assert os.path.isabs(d) and Path(d).parent == tmp_path / "env"
    e = resolve_tmp_dir("explicit")
    assert os.path.isabs(e) and Path(e).parent == tmp_path / "explicit"
    with ProteinWriter("P", 1, [1], 1, "rel", {}) as pw:
        assert os.path.isabs(pw.dir) and Path(pw.dir).parent == tmp_path / "rel"
        pw.add_run(0, "a", 1, fake_output([-6.0]), "ok")
        staged = pw.finalize(fake_sdf)
        assert all(os.path.isabs(p) for p in staged.values())
        os.chdir(tmp_path / "env")
        ResultsDir(tmp_path / "results").commit("P", staged, pw.summary)
    assert ResultsDir(tmp_path / "results").verify("P") == []


def test_lazy_package_exports():
    import pocketvina.io as pio
    assert pio.ResultsDir is ResultsDir and pio.Reader is posefile.Reader and pio.read_table is read_table
    assert "ProteinWriter" in dir(pio) and "Results" in pio.__all__
    assert not hasattr(pio, "no_such_name")


def test_atomic_place_same_filesystem(tmp_path):
    src, dst = tmp_path / "src.bin", tmp_path / "a" / "b" / "dst.bin"
    src.write_bytes(b"new")
    atomic_place(str(src), str(dst))
    assert dst.read_bytes() == b"new" and not src.exists()
    src.write_bytes(b"newer")
    atomic_place(src, dst)  # replaces
    assert dst.read_bytes() == b"newer"
    with pytest.raises(FileNotFoundError):
        atomic_place(tmp_path / "nothing", dst)
    assert dst.read_bytes() == b"newer"


def test_atomic_place_cross_device(tmp_path, monkeypatch):
    src, dst = tmp_path / "src.bin", tmp_path / "out" / "dst.bin"
    src.write_bytes(b"x" * (5 << 20))
    dst.parent.mkdir()
    dst.write_bytes(b"old")
    real_replace, calls = os.replace, []

    def replace(a, b):
        calls.append((os.fspath(a), os.fspath(b)))
        if os.fspath(a) == str(src):
            raise OSError(errno.EXDEV, "Invalid cross-device link")
        assert os.path.basename(a).startswith("dst.bin.tmp-") and os.path.dirname(a) == str(dst.parent)
        return real_replace(a, b)
    monkeypatch.setattr(os, "replace", replace)
    atomic_place(str(src), str(dst))
    assert dst.read_bytes() == b"x" * (5 << 20) and not src.exists() and len(calls) == 2
    assert os.listdir(dst.parent) == ["dst.bin"]


def test_atomic_place_other_errors_propagate(tmp_path, monkeypatch):
    src = tmp_path / "s"
    src.write_bytes(b"s")

    def replace(a, b):
        raise PermissionError(errno.EACCES, "denied")
    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(PermissionError):
        atomic_place(src, tmp_path / "d")
    assert src.exists()


def test_atomic_place_real_second_filesystem(tmp_path):
    shm = "/dev/shm"
    if not (os.path.isdir(shm) and os.access(shm, os.W_OK)) or os.stat(shm).st_dev == os.stat(tmp_path).st_dev:
        pytest.skip("no second writable filesystem")
    d = tempfile.mkdtemp(prefix="pocketvina_test_", dir=shm)
    try:
        src = os.path.join(d, "src")
        with open(src, "wb") as f:
            f.write(b"across")
        atomic_place(src, tmp_path / "d" / "dst")
        assert (tmp_path / "d" / "dst").read_bytes() == b"across" and not os.path.exists(src)
        assert os.listdir(tmp_path / "d") == ["dst"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("path, expected", [
    ("/x/AF-Q96LA5-F1-model_v4.pdb", "AF-Q96LA5-F1-model_v4"),
    ("AF-P1-F1-model_v4.pdb.gz", "AF-P1-F1-model_v4"),
    ("rel/1abc.cif", "1abc"),
    ("1ABC.PDB", "1ABC"),
    ("rec.pdbqt", "rec"),
    ("rec.pdbqt.gz", "rec"),
    ("pdb1abc.ent", "pdb1abc"),
    ("pdb1abc.ent.gz", "pdb1abc"),
    ("x.mmcif", "x"),
    ("x.MMCIF.gz", "x"),
    ("my protein (v2).pdb", "my_protein__v2_"),
    ("x.y.pdb", "x.y"),
    ("archive.tar.gz", "archive.tar.gz"),
    (Path("/p/q/r.cif"), "r"),
])
def test_protein_id_from_path(path, expected):
    assert protein_id_from_path(path) == expected


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a b", "../x", None, 5])
def test_check_protein_id_rejects(bad):
    with pytest.raises(ValueError):
        wr.check_protein_id(bad)


# --- ResultsDir ----------------------------------------------------------------------------------------------------

def test_init_run_writes_manifest_and_ligands(tmp_results):
    rd = ResultsDir(tmp_results)
    df = ligands_df()
    rd.init_run({"seed": 42, "num_modes": None, "box": (20, 20, 20), "out": Path("/x")}, df)
    m = json.loads(Path(rd.manifest_file).read_text())
    assert m["format"] == "pocketvina-atlas" and m["pose_format"] == "PVATLAS1" and m["n_ligands"] == 3
    assert m["params"] == {"seed": 42, "num_modes": None, "box": [20, 20, 20], "out": "/x"}
    assert m["ligands_sha256"] == wr.ligands_sha256(df) and m["pocketvina_version"]
    back = read_table(rd.ligands_file)
    assert back.ligand_id.tolist() == list(df.ligand_id) and back.ligand_idx.dtype == np.int32
    assert pq.read_schema(rd.ligands_file).equals(tables.LIGANDS)
    assert rd.done_proteins() == set() and not no_tmp_files(tmp_results)


def test_init_run_resume_and_conflicts(tmp_results):
    rd = ResultsDir(tmp_results)
    rd.init_run({"seed": 42, "box": [20, 20, 20]}, ligands_df())
    first = Path(rd.manifest_file).read_bytes()
    rd.init_run({"box": (20, 20, 20), "seed": 42}, ligands_df())  # same run, as JSON
    assert Path(rd.manifest_file).read_bytes() == first
    with pytest.raises(ValueError, match="other params: num_modes, seed"):
        rd.init_run({"seed": 7, "box": [20, 20, 20], "num_modes": 9}, ligands_df())
    with pytest.raises(ValueError, match="other ligands"):
        rd.init_run({"seed": 42, "box": [20, 20, 20]}, ligands_df(smiles_input=["N", "O", "S"]))
    assert Path(rd.manifest_file).read_bytes() == first
    rd.init_run({"seed": 7}, ligands_df(names=("x", "y")), force=True)
    m = rd.run_manifest()
    assert m["params"] == {"seed": 7} and m["n_ligands"] == 2
    assert read_table(rd.ligands_file).ligand_id.tolist() == ["x", "y"]
    rd.init_run({"seed": 7}, ligands_df(names=("x", "y")))


def test_init_run_force_keeps_proteins_only_for_the_same_ligand_list(tmp_path, tmp_results):
    """Committed proteins index ligands.parquet by ligand_idx: replacing the ligand list (force=True) used to keep
    them done, so their poses were shown under other ligands' names (or the reader failed)."""
    rd = ResultsDir(tmp_results)
    rd.init_run({"seed": 42}, ligands_df())
    rd.commit("P1", stage(tmp_path, "P1")[0], {})
    rd.commit("P2", stage(tmp_path, "P2")[0], {})
    # other params, same ligands (e.g. new conformers of another seed): proteins done stay done
    rd.init_run({"seed": 7}, ligands_df(seed=7, pdbqt="new conformer"), force=True)
    assert rd.done_proteins() == {"P1", "P2"}
    for other in (ligands_df(names=("molecule5", "molecule2", "molecule10")),   # reordered
                  ligands_df(smiles_input=["N", "O", "S"]),                     # same names, other molecules
                  ligands_df(names=("molecule2", "molecule5", "molecule10", "molecule11"))):  # one more
        rd.commit("P1", stage(tmp_path, "P1")[0], {})
        rd.init_run({"seed": 7}, other, force=True)
        assert rd.done_proteins() == set() and not rd.is_done("P1")
        assert read_table(rd.ligands_file).ligand_id.tolist() == list(other.ligand_id)
    rd.commit("P1", stage(tmp_path, "P1")[0], {})  # docked again against the new list
    rd.init_run({"seed": 7}, other)
    assert rd.done_proteins() == {"P1"} and rd.verify("P1") == []


def test_init_run_stages_ligands_in_tmp_dir(tmp_results, tmp_path, monkeypatch):
    # ligands.parquet (with every input PDBQT) used to be staged in $TMPDIR whatever tmp_dir the run was given
    monkeypatch.setenv("POCKETVINA_TMPDIR", str(tmp_path / "env_tmp"))
    (tmp_path / "env_tmp").mkdir()
    staged = []
    real = tables.write_table

    def spy(rows, schema, path):
        staged.append(path)
        return real(rows, schema, path)

    monkeypatch.setattr(tables, "write_table", spy)
    rd = ResultsDir(tmp_results)
    rd.init_run({}, ligands_df(), tmp_dir=str(tmp_path / "run_tmp"))
    os.unlink(rd.ligands_file)
    rd.init_run({}, ligands_df(), tmp_dir=str(tmp_path / "run_tmp"))  # restores it
    rd.init_run({"seed": 7}, ligands_df(), force=True, tmp_dir=str(tmp_path / "run_tmp"))
    assert len(staged) == 3 and all(Path(p).parent.parent == tmp_path / "run_tmp" for p in staged)
    assert os.listdir(tmp_path / "run_tmp") == [] and os.listdir(tmp_path / "env_tmp") == []


def test_init_run_restores_missing_ligands(tmp_results):
    rd = ResultsDir(tmp_results)
    rd.init_run({}, ligands_df())
    os.unlink(rd.ligands_file)  # as if the process died between MANIFEST.json and ligands.parquet
    rd.init_run({}, ligands_df())
    assert read_table(rd.ligands_file).ligand_id.tolist() == ["molecule2", "molecule5", "molecule10"]


@pytest.mark.parametrize("df, message", [
    (ligands_df().assign(ligand_idx=[0, 2, 1]), "ligand_idx must be"),
    (ligands_df(names=("a", "b", "a")), "duplicate ligand_id: \\['a'\\]"),
    (ligands_df().drop(columns=["status"]), "'status' is required"),
])
def test_init_run_rejects_bad_ligands(tmp_results, df, message):
    with pytest.raises(ValueError, match=message):
        ResultsDir(tmp_results).init_run({}, df)
    assert not os.path.exists(os.path.join(tmp_results, "MANIFEST.json"))


def test_init_run_rejects_foreign_manifest(tmp_results):
    os.makedirs(tmp_results)
    Path(tmp_results, "MANIFEST.json").write_text('{"format": "something else"}')
    with pytest.raises(ValueError, match="format 'something else'"):
        ResultsDir(tmp_results).init_run({}, ligands_df())
    with pytest.raises(TypeError):
        ResultsDir(tmp_results).init_run(["not", "a", "dict"], ligands_df())


def _race(n, fn):
    barrier = threading.Barrier(n)

    def run(t):
        barrier.wait()
        try:
            fn(t)
            return "ok"
        except ValueError:
            return "conflict"
    with ThreadPoolExecutor(n) as pool:
        return list(pool.map(run, range(n)))


@pytest.mark.parametrize("hard_links", [True, False])
def test_init_run_concurrent(tmp_results, monkeypatch, hard_links):
    if not hard_links:
        def no_link(a, b):
            raise OSError(errno.EPERM, "Operation not permitted")
        monkeypatch.setattr(os, "link", no_link)
    rd = ResultsDir(tmp_results)
    assert _race(8, lambda t: rd.init_run({"seed": 42}, ligands_df())) == ["ok"] * 8
    assert rd.run_manifest()["params"] == {"seed": 42} and not no_tmp_files(tmp_results)
    other = ResultsDir(os.path.join(tmp_results, "other"))
    results = _race(8, lambda t: other.init_run({"seed": t}, ligands_df()))
    assert results.count("ok") == 1
    assert other.run_manifest()["params"] == {"seed": results.index("ok")}
    assert read_table(other.ligands_file).ligand_id.tolist() == list(ligands_df().ligand_id)


def test_create_exclusive_lost_nfs_reply(tmp_path, monkeypatch):
    real_link = os.link

    def link(a, b):
        real_link(a, b)
        raise FileExistsError(errno.EEXIST, "File exists")  # the server did it, the reply got lost
    monkeypatch.setattr(os, "link", link)
    assert wr._create_exclusive(str(tmp_path / "m.json"), {"a": 1}) is True
    assert wr._create_exclusive(str(tmp_path / "m.json"), {"a": 2}) is False
    assert json.loads((tmp_path / "m.json").read_text()) == {"a": 1} and os.listdir(tmp_path) == ["m.json"]


def stage(tmp_path, protein_id, payload=b"pose bytes"):
    d = Path(tempfile.mkdtemp(dir=tmp_path))
    files = {f"poses/{protein_id}.poses": payload, f"tables/pairs/{protein_id}.parquet": b"pairs " + payload,
             f"structures/{protein_id}.pdb.gz": b"structure"}
    staged = {}
    for rel, data in files.items():
        p = d / rel.replace("/", "_")
        p.write_bytes(data)
        staged[rel] = str(p)
    return staged, files


def test_commit_moves_files_then_writes_manifest(tmp_path, tmp_results, monkeypatch):
    rd = ResultsDir(tmp_results)
    staged, files = stage(tmp_path, "P1")
    seen = []
    real = wr.atomic_place

    def spy(src, dst):
        seen.append(rd.is_done("P1"))
        return real(src, dst)
    monkeypatch.setattr(wr, "atomic_place", spy)
    m = rd.commit("P1", staged, {"n_runs": 3, "path": Path("/x")})
    assert seen == [False, False, False]
    assert rd.is_done("P1") and rd.done_proteins() == {"P1"} and not rd.is_done("P2")
    assert m == rd.protein_manifest("P1") and m["summary"] == {"n_runs": 3, "path": "/x"}
    for rel, data in files.items():
        assert Path(tmp_results, rel).read_bytes() == data
        assert m["files"][rel] == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    assert not any(os.path.exists(p) for p in staged.values())
    assert rd.verify("P1") == [] and not no_tmp_files(tmp_results)
    Path(tmp_results, "poses", "P1.poses").write_bytes(b"tampered")
    assert rd.verify("P1") == ["poses/P1.poses: changed"]
    assert rd.verify("P2") == ["P2: no manifest"]


def test_commit_again_replaces(tmp_path, tmp_results):
    rd = ResultsDir(tmp_results)
    rd.commit("P1", stage(tmp_path, "P1", b"v1")[0], {"v": 1})
    rd.commit("P1", stage(tmp_path, "P1", b"version 2")[0], {"v": 2})
    assert Path(tmp_results, "poses", "P1.poses").read_bytes() == b"version 2"
    assert rd.protein_manifest("P1")["summary"] == {"v": 2} and rd.verify("P1") == []


@pytest.mark.parametrize("rel", ["/abs/x", "../x", "a/../../x", ".", "MANIFEST.json", "manifest/P1.json",
                                 "manifest", ""])
def test_commit_rejects_paths(tmp_path, tmp_results, rel):
    src = tmp_path / "f"
    src.write_bytes(b"x")
    with pytest.raises(ValueError):
        ResultsDir(tmp_results).commit("P1", {rel: str(src)}, {})
    assert src.exists()


def test_commit_rejects_bad_input(tmp_path, tmp_results):
    rd = ResultsDir(tmp_results)
    with pytest.raises(FileNotFoundError):
        rd.commit("P1", {"poses/P1.poses": str(tmp_path / "missing")}, {})
    with pytest.raises(ValueError):
        rd.commit("../P1", {}, {})
    f = tmp_path / "f"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match="two staged files"):
        rd.commit("P1", {"a/b": str(f), "a//b": str(f)}, {})
    assert not rd.is_done("P1")


WORKER = r"""
import os, sys, tempfile, time
import pandas as pd
from pocketvina.io.writer import ResultsDir
results, pid, go = sys.argv[1:4]
open(go + "." + pid, "w").close()
deadline = time.time() + 60
while not os.path.exists(go) and time.time() < deadline:
    time.sleep(0.005)
rd = ResultsDir(results)
rd.init_run({"seed": 42}, pd.DataFrame(dict(ligand_idx=[0, 1], ligand_id=["a", "b"], smiles_input=["C", "CC"],
                                            prep_method="standard", status="ok")))
staged = {}
d = tempfile.mkdtemp()
for rel in (f"poses/{pid}.poses", f"tables/modes/{pid}.parquet"):
    p = os.path.join(d, rel.replace("/", "_"))
    with open(p, "wb") as f:
        f.write((rel * 1000).encode())
    staged[rel] = p
rd.commit(pid, staged, {"pid": pid})
"""


def test_processes_share_a_results_dir(tmp_path, tmp_results):
    """Several processes start the same run and commit different proteins into one results dir at once."""
    go = str(tmp_path / "go")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]),
               TMPDIR=str(tmp_path))
    pids = [f"P{q}" for q in range(6)]
    procs = [subprocess.Popen([sys.executable, "-c", WORKER, tmp_results, pid, go], env=env, cwd=str(tmp_path),
                              stderr=subprocess.PIPE) for pid in pids]
    try:
        deadline = time.time() + 120
        while not all(os.path.exists(f"{go}.{pid}") for pid in pids) and time.time() < deadline:
            time.sleep(0.01)
        Path(go).touch()
        errors = [p.communicate(timeout=120)[1].decode() for p in procs]
    finally:
        for p in procs:
            p.kill()
    assert [p.returncode for p in procs] == [0] * len(pids), errors
    rd = ResultsDir(tmp_results)
    assert rd.done_proteins() == set(pids) and all(rd.verify(pid) == [] for pid in pids)
    assert rd.run_manifest()["n_ligands"] == 2 and read_table(rd.ligands_file).ligand_id.tolist() == ["a", "b"]
    assert not no_tmp_files(tmp_results)


# --- ProteinWriter -------------------------------------------------------------------------------------------------

def combined_rows(real_run):
    with open(real_run.combined_csv, newline="") as f:
        return [dict(pocket=int(r["pocket rank"]), ligand=r["molecule_name"], mode=int(r["mode"]),
                     affinity=float(r["affinity (kcal/mol)"]), lb=float(r["dist from rmsd l.b."]),
                     ub=float(r["dist from rmsd u.b."])) for r in csv.DictReader(f)]


def write_fixture_protein(real_run, tmp_path, sdf_for=fake_sdf, **kw):
    """The fixture run through a ProteinWriter: 3 ligands x pockets 1, 2 (molecule10 has no output)."""
    calls = []

    def recording_sdf_for(i, k, model):
        calls.append((i, k, model))
        return sdf_for(i, k, model)
    pw = ProteinWriter(real_run.protein_id, len(real_run.ligands), list(real_run.pockets), 2, str(tmp_path / "tmp"),
                       {"source": "fixture"}, **kw)
    raws = {}
    for i, name in enumerate(real_run.ligands):
        for k in real_run.pockets:
            out = real_run.output(name, k)
            assert out.name == pdbqt.engine_output_name(real_run.receptor, k, real_run.ligand_pdbqt(name))
            raw = out.read_bytes() if out.exists() else None
            raws[i, k] = raw
            pw.add_run(i, name, k, raw, "ok" if raw else "no_output", runtime_s=1.5, seed=42)
    staged = pw.finalize(recording_sdf_for)
    return pw, staged, raws, calls


@pytest.fixture
def fixture_protein(real_run, tmp_path, tmp_results):
    pw, staged, raws, calls = write_fixture_protein(real_run, tmp_path)
    rd = ResultsDir(tmp_results)
    rd.init_run({"seed": 42}, ligands_df(real_run.ligands))
    rd.commit(real_run.protein_id, staged, pw.summary)
    pw.cleanup()
    return rd, pw, raws, calls


def test_fixture_protein_files(fixture_protein, real_run, tmp_path):
    rd, pw, raws, calls = fixture_protein
    pid = real_run.protein_id
    assert set(rd.protein_manifest(pid)["files"]) == {f"poses/{pid}.poses", f"tables/pocket_runs/{pid}.parquet",
                                                     f"tables/modes/{pid}.parquet", f"tables/pairs/{pid}.parquet"}
    assert rd.verify(pid) == [] and os.listdir(tmp_path / "tmp") == []
    for name, schema in (("pocket_runs", tables.POCKET_RUNS), ("modes", tables.MODES), ("pairs", tables.PAIRS)):
        assert pq.read_schema(Path(rd.path, "tables", name, f"{pid}.parquet")).equals(schema)
    s = pw.summary
    assert s["n_runs"] == 6 and s["runs_by_status"] == {"ok": 4, "no_output": 2} and s["n_modes"] == 36
    assert s["sdf_by_status"] == {"ok": 2, "no_pose": 1}
    assert s["raw_pdbqt_bytes"] == sum(len(raw) for raw in raws.values() if raw)
    assert rd.protein_manifest(pid)["summary"] == s


def test_fixture_pose_file_round_trip(fixture_protein, real_run):
    rd, pw, raws, calls = fixture_protein
    with posefile.Reader(os.path.join(rd.path, "poses", f"{real_run.protein_id}.poses")) as r:
        assert (r.N, r.K) == (3, 2)
        assert r.meta["protein_id"] == real_run.protein_id and r.meta["source"] == "fixture" and r.meta["K"] == 2
        for (i, k), raw in raws.items():
            assert r.modes(i, k) == raw  # original bytes, sha256-identical
        for i, k, model in calls:
            assert r.top(i) == fake_sdf(i, k, model)[0]
        assert r.top(2) is None


def test_fixture_tables(fixture_protein, real_run):
    rd, pw, raws, calls = fixture_protein
    pid = real_run.protein_id
    runs = read_table(os.path.join(rd.path, "tables", "pocket_runs", f"{pid}.parquet"))
    modes = read_table(os.path.join(rd.path, "tables", "modes", f"{pid}.parquet"))
    pairs = read_table(os.path.join(rd.path, "tables", "pairs", f"{pid}.parquet"))
    with posefile.Reader(os.path.join(rd.path, "poses", f"{pid}.poses")) as r:
        for run in runs.itertuples():
            raw = raws[run.ligand_idx, run.pocket]
            assert (run.offset, run.length) == r.block_range(run.ligand_idx, run.pocket)
            assert run.raw_length == (len(raw) if raw else 0) and run.seed == 42 and run.runtime_s == 1.5
            if raw:
                assert run.status == "ok" and run.n_modes == 9 and run.sha256 == hashlib.sha256(raw).digest()
                assert run.best_affinity == np.float32(min(m.affinity for m in pdbqt.parse_modes(raw)))
            else:
                assert run.status == "no_output" and run.n_modes == 0 and run.sha256 is None
                assert math.isnan(run.best_affinity) and run.ligand_id == "molecule10"
        assert [(x.ligand_idx, x.pocket) for x in runs.itertuples()] == sorted(raws)
        # modes: same scores as combined_results.csv; offsets point at the MODEL text
        names = list(real_run.ligands)
        got = sorted((m.pocket, names[m.ligand_idx], m.mode, m.affinity, m.rmsd_lb, m.rmsd_ub)
                     for m in modes.itertuples())
        want = sorted((c["pocket"], c["ligand"], c["mode"], np.float32(c["affinity"]), np.float32(c["lb"]),
                       np.float32(c["ub"])) for c in combined_rows(real_run))
        assert got == want
        assert list(zip(modes.ligand_idx, modes.pocket, modes["mode"])) == sorted(
            zip(modes.ligand_idx, modes.pocket, modes["mode"]))
        for m in modes.itertuples():
            assert (m.block_offset, m.block_length) == r.block_range(m.ligand_idx, m.pocket)
            block = r.modes(m.ligand_idx, m.pocket)
            assert block[m.raw_offset:m.raw_offset + m.raw_length] == pdbqt.model(raws[m.ligand_idx, m.pocket], m.mode)
        # pairs: lowest affinity over pockets and modes
        assert pairs.ligand_idx.tolist() == [0, 1, 2]
        for p in pairs.itertuples():
            rows = [c for c in combined_rows(real_run) if c["ligand"] == names[p.ligand_idx]]
            if not rows:
                assert p.sdf_status == "no_pose" and math.isnan(p.vina) and (p.best_pocket, p.best_mode) == (-1, -1)
                assert (p.top_offset, p.top_length, p.top_raw_length, p.top_sha256) == (0, 0, 0, None)
                continue
            best = min(rows, key=lambda c: (c["affinity"], c["pocket"], c["mode"]))
            assert (p.vina, p.best_pocket, p.best_mode) == (np.float32(best["affinity"]), best["pocket"], best["mode"])
            assert (p.rmsd_lb, p.rmsd_ub) == (np.float32(best["lb"]), np.float32(best["ub"]))
            assert p.sdf_status == "ok" and p.pb_status is None and p.ligand_id == names[p.ligand_idx]
            assert (p.top_offset, p.top_length) == r.top_range(p.ligand_idx)
            sdf = r.top(p.ligand_idx)
            assert p.top_raw_length == len(sdf) and p.top_sha256 == hashlib.sha256(sdf).digest()
        # section A in affinity order
        ranked = pairs[pairs.sdf_status == "ok"].sort_values(["vina", "ligand_idx"])
        assert ranked.top_offset.is_monotonic_increasing


def test_fixture_sdf_for_gets_best_model(fixture_protein, real_run):
    rd, pw, raws, calls = fixture_protein
    assert sorted(i for i, _, _ in calls) == [0, 1]
    pairs = read_table(os.path.join(rd.path, "tables", "pairs", f"{real_run.protein_id}.parquet"))
    for i, k, model in calls:
        p = pairs[pairs.ligand_idx == i].iloc[0]
        assert k == p.best_pocket and model == pdbqt.model(raws[i, k], int(p.best_mode))


def test_atlas_reader_reads_our_file(fixture_protein, real_run, atlas_pf):
    rd, pw, raws, calls = fixture_protein
    path = os.path.join(rd.path, "poses", f"{real_run.protein_id}.poses")
    a = atlas_pf.Reader(path)
    try:
        with posefile.Reader(path) as b:
            assert (a.N, a.K, a.A, a.B, a.meta) == (b.N, b.K, b.A, b.B, b.meta)
            for i in range(a.N):
                assert a.top(i) == b.top(i)
                for k in (1, 2):
                    assert a.modes(i, k) == raws[i, k]
                    assert a.block_range(i, k) == b.block_range(i, k)
    finally:
        a.close()


def test_same_pose_file_as_atlas_writer(real_run, tmp_path, atlas_pf):
    """ProteinWriter's pose file equals one the Atlas Writer makes from the same records in the same order."""
    pw, staged, raws, calls = write_fixture_protein(real_run, tmp_path, gzip_level=9)
    ours = Path(staged[f"poses/{real_run.protein_id}.poses"])
    pairs = read_table(staged[f"tables/pairs/{real_run.protein_id}.parquet"])
    order = pairs[pairs.sdf_status == "ok"].sort_values(["vina", "ligand_idx"]).ligand_idx.tolist()
    sdf = {i: fake_sdf(i, k, model)[0] for i, k, model in calls}
    w = atlas_pf.Writer(str(tmp_path / "atlas.poses"), 3, 2, pw.meta)
    for i in order:
        w.add_top(i, atlas_pf.gz_member(sdf[i], 9), len(sdf[i]))
    w.begin_modes()
    for (i, k), raw in sorted(raws.items(), key=lambda x: x[0][0] * 2 + x[0][1] - 1):
        if raw:
            w.add_block(i * 2 + k - 1, atlas_pf.gz_member(raw, 9))
    w.close()
    assert ours.read_bytes() == (tmp_path / "atlas.poses").read_bytes()
    pw.cleanup()


def test_best_pose_ties(tmp_path):
    """Lowest affinity; ties go to the lower pocket, then the lower mode. Section A ties go by ligand_idx."""
    pw = ProteinWriter("P", 4, [1, 2, 3], 3, str(tmp_path), {})
    pw.add_run(0, "a", 2, fake_output([-7.0, -7.0]), "ok")
    pw.add_run(0, "a", 1, fake_output([-6.0, -7.0, -7.0]), "ok")   # mode 2 of pocket 1 beats pocket 2
    pw.add_run(1, "b", 3, fake_output([-8.0]), "ok")
    pw.add_run(1, "b", 2, fake_output([-8.0, -8.0]), "ok")
    pw.add_run(2, "c", 1, fake_output([-7.0]), "ok")
    pw.add_run(3, "d", 3, fake_output([-9.5]), "ok")
    pw.add_run(3, "d", 1, fake_output([-9.4]), "ok")
    staged = pw.finalize(fake_sdf)
    pairs = read_table(staged["tables/pairs/P.parquet"]).set_index("ligand_idx")
    assert [tuple(pairs.loc[i, ["best_pocket", "best_mode"]]) for i in range(4)] == [(1, 2), (2, 1), (1, 1), (3, 1)]
    assert pairs.vina.tolist() == [-7.0, -8.0, -7.0, -9.5]
    assert pairs.sort_values("top_offset").index.tolist() == [3, 1, 0, 2]  # -9.5, -8.0, then -7.0 by ligand_idx
    runs = read_table(staged["tables/pocket_runs/P.parquet"])
    assert list(zip(runs.ligand_idx, runs.pocket)) == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 1), (3, 1), (3, 3)]
    with posefile.Reader(staged["poses/P.poses"]) as r:
        assert r.modes(0, 3) is None and r.modes(2, 2) is None
    pw.cleanup()


def test_statuses(tmp_path):
    pw = ProteinWriter("P", 5, [1], 1, str(tmp_path), {})
    broken = fake_output([-6.0, -5.0]).replace(b"ENDMDL\n", b"", 1)
    assert pw.add_run(0, "a", 1, broken, "ok", error="note") == "parse_error"
    assert pw.add_run(1, "b", 1, None, "ok") == "no_output"
    assert pw.add_run(2, "c", 1, b"", "ok") == "no_output"
    assert pw.add_run(3, "d", 1, fake_output([-6.0]), "engine_error", error="exit 1") == "engine_error"
    assert pw.add_run(4, "e", 1, None, "skipped") == "skipped"
    staged = pw.finalize(fake_sdf)
    runs = read_table(staged["tables/pocket_runs/P.parquet"]).set_index("ligand_idx")
    assert runs.status.tolist() == ["parse_error", "no_output", "no_output", "engine_error", "skipped"]
    assert runs.n_modes.tolist() == [0] * 5 and runs.best_affinity.isna().all()
    assert runs.error[0].startswith("note; 2 MODEL but 1 ENDMDL") and runs.error[3] == "exit 1"
    assert runs.sha256[0] == hashlib.sha256(broken).digest() and runs.sha256[1] is None
    with posefile.Reader(staged["poses/P.poses"]) as r:  # bytes are kept even when they are not used
        assert r.modes(0, 1) == broken and r.modes(3, 1) == fake_output([-6.0]) and r.modes(1, 1) is None
    assert len(read_table(staged["tables/modes/P.parquet"])) == 0
    assert set(read_table(staged["tables/pairs/P.parquet"]).sdf_status) == {"no_pose"}
    pw.cleanup()


@pytest.mark.parametrize("sdf_for, status", [
    (lambda i, k, m: (None, "convert_failed"), "convert_failed"),
    (lambda i, k, m: (b"", "ok"), "convert_failed"),
    (lambda i, k, m: (b"sdf", "convert_failed"), "convert_failed"),
    (lambda i, k, m: 1 / 0, "convert_failed"),
    (lambda i, k, m: ("text sdf\n", "ok"), "ok"),
])
def test_sdf_for_results(tmp_path, sdf_for, status):
    pw = ProteinWriter("P", 2, [1], 1, str(tmp_path), {})
    pw.add_run(0, "a", 1, fake_output([-6.0]), "ok")
    staged = pw.finalize(sdf_for)
    p = read_table(staged["tables/pairs/P.parquet"]).iloc[0]
    assert p.sdf_status == status and p.vina == -6.0 and p.best_pocket == 1
    with posefile.Reader(staged["poses/P.poses"]) as r:
        assert r.top(0) == (b"text sdf\n" if status == "ok" else None)
        assert (p.top_length > 0) == (status == "ok") and r.modes(0, 1) == fake_output([-6.0])
    pw.cleanup()


def test_sdf_for_unknown_status(tmp_path):
    with ProteinWriter("P", 1, [1], 1, str(tmp_path), {}) as pw:
        pw.add_run(0, "a", 1, fake_output([-6.0]), "ok")
        with pytest.raises(ValueError, match="sdf_status 'weird'"):
            pw.finalize(lambda i, k, m: (b"x", "weird"))
    assert not os.path.exists(pw.dir)


def test_add_run_rejects(tmp_path):
    with ProteinWriter("P", 2, [1, 3], 4, str(tmp_path), {}) as pw:
        pw.add_run(0, "a", 1, None, "no_output")
        for args, exc, message in [
            ((0, "a", 1, None, "no_output"), ValueError, "already added"),
            ((2, "c", 1, None, "no_output"), ValueError, "out of range"),
            ((-1, "c", 1, None, "no_output"), ValueError, "out of range"),
            ((1, "b", 2, None, "no_output"), ValueError, "not one of the pocket ranks"),
            ((1, "b", 1, None, "done"), ValueError, "unknown status"),
            ((0, "z", 3, None, "no_output"), ValueError, "was added as 'a'"),
            ((1, None, 1, None, "no_output"), ValueError, "ligand_id"),
            ((1, "b", 1, "MODEL 1\n", "ok"), TypeError, "bytes"),
        ]:
            with pytest.raises(exc, match=message):
                pw.add_run(*args)
        assert pw.n_runs == 1
        pw.add_run(1, "b", 3, None, "no_output")
        pw.finalize(fake_sdf)
        with pytest.raises(RuntimeError):
            pw.add_run(1, "b", 1, None, "no_output")
        with pytest.raises(RuntimeError):
            pw.finalize(fake_sdf)


@pytest.mark.parametrize("bad", [dict(seed="x"), dict(seed=float("nan")), dict(runtime_s="fast")])
def test_rejected_add_run_changes_nothing(tmp_path, bad):
    """A call rejected for a bad seed/runtime_s must not leave spooled bytes, modes or a best pose behind (they
    were credited to the next run)."""
    with ProteinWriter("P", 2, [1], 1, str(tmp_path), {}) as pw:
        with pytest.raises(ValueError):
            pw.add_run(0, "a", 1, fake_output([-9.0, -8.0, -7.0]), "ok", **bad)
        assert pw.n_runs == 0 and os.path.getsize(os.path.join(pw.dir, "blocks.spool")) == 0
        pw.add_run(1, "b", 1, fake_output([-5.0]), "ok", seed=1)
        staged = pw.finalize(fake_sdf)
        modes = read_table(staged["tables/modes/P.parquet"])
        assert list(zip(modes.ligand_idx, modes["mode"], modes.affinity)) == [(1, 1, -5.0)]
        pairs = read_table(staged["tables/pairs/P.parquet"])
        assert pairs.ligand_idx.tolist() == [1] and pw.summary["n_with_pose"] == 1
        with posefile.Reader(staged["poses/P.poses"]) as r:
            assert r.top(0) is None and r.modes(0, 1) is None and r.modes(1, 1) == fake_output([-5.0])


@pytest.mark.parametrize("args, exc", [
    (("P", 2, [1, 6], 5), ValueError),
    (("P", 2, [0], 5), ValueError),
    (("P", 2, [1, 1], 5), ValueError),
    (("P", 2, [1, 128], 130), ValueError),
    (("P", -1, [1], 5), ValueError),
    (("a/b", 2, [1], 5), ValueError),
])
def test_protein_writer_rejects(tmp_path, args, exc):
    with pytest.raises(exc):
        ProteinWriter(*args, str(tmp_path), {})
    with pytest.raises(TypeError):
        ProteinWriter("P", 1, [1], 1, str(tmp_path), {"bad": object()})
    with pytest.raises(TypeError):  # the pose file sorts meta keys: must fail here, not in finalize
        ProteinWriter("P", 1, [1], 1, str(tmp_path), {1: "a", "b": 2})


@pytest.mark.parametrize("level", [10, -2, "fast"])
def test_bad_gzip_level_fails_before_docking(tmp_path, level):
    with pytest.raises(ValueError):
        ProteinWriter("P", 1, [1], 1, str(tmp_path), {}, gzip_level=level)
    assert os.listdir(tmp_path) == []
    with ProteinWriter("P", 1, [1], 1, str(tmp_path), {}, gzip_level="0") as pw:  # zlib's 0..9 (and -1) work
        assert pw.add_run(0, "a", 1, fake_output([-6.0]), "ok") == "ok"
        with posefile.Reader(pw.finalize(fake_sdf)["poses/P.poses"]) as r:
            assert r.modes(0, 1) == fake_output([-6.0])


def test_spooled_to_disk_and_cleaned_up(tmp_path):
    rng = random.Random(0)  # random coordinates, so the spool outgrows its write buffer
    atoms = "".join(f"ATOM  {q:5d}  C   UNK A   1    {rng.uniform(-99, 99):8.3f}{rng.uniform(-99, 99):8.3f}"
                    f"{rng.uniform(-99, 99):8.3f}  0.00  0.00    +0.000 C \n" for q in range(1, 101))
    raw = fake_output([-6.0, -5.5, -5.0], ligand=f"ROOT\n{atoms}ENDROOT\nTORSDOF 0\n".encode())
    pw = ProteinWriter("P", 50, [1, 2], 2, str(tmp_path / "t"), {"m": 1}, uniprot="Q96LA5", gzip_level=1)
    assert Path(pw.dir).parent == tmp_path / "t"
    for i in range(50):
        for k in (1, 2):
            pw.add_run(i, f"L{i}", k, raw, "ok")
    spool = Path(pw.dir, "blocks.spool")
    total = 100 * len(posefile.gz_member(raw, 1))
    assert total > 2 * wr.SPOOL_BUFFER
    # on disk but for one write buffer, whatever the block size of the filesystem holding tmp_path
    assert total - wr.SPOOL_BUFFER <= spool.stat().st_size <= total
    staged = pw.finalize(fake_sdf)
    assert not spool.exists() and all(p.startswith(pw.dir) for p in staged.values())
    assert set(read_table(staged["tables/modes/P.parquet"]).uniprot) == {"Q96LA5"}
    assert set(read_table(staged["tables/pairs/P.parquet"]).uniprot) == {"Q96LA5"}
    with posefile.Reader(staged["poses/P.poses"]) as r:
        assert r.meta["m"] == 1 and r.meta["uniprot"] == "Q96LA5" and r.meta["pockets"] == [1, 2]
    pw.cleanup()
    assert os.listdir(tmp_path / "t") == []


def test_gzip_level(tmp_path):
    raw = fake_output([-6.0 - q / 10 for q in range(20)])
    sizes = {}
    for level in (1, 9):
        with ProteinWriter("P", 1, [1], 1, str(tmp_path), {}, gzip_level=level) as pw:
            pw.add_run(0, "a", 1, raw, "ok")
            staged = pw.finalize(fake_sdf)
            with posefile.Reader(staged["poses/P.poses"]) as r:
                assert r.modes(0, 1) == raw
                sizes[level] = r.block_range(0, 1)[1]
    assert sizes[1] > sizes[9]


def test_default_tmp_dir(tmp_path, clean_tmp_env):
    clean_tmp_env.setenv("POCKETVINA_TMPDIR", str(tmp_path))
    with ProteinWriter("P", 1, [1], 1, None, {}) as pw:
        assert Path(pw.dir).parent == tmp_path and Path(pw.dir).name.startswith("pocketvina_")
    assert os.listdir(tmp_path) == []


def test_empty_protein(tmp_path, tmp_results):
    """A protein without runs (or without pockets) still gives a valid pose file and empty tables."""
    for ranks, K in (([1, 2], 2), ([], 0)):
        with ProteinWriter("P", 3, ranks, K, str(tmp_path), {}) as pw:
            staged = pw.finalize(fake_sdf)
            with posefile.Reader(staged["poses/P.poses"]) as r:
                assert (r.N, r.K) == (3, K) and r.top(0) is None
            for name in ("pocket_runs", "modes", "pairs"):
                df = read_table(staged[f"tables/{name}/P.parquet"])
                assert len(df) == 0 and list(df.columns) == tables.SCHEMAS[name].names
            ResultsDir(tmp_results).commit("P", staged, pw.summary)
    assert ResultsDir(tmp_results).verify("P") == []


def test_fake_engine_end_to_end(real_run, tmp_path, tmp_results, fake_engine_env):
    """Fake engine outputs -> ProteinWriter -> commit; scores equal the engine's JSON, bytes the engine's PDBQT."""
    lig_dir = tmp_path / "ligands"
    lig_dir.mkdir()
    for name in real_run.ligands:
        shutil.copy(real_run.ligand_pdbqt(name), lig_dir / f"L{real_run.ligands.index(name):07d}.pdbqt")
    stems = sorted(p.name for p in lig_dir.iterdir())
    rd = ResultsDir(tmp_results)
    rd.init_run({"seed": 7}, ligands_df(real_run.ligands))
    pw = ProteinWriter(real_run.protein_id, 3, [1, 2], 5, str(tmp_path / "tmp"), {})
    expected = {}
    for k in (1, 2):
        out = tmp_path / f"out{k}"
        x, y, z = real_run.pocket_center(k)
        cfg = tmp_path / f"config{k}.txt"
        cfg.write_text(f"receptor = {real_run.receptor}\nligand_directory = {lig_dir}\noutput_directory = {out}\n"
                       f"center_x = {x}\ncenter_y = {y}\ncenter_z = {z}\nsize_x = 20\nsize_y = 20\nsize_z = 20\n"
                       f"thread = 8000\npocket_rank = {k}\nseed = 7\nnum_modes = 9\n")
        proc = subprocess.run([fake_engine_env.path, "--config", str(cfg)], capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr
        for i, name in enumerate(stems):
            path = out / pdbqt.engine_output_name(real_run.receptor, k, name)
            raw = path.read_bytes() if path.exists() else None
            if raw:
                entries = next(iter(json.loads(path.with_suffix(".json").read_text()).values()))
                expected[i, k] = (raw, [(e["mode"], e["affinity (kcal/mol)"], e["dist from rmsd l.b."],
                                         e["dist from rmsd u.b."]) for e in entries])
            pw.add_run(i, real_run.ligands[i], k, raw, "ok" if raw else "no_output", seed=7)
    assert sorted(expected) == [(0, 1), (0, 2), (1, 1), (1, 2)]  # molecule10 (Si) is rejected by the engine
    rd.commit(real_run.protein_id, pw.finalize(fake_sdf), pw.summary)
    pw.cleanup()
    modes = read_table(os.path.join(tmp_results, "tables", "modes", f"{real_run.protein_id}.parquet"))
    with posefile.Reader(os.path.join(tmp_results, "poses", f"{real_run.protein_id}.poses")) as r:
        assert r.K == 5 and r.modes(0, 3) is None and r.modes(2, 1) is None
        for (i, k), (raw, scores) in expected.items():
            assert r.modes(i, k) == raw
            got = modes[(modes.ligand_idx == i) & (modes.pocket == k)]
            assert [(m.mode, m.affinity, m.rmsd_lb, m.rmsd_ub) for m in got.itertuples()] == [
                (q, np.float32(a), np.float32(lb), np.float32(ub)) for q, a, lb, ub in scores]
    assert rd.done_proteins() == {real_run.protein_id} and rd.verify(real_run.protein_id) == []
