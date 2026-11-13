"""pocketvina.io.results.Results: tables, DuckDB views, poses, structures and exports of a results directory."""
import csv
import hashlib
import importlib.util
import io
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from rdkit import Chem

import fake_engine as fe
from pocketvina import convert, pdbqt
from pocketvina.io import results as results_mod
from pocketvina.io import sas, tables
from pocketvina.io.importer import import_legacy
from pocketvina.io.results import COMBINED_COLUMNS, Results, short_floats
from pocketvina.io.writer import ProteinWriter, ResultsDir

PID = "AF-Q96LA5-F1-model_v4"
PID2 = "AF-P12345-F1-model_v4"
NAMES = ("molecule2", "molecule5", "molecule10")
HAS_DUCKDB = importlib.util.find_spec("duckdb") is not None


def csv_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], sorted(rows[1:])


def sdf_records(data: bytes) -> list:
    return list(Chem.ForwardSDMolSupplier(io.BytesIO(data), removeHs=False))


def add_receptor_copy(root: Path, new_base: str):
    """Make the legacy run in ``root`` hold a second protein: a copy of the receptor with the same outputs."""
    s, p = root / "proteins" / "structures", root / "proteins" / "pockets"
    shutil.copy(s / f"{PID}.pdbqt", s / f"{new_base}.pdbqt")
    shutil.copy(p / f"{PID}.csv", p / f"{new_base}.csv")
    for f in sorted((root / "batch_1_out").glob(f"{PID}-pocket*_out.pdbqt")):
        shutil.copy(f, f.with_name(new_base + f.name[len(PID):]))


@pytest.fixture(scope="module")
def res(real_run, tmp_path_factory):
    path = import_legacy(str(real_run.root), str(tmp_path_factory.mktemp("results") / "results"),
                         str(real_run.smiles_csv))
    with Results(path) as r:
        yield r


@pytest.fixture(scope="module")
def two_proteins(real_run, tmp_path_factory):
    base = tmp_path_factory.mktemp("two")
    legacy = base / "legacy"
    shutil.copytree(real_run.root, legacy)
    add_receptor_copy(legacy, PID2)
    with Results(import_legacy(str(legacy), str(base / "results"), str(real_run.smiles_csv))) as r:
        yield r


@pytest.fixture
def res_copy(res, tmp_path):
    dst = tmp_path / "results"
    shutil.copytree(res.path, dst)
    with Results(dst) as r:
        yield r


# --- opening, proteins, tables -------------------------------------------------------------------------------------

def test_open(tmp_path, res):
    with pytest.raises(FileNotFoundError, match="MANIFEST.json"):
        Results(tmp_path)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(ValueError, match="format"):
        Results(tmp_path)
    assert repr(res) == f"Results({res.path!r})" and os.fspath(res) == res.path
    assert res.manifest()["params"]["source"] == "import_legacy"
    assert res.manifest(PID)["protein_id"] == PID


def test_proteins_and_tables(res, real_run):
    assert res.proteins() == [PID]
    lig = res.ligands()
    assert tuple(lig["ligand_id"]) == NAMES and list(lig.columns) == tables.LIGANDS.names
    assert list(res.ligands(columns=["ligand_id"]).columns) == ["ligand_id"]
    expected = dict(pockets=real_run.n_pockets, modes=36, pairs=3, pocket_runs=6)
    for name, n in expected.items():
        all_rows, one = getattr(res, name)(), getattr(res, name)(PID)
        assert len(all_rows) == n and all_rows.equals(one) and all_rows.equals(getattr(res, name)([PID]))
        # pairs() appends derived columns (ligand_efficiency, pb_valid) after the stored ones
        assert list(all_rows.columns)[:len(tables.SCHEMAS[name].names)] == tables.SCHEMAS[name].names
    assert res.table("proteins")["n_pockets_docked"].tolist() == [2]
    modes = res.modes(columns=["mode", "affinity"])
    assert list(modes.columns) == ["mode", "affinity"] and modes["affinity"].dtype == np.float32


def test_bad_arguments(res):
    for bad in ("nope", "../x", ""):
        with pytest.raises(KeyError, match="not in"):
            res.modes(bad)
    with pytest.raises(TypeError):
        res.pose_block(1, "molecule2", 1)
    with pytest.raises(ValueError, match="invalid table name"):
        res.table("../tables")
    with pytest.raises(KeyError, match="no table"):
        res.table("posebusters")
    with pytest.raises(ValueError, match="not split by protein"):
        res.table("ligands", PID)


def test_ligand_resolution(res):
    assert [res.ligand_idx(n) for n in NAMES] == [0, 1, 2]
    assert res.ligand_idx(np.int64(1)) == 1 and res.ligand_id(2) == "molecule10"
    with pytest.raises(KeyError):
        res.ligand_idx("nope")
    with pytest.raises(KeyError):
        res.ligand_idx(3)
    with pytest.raises(KeyError):
        res.ligand_idx(-1)
    with pytest.raises(TypeError):
        res.ligand_idx(True)
    with pytest.raises(TypeError):
        res.ligand_idx(1.0)


# --- poses and structures ------------------------------------------------------------------------------------------

def test_pose_blocks_byte_identical(res, real_run):
    for name in real_run.docked:
        for k in real_run.pockets:
            raw = real_run.output(name, k).read_bytes()
            assert res.pose_block(PID, name, k).encode("utf-8", "surrogateescape") == raw
            assert res.pose_block(PID, NAMES.index(name), k).encode() == raw


def test_pose_models(res, real_run):
    modes = res.modes(PID)
    for row in modes.itertuples():
        raw = real_run.output(NAMES[row.ligand_idx], row.pocket).read_bytes()
        model = res.pose_model(PID, int(row.ligand_idx), int(row.pocket), int(row.mode)).encode()
        assert model == pdbqt.model(raw, row.mode) == raw[row.raw_offset:row.raw_offset + row.raw_length]
        assert model.startswith(b"MODEL %d\n" % row.mode) and model.rstrip().endswith(b"ENDMDL")
    with pytest.raises(IndexError):
        res.pose_model(PID, "molecule2", 1, 10)


def test_missing_poses(res):
    with pytest.raises(KeyError, match="no engine output"):
        res.pose_block(PID, "molecule10", 1)
    for pocket in (0, 3, -1):
        with pytest.raises(KeyError, match="no pocket"):
            res.pose_block(PID, "molecule2", pocket)
    with pytest.raises(KeyError):
        res.pose_block(PID, "nope", 1)
    with pytest.raises(KeyError):
        res.top_sdf("nope", "molecule2")


def test_top_sdf(res, real_run):
    pairs = res.pairs(PID).set_index("ligand_id")
    smiles = real_run.smiles()
    for name in real_run.docked:
        text = res.top_sdf(PID, name)
        assert hashlib.sha256(text.encode()).digest() == pairs.loc[name, "top_sha256"]
        (mol,) = sdf_records(text.encode())
        assert mol.GetProp("_Name") == name and int(mol.GetProp("pocket")) == pairs.loc[name, "best_pocket"]
        assert convert.same_molecule(mol, smiles[name])
    assert res.top_sdf(PID, "molecule10") is None
    assert res.top_sdf(PID, 0) == res.top_sdf(PID, "molecule2")


def test_structure(res, real_run):
    assert res.structure(PID) == real_run.receptor.read_bytes()
    assert res.structure_path(PID) == os.path.join(res.path, "structures", f"{PID}.pdbqt.gz")


def test_structure_fallback_and_missing(res_copy, real_run):
    manifest_file = Path(res_copy.path, "manifest", f"{PID}.json")
    manifest = json.loads(manifest_file.read_text())
    del manifest["files"][f"structures/{PID}.pdbqt.gz"]  # e.g. a manifest of another writer
    manifest_file.write_text(json.dumps(manifest))
    plain = Path(res_copy.path, "structures", f"{PID}.pdb")
    plain.write_bytes(b"ATOM\n")
    assert res_copy.structure_path(PID) == os.path.join(res_copy.path, "structures", f"{PID}.pdbqt.gz")
    Path(res_copy.structure_path(PID)).unlink()
    assert res_copy.structure(PID) == b"ATOM\n"
    plain.unlink()
    with pytest.raises(FileNotFoundError):
        res_copy.structure(PID)


def test_sas(res_copy, caplog):
    """Results.sas: None for results without SAS points (here imported from a legacy run), also when a file is there
    that the protein's manifest does not list (left by an earlier run of the protein); the points of a listed file;
    None and a warning for a damaged one. Docking runs write it: test_pipeline."""
    assert res_copy.sas(PID) is None
    data = {"protein": PID, "pockets": {"1": [[1.25, -2.5, 3.75, 0.5]]}, "other_pockets": {"3": [[1.2, 2.5, 3.8, 0.3]]},
            "rest": [[0.1, 0.2, 0.3, 0.01]]}
    path = Path(res_copy.path, "sas", f"{PID}.json.gz")
    path.parent.mkdir()
    path.write_bytes(sas.encode(data))
    assert res_copy.sas(PID) is None
    manifest_file = Path(res_copy.path, "manifest", f"{PID}.json")
    manifest = json.loads(manifest_file.read_text())
    raw = path.read_bytes()
    manifest["files"][f"sas/{PID}.json.gz"] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    manifest_file.write_text(json.dumps(manifest))
    assert res_copy.sas(PID) == data
    path.write_bytes(raw[:len(raw) // 2])
    with caplog.at_level(logging.WARNING, logger="pocketvina.io.results"):
        assert res_copy.sas(PID) is None
    assert f"{PID}: cannot read the SAS points" in caplog.text
    with pytest.raises(KeyError, match="not in"):
        res_copy.sas("1fbl")


# --- DuckDB --------------------------------------------------------------------------------------------------------

def test_duckdb_views(res):
    pytest.importorskip("duckdb")
    con = res.duckdb()
    counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in results_mod.TABLES}
    assert counts == dict(ligands=3, proteins=1, pockets=22, pocket_runs=6, modes=36, pairs=3)
    best = con.execute("""
        SELECT l.ligand_id, min(m.affinity) AS vina FROM modes m JOIN ligands l USING (ligand_idx)
        GROUP BY l.ligand_id ORDER BY vina""").fetchdf()
    pairs = res.pairs().dropna(subset=["vina"]).sort_values("vina")
    assert list(best["ligand_id"]) == list(pairs["ligand_id"])
    assert np.array_equal(best["vina"].to_numpy(np.float32), pairs["vina"].to_numpy(np.float32))
    cols = [r[0] for r in con.execute("DESCRIBE modes").fetchall()]
    assert cols == tables.MODES.names


def _plain_values(df: pd.DataFrame) -> list:
    """Rows of a DataFrame as tuples of plain values (None for NULL / NaN / NA, bytes for binary), to compare a
    DuckDB result with a pandas one whatever the dtypes."""
    def plain(v):
        if isinstance(v, (bytes, bytearray, memoryview)):
            return bytes(v)
        return None if pd.isna(v) else v.item() if isinstance(v, np.generic) else v
    return [tuple(plain(v) for v in row) for row in df.itertuples(index=False)]


def _sorted_pairs(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["protein_id", "ligand_idx"], kind="stable").reset_index(drop=True)


def write_pb_part(results_path, pid: str, rows: list) -> Path:
    """A ``tables/posebusters/<pid>.parquet`` with these (ligand_idx, ligand_id, pocket, mode, pb_status, pb_valid)
    rows (no pose file checksum in its metadata: read as current)."""
    path = Path(results_path, "tables", "posebusters", f"{pid}.parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    tables.write_table([dict(protein_id=pid, ligand_idx=i, ligand_id=lid, pocket=k, mode=m, pb_config="dock",
                             pb_version="test", pb_status=status, pb_valid=valid, pb_mask=0, pb_na_mask=0)
                        for i, lid, k, m, status, valid in rows], tables.POSEBUSTERS, path)
    return path


def test_duckdb_pairs_view(res):
    """The pairs view has the columns of pairs() and its values: ligand_efficiency, and pb_valid NULL without
    PoseBusters results (pairs() then has no pb_valid column)."""
    pytest.importorskip("duckdb")
    con = res.duckdb()
    assert [r[0] for r in con.execute("DESCRIBE pairs").fetchall()] == results_mod.PAIRS_VIEW.names
    assert results_mod.PAIRS_VIEW.names == tables.PAIRS.names + ["pb_valid", "ligand_efficiency"]
    got = _sorted_pairs(con.execute("SELECT * FROM pairs").fetchdf())
    want = _sorted_pairs(res.pairs())
    assert "pb_valid" not in want.columns and got["pb_valid"].isna().all()
    assert _plain_values(got.drop(columns=["pb_valid"])) == _plain_values(want)
    n_heavy = res.ligands().set_index("ligand_idx").n_heavy
    docked = got[got.vina.notna()]
    assert docked.ligand_efficiency.tolist() == [-short_floats([v])[0] / n_heavy[i]
                                                 for i, v in zip(docked.ligand_idx, docked.vina)]
    assert con.execute("SELECT ligand_id, round(ligand_efficiency, 4) FROM pairs WHERE vina < -6 "
                       "ORDER BY ligand_efficiency DESC").fetchall() == [("molecule2", round(6.3 / n_heavy[0], 4))]


def test_best_scores(two_proteins, tmp_path):
    """One row per docked protein x ligand, best score first (viz.table without the SMILES), all rows by default;
    the same columns once there are PoseBusters results (PoseBusters runs only sometimes)."""
    from pocketvina import viz

    df = two_proteins.best_scores()
    assert list(df.columns) == ["protein_id", "ligand_id", "vina", "best_pocket", "best_mode", "ligand_efficiency"]
    pd.testing.assert_frame_equal(df, viz.table(two_proteins, top=None).drop(columns="smiles_docked"))
    assert len(df) == 4 and df["vina"].is_monotonic_increasing  # molecule10 has no pose
    assert set(zip(df.protein_id, df.ligand_id)) == {(p, m) for p in (PID, PID2) for m in ("molecule2", "molecule5")}
    assert df["vina"].tolist() == pytest.approx([-6.3, -6.3, -5.2, -5.2])
    pd.testing.assert_frame_equal(two_proteins.best_scores(top=3), df.head(3))
    pd.testing.assert_frame_equal(two_proteins.best_scores(protein=[PID2, PID]), df)
    only = two_proteins.best_scores(protein=PID2, top=1)
    assert only[["protein_id", "ligand_id"]].values.tolist() == [[PID2, "molecule2"]]
    with pytest.raises(KeyError, match="protein 'nope'"):
        two_proteins.best_scores(protein="nope")
    with pytest.raises(ValueError, match="top must be"):
        two_proteins.best_scores(top=-1)

    path = tmp_path / "results"
    shutil.copytree(two_proteins.path, path)
    write_pb_part(path, PID, [(0, "molecule2", 2, 1, "valid", True), (1, "molecule5", 1, 1, "invalid", False)])
    with Results(path) as r:
        assert {"pb_status", "pb_valid"} <= set(viz.table(r, top=None).columns)  # viz.table shows them, as before
        pd.testing.assert_frame_equal(r.best_scores(), df)
        pd.testing.assert_frame_equal(r.best_scores(protein=PID, top=1),
                                      df[df.protein_id == PID].head(1).reset_index(drop=True))


def test_best_scores_head(two_proteins, tmp_path, monkeypatch):
    """_best_scores_head (the table at the end of a run): the first n rows of best_scores() (same values, order with
    tied scores, dtypes) and its number of rows, read with DuckDB, not by loading every pair; also with PoseBusters
    results, for some proteins, and with nothing docked."""
    pytest.importorskip("duckdb")
    df = two_proteins.best_scores()
    path = tmp_path / "results"
    shutil.copytree(two_proteins.path, path)
    write_pb_part(path, PID, [(0, "molecule2", 2, 1, "valid", True)])

    def no_full_read(*args, **kwargs):
        raise AssertionError("every pair read")

    with Results(path) as r:
        monkeypatch.setattr(Results, "pairs", no_full_read)
        for n in (0, 1, 3, 4, 50):
            head, total = r._best_scores_head(None, n)
            pd.testing.assert_frame_equal(head, df.head(n))
            assert total == 4
        head, total = r._best_scores_head([PID2], 1)
        pd.testing.assert_frame_equal(head, df[df.protein_id == PID2].head(1).reset_index(drop=True))
        assert total == 2
        monkeypatch.undo()
        head, total = r._best_scores_head([], 50)
        assert head.empty and list(head.columns) == list(results_mod.BEST_SCORES_COLUMNS) and total == 0


def test_results_from_the_package():
    """``from pocketvina import Results`` works, but ``from pocketvina import *`` does not load pyarrow for it."""
    import pocketvina

    assert pocketvina.Results is Results
    assert "Results" not in pocketvina.__all__


def test_duckdb_pairs_view_with_posebusters(two_proteins, tmp_path):
    """pb_status / pb_valid of the PoseBusters row of the best pose, joined as pairs() does: rows of other poses do
    not count (pb_status stays the pairs table's), the first row of a pose seen twice wins, and parts that do not
    match the protein's pose file are left out."""
    pytest.importorskip("duckdb")
    path = tmp_path / "results"
    shutil.copytree(two_proteins.path, path)
    best = two_proteins.pairs().set_index(["protein_id", "ligand_idx"])
    k2, m2 = (int(v) for v in best.loc[(PID, 0), ["best_pocket", "best_mode"]])
    k5, m5 = (int(v) for v in best.loc[(PID, 1), ["best_pocket", "best_mode"]])
    k10, m10 = (int(v) for v in best.loc[(PID, 2), ["best_pocket", "best_mode"]])  # no pose
    write_pb_part(path, PID, [(0, "molecule2", k2, m2, "valid", True),
                              (0, "molecule2", k2, m2, "invalid", False),  # the same pose again: ignored
                              (1, "molecule5", k5, m5 + 1, "invalid", False),  # another pose
                              (2, "molecule10", k10, m10, "no_pose", None)])
    write_pb_part(path, PID2, [(1, "molecule5", k5, m5, "timeout", None), (0, "molecule2", k2, m2, "invalid", False)])
    with Results(path) as r:
        con = r.duckdb()
        got = _sorted_pairs(con.execute("SELECT * FROM pairs").fetchdf())
        want = _sorted_pairs(r.pairs())
        assert list(got.columns) == list(want.columns) == results_mod.PAIRS_VIEW.names
        assert _plain_values(got) == _plain_values(want)
        assert [(p, i, s, v) for p, i, s, v in _plain_values(got[["protein_id", "ligand_idx", "pb_status",
                                                                   "pb_valid"]])] == sorted([
            (PID, 0, "valid", True), (PID, 1, None, None), (PID, 2, "no_pose", None),
            (PID2, 0, "invalid", False), (PID2, 1, "timeout", None), (PID2, 2, None, None)])
        assert con.execute("SELECT count(*) FROM posebusters").fetchone() == (6,)
        assert con.execute("SELECT count(*) FROM pairs WHERE pb_valid").fetchone() == (1,)

    # a part made from another pose file (the protein was docked again since): left out, as by pairs()
    part = Path(path, "tables", "posebusters", f"{PID2}.parquet")
    table = pq.read_table(part)
    pq.write_table(table.replace_schema_metadata({**(table.schema.metadata or {}),
                                                  results_mod.PB_POSES_META: b"0" * 64}), part)
    with Results(path) as r:
        got = _sorted_pairs(r.duckdb().execute("SELECT * FROM pairs").fetchdf())
        assert _plain_values(got) == _plain_values(_sorted_pairs(r.pairs()))
        assert got[got.protein_id == PID2].pb_status.isna().all()


def test_duckdb_pairs_view_first_part_wins(two_proteins, tmp_path):
    """A pose found in two PoseBusters parts takes the row of the first part in protein order, as pairs() does,
    even when that row comes later in its file than the other part's row."""
    pytest.importorskip("duckdb")
    path = tmp_path / "results"
    shutil.copytree(two_proteins.path, path)
    k, m = (int(v) for v in two_proteins.pairs(PID).set_index("ligand_idx").loc[0, ["best_pocket", "best_mode"]])
    assert two_proteins.proteins() == [PID2, PID]  # PID2's part is read first
    folder = Path(path, "tables", "posebusters")
    folder.mkdir(parents=True, exist_ok=True)
    for pid, rows in ((PID, [(PID, m, "valid", True)]),
                      # the third row is PID's best pose of molecule2 again, at a later row than in PID's own part
                      (PID2, [(PID2, m, "invalid", False), (PID2, m + 1, "timeout", None), (PID, m, "error", None)])):
        tables.write_table([dict(protein_id=p, ligand_idx=0, ligand_id="molecule2", pocket=k, mode=mode,
                                 pb_config="dock", pb_version="test", pb_status=status, pb_valid=valid, pb_mask=0,
                                 pb_na_mask=0) for p, mode, status, valid in rows],
                           tables.POSEBUSTERS, folder / f"{pid}.parquet")
    with Results(path) as r:
        want = _sorted_pairs(r.pairs())
        got = _sorted_pairs(r.duckdb().execute("SELECT * FROM pairs").fetchdf())
        assert _plain_values(got) == _plain_values(want)
        row = got[(got.protein_id == PID) & (got.ligand_idx == 0)]
        assert (row.pb_status.tolist(), row.pb_valid.isna().tolist()) == (["error"], [True])


def test_empty_results(tmp_path):
    rd = ResultsDir(tmp_path / "results")
    rd.init_run({"source": "test"}, pd.DataFrame(dict(ligand_idx=[0], ligand_id=["a"], smiles_input=["C"],
                                                      prep_method="standard", status="ok")))
    with Results(rd.path) as r:
        assert r.proteins() == []
        assert len(r.modes()) == 0 and list(r.modes().columns) == tables.MODES.names
        header, rows = csv_rows(r.to_combined_csv(tmp_path / "c.csv"))
        assert header == list(COMBINED_COLUMNS) and rows == []
        assert r.export(tmp_path / "out", "all", "pdbqt") == []
        if HAS_DUCKDB:
            con = r.duckdb()
            assert con.execute("SELECT count(*) FROM modes").fetchone() == (0,)
            assert con.execute("SELECT count(*) FROM ligands").fetchone() == (1,)
            assert con.execute("SELECT count(*) FROM pairs").fetchone() == (0,)
            assert [c[0] for c in con.execute("DESCRIBE pairs").fetchall()] == results_mod.PAIRS_VIEW.names


def test_uncommitted_parts_are_ignored(res_copy):
    root = Path(res_copy.path)
    for name in ("modes", "pairs"):  # parts of a protein whose commit did not finish
        shutil.copy(root / "tables" / name / f"{PID}.parquet", root / "tables" / name / "half_done.parquet")
    assert len(res_copy.modes()) == 36 and len(res_copy.pairs()) == 3
    if HAS_DUCKDB:
        assert res_copy.duckdb().execute("SELECT count(*) FROM modes").fetchone() == (36,)
    (root / "manifest" / f"{PID}.json").unlink()
    assert res_copy.proteins() == [] and len(res_copy.modes()) == 0
    with pytest.raises(KeyError):
        res_copy.pose_block(PID, "molecule2", 1)


def test_duckdb_views_ignore_parts_added_later(res_copy):
    """Another process committing a protein after duckdb(): its parts appear before its manifest."""
    pytest.importorskip("duckdb")
    con = res_copy.duckdb()
    folder = Path(res_copy.path, "tables")
    for name in ("modes", "pairs"):
        shutil.copy(folder / name / f"{PID}.parquet", folder / name / "being_committed.parquet")
    assert con.execute("SELECT count(*) FROM modes").fetchone() == (36,)
    assert con.execute("SELECT count(*) FROM pairs").fetchone() == (3,)


def test_key_value_folder_in_path_is_not_a_column(res, tmp_path):
    """A results folder below "run=1/": no "run" column (Hive partitioning) in the tables or the DuckDB views."""
    root = tmp_path / "run=1" / "results"
    shutil.copytree(res.path, root)
    write_pb_part(root, PID, [(0, "molecule2", 1, 1, "valid", True)])
    with Results(root) as r:
        for name in results_mod.TABLES:
            assert list(r.table(name).columns) == tables.SCHEMAS[name].names, name
        assert "run" not in r.pairs().columns and "run" not in r.posebusters().columns
        if HAS_DUCKDB:
            con = r.duckdb()
            for name in (*results_mod.TABLES, "posebusters"):
                assert "run" not in [c[0] for c in con.execute(f"DESCRIBE {name}").fetchall()], name
            assert [c[0] for c in con.execute("DESCRIBE pairs").fetchall()] == results_mod.PAIRS_VIEW.names
            assert con.execute("SELECT count(*) FROM modes").fetchone() == (36,)


def test_two_proteins(two_proteins, real_run, tmp_path):
    r = two_proteins
    assert r.proteins() == sorted([PID, PID2])
    modes = r.modes()
    assert len(modes) == 72 and list(dict.fromkeys(modes["protein_id"])) == sorted([PID, PID2])
    assert len(r.modes([PID2])) == 36 and r.pairs(PID2)["uniprot"].tolist() == ["P12345"] * 3
    assert r.pose_block(PID2, "molecule5", 2).encode() == real_run.output("molecule5", 2).read_bytes()
    if HAS_DUCKDB:
        con = r.duckdb()
        assert con.execute("SELECT count(DISTINCT protein_id) FROM modes").fetchone() == (2,)
    header, rows = csv_rows(r.to_combined_csv(tmp_path / "c.csv"))
    assert {row[0].rsplit("-pocket", 1)[0] for row in rows} == {PID, PID2} and len(rows) == 72
    _, one = csv_rows(r.to_combined_csv(tmp_path / "one.csv", protein=PID2))
    assert len(one) == 36


def test_open_pose_files_are_bounded(two_proteins, monkeypatch):
    monkeypatch.setattr(results_mod, "MAX_OPEN_POSE_FILES", 1)
    r = two_proteins
    r.close()
    r.pose_block(PID, "molecule2", 1)
    first = r._readers[PID]
    r.pose_block(PID2, "molecule2", 1)
    assert list(r._readers) == [PID2] and first.f.closed
    r.close()
    assert r._readers == {} and r.pose_block(PID, "molecule2", 1)


# --- combined_results.csv and exports ------------------------------------------------------------------------------

def test_combined_csv_reproduces_fixture(res, real_run, tmp_path):
    path = res.to_combined_csv(tmp_path / "sub" / "combined_results.csv")
    assert path == str(tmp_path / "sub" / "combined_results.csv")
    header, rows = csv_rows(path)
    expected_header, expected = csv_rows(real_run.combined_csv)
    assert header == expected_header == list(COMBINED_COLUMNS)
    assert rows == expected  # same text, so the same values
    assert not [p for p in (tmp_path / "sub").iterdir() if ".tmp-" in p.name]


def test_combined_csv_without_smiles(real_run, tmp_path):
    with Results(import_legacy(str(real_run.root), str(tmp_path / "results"))) as r:
        assert csv_rows(r.to_combined_csv(tmp_path / "c.csv")) == csv_rows(real_run.combined_csv)


def test_csv_exports_with_old_pandas(res, real_run, tmp_path, monkeypatch):
    """pandas < 1.5 has ``line_terminator``, not ``lineterminator``."""
    to_csv = pd.DataFrame.to_csv

    def old_to_csv(self, *args, **kw):
        if "lineterminator" in kw:
            raise TypeError("to_csv() got an unexpected keyword argument 'lineterminator'")
        kw["lineterminator"] = kw.pop("line_terminator", None)
        return to_csv(self, *args, **kw)

    monkeypatch.setattr(pd.DataFrame, "to_csv", old_to_csv)
    assert csv_rows(res.to_combined_csv(tmp_path / "c.csv")) == csv_rows(real_run.combined_csv)
    (best,) = res.export(tmp_path / "out", "best", "csv")
    assert b"\r" not in Path(best).read_bytes() and len(csv_rows(best)[1]) == 3


def test_short_floats():
    values = np.array([-5.2, 0.0, -4.0, 3.551, 9.882, -12.3, 1234.567], dtype=np.float32)
    assert short_floats(values).tolist() == [-5.2, 0.0, -4.0, 3.551, 9.882, -12.3, 1234.567]
    assert np.isnan(short_floats([np.nan])[0])


def test_export_pdbqt(res, real_run, tmp_path):
    out = tmp_path / "out"
    written = res.export(out, "all", "pdbqt")
    assert len(written) == 4
    for name in real_run.docked:
        for k in real_run.pockets:
            path = out / PID / f"{PID}-pocket{k}-{name}_out.pdbqt"
            assert str(path) in written and path.read_bytes() == real_run.output(name, k).read_bytes()
    best = res.export(out, "best", "pdbqt")
    pairs = res.pairs(PID).set_index("ligand_id")
    assert len(best) == 2
    for name in real_run.docked:
        k, mode = int(pairs.loc[name, "best_pocket"]), int(pairs.loc[name, "best_mode"])
        path = out / PID / f"{PID}-pocket{k}-{name}_best.pdbqt"
        assert path.read_bytes() == pdbqt.model(real_run.output(name, k).read_bytes(), mode)


def test_export_sdf(res, real_run, tmp_path):
    out = tmp_path / "out"
    best = res.export(out, "best", "sdf")
    pairs = res.pairs(PID).set_index("ligand_id")
    assert len(best) == 2
    for name in real_run.docked:
        path = out / PID / f"{PID}-pocket{pairs.loc[name, 'best_pocket']}-{name}_best.sdf"
        assert path.read_text() == res.top_sdf(PID, name)
    every = res.export(out, "all", "sdf", protein=PID)
    assert len(every) == 4
    smiles = real_run.smiles()
    for name in real_run.docked:
        for k in real_run.pockets:
            mols = sdf_records((out / PID / f"{PID}-pocket{k}-{name}_out.sdf").read_bytes())
            assert len(mols) == real_run.n_modes
            assert [m.GetProp("mode") for m in mols] == [str(q) for q in range(1, real_run.n_modes + 1)]
            assert {m.GetProp("ligand_idx") for m in mols} == {str(NAMES.index(name))}
            assert all(convert.same_molecule(m, smiles[name]) for m in mols)
            scores = [float(m.GetProp("affinity")) for m in mols]
            assert scores == [rec.affinity for rec in pdbqt.parse_modes(real_run.output(name, k).read_bytes())]


def test_export_sdf_all_without_smiles(real_run, tmp_path, caplog):
    with Results(import_legacy(str(real_run.root), str(tmp_path / "results"))) as r, \
            caplog.at_level(logging.WARNING, logger="pocketvina.io.results"):
        assert r.export(tmp_path / "out", "best", "sdf") == []
        assert r.export(tmp_path / "out", "all", "sdf") == []
    assert "no SMILES" in caplog.text


def test_export_csv(res, real_run, tmp_path):
    (combined,) = res.export(tmp_path / "out", "all", "csv")
    assert combined == str(tmp_path / "out" / "combined_results.csv")
    assert csv_rows(combined) == csv_rows(real_run.combined_csv)
    (best,) = res.export(tmp_path / "out", "best", "csv")
    header, rows = csv_rows(best)
    assert header == list(results_mod.BEST_COLUMNS)
    assert rows == sorted([
        [PID, "0", "molecule2", "-6.3", "2", "1", "0.0", "0.0", "ok", ""],
        [PID, "1", "molecule5", "-5.2", "1", "1", "0.0", "0.0", "ok", ""],
        [PID, "2", "molecule10", "", "", "", "", "", "no_pose", ""],
    ])


def test_export_rejects(res, tmp_path):
    with pytest.raises(ValueError, match="what"):
        res.export(tmp_path, "top", "sdf")
    with pytest.raises(ValueError, match="fmt"):
        res.export(tmp_path, "best", "mol2")
    with pytest.raises(KeyError):
        res.export(tmp_path, "best", "sdf", protein="nope")


def test_export_file_names_are_safe(tmp_path):
    """Ligand ids that are not file names (a writer-made results dir, not an import)."""
    ids = ["a/b", "A/B", "..", "ok-1", "c_6", "C", "c"]  # "c" -> "c_6" is taken by ligand 4
    rd = ResultsDir(tmp_path / "results")
    rd.init_run({"source": "test"}, pd.DataFrame(dict(ligand_idx=range(len(ids)), ligand_id=ids, smiles_input="C",
                                                      prep_method="standard", status="ok")))
    ligand = ("ROOT\nATOM      1  C   UNK A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C \n"
              "ENDROOT\nTORSDOF 0\n")
    with ProteinWriter("prot", len(ids), [1], 1, tmp_path / "tmp", {}) as w:
        for i, lig in enumerate(ids):
            w.add_run(i, lig, 1, fe.render_pdbqt(ligand, [(-5.0 - i, 0.0, 0.0)]).encode(), "ok")
        staged = w.finalize(lambda i, k, model: (b"x\n$$$$\n", "ok"))
        rd.commit("prot", staged, w.summary)
    with Results(rd.path) as r:
        written = r.export(tmp_path / "out", "all", "pdbqt")
        names = sorted(os.path.basename(p) for p in written)
        assert names == sorted(["prot-pocket1-a_b_out.pdbqt", "prot-pocket1-A_B_1_out.pdbqt",
                                "prot-pocket1-ligand2_out.pdbqt", "prot-pocket1-ok-1_out.pdbqt",
                                "prot-pocket1-c_6_out.pdbqt", "prot-pocket1-C_out.pdbqt",
                                "prot-pocket1-c_6_2_out.pdbqt"])
        assert len({n.lower() for n in names}) == len(ids) == len(os.listdir(tmp_path / "out" / "prot"))
        assert r.top_sdf("prot", "a/b") == "x\n$$$$\n"
        assert r.pose_block("prot", "..", 1).startswith("MODEL 1\nREMARK VINA RESULT:      -7.0")
        with pytest.raises(FileNotFoundError):
            r.structure("prot")


def test_export_file_names_of_long_ligand_ids(tmp_path):
    """IUPAC-style ids longer than a file name allows (run_atlas docks them as L<idx> files, so it accepts them)."""
    long = "N-[2-(4-methylpiperazin-1-yl)ethyl]-" * 8
    ids = [long + "benzamide", long + "pyridine", "short"]  # the first two share their first MAX_STEM characters
    rd = ResultsDir(tmp_path / "results")
    rd.init_run({"source": "test"}, pd.DataFrame(dict(ligand_idx=range(len(ids)), ligand_id=ids, smiles_input="C",
                                                      prep_method="standard", status="ok")))
    ligand = ("ROOT\nATOM      1  C   UNK A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C \n"
              "ENDROOT\nTORSDOF 0\n")
    with ProteinWriter("prot", len(ids), [1], 1, tmp_path / "tmp", {}) as w:
        for i, lig in enumerate(ids):
            w.add_run(i, lig, 1, fe.render_pdbqt(ligand, [(-5.0 - i, 0.0, 0.0)]).encode(), "ok")
        rd.commit("prot", w.finalize(lambda i, k, model: (b"x\n$$$$\n", "ok")), w.summary)
    with Results(rd.path) as r:
        names = [os.path.basename(p) for p in r.export(tmp_path / "out", "best", "pdbqt")]
        stem = results_mod._UNSAFE_CHARS.sub("_", long)[:results_mod.MAX_STEM]
        assert sorted(names) == sorted([f"prot-pocket1-{stem}_best.pdbqt", f"prot-pocket1-{stem}_1_best.pdbqt",
                                        "prot-pocket1-short_best.pdbqt"])
        assert len(r.export(tmp_path / "out", "best", "sdf")) == 3
