"""pocketvina.bust (PoseBusters checks of a results directory), the PoseBusters accessors of io.results.Results and
the ``posebusters`` option of pipeline.run_atlas.

Results used: ``imported`` (the real run of tests/data imported with io.importer: molecule2 and molecule5 docked in
two pockets of a PDBQT receptor, molecule10 without a pose) and ``fake_run`` (1fbl.pdb docked with the fake engine:
a PDB structure with waters and metal ions; the poses lie where the ligands were embedded, away from the pocket).
Expected rows come from PoseBusters' own path-based call (``PoseBusters(config).bust(<sdf files>, mol_cond=<pdb
file>)``) on the same poses and protein. Workers are forked (``bust.START_METHOD``), so a PoseBusters check
monkeypatched here is the one they run (timeouts, dying workers).

The whole module is skipped without PoseBusters (it is an optional dependency); about 30 s with it.
"""
import functools
import json
import multiprocessing
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from rdkit import Chem

from pocketvina import PocketVinaConfig, PocketVinaGPU, bust, pdbqt, pipeline
from pocketvina.convert import pose_to_sdf
from pocketvina.io import tables
from pocketvina.io.importer import import_legacy
from pocketvina.io.results import Results
from pocketvina.io.writer import ResultsDir
from test_pipeline import FIXTURE_RECEPTOR, atlas_config, cropped_receptor, make_inputs, stub_predict_pockets

posebusters = pytest.importorskip("posebusters")
from posebusters import PoseBusters  # noqa: E402
from posebusters.tools.loading import safe_load_mol, safe_supply_mols  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
FAKE_ENGINE = TESTS_DIR / "fake_engine.py"
PID = "AF-Q96LA5-F1-model_v4"
FAKE_PID = "1fbl"
NAMES = bust.check_names("dock")
ALL_BITS = (1 << len(NAMES)) - 1
WORKERS = 2


# --- fixtures and helpers ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def imported(real_run, tmp_path_factory) -> str:
    """Results of the tests/data run (read only: tests that write copy it with ``copy_results``)."""
    return import_legacy(str(real_run.root), str(tmp_path_factory.mktemp("imported") / "results"),
                         str(real_run.smiles_csv))


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory, example_dir) -> str:
    """Results of 1fbl.pdb docked with the fake engine: molecule1 and molecule2 in pocket 1 (read only)."""
    base = tmp_path_factory.mktemp("fake_run")
    with pytest.MonkeyPatch.context() as mp:
        install_fake_engine(mp, base)
        mp.setattr(pipeline, "predict_pockets", stub_predict_pockets(n_pockets=1))
        mp.setattr(pipeline, "receptor_pdbqt", cropped_receptor())
        plist, smiles = make_inputs(base, example_dir, ligands=("molecule1", "molecule2"),
                                    proteins=[example_dir / "proteins" / "1fbl.pdb"])
        return pipeline.run_atlas(atlas_config(plist, smiles, base / "results", base / "tmp", top_pockets=1))


@pytest.fixture(scope="module")
def busted(imported, tmp_path_factory) -> str:
    """``imported`` with the PoseBusters results of the best poses (read only)."""
    path = copy_results(imported, tmp_path_factory.mktemp("busted"))
    assert bust.bust_results(path, workers=WORKERS, tmp_dir=str(tmp_path_factory.mktemp("busted_tmp"))) == [PID]
    return path


@pytest.fixture
def stubs(monkeypatch):
    """Fixture pockets instead of P2Rank, and cropped receptors (as in test_pipeline.py)."""
    monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets())
    monkeypatch.setattr(pipeline, "receptor_pdbqt", cropped_receptor())


@pytest.fixture
def forked(monkeypatch):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("needs the fork start method")
    monkeypatch.setattr(bust, "START_METHOD", "fork")


def install_fake_engine(mp, folder: Path):
    """``$POCKETVINA_ENGINE`` = the fake engine (as the ``fake_engine_env`` fixture, for a module-scoped run)."""
    exe = folder / "PocketVina-GPU"
    exe.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE_ENGINE))} "$@"\n')
    exe.chmod(0o755)
    mp.setenv("POCKETVINA_ENGINE", str(exe if os.access(exe, os.X_OK) else FAKE_ENGINE))
    mp.setenv("POCKETVINA_KERNEL_CACHE", str(folder / "kernel_cache"))
    mp.delenv("POCKETVINA_FAKE_ENGINE_SLEEP", raising=False)
    mp.delenv("POCKETVINA_FAKE_ENGINE_LOG", raising=False)


def copy_results(src, dst_parent) -> str:
    dst = Path(dst_parent) / "results"
    shutil.copytree(src, dst)
    return str(dst)


def files_in(root) -> set:
    root = Path(root)
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def pb_part(results, pid=PID) -> Path:
    return Path(results) / "tables" / "posebusters" / f"{pid}.parquet"


def mask_values(pb_mask: int, pb_na_mask: int) -> list:
    """The per-check results encoded in the masks: True (passed), False (failed), None (no result)."""
    return [None if pb_na_mask >> b & 1 else bool(pb_mask >> b & 1) for b in range(len(NAMES))]


def pose_sdf(r: Results, pid: str, row) -> str:
    """SDF of the pose of a posebusters row, as bust_results checks it (stored best pose or converted mode)."""
    best = r.pairs(pid).set_index("ligand_idx").loc[row.ligand_idx]
    if (row.pocket, row.mode) == (best.best_pocket, best.best_mode):
        return r.top_sdf(pid, int(row.ligand_idx))
    model = pdbqt.model(r.pose_block(pid, int(row.ligand_idx), int(row.pocket)).encode(), int(row.mode))
    smiles = r.ligands().set_index("ligand_idx").loc[row.ligand_idx, "smiles_docked"]
    sdf, status = pose_to_sdf(model, smiles, {"_Name": row.ligand_id})
    assert status == "ok"
    return sdf.decode()


def path_based(r: Results, pid: str, rows, tmp_path: Path, config="dock") -> pd.DataFrame:
    """PoseBusters' own result for the poses of ``rows``: SDF and PDB files, loaded by PoseBusters itself."""
    folder = tmp_path / "path_based"
    folder.mkdir()
    protein = folder / "protein.pdb"
    protein.write_text(bust._protein_pdb(r, pid))
    sdfs = []
    for n, row in enumerate(rows):
        sdfs.append(folder / f"pose{n}.sdf")
        sdfs[-1].write_text(pose_sdf(r, pid, row))
    return PoseBusters(config).bust(sdfs, mol_cond=protein)


def patch_check(monkeypatch, make):
    """Replace PoseBusters' energy ratio check by ``make(original)`` (same signature, so PoseBusters calls it)."""
    import posebusters.posebusters as pb_module

    original = pb_module.module_dict["energy_ratio"]
    monkeypatch.setitem(pb_module.module_dict, "energy_ratio", functools.wraps(original)(make(original)))


# --- check names and loading ---------------------------------------------------------------------------------------

def test_check_names():
    assert NAMES[:3] == ["mol_pred_loaded", "mol_cond_loaded", "sanitization"] and len(NAMES) == 22
    for name in ("dock_fast", "redock", "mol", "gen"):
        assert 0 < len(bust.check_names(name)) <= bust.MAX_CHECKS
    _, cfg = bust._config("dock")
    assert bust.check_names(cfg) == NAMES
    assert bust._config(cfg)[0].startswith("custom-") and bust._config(cfg)[0] == bust._config(dict(cfg))[0]
    with pytest.raises(ValueError, match="unknown PoseBusters config"):
        bust.check_names("dock2")
    with pytest.raises(ValueError, match="int32 masks"):
        bust.check_names(dict(modules=[dict(name="Loading", function="loading",
                                            chosen_binary_test_output=[f"c{i}" for i in range(32)])]))


def test_protein_without_crystal_waters(fake_run):
    """PoseBusters sees the structure as docked: no HOH (the docking receptor has none), ions and ligands kept."""
    with Results(fake_run) as r:
        original = r.structure(FAKE_PID).decode()
        text = bust._protein_pdb(r, FAKE_PID)
    water = [line for line in original.splitlines(True) if line.startswith(("ATOM", "HETATM")) and "HOH" in line]
    assert len(water) == 295
    assert text == "".join(line for line in original.splitlines(True) if line not in water)
    assert {line[17:20] for line in text.splitlines() if line.startswith("HETATM")} == {" CA", "HTA", " ZN"}


def test_workers_share_the_cpus(imported, tmp_path, monkeypatch, forked):
    """Each worker runs PoseBusters' energy check with (CPUs this process may use) // workers threads; its default,
    0, would start a thread per core of the machine in every worker. A config that sets num_threads keeps it."""
    _, cfg = bust._config("dock")
    limited = bust._limit_threads(cfg, 3)
    energy = [m for m in limited["modules"] if m["function"] == "energy_ratio"]
    assert len(energy) == 1 and energy[0]["parameters"]["num_threads"] == 3
    assert all("num_threads" not in m.get("parameters", {}) for m in cfg["modules"])  # a copy
    assert bust._limit_threads(limited, 5) == limited
    seen = tmp_path / "num_threads.txt"

    def make(original):
        def check(*args, **kwargs):
            with open(seen, "a") as f:
                f.write(f"{kwargs.get('num_threads')}\n")
            return original(*args, **kwargs)
        return check

    patch_check(monkeypatch, make)
    monkeypatch.setattr(bust, "_available_cpus", lambda: 7)
    path = copy_results(imported, tmp_path)
    assert bust.bust_results(path, workers=WORKERS, tmp_dir=str(tmp_path / "tmp")) == [PID]
    assert seen.read_text().split() == ["3", "3"]  # 7 // 2, for the two poses
    with Results(path) as r:
        assert r.posebusters(PID).pb_status.tolist() == ["valid", "valid", "no_pose"]


@pytest.mark.parametrize("process_mol", [True, False], ids=["in_memory", "temp_file"])
def test_loaders_match_posebusters_file_loaders(imported, tmp_path, monkeypatch, process_mol):
    """Protein and pose loaded from text are the molecules PoseBusters loads from .pdb / .sdf files (also with the
    temp-file fallback for a PoseBusters without ``_process_mol``)."""
    _, cfg = bust._config("dock")
    with Results(imported) as r:
        text = bust._protein_pdb(r, PID)
        sdf = r.top_sdf(PID, "molecule5")
    pdb_file, sdf_file = tmp_path / "protein.pdb", tmp_path / "pose.sdf"
    pdb_file.write_text(text)
    sdf_file.write_text(sdf)
    want_protein = safe_load_mol(pdb_file, **cfg["loading"]["mol_cond"])
    want_pose = next(safe_supply_mols(sdf_file, **cfg["loading"]["mol_pred"]))
    if not process_mol:
        monkeypatch.setattr(bust, "_pb_process_mol", lambda: None)
    work = tmp_path / "work"
    work.mkdir()
    protein = bust._load_protein(text, cfg, str(work))
    pose = bust._load_pose(sdf.encode(), cfg, str(work))
    assert protein.ToBinary() == want_protein.ToBinary()
    assert Chem.MolToPDBBlock(protein) == Chem.MolToPDBBlock(want_protein)
    assert pose.ToBinary() == want_pose.ToBinary() and Chem.MolToMolBlock(pose) == Chem.MolToMolBlock(want_pose)
    assert protein.GetNumAtoms() == text.count("\nATOM") + text.startswith("ATOM")
    assert bust._load_protein("not a PDB file\n", cfg, str(work)) is None
    with pytest.raises(ValueError, match="Could not load molecule"):
        bust._load_pose(b"not an SDF\n", cfg, str(work))
    assert os.listdir(work) == []


# --- results equal PoseBusters' path-based call --------------------------------------------------------------------

@pytest.mark.parametrize("dataset, select, top_k, n_checked", [("imported", "top_k", 2, 4), ("fake_run", "best", 1, 2)])
def test_rows_equal_path_based_posebusters(request, tmp_path, dataset, select, top_k, n_checked):
    """bust_results (protein loaded once per worker in memory, poses as Mols) gives PoseBusters' path-based results;
    masks, pb_valid and failed_checks encode them with check_names."""
    path = copy_results(request.getfixturevalue(dataset), tmp_path)
    pid = PID if dataset == "imported" else FAKE_PID
    assert bust.bust_results(path, select=select, top_k=top_k, workers=WORKERS, tmp_dir=str(tmp_path / "tmp")) \
        == [pid]
    assert json.loads(pq.read_schema(pb_part(path, pid)).metadata[b"pocketvina.pb_checks"]) == NAMES
    with Results(path) as r:
        df = r.posebusters(pid)
        assert list(df.columns) == tables.POSEBUSTERS.names
        assert r.posebusters_checks(pid) == NAMES
        assert set(df.pb_config) == {"dock"} and set(df.pb_version) == {posebusters.__version__}
        checked = df[df.pb_status.isin(["valid", "invalid"])]
        assert len(checked) == n_checked
        expected = path_based(r, pid, list(checked.itertuples()), tmp_path)
        assert list(expected.columns) == NAMES
        for row, (_, want) in zip(checked.itertuples(), expected.iterrows()):
            values = [bust._flag(want[n]) for n in NAMES]
            assert mask_values(row.pb_mask, row.pb_na_mask) == values, (row.ligand_id, row.pocket, row.mode)
            failed = [n for n, v in zip(NAMES, values) if v is False]
            assert row.pb_valid == (not failed) and row.pb_status == ("invalid" if failed else "valid")
            assert row.failed_checks == (",".join(failed) or None)
            assert row.n_heavy_docked == Chem.MolFromMolBlock(pose_sdf(r, pid, row)).GetNumHeavyAtoms()
            assert row.runtime_s > 0 and row.error is None
        if dataset == "imported":
            assert set(checked.pb_status) == {"valid"}
            assert sorted(zip(checked.ligand_id, checked.pocket, checked["mode"])) == [
                ("molecule2", 2, 1), ("molecule2", 2, 2), ("molecule5", 1, 1), ("molecule5", 2, 1)]
            no_pose = df[df.pb_status == "no_pose"]
            assert no_pose[["ligand_id", "pocket", "mode"]].values.tolist() == [["molecule10", -1, -1]]
            assert no_pose.pb_valid.isna().all() and (no_pose.pb_na_mask == ALL_BITS).all()
            assert (no_pose.pb_mask == 0).all() and no_pose.runtime_s.isna().all()
        else:
            assert set(checked.pb_status) == {"invalid"}  # the fake poses are not in the pocket
            assert all("protein-ligand_maximum_distance" in f for f in checked.failed_checks)


def test_per_pocket_selection(imported, tmp_path):
    path = copy_results(imported, tmp_path)
    bust.bust_results(path, select="per_pocket", workers=WORKERS, tmp_dir=str(tmp_path / "tmp"))
    with Results(path) as r:
        df = r.posebusters(PID)
        modes = r.modes(PID)
    got = list(zip(df.ligand_id, df.pocket, df["mode"], df.pb_status))
    best = modes.sort_values(["affinity", "mode"]).groupby(["ligand_idx", "pocket"]).head(1)
    assert sorted(got) == sorted(
        [("molecule2" if i == 0 else "molecule5", k, m, "valid") for i, k, m in zip(best.ligand_idx, best.pocket,
                                                                                     best["mode"])]
        + [("molecule10", 1, -1, "no_pose"), ("molecule10", 2, -1, "no_pose")])


# --- timeouts and failing workers ----------------------------------------------------------------------------------

def test_slow_poses_are_retried_then_timeout_rows(imported, tmp_path, monkeypatch, forked, caplog):
    """A pose still running at the first timeout is checked again in a fresh worker with the next one, and is a
    "timeout" row after the last; no pose is dropped and no worker is left behind. No check races a limit: a slow
    check sleeps 60 s, far longer than the limits, and a check that must finish gets 60 s; whether a worker is slow is
    decided here before the pool starts it, in the order the pool starts them, not by a race between workers. Nor is
    the time of the whole run asserted: a slow check that was not stopped leaves a file when its 60 s are over."""
    mode, slow, start_worker = ["retried"], [], bust._Worker.__init__

    def numbered_worker(self, *args, **kwargs):  # here, before the pool forks its next worker, which inherits the
        # list (its own entry last): "retried": a run's first WORKERS workers (each pose's first check) are slow;
        # "timeout": all are
        slow.append(mode[0] == "timeout" or len(slow) < WORKERS)
        start_worker(self, *args, **kwargs)

    def make(original):
        def check(*args, **kwargs):
            if slow[-1]:
                time.sleep(60)
                (tmp_path / f"not_stopped_{os.getpid()}").touch()
            # 5 conformations instead of the config's 50: a quicker check (with 50: 10 s on 2 CPUs)
            return original(*args, **dict(kwargs, ensemble_number_conformations=5))
        return check

    monkeypatch.setattr(bust._Worker, "__init__", numbered_worker)
    patch_check(monkeypatch, make)
    caplog.set_level("INFO", logger="pocketvina.bust")
    path = copy_results(imported, tmp_path / "retried")
    assert bust.bust_results(path, timeouts=(1.5, 60), workers=WORKERS, tmp_dir=str(tmp_path / "tmp")) == [PID]
    with Results(path) as r:
        df = r.posebusters(PID).set_index("ligand_id")
    assert df.pb_status.to_dict() == {"molecule2": "valid", "molecule5": "valid", "molecule10": "no_pose"}
    assert caplog.text.count("longer than 1.5 s, trying again with 60 s") == 2
    assert multiprocessing.active_children() == [] and not list(tmp_path.glob("not_stopped_*"))

    mode[0] = "timeout"
    path = copy_results(imported, tmp_path / "timeout")
    assert bust.bust_results(path, timeouts=(1.5, 3), workers=WORKERS, tmp_dir=str(tmp_path / "tmp")) == [PID]
    with Results(path) as r:
        df = r.posebusters(PID).set_index("ligand_id")
    assert df.pb_status.to_dict() == {"molecule2": "timeout", "molecule5": "timeout", "molecule10": "no_pose"}
    for slow in (df.loc["molecule2"], df.loc["molecule5"]):
        assert pd.isna(slow.pb_valid) and slow.pb_mask == 0 and slow.pb_na_mask == ALL_BITS
        assert "timed out" in slow.error and slow.runtime_s >= 4.5 and pd.isna(slow.n_heavy_docked)
    assert caplog.text.count("longer than 1.5 s, trying again with 3 s") == 2
    assert multiprocessing.active_children() == []
    assert os.listdir(tmp_path / "tmp") == []
    assert not list(tmp_path.glob("not_stopped_*"))  # the slow checks were stopped, not waited for


def test_dead_worker_gives_error_rows(imported, tmp_path, monkeypatch, forked):
    def make(original):
        def check(*args, **kwargs):
            os._exit(3)
        return check

    patch_check(monkeypatch, make)
    path = copy_results(imported, tmp_path)
    assert bust.bust_results(path, workers=WORKERS, tmp_dir=str(tmp_path / "tmp")) == [PID]
    with Results(path) as r:
        df = r.posebusters(PID).set_index("ligand_id")
    assert df.pb_status.to_dict() == {"molecule2": "error", "molecule5": "error", "molecule10": "no_pose"}
    assert all("worker process died (exit code 3)" in e for e in df.error[:2])
    assert multiprocessing.active_children() == []


def test_unreadable_structure_gives_error_rows(imported, tmp_path):
    path = copy_results(imported, tmp_path)
    structure = Path(path) / "structures" / f"{PID}.pdbqt.gz"
    structure.write_bytes(b"not gzip")
    assert bust.bust_results(path, workers=WORKERS, tmp_dir=str(tmp_path / "tmp")) == [PID]
    with Results(path) as r:
        df = r.posebusters(PID)
    assert df.pb_status.tolist() == ["error", "error", "no_pose"]
    assert df.error[0].startswith("cannot read the protein structure")


def test_unreadable_pose_gives_an_error_row(imported, tmp_path, monkeypatch):
    """A pose that cannot be read is an "error" row; the other poses of the protein are still checked."""
    top_sdf = Results.top_sdf

    def broken(self, protein, ligand):
        if self.ligand_id(ligand) == "molecule2":
            raise ValueError("damaged pose file")
        return top_sdf(self, protein, ligand)

    monkeypatch.setattr(Results, "top_sdf", broken)
    path = copy_results(imported, tmp_path)
    assert bust.bust_results(path, workers=WORKERS, tmp_dir=str(tmp_path / "tmp")) == [PID]
    monkeypatch.undo()
    with Results(path) as r:
        df = r.posebusters(PID).set_index("ligand_id")
    assert df.pb_status.to_dict() == {"molecule2": "error", "molecule5": "valid", "molecule10": "no_pose"}
    assert df.loc["molecule2"].error == "cannot read the pose: ValueError: damaged pose file"
    assert multiprocessing.active_children() == []


def test_pool_is_clean_after_an_exception(imported, tmp_path):
    """A check stopped by an exception leaves no busy worker behind: the background process goes on with the
    same pool, and an old worker would answer for the next protein's poses."""
    _, cfg = bust._config("dock")
    with Results(imported) as r:
        text = bust._protein_pdb(r, PID)
        sdf = r.top_sdf(PID, "molecule5").encode()

    class Stop(BaseException):
        pass

    def stop():
        raise Stop

    pool = bust._Pool(cfg, 2, (60,), str(tmp_path))
    try:
        with pytest.raises(Stop):  # pose 0 is being checked by a worker when pose 1 raises
            pool.check(text, [lambda: ("sdf", sdf), stop], label="first")
        assert all(w.task is None for w in pool.workers)
        got = pool.check(text, [lambda: ("sdf", sdf)] * 3, label="second")
        assert [g.get("status") for g in got] == [None] * 3 and all(False not in g["values"] for g in got)
    finally:
        pool.close()
    assert multiprocessing.active_children() == []


def _running(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG and /proc are Linux only")
def test_busy_worker_ends_with_a_killed_parent(imported, tmp_path):
    """A parent killed with SIGKILL cannot stop a pose past its timeout: its busy worker must not run on."""
    with Results(imported) as r:
        (tmp_path / "protein.pdb").write_text(bust._protein_pdb(r, PID))
        (tmp_path / "pose.sdf").write_text(r.top_sdf(PID, "molecule5"))
    marker = tmp_path / "worker.pid"
    code = (
        "import functools, os, sys, time\n"
        "import posebusters.posebusters as pb_module\n"
        "from pocketvina import bust\n"
        "original = pb_module.module_dict['energy_ratio']\n"
        "def slow(*args, **kwargs):\n"
        "    with open(sys.argv[3] + '.tmp', 'w') as f:\n"
        "        f.write(str(os.getpid()))\n"
        "    os.replace(sys.argv[3] + '.tmp', sys.argv[3])\n"
        "    time.sleep(120)\n"
        "pb_module.module_dict['energy_ratio'] = functools.wraps(original)(slow)\n"
        "bust.START_METHOD = 'fork'\n"
        "_, cfg = bust._config('dock')\n"
        "text, sdf = open(sys.argv[1]).read(), open(sys.argv[2], 'rb').read()\n"
        "bust._Pool(cfg, 1, (300,), os.path.dirname(sys.argv[3])).check(text, [lambda: ('sdf', sdf)])\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))
    proc = subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "protein.pdb"), str(tmp_path / "pose.sdf"),
                             str(marker)], cwd=str(tmp_path), env=env)
    worker = None
    try:
        deadline = time.monotonic() + 120
        while not marker.exists():
            assert proc.poll() is None and time.monotonic() < deadline, "the worker did not start its check"
            time.sleep(0.1)
        worker = int(marker.read_text())
        proc.kill()
        proc.wait(30)
        deadline = time.monotonic() + 10
        while _running(worker) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not _running(worker), "the worker went on checking its pose after its parent was killed"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if worker is not None and _running(worker):
            os.kill(worker, signal.SIGKILL)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG and /proc are Linux only")
def test_background_buster_ends_with_a_killed_docking_process(imported, tmp_path):
    """A docking process killed with SIGKILL (e.g. by the OOM killer) cannot terminate() its BackgroundBuster: the
    background process and its busy worker must not run on (the run, resumed, checks the protein again)."""
    path = copy_results(imported, tmp_path)
    marker = tmp_path / "busy.pids"
    serve = (
        "import functools, os, sys, time\n"
        "import posebusters.posebusters as pb_module\n"
        "original = pb_module.module_dict['energy_ratio']\n"
        "def slow(*args, **kwargs):\n"
        f"    with open({str(marker)!r} + '.tmp', 'w') as f:\n"
        "        f.write(f'{os.getppid()} {os.getpid()}')\n"
        f"    os.replace({str(marker)!r} + '.tmp', {str(marker)!r})\n"
        "    time.sleep(120)\n"
        "pb_module.module_dict['energy_ratio'] = functools.wraps(original)(slow)\n"
        "from pocketvina.bust import _serve\n"
        "sys.exit(_serve(sys.argv[1]))\n")
    code = (
        "import sys, time\n"
        "from pocketvina import bust\n"
        "bust._SERVE_CODE = sys.argv[3]\n"
        "b = bust.BackgroundBuster(sys.argv[1], workers=1, tmp_dir=sys.argv[2])\n"
        f"b.submit({PID!r})\n"
        "time.sleep(300)\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))
    proc = subprocess.Popen([sys.executable, "-c", code, path, str(tmp_path / "tmp"), serve], cwd=str(tmp_path),
                            env=env)
    pids = []
    try:
        deadline = time.monotonic() + 120
        while not marker.exists():
            assert proc.poll() is None and time.monotonic() < deadline, "the background check did not start"
            time.sleep(0.1)
        pids = [int(p) for p in marker.read_text().split()]  # the background process, its busy worker
        proc.kill()
        proc.wait(30)
        deadline = time.monotonic() + 15
        while any(map(_running, pids)) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not any(map(_running, pids)), "the PoseBusters checks went on after the docking process was killed"
        assert not pb_part(path).exists()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for pid in pids:
            if _running(pid):
                os.kill(pid, signal.SIGKILL)


def test_forkserver_workers_stay_while_idle(imported, tmp_path, monkeypatch):
    """With the "forkserver" start method the workers are children of the fork server, not of the caller: they
    must not take that for a dead parent when idle (e.g. between the proteins of a docking run)."""
    if "forkserver" not in multiprocessing.get_all_start_methods():
        pytest.skip("needs the forkserver start method")
    monkeypatch.setattr(bust, "START_METHOD", "forkserver")
    monkeypatch.setattr(bust, "_WATCHDOG_S", 0.2)
    _, cfg = bust._config("dock")
    with Results(imported) as r:
        text = bust._protein_pdb(r, PID)
        sdf = r.top_sdf(PID, "molecule5").encode()
    pool = bust._Pool(cfg, 1, (60,), str(tmp_path))
    try:
        assert "values" in pool.check(text, [lambda: ("sdf", sdf)])[0]
        worker = pool.workers[0].proc
        time.sleep(1.5)  # idle for several watchdog periods
        assert worker.is_alive()
        assert "values" in pool.check(text, [lambda: ("sdf", sdf)])[0]
        assert [w.proc for w in pool.workers] == [worker]
    finally:
        pool.close()
    assert multiprocessing.active_children() == []


# --- resume, overwrite, temp files ---------------------------------------------------------------------------------

def test_resume_overwrite_and_no_files_outside_tmp(imported, tmp_path, monkeypatch, caplog):
    path = copy_results(imported, tmp_path)
    tmp, cwd, env_tmp = tmp_path / "tmp", tmp_path / "cwd", tmp_path / "env_tmp"
    for d in (cwd, env_tmp):
        d.mkdir()
    for var in ("TMPDIR", "POCKETVINA_TMPDIR"):
        monkeypatch.setenv(var, str(env_tmp))
    monkeypatch.chdir(cwd)
    before = files_in(path)
    assert bust.bust_results(path, workers=WORKERS, tmp_dir=str(tmp)) == [PID]
    assert files_in(path) == before | {f"tables/posebusters/{PID}.parquet"}
    assert os.listdir(tmp) == [] and os.listdir(cwd) == [] and os.listdir(env_tmp) == []
    part = pb_part(path)
    data = part.read_bytes()

    caplog.set_level("INFO", logger="pocketvina.bust")
    assert bust.bust_results(path, workers=WORKERS, proteins=[PID]) == []  # done: skipped
    assert part.read_bytes() == data and "already there, skipped" in caplog.text
    assert bust.bust_results(path, workers=WORKERS, select="per_pocket") == []  # other settings: kept
    assert part.read_bytes() == data and "overwrite=True replaces them" in caplog.text
    assert bust.bust_results(path, workers=WORKERS, select="per_pocket", overwrite=True) == [PID]
    assert len(pq.read_table(part)) == 6
    meta = pq.read_schema(part).metadata
    assert json.loads(meta[b"pocketvina.pb_select"]) == dict(select="per_pocket", top_k=1)
    assert meta[b"pocketvina.poses_sha256"].decode() == \
        ResultsDir(path).protein_manifest(PID)["files"][f"poses/{PID}.poses"]["sha256"]

    # a protein docked again after its checks (other pose file): the old part is ignored, and checked again
    table = pq.read_table(part)
    pq.write_table(table.replace_schema_metadata({**meta, b"pocketvina.poses_sha256": b"0" * 64}), part)
    with Results(path) as r:
        assert r.posebusters().empty and "pb_valid" not in r.pairs().columns
        assert "posebusters" not in {t[0] for t in r.duckdb().execute("SHOW TABLES").fetchall()}
    assert bust.bust_results(path, workers=WORKERS, select="per_pocket") == [PID]
    with Results(path) as r:
        assert len(r.posebusters()) == 6
    assert os.listdir(env_tmp) == [] and os.listdir(cwd) == []
    assert multiprocessing.active_children() == []


@pytest.mark.parametrize("kw, err, match", [
    (dict(select="all"), ValueError, "select"),
    (dict(top_k=0), ValueError, "top_k"),
    (dict(timeouts=(0,)), ValueError, "timeouts"),
    (dict(timeouts=()), ValueError, "timeouts"),
    (dict(workers=0), ValueError, "workers"),
    (dict(config="nope"), ValueError, "unknown PoseBusters config"),
    (dict(config=3), TypeError, "config"),
    (dict(proteins=[PID, "nope"]), KeyError, "nope"),
])
def test_invalid_arguments(imported, tmp_path, kw, err, match):
    before = files_in(imported)
    with pytest.raises(err, match=match):
        bust.bust_results(imported, tmp_dir=str(tmp_path / "tmp"), **kw)
    assert files_in(imported) == before
    assert not (tmp_path / "tmp").exists() or os.listdir(tmp_path / "tmp") == []


def test_import_without_posebusters(tmp_path):
    """``import pocketvina`` and the atlas modules work without PoseBusters; checks raise ImportError."""
    code = (
        "import sys\n"
        "sys.modules['posebusters'] = None\n"
        "import pocketvina, pocketvina.bust, pocketvina.pipeline, pocketvina.io.results\n"
        "try:\n"
        "    pocketvina.bust.bust_results(sys.argv[1])\n"
        "except ImportError as e:\n"
        "    print('ImportError:', e)\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))
    proc = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "ImportError: PoseBusters is not installed: pip install posebusters" in proc.stdout


# --- Results accessors ---------------------------------------------------------------------------------------------

def test_results_accessors(imported, busted, tmp_path):
    with Results(imported) as r:  # no PoseBusters results
        pairs = r.pairs()
        assert "pb_valid" not in pairs.columns and pairs.pb_status.isna().all()
        assert r.pairs(columns=["ligand_id", "pb_valid"]).pb_valid.isna().all()
        empty = r.posebusters()
        assert empty.empty and list(empty.columns) == tables.POSEBUSTERS.names
        assert "posebusters" not in {t[0] for t in r.duckdb().execute("SHOW TABLES").fetchall()}
        with pytest.raises(KeyError, match="no PoseBusters results"):
            r.posebusters_checks(PID)

    with Results(busted) as r:
        pairs = r.pairs()
        stored = pd.read_parquet(Path(busted) / "tables" / "pairs" / f"{PID}.parquet")
        assert list(pairs.columns) == tables.PAIRS.names + ["pb_valid", "ligand_efficiency"]
        assert pairs.drop(columns=["pb_status", "pb_valid", "ligand_efficiency"]).equals(stored.drop(
            columns=["pb_status"]))
        assert pairs.pb_status.tolist() == ["valid", "valid", "no_pose"]
        assert pairs.pb_valid.tolist()[:2] == [True, True] and pd.isna(pairs.pb_valid[2])
        n_heavy = r.ligands().set_index("ligand_idx").n_heavy
        assert pairs.ligand_efficiency[:2].tolist() == pytest.approx(
            [6.3 / n_heavy[0], 5.2 / n_heavy[1]], abs=1e-12)
        assert np.isnan(pairs.ligand_efficiency[2])
        sub = r.pairs(PID, columns=["ligand_efficiency", "ligand_id", "pb_valid"])
        assert list(sub.columns) == ["ligand_efficiency", "ligand_id", "pb_valid"]
        assert sub.ligand_efficiency.equals(pairs.ligand_efficiency)
        assert r.pairs(columns=["ligand_idx", "best_pocket"]).equals(stored[["ligand_idx", "best_pocket"]])
        # proteins as a generator: read for the pairs and again for the PoseBusters parts
        assert r.pairs(p for p in [PID]).equals(pairs)
        assert r.pairs(iter([PID]), columns=["pb_status", "pb_valid"]).equals(pairs[["pb_status", "pb_valid"]])

        pb = r.posebusters()
        assert len(pb) == 3 and r.posebusters(PID, columns=["pb_status"]).pb_status.tolist() == \
            ["valid", "valid", "no_pose"]
        con = r.duckdb()
        assert con.execute("SELECT pb_status, count(*) FROM posebusters GROUP BY 1 ORDER BY 1").fetchall() == \
            [("no_pose", 1), ("valid", 2)]
        joined = con.execute("SELECT p.ligand_id, b.pb_valid FROM pairs p JOIN posebusters b ON p.protein_id = "
                             "b.protein_id AND p.ligand_idx = b.ligand_idx AND p.best_pocket = b.pocket AND "
                             "p.best_mode = b.mode ORDER BY 1").fetchall()
        assert joined == [("molecule10", None), ("molecule2", True), ("molecule5", True)]
        # the pairs view has pairs()'s PoseBusters columns and ligand efficiency
        view = con.execute("SELECT * FROM pairs ORDER BY ligand_idx").fetchdf()
        by_idx = pairs.sort_values("ligand_idx").reset_index(drop=True)
        assert list(view.columns) == list(pairs.columns)
        assert view.pb_status.tolist() == by_idx.pb_status.tolist() == ["valid", "valid", "no_pose"]
        assert [None if pd.isna(v) else bool(v) for v in view.pb_valid] == [True, True, None]
        np.testing.assert_array_equal(view.ligand_efficiency.to_numpy(), by_idx.ligand_efficiency.to_numpy())
        assert con.execute("SELECT ligand_id FROM pairs WHERE pb_valid ORDER BY ligand_efficiency DESC").fetchall() \
            == [(lid,) for lid in by_idx[by_idx.pb_valid.fillna(False)].sort_values(
                "ligand_efficiency", ascending=False).ligand_id]
        csv_path = r.export(tmp_path / "export", fmt="csv")[0]
        assert pd.read_csv(csv_path).pb_status.tolist() == ["valid", "valid", "no_pose"]


# --- run_atlas(posebusters=True) -----------------------------------------------------------------------------------

def test_run_atlas_posebusters_option(tmp_path, example_dir, fake_engine_env, stubs):
    """Off by default; with posebusters=True a background process checks every protein after its commit, and the
    proteins found done when a run is resumed; the run waits for it and leaves no temp files."""
    ligands = ("molecule1", "molecule2")
    one = [example_dir / "proteins" / "1fbl.pdb"]
    out, tmp = tmp_path / "results", tmp_path / "tmp"
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=ligands, proteins=one)
    pipeline.run_atlas(PocketVinaGPU(atlas_config(plist, smiles, out, tmp, top_pockets=1)))
    assert not (out / "tables" / "posebusters").exists()

    plist, smiles = make_inputs(tmp_path, example_dir, ligands=ligands, proteins=one + [FIXTURE_RECEPTOR])
    config = atlas_config(plist, smiles, out, tmp, top_pockets=1, posebusters=True, posebusters_workers=2)
    pipeline.run_atlas(PocketVinaGPU(config))
    assert sorted(os.listdir(out / "tables" / "posebusters")) == [f"{FAKE_PID}.parquet", f"{PID}.parquet"]
    assert os.listdir(tmp) == []
    with Results(out) as r:
        pb = r.posebusters()
        assert len(pb) == 4 and set(pb.pb_status) <= {"valid", "invalid"}
        assert set(zip(pb.protein_id, pb.ligand_id)) == {(p, lig) for p in (FAKE_PID, PID) for lig in ligands}
        assert r.pairs().pb_valid.notna().all()
    assert len(fake_engine_env.calls()) == 2  # 1fbl was docked once


def test_background_buster_close_and_terminate(busted, tmp_path, caplog):
    tmp = tmp_path / "tmp"
    before = files_in(busted)
    b = bust.BackgroundBuster(busted, workers=1, tmp_dir=str(tmp))
    b.submit(PID)  # checked already: skipped
    b.submit("nope")  # not a committed protein: logged, exit code 1
    assert b.close() == 1 and b.submitted == 2
    stopped = bust.BackgroundBuster(busted, workers=1, tmp_dir=str(tmp))
    stopped.terminate()
    assert stopped.proc.returncode is not None
    stopped.submit(PID)  # after it stopped: logged, not raised
    assert "the PoseBusters process stopped" in caplog.text and stopped.submitted == 0
    assert files_in(busted) == before and os.listdir(tmp) == []


def test_background_buster_submit_does_not_wait(tmp_path, monkeypatch):
    """submit() returns at once even when the process reads nothing for a while (a resumed run submits every
    protein found done at the start: more than a pipe holds); close() passes on every id."""
    n = 20000  # about 500 kB of ids
    go = tmp_path / "go"
    code = (f"import os, sys, time\n"
            f"while not os.path.exists({str(go)!r}): time.sleep(0.05)\n"
            f"sys.exit(0 if [line.strip() for line in sys.stdin] == [f'AF-{{i}}' for i in range({n})] else 1)\n")
    monkeypatch.setattr(bust, "_SERVE_CODE", code)
    b = bust.BackgroundBuster(tmp_path / "results", tmp_dir=str(tmp_path / "tmp"))
    try:
        submitting = threading.Thread(target=lambda: [b.submit(f"AF-{i}") for i in range(n)], daemon=True)
        submitting.start()
        submitting.join(60)
        assert not submitting.is_alive(), "submit() waited for the background process"
    finally:
        go.touch()
    assert b.close() == 0 and b.submitted == n
    assert os.listdir(tmp_path / "tmp") == []


def test_posebusters_config_options(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",),
                                proteins=[example_dir / "proteins" / "1fbl.pdb"])
    with pytest.raises(ValueError, match="posebusters must be True or False"):
        atlas_config(plist, smiles, tmp_path / "out", tmp_path / "tmp", posebusters="yes")
    with pytest.raises(ValueError, match="posebusters_workers"):
        atlas_config(plist, smiles, tmp_path / "out", tmp_path / "tmp", posebusters_workers=0)
    assert PocketVinaConfig(protein_paths=str(plist), smiles_file=str(smiles)).posebusters is False

    def missing():
        raise ImportError("PoseBusters is not installed: pip install posebusters")

    monkeypatch.setattr(bust, "require_posebusters", missing)
    config = atlas_config(plist, smiles, tmp_path / "out", tmp_path / "tmp", posebusters=True)
    with pytest.raises(ImportError, match="pip install posebusters"):
        pipeline.run_atlas(PocketVinaGPU(config))
    assert fake_engine_env.calls() == [] and not (tmp_path / "out").exists()  # nothing docked
