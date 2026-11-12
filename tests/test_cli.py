"""pocketvina.cli (the ``pocketvina`` command, ``python -m pocketvina``) with ``click.testing.CliRunner``.

Docking and the GPU benchmark run the fake engine (tests/fake_engine.py) with the fixture pockets instead of P2Rank;
the commands that read results use the tests/data run imported with io.importer. ``bench gpu`` samples a fake
``nvidia-smi`` that reports the benchmark's engines (found in /proc) on a fake GPU.
"""
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from pocketvina import cli, gpu, pipeline
from pocketvina import engine as pv_engine
from pocketvina import ligands as pv_ligands
from pocketvina import main as pv_main
from pocketvina.io.importer import import_legacy
from pocketvina.io.results import Results
from pocketvina.io.tables import read_table
from pocketvina.main import P2RankProcessor
from test_gpu import fake_gpus, wrapper_engine
from test_pipeline import (
    FIXTURE_RECEPTOR,
    POCKETS_CSV,
    _alive,
    cropped_receptor,
    make_database,
    make_inputs,
    stub_predict_pockets,
    write_pockets,
)
from test_viz import HAS_PY3DMOL, needs_py3dmol  # `show --backend py3dmol`: the optional viz extra

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
PID = "AF-Q96LA5-F1-model_v4"
LIGANDS = ("molecule1", "molecule2")
COMMANDS = (("dock", "--proteins"), ("bust", "--select"), ("show", "--ligand"), ("export", "--what"),
            ("info", "--json"), ("query", "--csv"), ("import-legacy", "--smiles"), ("bench", "gpu"),
            ("bench gpu", "--n-ligands"), ("setup", "p2rank"), ("setup p2rank", "--from"), ("atlas", "download"),
            ("atlas search", "--top"), ("atlas show", "--ligand"), ("atlas export", "--fmt"),
            ("atlas download", "--out"))
OOM_LINE = "Err-4:CL_MEM_OBJECT_ALLOCATION_FAILURE"


def _has_toml_reader() -> bool:
    try:
        import tomllib  # noqa: F401
    except ImportError:
        try:
            import tomli  # noqa: F401
        except ImportError:
            return False
    return True


# dock --config needs tomllib (Python 3.11+) or the optional tomli (see test_dock_config_file_needs_tomli_...)
needs_toml = pytest.mark.skipif(not _has_toml_reader(), reason="Python < 3.11 without tomli: no TOML reader")


# --- fixtures and helpers ------------------------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _restore_root_logger():
    """The commands set the root logger's level (-v/-q); give the other tests theirs back."""
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    yield
    root.setLevel(level)
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)


@pytest.fixture
def stubs(monkeypatch):
    """Fixture pockets instead of P2Rank (atlas and "files" format), cropped receptors (fast OpenBabel)."""
    monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets())
    monkeypatch.setattr(pipeline, "receptor_pdbqt", cropped_receptor())
    monkeypatch.setattr(pv_main, "receptor_pdbqt", cropped_receptor())

    def run_p2rank(self, pdb_file):
        base = os.path.splitext(os.path.basename(pdb_file))[0]
        write_pockets(os.path.join(self.pockets_dir, f"{base}.csv"), 3)
        return True

    monkeypatch.setattr(P2RankProcessor, "run_p2rank", run_p2rank)


@pytest.fixture(scope="module")
def imported(real_run, tmp_path_factory) -> str:
    """The tests/data run as an atlas results directory (read only: copy it to write)."""
    return import_legacy(str(real_run.root), str(tmp_path_factory.mktemp("imported") / "results"),
                         str(real_run.smiles_csv))


def invoke(*args):
    return CliRunner().invoke(cli.cli, [str(a) for a in args], catch_exceptions=False)


def ok(result):
    assert result.exit_code == 0, result.output
    return result


def stderr_of(result) -> str:
    """The command's standard error. Click < 8.2 (the last for Python 3.9 is 8.1) mixes it into ``output`` and
    raises on ``result.stderr``."""
    try:
        return result.stderr
    except ValueError:  # "stderr not separately captured"
        return result.output


def inputs(tmp_path, example_dir, proteins=(FIXTURE_RECEPTOR,), ligands=LIGANDS):
    return make_inputs(tmp_path, example_dir, ligands=ligands, proteins=list(proteins))


def manifest_params(out) -> dict:
    return json.loads((Path(out) / "MANIFEST.json").read_text())["params"]


def copy_results(src, dst) -> str:
    shutil.copytree(src, dst)
    return str(dst)


# --- group, help, exit codes ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("command, option", [((), "dock")] + [(c.split(), o) for c, o in COMMANDS],
                         ids=["pocketvina"] + [c for c, _ in COMMANDS])
def test_help(command, option):
    result = ok(invoke(*command, "--help"))
    assert result.stdout.startswith("Usage: ") and option in result.stdout
    assert ok(invoke(*command, "-h")).stdout == result.stdout


def test_dock_help_documents_the_config_file_and_exit_codes():
    text = ok(invoke("dock", "--help")).stdout
    for needle in ("--config", "run.toml", "snake_case", "[default: atlas]", "Exit codes", "nothing docked"):
        assert needle in text, needle


def test_version_and_usage_errors_exit_with_1():
    from pocketvina import __version__

    assert __version__ in ok(invoke("--version")).stdout
    for args in (["dock", "--bogus"], ["info"], ["export", "--results", ".", "--what", "nope", "--out", "x"],
                 ["nope"], ["query"]):
        result = invoke(*args)
        assert result.exit_code == cli.EXIT_USAGE, (args, result.output)
        assert "Error" in result.output
    result = invoke("-v", "info", "--results", ".")  # -v: the traceback too
    assert result.exit_code == cli.EXIT_USAGE
    assert "Traceback" in stderr_of(result) and "MANIFEST.json" in stderr_of(result)
    assert "Traceback" not in invoke("info", "--results", ".").output


def test_python_m_pocketvina(tmp_path):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))
    run = subprocess.run([sys.executable, "-m", "pocketvina", "--help"], cwd=tmp_path, env=env, capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert "Usage: python -m pocketvina" in run.stdout and "bench" in run.stdout
    run = subprocess.run([sys.executable, "-m", "pocketvina", "info"], cwd=tmp_path, env=env, capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 1 and "Missing option '--results'" in run.stderr
    assert os.listdir(tmp_path) == []


# --- dock ----------------------------------------------------------------------------------------------------------

def test_dock_atlas(tmp_path, example_dir, fake_engine_env, stubs):
    """The flags reach PocketVinaConfig (the engine config and the MANIFEST params); a rerun resumes (nothing to
    dock), other settings need --force."""
    plist, smiles = inputs(tmp_path, example_dir)
    out, tmp = tmp_path / "results", tmp_path / "tmp"
    result = ok(invoke("dock", "--proteins", plist, "--ligands", smiles, "--out", out, "--top-pockets", 2, "--seed", 7,
                       "--box", 18, 18, 18, "--thread", 2000, "--num-modes", 4, "--energy-range", 3.5,
                       "--ligands-per-call", 1, "--engine-timeout", 60, "--tmp-dir", tmp))
    assert f"results: {out}" in result.stdout
    assert "proteins: 1 docked, 0 done before, 0 failed (of 1)" in result.stdout
    assert "highest-affinity poses: 2" in result.stdout
    params = manifest_params(out)
    assert {k: params[k] for k in ("output_format", "seed", "box_size", "thread", "num_modes", "energy_range",
                                   "top_pockets", "ligands_per_call", "pocket_source")} == dict(
        output_format="atlas", seed=7, box_size=[18, 18, 18], thread=2000, num_modes=4, energy_range=3.5,
        top_pockets=2, ligands_per_call=1, pocket_source="p2rank")
    calls = fake_engine_env.calls()
    assert len(calls) == 4  # 2 pockets x 2 ligands, one per call
    for c in calls:
        o = c["options"]
        assert (o["seed"], o["size_x"], o["num_modes"], o["thread"], o["energy_range"]) == (7, 18.0, 4, 2000, 3.5)
    runs = read_table(out / "tables" / "pocket_runs" / f"{PID}.parquet")
    assert len(runs) == 4 and set(runs.status) == {"ok"} and set(runs.n_modes) == {4}
    assert os.listdir(tmp) == []

    result = ok(invoke("dock", "--proteins", plist, "--ligands", smiles, "--out", out, "--top-pockets", 2, "--seed", 7,
                       "--box", 18, 18, 18, "--thread", 2000, "--num-modes", 4, "--energy-range", 3.5,
                       "--ligands-per-call", 1, "--tmp-dir", tmp))
    assert "proteins: 0 docked, 1 done before, 0 failed (of 1)" in result.stdout
    assert len(fake_engine_env.calls()) == 4

    result = invoke("dock", "--proteins", plist, "--ligands", smiles, "--out", out, "--seed", 8, "--tmp-dir", tmp)
    assert result.exit_code == cli.EXIT_USAGE and "holds another run" in result.output
    assert "--force" in result.output and "force=True" not in result.output  # the option, not the Python argument
    result = ok(invoke("dock", "--proteins", plist, "--ligands", smiles, "--out", out, "--seed", 8, "--force",
                       "--tmp-dir", tmp))
    assert "1 done before" in result.stdout and manifest_params(out)["seed"] == 8


def test_dock_ligands_per_call_zero_docks_all_ligands_of_a_pocket_in_one_call(tmp_path, example_dir,
                                                                               fake_engine_env, stubs):
    """--ligands-per-call 0 = all ligands of a pocket in one engine call (main.ligand_chunk_size), as in Python."""
    plist, smiles = inputs(tmp_path, example_dir)
    out = tmp_path / "results"
    ok(invoke("dock", "--proteins", plist, "--ligands", smiles, "--out", out, "--top-pockets", 2,
              "--ligands-per-call", 0, "--tmp-dir", tmp_path / "tmp"))
    assert manifest_params(out)["ligands_per_call"] == len(LIGANDS)
    assert len(fake_engine_env.calls()) == 2  # one per pocket
    result = invoke("dock", "--proteins", plist, "--ligands", smiles, "--out", out, "--ligands-per-call", -1)
    assert result.exit_code == cli.EXIT_USAGE and "--ligands-per-call" in result.output
    (option,) = [p for p in cli.dock.params if p.name == "ligands_per_call"]
    assert option.help.endswith(f"[default: --batch-size if set, else {cli.DEFAULT_LIGANDS_PER_CALL}]")


def test_dock_force_with_other_ligands_counts_the_proteins_docked_again(tmp_path, example_dir, fake_engine_env,
                                                                       stubs, monkeypatch):
    """--force with another ligand list: the proteins done before are docked again (ResultsDir.init_run), so they
    count as docked, and an engine call of theirs that fails is a partial failure (exit code 2)."""
    plist, smiles = inputs(tmp_path, example_dir, ligands=LIGANDS[:1])
    args = ["dock", "--proteins", plist, "--out", tmp_path / "results", "--top-pockets", 2,
            "--tmp-dir", tmp_path / "tmp"]
    ok(invoke(*args, "--ligands", smiles))
    _, smiles = inputs(tmp_path, example_dir, ligands=LIGANDS)
    half = wrapper_engine(tmp_path, fake_engine_env, 'if "pocket_rank = 2" in config:\n    sys.exit(1)', name="half")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(half))
    result = invoke(*args, "--ligands", smiles, "--force")
    assert result.exit_code == cli.EXIT_PARTIAL, result.output
    assert "proteins: 1 docked, 0 done before, 0 failed (of 1)" in result.stdout
    assert "engine calls failed: 1" in result.stdout
    data = json.loads(ok(invoke("info", "--results", tmp_path / "results", "--json")).stdout)
    assert data["problems"] == {PID: "1 of 2 engine runs failed"}
    text = ok(invoke("info", "--results", tmp_path / "results")).stdout
    assert f"problems (1 proteins):\n  {PID}: 1 of 2 engine runs failed" in text
    result = ok(invoke(*args, "--ligands", smiles))  # resumed: nothing docked now
    assert "proteins: 0 docked, 1 done before" in result.stdout and "engine calls failed" not in result.stdout


@needs_toml
def test_dock_options_reach_the_config(tmp_path, example_dir, monkeypatch):
    """Every dock option is passed on to PocketVinaConfig, from the command line and from --config alike."""
    plist, smiles = inputs(tmp_path, example_dir)
    seen = []

    def run_atlas(pv, force=False):
        seen.append((pv.config, force))
        return pv.config.output_dir

    monkeypatch.setattr(pipeline, "run_atlas", run_atlas)
    out, tmp = tmp_path / "results", tmp_path / "tmp"
    invoke("dock", "--proteins", plist, "--ligands", smiles, "--out", out, "--top-pockets", 3, "--box", 18, 19, 21.5,
           "--thread", 4000, "--num-modes", 5, "--energy-range", 2.5, "--seed", 9, "--ligand-prep", "legacy",
           "--ligands-per-call", 7, "--batch-size", 4, "--gpus", "1,0", "--procs-per-gpu", "auto",
           "--vram-per-proc", 3000, "--engine-timeout", 90, "--posebusters", "--posebusters-workers", 2,
           "--tmp-dir", tmp, "--p2rank-model", "default", "--p2rank-threads", 2, "--gzip-level", 3, "--force")
    cfg, force = seen[-1]
    expected = dict(protein_paths=str(plist), smiles_file=str(smiles), output_dir=str(out), output_format="atlas",
                    top_pockets=3, num_modes=5, energy_range=2.5, seed=9, ligand_prep="legacy", ligands_per_call=7,
                    batch_size=4, gpus="1,0", procs_per_gpu="auto", vram_per_proc_mib=3000, engine_timeout=90.0,
                    posebusters=True, posebusters_workers=2, tmp_dir=str(tmp), gzip_level=3)
    assert {k: getattr(cfg, k) for k in expected} == expected and force
    pc = cfg.pocket_config
    assert (pc.size_x, pc.size_y, pc.size_z, pc.thread) == (18, 19, 21.5, 4000)
    assert (cfg.p2rank_config.model, cfg.p2rank_config.threads) == ("default", 2)

    run_toml = tmp_path / "run.toml"
    run_toml.write_text(f'proteins = "{plist}"\nligands = "{smiles}"\nout = "{out}"\ntop_pockets = 3\n'
                        'box = [18, 19, 21.5]\nthread = 4000\nnum_modes = 5\nenergy_range = 2.5\nseed = 9\n'
                        'ligand_prep = "legacy"\nligands_per_call = 7\nbatch_size = 4\ngpus = [1, 0]\n'
                        'procs_per_gpu = "auto"\nvram_per_proc = 3000\nengine_timeout = 90\nposebusters = true\n'
                        f'posebusters_workers = 2\ntmp_dir = "{tmp}"\np2rank_model = "default"\np2rank_threads = 2\n'
                        'gzip_level = 3\nforce = true\n')
    invoke("dock", "--config", run_toml)
    assert len(seen) == 2 and seen[1] == seen[0]


@needs_toml
def test_dock_protein_source_on_the_command_line_overrides_config(tmp_path, example_dir, monkeypatch):
    """--proteins, or --database-path/--protein-list, given on the command line replace the --config file's proteins
    of the other kind (they used to exclude each other); both kinds in one place stay an error."""
    plist, smiles = inputs(tmp_path, example_dir)
    db, protein_list = make_database(tmp_path)
    seen = []
    monkeypatch.setattr(pipeline, "run_atlas", lambda pv, force=False: seen.append(pv.config))
    run_toml = tmp_path / "run.toml"
    common = f'ligands = "{smiles}"\nout = "{tmp_path / "results"}"\n'
    run_toml.write_text(f'proteins = "{plist}"\n' + common)
    invoke("dock", "--config", run_toml, "--database-path", db, "--protein-list", protein_list)
    assert (seen[-1].protein_paths, seen[-1].database_path, seen[-1].protein_list) == (None, str(db), str(protein_list))
    run_toml.write_text(f'database_path = "{db}"\nprotein_list = "{protein_list}"\n' + common)
    invoke("dock", "--config", run_toml, "--proteins", plist)
    assert (seen[-1].protein_paths, seen[-1].database_path, seen[-1].protein_list) == (str(plist), None, None)
    invoke("dock", "--config", run_toml, "--protein-list", protein_list)  # the file's --database-path stays
    assert (seen[-1].database_path, seen[-1].protein_list) == (str(db), str(protein_list))
    assert len(seen) == 3
    run_toml.write_text(f'proteins = "{plist}"\ndatabase_path = "{db}"\nprotein_list = "{protein_list}"\n' + common)
    for extra in ([], ["--proteins", plist, "--database-path", db]):
        result = invoke("dock", "--config", run_toml, *extra)
        assert result.exit_code == cli.EXIT_USAGE and "exclude each other" in result.output, (extra, result.output)
    assert len(seen) == 3


def test_dock_proteins_folder_or_structure(tmp_path, example_dir, fake_engine_env, stubs):
    folder = tmp_path / "structures"
    folder.mkdir()
    shutil.copy(FIXTURE_RECEPTOR, folder)
    shutil.copy(example_dir / "proteins" / "1fbl.pdb", folder)
    (folder / "notes.txt").write_text("not a structure\n")
    (folder / ".hidden.pdb").write_text("")
    _, smiles = inputs(tmp_path, example_dir, ligands=LIGANDS[:1])
    tmp = tmp_path / "tmp"
    result = ok(invoke("dock", "--proteins", folder, "--ligands", smiles, "--out", tmp_path / "r1", "--top-pockets", 1,
                       "--tmp-dir", tmp))
    assert "proteins: 2 docked, 0 done before, 0 failed (of 2)" in result.stdout
    assert Results(tmp_path / "r1").proteins() == sorted([PID, "1fbl"])
    result = ok(invoke("dock", "--proteins", folder / FIXTURE_RECEPTOR.name, "--ligands", smiles, "--out",
                       tmp_path / "r2", "--top-pockets", 1, "--tmp-dir", tmp))
    assert "proteins: 1 docked" in result.stdout and Results(tmp_path / "r2").proteins() == [PID]
    assert os.listdir(tmp) == []  # the protein list written for the folder is removed
    empty = tmp_path / "empty"
    empty.mkdir()
    result = invoke("dock", "--proteins", empty, "--ligands", smiles, "--out", tmp_path / "r3", "--tmp-dir", tmp)
    assert result.exit_code == cli.EXIT_USAGE and "no structure files" in result.output


@needs_toml
def test_dock_config_file(tmp_path, example_dir, fake_engine_env, stubs):
    """TOML keys in snake_case (or PocketVinaConfig field names), paths relative to the file, flags override it."""
    plist, smiles = inputs(tmp_path, example_dir)
    run_toml = plist.parent / "run.toml"
    run_toml.write_text(f'proteins = "{plist.name}"\nsmiles_file = "{smiles.name}"\nout = "../results"\n'
                        'top_pockets = 1\nseed = 5\nbox = 22\nnum_modes = 3\nprocs_per_gpu = "auto"\n'
                        'engine_timeout = 120\nligand_prep = "standard"\nposebusters = false\n')
    result = ok(invoke("dock", "--config", run_toml, "--seed", 9, "--tmp-dir", tmp_path / "tmp"))
    assert f"results: {tmp_path / 'results'}" in result.stdout
    params = manifest_params(tmp_path / "results")
    assert (params["seed"], params["top_pockets"], params["box_size"], params["num_modes"]) == (9, 1, [22, 22, 22], 3)
    assert {c["options"]["seed"] for c in fake_engine_env.calls()} == {9}

    cases = {'bogus = 1\n': "unknown key 'bogus'", 'top_pockets = 0\n': "top_pockets",
             'ligands = "x.csv"\n': "set the same option", 'box = [1, 2]\n': "Takes 3 values",
             'seed = "abc"\n': "not an integer or 'random'", 'top_pockets = [1]\n': "not a valid value",
             # click would cast these silently (2.5 -> 2, true -> 1)
             'num_modes = 2.5\n': "not a valid value", 'top_pockets = true\n': "not a valid value",
             'box = [20, true, 20]\n': "not a valid value", 'box = 0\n': "box: 0", 'thread = 10\n': "thread: 10",
             # flags take true/false (click's bool type fails on a number with an AttributeError traceback)
             'posebusters = 1\n': "not a valid value", 'force = 2.5\n': "not a valid value",
             'posebusters = "yes"\n': "not a valid value",
             # a path is a string (click.Path would os.stat() the number as a file descriptor, then crash)
             'out = 5\n': "not a valid value", 'smiles_file = 1\n': "not a valid value",
             'tmp_dir = 1.5\n': "not a valid value",
             'seed = 2026-10-28\n': "not a valid value", '[dock]\nseed = 1\n': "unknown key 'dock'",
             'proteins = "missing.txt"\n': "does not exist", 'top_pockets = \n': "not valid TOML"}
    base = run_toml.read_text()
    for extra, message in cases.items():
        key = extra.split("=")[0].strip()
        run_toml.write_text("".join(line for line in base.splitlines(True) if line.split("=")[0].strip() != key)
                            + extra)
        result = invoke("dock", "--config", run_toml)
        assert result.exit_code == cli.EXIT_USAGE and message in result.output, (extra, result.output)
    run_toml.write_text(base.replace('out = "../results"\n', ""))
    result = invoke("dock", "--config", run_toml)
    assert result.exit_code == cli.EXIT_USAGE and "missing --out" in result.output
    run_toml.write_bytes(base.encode() + b'p2rank_model = "\xff"\n')  # not UTF-8 (tomllib: UnicodeDecodeError)
    result = invoke("dock", "--config", run_toml)
    assert result.exit_code == cli.EXIT_USAGE and "not valid TOML" in result.output, result.output


def test_dock_ctrl_c_exits_with_130(tmp_path, example_dir, monkeypatch):
    """Ctrl-C: "Aborted!" and exit code 130 (not 1, a usage error), and the protein list written for a folder of
    structures is removed."""
    def run_atlas(pv, force=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "run_atlas", run_atlas)
    folder = tmp_path / "structures"
    folder.mkdir()
    shutil.copy(FIXTURE_RECEPTOR, folder)
    _, smiles = inputs(tmp_path, example_dir, ligands=LIGANDS[:1])
    result = invoke("dock", "--proteins", folder, "--ligands", smiles, "--out", tmp_path / "results",
                    "--tmp-dir", tmp_path / "tmp")
    assert result.exit_code == cli.EXIT_ABORTED and "Aborted!" in result.output, result.output
    assert os.listdir(tmp_path / "tmp") == []


def test_dock_config_file_needs_tomli_before_python_3_11(tmp_path, monkeypatch):
    run_toml = tmp_path / "run.toml"
    run_toml.write_text("seed = 1\n")
    monkeypatch.setitem(sys.modules, "tomllib", None)
    monkeypatch.setitem(sys.modules, "tomli", None)
    result = invoke("dock", "--config", run_toml)
    assert result.exit_code == cli.EXIT_USAGE and "pip install tomli" in result.output


def test_dock_random_seed(tmp_path, example_dir, fake_engine_env, stubs):
    plist, smiles = inputs(tmp_path, example_dir, ligands=LIGANDS[:1])
    ok(invoke("dock", "--proteins", plist, "--ligands", smiles, "--out", tmp_path / "r", "--top-pockets", 1,
              "--seed", "random", "--tmp-dir", tmp_path / "tmp"))
    assert manifest_params(tmp_path / "r")["seed"] is None
    assert "seed" not in fake_engine_env.calls()[0]["options"]


def test_dock_exit_codes(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    """2: some proteins failed (rerun to retry); 3: nothing docked; 1: settings the run cannot use."""
    tmp = tmp_path / "tmp"
    plist, smiles = inputs(tmp_path, example_dir, proteins=(FIXTURE_RECEPTOR, tmp_path / "missing.pdb"))
    result = invoke("dock", "--proteins", plist, "--ligands", smiles, "--out", tmp_path / "partial", "--top-pockets",
                    1, "--tmp-dir", tmp)
    assert result.exit_code == cli.EXIT_PARTIAL, result.output
    assert "proteins: 1 docked, 0 done before, 1 failed (of 2)" in result.stdout
    assert "failed: missing: FileNotFoundError" in result.stdout

    missing_only = tmp_path / "missing_only.txt"
    missing_only.write_text("missing.pdb\n")
    result = invoke("dock", "--proteins", missing_only, "--ligands", smiles, "--out", tmp_path / "none",
                    "--tmp-dir", tmp)
    assert result.exit_code == cli.EXIT_NOTHING and "0 docked" in result.stdout

    db, protein_list = make_database(tmp_path)  # one protein with pockets, one missing
    result = invoke("dock", "--database-path", db, "--protein-list", protein_list, "--ligands", smiles, "--out",
                    tmp_path / "db_results", "--tmp-dir", tmp)
    assert result.exit_code == cli.EXIT_PARTIAL and "1 docked" in result.stdout and "1 failed" in result.stdout
    assert manifest_params(tmp_path / "db_results")["pocket_source"] == "database"

    monkeypatch.setenv("POCKETVINA_ENGINE", str(tmp_path / "no_such_engine"))
    result = invoke("dock", "--proteins", plist, "--ligands", smiles, "--out", tmp_path / "no_engine",
                    "--tmp-dir", tmp)
    assert result.exit_code == cli.EXIT_NOTHING and "cannot run the PocketVina engine" in result.output
    monkeypatch.setenv("POCKETVINA_ENGINE", str(fake_engine_env))

    usage = [["--ligands", smiles, "--out", tmp_path / "x"],
             ["--proteins", plist, "--out", tmp_path / "x"],
             ["--proteins", plist, "--ligands", smiles],
             ["--proteins", plist, "--database-path", db, "--protein-list", protein_list, "--ligands", smiles,
              "--out", tmp_path / "x"],
             ["--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--num-modes", 0],
             ["--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--procs-per-gpu", "0"],
             ["--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--seed", "abc"],
             ["--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--gpus", "x"],
             ["--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--gpus", "0"],  # no GPU here
             ["--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--seed", -1],
             # the engine takes none of these: no MANIFEST.json registering them in --out (checked below)
             ["--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--box", 0, 20, 20],
             ["--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--box", 20, -3, 20],
             ["--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--box", 20, 20, "nan"],
             ["--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--box", "inf", 20, 20],
             ["--format", "files", "--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--box", 0, 1, 1],
             ["--proteins", plist, "--ligands", smiles, "--out", tmp_path / "x", "--thread", 2**31]]
    for args in usage:
        result = invoke("dock", *args, "--tmp-dir", tmp)
        assert result.exit_code == cli.EXIT_USAGE and "Error" in result.output, (args, result.output)
    assert not (tmp_path / "x").exists()
    assert os.listdir(tmp) == []


NO_PYARROW_DRIVER = """\
import sys
sys.modules["pyarrow"] = None  # import pyarrow fails
from pocketvina import cli
cli.main(sys.argv[1:])
"""


def test_dock_atlas_without_pyarrow(tmp_path, example_dir):
    """The atlas format needs pyarrow: without it, an error message and exit code 1, not a traceback."""
    _, smiles = inputs(tmp_path, example_dir, ligands=LIGANDS[:1])
    driver = tmp_path / "driver.py"
    driver.write_text(NO_PYARROW_DRIVER)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))
    run = subprocess.run([sys.executable, str(driver), "dock", "--proteins", str(FIXTURE_RECEPTOR), "--ligands",
                          str(smiles), "--out", str(tmp_path / "results"), "--tmp-dir", str(tmp_path / "tmp")],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert run.returncode == cli.EXIT_USAGE, run.stderr
    assert "Error: " in run.stderr and "pyarrow" in run.stderr and "Traceback" not in run.stderr
    assert os.listdir(tmp_path / "tmp") == []


def test_dock_nothing_docked_when_the_engine_writes_no_pose(tmp_path, example_dir, fake_engine_env, stubs,
                                                            monkeypatch):
    engine = wrapper_engine(tmp_path, fake_engine_env, "sys.exit(0)", name="silent_engine")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(engine))
    plist, smiles = inputs(tmp_path, example_dir, ligands=LIGANDS[:1])
    result = invoke("dock", "--proteins", plist, "--ligands", smiles, "--out", tmp_path / "r", "--top-pockets", 1,
                    "--tmp-dir", tmp_path / "tmp")
    assert result.exit_code == cli.EXIT_NOTHING and "highest-affinity poses: 0" in result.stdout


def test_dock_files_format(tmp_path, example_dir, fake_engine_env, stubs):
    plist, smiles = inputs(tmp_path, example_dir)
    out = tmp_path / "out"
    result = ok(invoke("dock", "--format", "files", "--proteins", plist, "--ligands", smiles, "--out", out,
                       "--top-pockets", 2, "--seed", 3, "--batch-size", 1, "--tmp-dir", tmp_path / "tmp"))
    assert "proteins with poses: 1 of 1" in result.stdout
    df = pd.read_csv(out / "combined_results.csv")
    assert set(df.protein) == {f"{PID}-pocket1", f"{PID}-pocket2"} and set(df.molecule_name) == set(LIGANDS)
    calls = fake_engine_env.calls()
    assert len(calls) == 4 and {c["options"]["seed"] for c in calls} == {3}  # 2 pockets x 2 batches of 1

    broken = wrapper_engine(tmp_path, fake_engine_env, "sys.exit(1)", name="broken_engine")
    os.environ["POCKETVINA_ENGINE"] = str(broken)  # restored by fake_engine_env's monkeypatch
    result = invoke("dock", "--format", "files", "--proteins", plist, "--ligands", smiles, "--out", out,
                    "--top-pockets", 2, "--seed", 3, "--batch-size", 1, "--tmp-dir", tmp_path / "tmp")
    assert result.exit_code == cli.EXIT_NOTHING and "(0 rows)" in result.stdout


DOCK_SIGTERM_DRIVER = """\
import os, subprocess, sys
sys.path.insert(0, {tests_dir!r})
import test_pipeline
from pocketvina import cli

test_pipeline.install_stubs()
popen = subprocess.Popen

class RecordingPopen(popen):  # tells the test the engine's pid
    def __init__(self, cmd, *a, **kw):
        super().__init__(cmd, *a, **kw)
        if cmd[0] == os.environ["POCKETVINA_ENGINE"]:
            with open({pid_file!r} + ".tmp", "w") as f:
                f.write(str(self.pid))
            os.replace({pid_file!r} + ".tmp", {pid_file!r})

subprocess.Popen = RecordingPopen
cli.main(sys.argv[1:])
"""


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX signals")
def test_dock_sigterm_removes_the_protein_list(tmp_path, example_dir, fake_engine_env, monkeypatch):
    """SIGTERM while a folder of structures is docked: the command dies of SIGTERM, its engine too, and the protein
    list it wrote for the folder is removed (the run's own SIGTERM handler used to end the process before)."""
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "60")
    folder = tmp_path / "structures"
    folder.mkdir()
    shutil.copy(FIXTURE_RECEPTOR, folder)
    _, smiles = inputs(tmp_path, example_dir, ligands=LIGANDS[:1])
    tmp, pid_file, log, driver = (tmp_path / n for n in ("tmp", "engine.pid", "driver.log", "driver.py"))
    driver.write_text(DOCK_SIGTERM_DRIVER.format(tests_dir=str(TESTS_DIR), pid_file=str(pid_file)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])))
    args = ["dock", "--proteins", folder, "--ligands", smiles, "--out", tmp_path / "results", "--top-pockets", 1,
            "--tmp-dir", tmp]
    engine_pid = None
    with open(log, "wb") as stderr:
        proc = subprocess.Popen([sys.executable, str(driver)] + [str(a) for a in args], env=env, cwd=str(tmp_path),
                                stdout=subprocess.DEVNULL, stderr=stderr)
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
    assert os.listdir(tmp) == []


# --- commands on a results directory -------------------------------------------------------------------------------

def test_commands_on_a_docked_run(tmp_path, example_dir, fake_engine_env, stubs):
    """dock, then info, query, query best, export and show on what it wrote (fake engine poses)."""
    plist, smiles = inputs(tmp_path, example_dir)
    out = tmp_path / "results"
    ok(invoke("-q", "dock", "--proteins", plist, "--ligands", smiles, "--out", out, "--top-pockets", 2,
              "--tmp-dir", tmp_path / "tmp"))
    data = json.loads(ok(invoke("info", "--results", out, "--json", "--proteins", plist)).stdout)
    assert data["proteins"] == 1 and data["pocket_runs"] == dict(total=4, by_status={"ok": 4})
    assert data["pairs"] == dict(total=2, with_pose=2) and data["not_done"] == [] and data["params"]["seed"] == 42
    best = Results(out).pairs().sort_values("vina")
    lines = ok(invoke("query", "best", "--results", out, "--top", 5)).stdout.splitlines()
    assert [line.split()[1] for line in lines[1:]] == list(best.ligand_id)
    result = ok(invoke("query", "--results", out, "--csv", "-", "SELECT pocket, count(*) AS n FROM pocket_runs "
                                                               "GROUP BY pocket ORDER BY pocket"))
    assert result.stdout == "pocket,n\n1,2\n2,2\n"
    assert ok(invoke("export", "--results", out, "--format", "pdbqt", "--what", "all", "--out", tmp_path / "e")
              ).stdout.startswith("4 files written")
    if HAS_PY3DMOL:
        page = tmp_path / "page.html"
        ok(invoke("show", "--results", out, "--protein", PID, "--ligand", "molecule1", "--pocket", 1, "--out", page,
                  "--backend", "py3dmol"))
        assert page.is_file()


def test_import_legacy(tmp_path, real_run, imported):
    out = tmp_path / "results"
    result = ok(invoke("import-legacy", real_run.root, "--results", out, "--smiles", real_run.smiles_csv,
                       "--tmp-dir", tmp_path / "tmp"))
    assert result.stdout.strip() == f"{out}: 1 proteins"
    pd.testing.assert_frame_equal(Results(out).pairs(), Results(imported).pairs())
    result = invoke("import-legacy", tmp_path / "tmp", "--results", tmp_path / "other")
    assert result.exit_code == cli.EXIT_USAGE and "not a legacy PocketVina output_dir" in result.output


def test_import_legacy_unwinds_on_sigterm(tmp_path, real_run, imported, monkeypatch):
    """The import runs under engine.exit_on_sigterm: SIGTERM unwinds it, so its temp folder is removed first."""
    if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
        pytest.skip("SIGTERM has a handler here (exit_on_sigterm leaves it alone)")
    from pocketvina.io import importer

    handlers = []

    def fake_import(output_dir, results, smiles=None, **kw):
        handlers.append(signal.getsignal(signal.SIGTERM))
        return imported

    monkeypatch.setattr(importer, "import_legacy", fake_import)
    ok(invoke("import-legacy", real_run.root, "--results", tmp_path / "results"))
    assert handlers and handlers[0] is not signal.SIG_DFL
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_info(tmp_path, imported):
    result = ok(invoke("info", "--results", imported))
    text = result.stdout
    for needle in (f"results: {imported}", "format: pocketvina-atlas 1", "params:", "ligands: 3 (", "proteins: 1 done",
                   "highest-affinity poses: 2 of 3"):
        assert needle in text, needle
    data = json.loads(ok(invoke("info", "--results", imported, "--json")).stdout)
    assert data["proteins"] == 1 and data["ligands"]["total"] == 3 and data["pairs"] == dict(total=3, with_pose=2)
    assert data["pocket_runs"]["total"] == 6 and data["problems"] == {} and data["incomplete"] == []
    assert "not_done" not in data

    copy = copy_results(imported, tmp_path / "copy")
    (Path(copy) / "poses" / "ghost.poses").write_bytes(b"")
    plist = tmp_path / "proteins.txt"
    plist.write_text(f"{PID}.pdbqt\nother.pdb\n")
    data = json.loads(ok(invoke("info", "--results", copy, "--json", "--proteins", plist)).stdout)
    assert data["incomplete"] == ["ghost"] and data["not_done"] == ["other"]
    text = ok(invoke("info", "--results", copy, "--proteins", plist)).stdout
    assert "incomplete (1" in text and "not done (1 of the --proteins list" in text and "  other" in text
    result = invoke("info", "--results", tmp_path)
    assert result.exit_code == cli.EXIT_USAGE and "not a PocketVina results directory" in result.output


def test_query(tmp_path, imported):
    result = ok(invoke("query", "--results", imported, "SELECT ligand_id, vina FROM pairs ORDER BY ligand_idx"))
    lines = result.stdout.splitlines()
    assert lines[0].split() == ["ligand_id", "vina"] and [line.split()[0] for line in lines[1:]] == [
        "molecule2", "molecule5", "molecule10"]
    out = tmp_path / "q" / "modes.csv"
    result = ok(invoke("query", "--results", imported, "--csv", out, "SELECT * FROM modes WHERE pocket = 1"))
    assert f"18 rows written to {out}" in result.stdout and len(pd.read_csv(out)) == 18
    result = ok(invoke("query", "--results", imported, "--csv", "-", "SELECT count(*) AS n FROM pocket_runs"))
    assert result.stdout == "n\n6\n"
    result = ok(invoke("query", "--results", imported, "SELECT * FROM pairs WHERE vina > 100"))
    assert "(0 rows)" in result.stdout

    result = ok(invoke("query", "best", "--results", imported, "--top", 1))
    lines = result.stdout.splitlines()
    assert len(lines) == 2 and "ligand_efficiency" in lines[0]
    best = Results(imported).pairs().sort_values("vina").iloc[0]
    assert lines[1].split()[1] == best.ligand_id
    result = ok(invoke("query", "best", "--results", imported, "--protein", PID, "--csv", "-"))
    assert len(pd.read_csv(pd.io.common.StringIO(result.stdout))) == 2

    for args, message in ((["SELECT * FROM nope"], "query failed"), (["--top", 3, "SELECT 1"], "--top applies"),
                          (["best", "--protein", "nope"], "nope")):
        result = invoke("query", "--results", imported, *args)
        assert result.exit_code == cli.EXIT_USAGE and message in result.output, (args, result.output)


@needs_py3dmol
def test_show(tmp_path, imported, monkeypatch):
    out = tmp_path / "views" / "page.html"
    result = ok(invoke("show", "--results", imported, "--protein", PID, "--ligand", "molecule2", "--out", out,
                       "--backend", "py3dmol"))
    assert result.stdout.strip() == str(out)
    assert "<html" in out.read_text().lower()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # --out ~/... (not expanded by a shell): the path printed
    result = ok(invoke("show", "--results", imported, "--protein", PID, "--ligand", "molecule2", "--out",
                       "~/page.html", "--backend", "py3dmol"))
    assert result.stdout.strip() == str(tmp_path / "home" / "page.html")
    assert (tmp_path / "home" / "page.html").is_file()
    monkeypatch.chdir(tmp_path)
    result = ok(invoke("show", "--results", imported, "--protein", PID, "--ligand", "1", "--pocket", 2, "--mode", 3,
                       "--backend", "py3dmol"))
    assert result.stdout.strip() == str(tmp_path / f"{PID}-molecule5-pocket2-mode3.html")
    assert (tmp_path / f"{PID}-molecule5-pocket2-mode3.html").is_file()
    for args, message in ((["--protein", "nope", "--ligand", "molecule2"], "nope"),
                          (["--protein", PID, "--ligand", "nope"], "no ligand 'nope'"),
                          (["--protein", PID, "--ligand", "molecule10"], "molecule10"),
                          (["--protein", PID, "--ligand", "molecule2", "--mode", 2], "--mode needs --pocket")):
        result = invoke("show", "--results", imported, "--backend", "py3dmol", *args)
        assert result.exit_code == cli.EXIT_USAGE and message in result.output, (args, result.output)


def test_export(tmp_path, imported, monkeypatch):
    out = tmp_path / "sdf"
    result = ok(invoke("export", "--results", imported, "--what", "best", "--format", "sdf", "--out", out))
    assert result.stdout.strip() == f"2 files written to {out}"
    monkeypatch.setenv("HOME", str(tmp_path / "home"))  # --out ~/... (not expanded by a shell): the path printed
    result = ok(invoke("export", "--results", imported, "--out", "~/poses"))
    assert result.stdout.strip() == f"2 files written to {tmp_path / 'home' / 'poses'}"
    assert len(list((tmp_path / "home" / "poses" / PID).iterdir())) == 2
    pairs = Results(imported).pairs()
    expected = sorted(f"{PID}-pocket{k}-{lig}_best.sdf" for lig, k in zip(pairs.ligand_id, pairs.best_pocket) if k > 0)
    assert sorted(p.name for p in (out / PID).iterdir()) == expected and len(expected) == 2
    result = ok(invoke("export", "--results", imported, "--what", "all", "--format", "csv", "--out", tmp_path / "csv",
                       "--protein", PID))
    assert result.stdout.startswith("1 files written")
    assert len(pd.read_csv(tmp_path / "csv" / "combined_results.csv")) == 36
    result = invoke("export", "--results", imported, "--out", out, "--protein", "nope")
    assert result.exit_code == cli.EXIT_USAGE and "nope" in result.output


def test_bust(tmp_path, imported):
    pytest.importorskip("posebusters")
    results = copy_results(imported, tmp_path / "results")
    result = ok(invoke("bust", "--results", results, "--workers", 1, "--timeouts", "60,300",
                       "--tmp-dir", tmp_path / "tmp"))
    assert "PoseBusters (dock, best): 1 proteins checked, 0 had results already" in result.stdout
    pb = Results(results).posebusters()
    assert len(pb) == 3 and "pb_status: " in result.stdout
    assert set(pb.pb_status) <= {"valid", "invalid", "no_pose"} and (pb.pb_status == "no_pose").sum() == 1
    result = ok(invoke("bust", "--results", results, "--workers", 1, "--protein", PID))
    assert "0 proteins checked, 1 had results already" in result.stdout
    assert os.listdir(tmp_path / "tmp") == []
    for args in (["--timeouts", "abc"], ["--timeouts", "0"], ["--protein", "nope"], ["--config", "nope"]):
        result = invoke("bust", "--results", results, "--workers", 1, *args)
        assert result.exit_code == cli.EXIT_USAGE, (args, result.output)


def test_bust_options_reach_bust_results(tmp_path, imported, monkeypatch):
    from pocketvina import bust as pb

    seen = []
    monkeypatch.setattr(pb, "require_posebusters", lambda: None)
    monkeypatch.setattr(pb, "bust_results", lambda results, **kw: seen.append((results, kw)) or [])
    result = ok(invoke("bust", "--results", imported, "--select", "per_pocket", "--top-k", 3, "--config", "dock_fast",
                       "--workers", 2, "--timeouts", "none", "--overwrite", "--protein", PID,
                       "--tmp-dir", tmp_path / "t"))
    assert seen == [(str(imported), dict(select="per_pocket", top_k=3, config="dock_fast", workers=2, timeouts=None,
                                         proteins=[PID], overwrite=True, tmp_dir=str(tmp_path / "t")))]
    assert "0 proteins checked, 1 had results already" in result.stdout and "pb_status: no rows" in result.stdout
    ok(invoke("bust", "--results", imported, "--timeouts", "20, 200"))
    assert seen[-1][1] | dict(timeouts=None) == dict(select="best", top_k=1, config="dock", workers=None,
                                                     timeouts=None, proteins=None, overwrite=False, tmp_dir=None)
    assert seen[-1][1]["timeouts"] == (20.0, 200.0)


def test_bust_without_posebusters(tmp_path, imported, monkeypatch):
    monkeypatch.setitem(sys.modules, "posebusters", None)
    result = invoke("bust", "--results", imported)
    assert result.exit_code == cli.EXIT_USAGE and "pip install posebusters" in result.output


# --- bench gpu -----------------------------------------------------------------------------------------------------

def bench_inputs(tmp_path, example_dir):
    _, smiles = inputs(tmp_path, example_dir)
    return smiles


def pocket1_center():
    row = pd.read_csv(POCKETS_CSV, skipinitialspace=True).iloc[0]
    return [row["center_x"], row["center_y"], row["center_z"]]


def overlap(a, b) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def test_bench_gpu(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    """Each of the N processes docks the same ligands into the pocket at the same time; a warm-up call first. No
    timing margins: the 2 calls of the 2-process run end only once both have started
    (``POCKETVINA_FAKE_ENGINE_WAIT_FOR``, set for them by a wrapper engine), so they overlap; run one after the other,
    they would not (the first would end only after the 30 s fallback of the thread that writes "go", counted from
    the first call's start, so a slow run before it does not use it up)."""
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "0.5")
    gate = tmp_path / "engine_gate"
    gate.mkdir()
    engine = wrapper_engine(tmp_path, fake_engine_env, f"""if "procs2_" in env["POCKETVINA_ENGINE_SLOT"]:
    env["POCKETVINA_FAKE_ENGINE_WAIT_FOR"] = {str(gate)!r}""")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(engine))

    def release():  # "go" once both calls of the 2-process run have started (30 s after the first at most)
        while not list(gate.glob("started-*")):
            if done.wait(0.05):  # the benchmark ended before that run
                return
        deadline = time.monotonic() + 30
        while len(list(gate.glob("started-*"))) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        (gate / "go").touch()

    done = threading.Event()
    releaser = threading.Thread(target=release, daemon=True)
    releaser.start()
    smiles = bench_inputs(tmp_path, example_dir)
    out, tmp = tmp_path / "bench", tmp_path / "tmp"
    try:
        result = ok(invoke("bench", "gpu", "--protein", FIXTURE_RECEPTOR, "--ligands", smiles, "--n-ligands", 3,
                           "--procs", "1,2", "--pocket", 2, "--out", out, "--seed", 11, "--thread", 1000,
                           "--tmp-dir", tmp))
    finally:
        done.set()
        releaser.join()
    assert "GPU: default (no nvidia-smi)" in result.stdout and "ligands_per_hour" in result.stdout
    lines = result.stdout.splitlines()
    head = next(n for n, line in enumerate(lines) if line.split()[:1] == ["procs"])
    table = [dict(zip(lines[head].split(), line.split())) for line in lines[head + 1:head + 3]]
    assert [r["procs"] for r in table] == ["1", "2"] and "None" not in result.stdout  # not sampled: "-"
    assert {r[c] for r in table for c in ("peak_vram_mib", "gpu_util_mean", "gpu_mem_used_max_mib")} == {"-"}
    df = pd.read_csv(out / cli.BENCH_CSV)
    assert list(df.columns) == list(cli.BENCH_COLUMNS)
    assert list(df.procs) == [1, 2] and list(df.ligands_docked) == [3, 6] and list(df.failed_calls) == [0, 0]
    assert df.speedup[0] == 1.0 and df.slowdown[0] == 1.0 and df.peak_vram_mib.isna().all()
    assert (df.wall_s > 0.5).all() and (df.ligands_per_hour > 0).all()
    report = json.loads((out / cli.BENCH_JSON).read_text())
    assert report["gpu"] is None and report["n_ligands"] == 3 and report["pocket"] == 2
    assert report["warmup"]["ok"] and report["warmup"]["ligands"] == 1
    assert [c["procs"] for c in report["calls"]] == [1, 2, 2] and all(c["docked"] == 3 for c in report["calls"])

    calls = fake_engine_env.calls()
    assert len(calls) == 4
    center = [round(v, 4) for v in pocket1_center()]
    stub_center = [calls[0]["options"][f"center_{a}"] for a in "xyz"]
    assert stub_center != center  # pocket 2 of the predictions
    for c in calls:
        o = c["options"]
        assert (o["seed"], o["thread"], o["pocket_rank"], o["size_x"]) == (11, 1000, 2, 20.0)
        assert os.path.basename(o["receptor"]) == f"{PID}.pdbqt"
        assert c["env"]["POCKETVINA_ENGINE_SLOT"].startswith("bench")
    assert [len(c["outputs"]) for c in calls] == [1, 3, 3, 3]
    # both calls of the 2-process run got "go" (none gave up waiting); the overlap check shows that both had started
    # before either ended ("go" also comes after the releaser's 30 s fallback, so it alone does not)
    assert [c.get("go") for c in calls] == [None, None, True, True]
    assert overlap(calls[2], calls[3]) and not overlap(calls[1], calls[2])
    assert os.listdir(tmp) == []


def test_bench_gpu_options_reach_the_engine(tmp_path, example_dir, fake_engine_env, stubs, monkeypatch):
    """--box, --ligand-prep, --seed and --p2rank-model reach the engine config, the ligand preparation and P2Rank."""
    models, preps = [], []
    stub = pipeline.predict_pockets

    def predict_pockets(structure, work_dir, p2rank_config=None, **kw):
        models.append(p2rank_config.model)
        return stub(structure, work_dir, p2rank_config, **kw)

    real_prepare = pv_ligands.prepare_ligands

    def prepare_ligands(rows, **kw):
        preps.append(kw)
        return real_prepare(rows, **dict(kw, method="standard"))  # legacy needs the obabel executable

    monkeypatch.setattr(pipeline, "predict_pockets", predict_pockets)
    monkeypatch.setattr(pv_ligands, "prepare_ligands", prepare_ligands)
    smiles = bench_inputs(tmp_path, example_dir)
    out = tmp_path / "bench"
    ok(invoke("bench", "gpu", "--protein", FIXTURE_RECEPTOR, "--ligands", smiles, "--n-ligands", 1, "--procs", 1,
              "--box", 18, 19, 21.5, "--ligand-prep", "legacy", "--seed", 5, "--p2rank-model", "default",
              "--out", out, "--tmp-dir", tmp_path / "tmp"))
    assert models == ["default"] and preps == [dict(method="legacy", seed=5)]
    calls = fake_engine_env.calls()
    assert len(calls) == 2
    for c in calls:
        o = c["options"]
        assert (o["size_x"], o["size_y"], o["size_z"], o["seed"]) == (18.0, 19.0, 21.5, 5)
    report = json.loads((out / cli.BENCH_JSON).read_text())
    assert (report["box"], report["ligand_prep"], report["seed"]) == ([18.0, 19.0, 21.5], "legacy", 5)


def test_bench_gpu_split_and_center(tmp_path, example_dir, fake_engine_env, monkeypatch):
    smiles = bench_inputs(tmp_path, example_dir)
    out = tmp_path / "bench"
    monkeypatch.setenv("HOME", str(tmp_path))  # --out ~/bench (not expanded by a shell): the path printed is where
    result = ok(invoke("bench", "gpu", "--protein", FIXTURE_RECEPTOR, "--ligands", smiles, "--n-ligands", 3,
                       "--procs", "2,1", "--split", "--center", *pocket1_center(), "--out", "~/bench",
                       "--tmp-dir", tmp_path / "tmp"))
    assert f"written: {out / cli.BENCH_CSV}, {out / cli.BENCH_JSON}" in result.stdout
    df = pd.read_csv(out / cli.BENCH_CSV)
    assert list(df.procs) == [2, 1] and list(df.ligands_docked) == [3, 3]
    assert df.speedup[1] == 1.0 and df.slowdown[1] == 1.0  # relative to the fewest processes
    calls = fake_engine_env.calls()
    assert [len(c["outputs"]) for c in (calls[0], calls[3])] == [1, 3]  # warm-up, then 1 process
    assert sorted(len(c["outputs"]) for c in calls[1:3]) == [1, 2]  # 2 processes share 3 ligands
    assert [calls[0]["options"][f"center_{a}"] for a in "xyz"] == pytest.approx(pocket1_center())


def fake_nvidia_smi(tmp_path, uuid, gate) -> Path:
    """An ``nvidia-smi`` reporting GPU ``uuid`` at 37 % and every process of this test (``$POCKETVINA_FAKE_ENGINE_LOG``
    in its environment, so not the engines of other test sessions) with POCKETVINA_ENGINE_SLOT in its environment as
    a compute app using 1234 MiB on it. ``gate`` holds a folder ``procs<n>`` per configuration of the benchmark: once
    a sample has listed all its engines (``procs<n>_1`` to ``procs<n>_<n>``), the next run (the benchmark has read
    that sample) writes ``go`` into it."""
    bin_dir = tmp_path / "smi_bin"
    bin_dir.mkdir()
    script = bin_dir / "fake_smi.py"
    script.write_text(f"""import os, sys
args = " ".join(sys.argv[1:])
gate = {str(gate)!r}
for config in os.listdir(gate):
    if os.path.exists(os.path.join(gate, config, "seen")):
        open(os.path.join(gate, config, "go"), "w").close()
if "--query-gpu=uuid,utilization.gpu,memory.used" in args:
    print("{uuid}, 37, 5000")
elif "--query-compute-apps=gpu_uuid,pid,used_memory" in args:
    test = ("POCKETVINA_FAKE_ENGINE_LOG=" + os.environ["POCKETVINA_FAKE_ENGINE_LOG"]).encode()
    tags = set()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            env = open(f"/proc/{{pid}}/environ", "rb").read().split(b"\\0")
        except OSError:
            continue
        slot = [v.decode() for v in env if v.startswith(b"POCKETVINA_ENGINE_SLOT=")]
        if slot and test in env:
            print(f"{uuid}, {{pid}}, 1234")
            tags.add(slot[0].rsplit("-", 1)[-1])
    print("GPU-someone-else, 1, 999")
    for config in os.listdir(gate):
        if all(f"{{config}}_{{p}}" in tags for p in range(1, int(config[len("procs"):]) + 1)):
            open(os.path.join(gate, config, "seen"), "w").close()
else:
    sys.exit(1)
""")
    exe = bin_dir / "nvidia-smi"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    exe.chmod(0o755)
    if not os.access(exe, os.X_OK):
        pytest.skip(f"cannot execute {exe} (noexec temp dir?)")
    return bin_dir


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="the benchmark finds its engines in /proc")
def test_bench_gpu_samples_nvidia_smi(tmp_path, example_dir, fake_engine_env, monkeypatch):
    """nvidia-smi is sampled while each configuration runs: the GPU's use and the memory of the benchmark's own
    engines (not of another process with POCKETVINA_ENGINE_SLOT on the GPU). No timing margins: the engines of a
    configuration end only once a sample has seen them all (``POCKETVINA_FAKE_ENGINE_WAIT_FOR``, see
    ``fake_nvidia_smi``); without such a sample they give up waiting after 30 s and the test fails."""
    g = fake_gpus(2)[1]
    monkeypatch.setattr(gpu, "list_gpus", lambda: fake_gpus(2))
    gate = tmp_path / "engine_gate"
    for config in ("procs1", "procs2"):
        (gate / config).mkdir(parents=True)
    monkeypatch.setenv("PATH", f"{fake_nvidia_smi(tmp_path, g.uuid, gate)}{os.pathsep}{os.environ['PATH']}")
    # the calls of a configuration wait in its gate folder; exec: one process per engine, as nvidia-smi lists them
    engine = wrapper_engine(tmp_path, fake_engine_env, f"""tag = env["POCKETVINA_ENGINE_SLOT"].rsplit("-", 1)[1]
if tag.startswith("procs"):
    env["POCKETVINA_FAKE_ENGINE_WAIT_FOR"] = os.path.join({str(gate)!r}, tag.split("_")[0])
os.execv({str(fake_engine_env)!r}, [{str(fake_engine_env)!r}] + argv)""")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(engine))
    # an engine of another benchmark on the same GPU: listed by nvidia-smi, not counted
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                             env=dict(os.environ, POCKETVINA_ENGINE_SLOT="bench1-0123abcd-other"))
    try:
        smiles = bench_inputs(tmp_path, example_dir)
        out = tmp_path / "bench"
        result = ok(invoke("bench", "gpu", "--protein", FIXTURE_RECEPTOR, "--ligands", smiles, "--n-ligands", 2,
                           "--procs", "1,2", "--center", 0, 0, 0, "--gpu", 1, "--sample-interval", 0.1, "--out", out,
                           "--tmp-dir", tmp_path / "tmp"))
    finally:
        other.kill()
        other.wait()
    assert f"GPU: 1 Tesla P40 ({g.uuid})" in result.stdout
    report = json.loads((out / cli.BENCH_JSON).read_text())
    assert report["gpu"]["uuid"] == g.uuid
    rows = report["configurations"]
    assert [r["peak_vram_mib"] for r in rows] == [1234, 1234] and [r["procs_seen"] for r in rows] == [1, 2]
    assert [r["gpu_util_mean"] for r in rows] == [37, 37] and [r["gpu_mem_used_max_mib"] for r in rows] == [5000] * 2
    calls = fake_engine_env.calls()
    assert {c["env"]["CUDA_VISIBLE_DEVICES"] for c in calls} == {g.uuid}
    assert [c.get("go") for c in calls] == [None, True, True, True]  # the warm-up, then the configurations' calls


def test_bench_gpu_failures(tmp_path, example_dir, fake_engine_env, monkeypatch):
    """A call that runs out of GPU memory: exit code 2 and the call in the table; no engine call works: 3."""
    smiles = bench_inputs(tmp_path, example_dir)
    args = ["bench", "gpu", "--protein", FIXTURE_RECEPTOR, "--ligands", smiles, "--n-ligands", 2, "--center", 0, 0, 0,
            "--tmp-dir", tmp_path / "tmp"]
    engine = wrapper_engine(tmp_path, fake_engine_env, f"""if "procs2_2" in argv[1]:
    print({OOM_LINE!r})
    sys.exit(1)""", name="oom_engine")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(engine))
    result = invoke(*args, "--procs", "1,2", "--out", tmp_path / "oom")
    assert result.exit_code == cli.EXIT_PARTIAL, result.output
    assert f"failed: procs2_2: {OOM_LINE}" in result.output
    rows = json.loads((tmp_path / "oom" / cli.BENCH_JSON).read_text())["configurations"]
    assert [(r["failed_calls"], r["out_of_memory_calls"], r["ligands_docked"]) for r in rows] == [(0, 0, 2), (1, 1, 2)]

    broken = wrapper_engine(tmp_path, fake_engine_env, "print('no OpenCL here')\nsys.exit(1)", name="broken_engine")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(broken))
    result = invoke(*args, "--out", tmp_path / "broken")
    assert result.exit_code == cli.EXIT_NOTHING and "warm-up engine call failed: no OpenCL here" in result.output
    assert not (tmp_path / "broken").exists() and os.listdir(tmp_path / "tmp") == []
    monkeypatch.setenv("POCKETVINA_ENGINE", str(tmp_path / "no_such_engine"))
    result = invoke(*args, "--out", tmp_path / "missing")
    assert result.exit_code == cli.EXIT_NOTHING and "cannot run the PocketVina engine" in result.output


def test_bench_gpu_warmup_skips_ligands_the_engine_rejects(tmp_path, example_dir, fake_engine_env, monkeypatch):
    """The engine builds its kernels only after it has parsed a ligand ("No valid ligands" exits 1 before): a first
    ligand it rejects is followed by the next one, not reported as a failed warm-up; all rejected: exit code 3."""
    real_prepare = pv_ligands.prepare_ligands
    bad = set()

    def without_torsdof(rows, **kw):
        records = real_prepare(rows, **kw)
        for rec in records:
            if rec.ligand_id in bad:
                rec.pdbqt = "".join(line for line in rec.pdbqt.splitlines(True) if "TORSDOF" not in line)
        return records

    monkeypatch.setattr(pv_ligands, "prepare_ligands", without_torsdof)
    smiles = bench_inputs(tmp_path, example_dir)
    args = ["bench", "gpu", "--protein", FIXTURE_RECEPTOR, "--ligands", smiles, "--n-ligands", 2, "--procs", 1,
            "--center", 0, 0, 0, "--tmp-dir", tmp_path / "tmp"]
    bad.add(LIGANDS[0])
    ok(invoke(*args, "--out", tmp_path / "first"))
    calls = fake_engine_env.calls()
    assert [(c["returncode"], len(c["outputs"])) for c in calls] == [(1, 0), (0, 1), (0, 1)]  # rejected, next, run
    report = json.loads((tmp_path / "first" / cli.BENCH_JSON).read_text())
    assert report["warmup"]["docked"] == 1 and report["configurations"][0]["ligands_docked"] == 1

    bad.add(LIGANDS[1])
    result = invoke(*args, "--out", tmp_path / "all")
    assert result.exit_code == cli.EXIT_NOTHING, result.output
    assert "the engine rejected the first 2 prepared ligands" in result.output
    assert len(fake_engine_env.calls()) == 5 and os.listdir(tmp_path / "tmp") == []


def test_bench_gpu_nothing_docked(tmp_path, example_dir, fake_engine_env, monkeypatch):
    """Engine calls that exit with code 0 but write no pose: exit code 3, at the warm-up or after the run."""
    smiles = bench_inputs(tmp_path, example_dir)
    args = ["bench", "gpu", "--protein", FIXTURE_RECEPTOR, "--ligands", smiles, "--n-ligands", 2, "--center", 0, 0, 0,
            "--tmp-dir", tmp_path / "tmp"]
    silent = wrapper_engine(tmp_path, fake_engine_env, "sys.exit(0)", name="silent_engine")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(silent))
    result = invoke(*args, "--out", tmp_path / "silent")
    assert result.exit_code == cli.EXIT_NOTHING and "warm-up engine call docked nothing" in result.output
    no_pose = wrapper_engine(tmp_path, fake_engine_env, 'if "procs" in argv[1]:\n    sys.exit(0)', name="no_pose")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(no_pose))
    result = invoke(*args, "--procs", "1,2", "--out", tmp_path / "bench")
    assert result.exit_code == cli.EXIT_NOTHING and "no engine call docked a ligand" in result.output
    rows = json.loads((tmp_path / "bench" / cli.BENCH_JSON).read_text())["configurations"]
    assert [(r["ligands_docked"], r["failed_calls"]) for r in rows] == [(0, 0), (0, 0)]


def test_bench_gpu_timed_out_call_docks_nothing(tmp_path, example_dir, fake_engine_env, monkeypatch):
    """A call killed after --engine-timeout docked nothing, whatever outputs it had written (they may be cut short;
    as in pipeline._run_call)."""
    hang = wrapper_engine(tmp_path, fake_engine_env, f"""if "procs" in argv[1]:
    subprocess.call([{str(fake_engine_env)!r}] + argv)  # writes the poses, then hangs
    import time
    time.sleep(60)""", name="hang_engine")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(hang))
    smiles = bench_inputs(tmp_path, example_dir)
    result = invoke("bench", "gpu", "--protein", FIXTURE_RECEPTOR, "--ligands", smiles, "--n-ligands", 2, "--procs", 1,
                    "--center", 0, 0, 0, "--engine-timeout", 3, "--out", tmp_path / "b", "--tmp-dir", tmp_path / "tmp")
    assert result.exit_code == cli.EXIT_NOTHING and "no engine call docked a ligand" in result.output, result.output
    call = json.loads((tmp_path / "b" / cli.BENCH_JSON).read_text())["calls"][0]
    assert call["timed_out"] and call["docked"] == 0 and not call["ok"]
    assert fake_engine_env.calls()[-1]["outputs"] and os.listdir(tmp_path / "tmp") == []


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="the benchmark finds its engines in /proc")
def test_bench_gpu_stop_kills_engines_that_start_late(tmp_path, example_dir, fake_engine_env, monkeypatch):
    """A configuration stopped (Ctrl-C, SIGTERM, here a call that raises) while one of its threads has not started
    its engine yet: that engine, started after the first kill, is killed too instead of being waited for."""
    slow = wrapper_engine(tmp_path, fake_engine_env, """if "procs" in argv[1]:
    os.environ["POCKETVINA_FAKE_ENGINE_SLEEP"] = "60"
""", name="slow_engine")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(slow))
    real_write_config, real_run_engine = pv_engine.write_config, pv_engine.run_engine
    second_running = threading.Event()
    late = []  # the second call's engine run: started after the stop (not a file the engine writes: it may be
    #            killed before its interpreter gets that far)

    def run_engine(config, *args, **kw):
        result = real_run_engine(config, *args, **kw)
        if os.path.basename(config) == "procs2_2.txt":
            late.append(result)
        return result

    def write_config(path, *args, **kw):
        name = os.path.basename(path)
        if name == "procs2_1.txt":
            # only once the second call runs: a call still queued is cancelled and never starts its engine (on a
            # slow machine the stop could come before the second thread takes its call)
            assert second_running.wait(30)
            raise RuntimeError("stopped")
        if name == "procs2_2.txt":
            second_running.set()
            time.sleep(1.0)  # its engine starts after the stop
        return real_write_config(path, *args, **kw)

    monkeypatch.setattr(pv_engine, "write_config", write_config)
    monkeypatch.setattr(pv_engine, "run_engine", run_engine)
    smiles = bench_inputs(tmp_path, example_dir)
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="stopped"):
        invoke("bench", "gpu", "--protein", FIXTURE_RECEPTOR, "--ligands", smiles, "--n-ligands", 1, "--procs", 2,
               "--center", 0, 0, 0, "--out", tmp_path / "b", "--tmp-dir", tmp_path / "tmp")
    assert time.monotonic() - t0 < 30  # the late engine sleeps 60 s: it was killed, not waited for
    assert len(late) == 1 and not late[0].ok and not late[0].timed_out, late
    assert os.listdir(tmp_path / "tmp") == [] and not (tmp_path / "b").exists()


def test_bench_gpu_usage_errors(tmp_path, example_dir, fake_engine_env, stubs):
    smiles = bench_inputs(tmp_path, example_dir)
    base = ["bench", "gpu", "--protein", FIXTURE_RECEPTOR, "--ligands", smiles, "--out", tmp_path / "b",
            "--n-ligands", 2, "--tmp-dir", tmp_path / "tmp"]
    for extra, message in ((["--procs", "0"], "integers >= 1"), (["--procs", "a"], "process counts"),
                           (["--procs", "3", "--split"], "at least as many ligands"),
                           (["--gpu", "0"], "nvidia-smi lists no GPU"), (["--pocket", 99], "no pocket 99"),
                           (["--thread", 10], "--thread"), (["--box", 20, 0, 20], "--box"),
                           (["--box", "nan", 20, 20], "--box"),
                           # nan/inf pass click's ranges: a sampler busy loop or crash, a late engine error
                           (["--sample-interval", "inf"], "--sample-interval"),
                           (["--sample-interval", "nan"], "--sample-interval"),
                           (["--engine-timeout", "inf"], "--engine-timeout"),
                           (["--center", "nan", 0, 0], "--center")):
        result = invoke(*base, *extra)
        assert result.exit_code == cli.EXIT_USAGE and message in result.output, (extra, result.output)
    bad = tmp_path / "bad.csv"
    bad.write_text("molecule_name,smiles\nx,not_a_smiles\n")
    result = invoke(*base[:5], bad, *base[6:])
    assert result.exit_code == cli.EXIT_NOTHING and "none of the 1 ligands could be prepared" in result.output
    assert fake_engine_env.calls() == [] and os.listdir(tmp_path / "tmp") == []


def test_bench_gpu_out_must_be_writable_before_it_runs(tmp_path, example_dir, fake_engine_env):
    """An --out it cannot write: exit code 1 before the benchmark, not after it (its results would be lost)."""
    smiles = bench_inputs(tmp_path, example_dir)
    base = ["bench", "gpu", "--protein", FIXTURE_RECEPTOR, "--ligands", smiles, "--n-ligands", 1, "--procs", 1,
            "--center", 0, 0, 0, "--tmp-dir", tmp_path / "tmp"]
    a_file = tmp_path / "a_file"
    a_file.write_text("")
    result = invoke(*base, "--out", a_file / "bench")
    assert result.exit_code == cli.EXIT_USAGE and f"{a_file} is not a folder" in result.output, result.output
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o555)
    try:
        if os.access(read_only, os.W_OK):
            pytest.skip("root (or an ACL) may write into a read-only folder")
        result = invoke(*base, "--out", read_only / "new" / "bench")
        assert result.exit_code == cli.EXIT_USAGE and f"cannot write into {read_only}" in result.output, \
            result.output
    finally:
        read_only.chmod(0o755)
    assert fake_engine_env.calls() == [] and not (tmp_path / "tmp").exists()  # checked before any work
