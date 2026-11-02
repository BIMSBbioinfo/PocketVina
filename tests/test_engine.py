"""pocketvina.engine: config files, engine environment, kernel cache folder and output collection.

Everything except the ``test_real_engine_*`` tests runs the fake engine (tests/fake_engine.py); those need a GPU and
``POCKETVINA_GPU_TESTS=1``.
"""
import hashlib
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time

import numpy as np
import pytest

import fake_engine as fe
from pocketvina import engine
from pocketvina.engine import EngineResult, collect_outputs, kernel_cache_dir, run_engine, write_config

STEMS = ("L0000000", "L0000001")  # ligands.write_ligand_dir names: molecule2, molecule5


@pytest.fixture(autouse=True)
def _fresh_engine_caches():
    engine._gpu_identity.cache_clear()
    engine._file_sha256_cached.cache_clear()
    yield
    engine._gpu_identity.cache_clear()


def _skip_unless_executable(path):
    # tmp_path on a noexec mount (a hardened /tmp on some clusters), see conftest.fake_engine_env
    if not os.access(path, os.X_OK):
        pytest.skip(f"cannot execute {path} (noexec temp dir?); set --basetemp elsewhere")


@pytest.fixture
def fake_smi(tmp_path, monkeypatch):
    """Put a fake ``nvidia-smi`` first on PATH; call it with the CSV rows to print (None: exit 1)."""
    bin_dir = tmp_path / "smi_bin"
    bin_dir.mkdir()
    exe = bin_dir / "nvidia-smi"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    def set_rows(rows):
        body = "exit 1\n" if rows is None else "cat <<'EOF'\n" + "".join(r + "\n" for r in rows) + "EOF\n"
        exe.write_text("#!/bin/sh\n" + body)
        exe.chmod(0o755)
        engine._gpu_identity.cache_clear()

    set_rows(["0, GPU-aaaa-1111, Tesla P40, 545.23.08"])
    _skip_unless_executable(exe)
    return set_rows


@pytest.fixture
def job(tmp_path, real_run):
    """A ligand dir with two docking fixture ligands (as ``L000000{0,1}.pdbqt``) and a copy of the receptor."""
    lig_dir = tmp_path / "ligands"
    lig_dir.mkdir()
    for stem, name in zip(STEMS, real_run.docked):
        shutil.copy(real_run.ligand_pdbqt(name), lig_dir / f"{stem}.pdbqt")
    receptor = tmp_path / f"{real_run.protein_id}.pdbqt"
    shutil.copy(real_run.receptor, receptor)
    return {"receptor": receptor, "ligand_dir": lig_dir, "center": real_run.pocket_center(1), "size": (20, 20, 20),
            "thread": 8000, "pocket": 1}


def _config(path):
    return dict(fe.parse_config(open(path).read()))


# --- engine_path / kernel_cache_dir --------------------------------------------------------------------------------

def test_engine_path_default(monkeypatch):
    monkeypatch.delenv("POCKETVINA_ENGINE", raising=False)
    assert engine.engine_path() == os.path.join(engine.PACKAGE_DIR, "PocketVina-GPU")
    assert os.path.dirname(engine.engine_path()) == os.path.dirname(os.path.abspath(engine.__file__))


def test_engine_path_from_env(fake_engine_env, monkeypatch, tmp_path):
    assert engine.engine_path() == str(fake_engine_env)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POCKETVINA_ENGINE", "./bin/engine")
    assert engine.engine_path() == str(tmp_path / "bin" / "engine")
    bin_dir = tmp_path / "onpath"
    bin_dir.mkdir()
    exe = bin_dir / "my-engine"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    _skip_unless_executable(exe)  # shutil.which only finds executables
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("POCKETVINA_ENGINE", "my-engine")
    assert engine.engine_path() == str(exe)


def _installed_engine(prefix):
    """``make install`` layout: ``<prefix>/bin/pocketvina-engine`` + ``<prefix>/share/pocketvina/OpenCL/...``."""
    exe = prefix / "bin" / engine.ENGINE_COMMAND
    exe.parent.mkdir(parents=True)
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    _skip_unless_executable(exe)
    for sub in engine.KERNEL_SOURCE_DIRS:
        (prefix / "share" / "pocketvina" / sub).mkdir(parents=True)
    (prefix / "share" / "pocketvina" / "OpenCL" / "src" / "kernels" / "kernel1.cl").write_text("// k1\n")
    return exe


def test_engine_path_finds_an_installed_engine(monkeypatch, tmp_path):
    """Without the package binary (e.g. a wheel built without it): pocketvina-engine on PATH, then in this
    Python's bin/ (a conda env not activated); the kernel sources come from <prefix>/share/pocketvina."""
    monkeypatch.delenv("POCKETVINA_ENGINE", raising=False)
    monkeypatch.delenv("POCKETVINA_KERNEL_SRC", raising=False)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    monkeypatch.setattr(engine, "PACKAGE_DIR", str(pkg))
    monkeypatch.setattr(engine.sys, "prefix", str(tmp_path / "no_env"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty_bin"))
    assert engine.engine_path() == str(pkg / engine.ENGINE_NAME)  # none: run_engine reports this path missing
    assert engine.kernel_source_dir() == str(pkg)

    env_exe = _installed_engine(tmp_path / "conda_env")
    monkeypatch.setattr(engine.sys, "prefix", str(tmp_path / "conda_env"))
    assert engine.engine_path() == str(env_exe)
    assert engine.kernel_source_dir() == str(tmp_path / "conda_env" / "share" / "pocketvina")

    path_exe = _installed_engine(tmp_path / "prefix")
    monkeypatch.setenv("PATH", f"{path_exe.parent}{os.pathsep}{tmp_path / 'empty_bin'}")
    share = str(tmp_path / "prefix" / "share" / "pocketvina")
    assert engine.engine_path() == str(path_exe) and engine.kernel_source_dir() == share
    assert engine.engine_env()["POCKETVINA_KERNEL_SRC"] == share
    monkeypatch.setenv("POCKETVINA_KERNEL_SRC", str(tmp_path / "src"))
    assert engine.kernel_source_dir() == str(tmp_path / "src")
    monkeypatch.delenv("POCKETVINA_KERNEL_SRC")

    packaged = pkg / engine.ENGINE_NAME  # the package binary comes first, with the package's kernel sources
    packaged.write_text("#!/bin/sh\n")
    packaged.chmod(0o644)  # not executable: not used
    assert engine.engine_path() == str(path_exe)
    packaged.chmod(0o755)
    assert engine.engine_path() == str(packaged) and engine.kernel_source_dir() == str(pkg)


def test_kernel_sources_of_an_engine_linked_into_a_bin_folder(monkeypatch, tmp_path):
    """$POCKETVINA_ENGINE (or PATH) may name a symlink to an installed engine: its prefix has the kernels."""
    exe = _installed_engine(tmp_path / "prefix")
    link = tmp_path / "bin" / "my-engine"
    link.parent.mkdir()
    link.symlink_to(exe)
    monkeypatch.delenv("POCKETVINA_KERNEL_SRC", raising=False)
    monkeypatch.setenv("POCKETVINA_ENGINE", str(link))
    assert engine.engine_path() == str(link)
    assert engine.installed_kernel_dir(str(link)) == str(tmp_path / "prefix" / "share" / "pocketvina")
    assert engine.kernel_source_dir() == str(tmp_path / "prefix" / "share" / "pocketvina")
    assert engine.installed_kernel_dir(str(tmp_path / "bin" / "missing")) is None


def test_kernel_cache_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POCKETVINA_KERNEL_CACHE", "kc")
    assert kernel_cache_dir() == str(tmp_path / "kc")
    assert not (tmp_path / "kc").exists()  # the engine creates it


def test_kernel_cache_dir_default_location(monkeypatch, tmp_path, fake_smi):
    monkeypatch.delenv("POCKETVINA_KERNEL_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    path = kernel_cache_dir()
    parent, key = os.path.split(path)
    assert parent == str(tmp_path / "home" / ".cache" / "pocketvina" / "kernels")
    assert re.fullmatch(r"[0-9a-f]{16}", key) and key == engine.kernel_cache_key()
    assert kernel_cache_dir() == path
    assert not os.path.exists(path)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert kernel_cache_dir() == str(tmp_path / "xdg" / "pocketvina" / "kernels" / key)
    monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")  # invalid in the XDG spec: ignored
    assert kernel_cache_dir() == path
    assert not path.startswith(engine.PACKAGE_DIR + os.sep)


def test_kernel_cache_key_tracks_engine_gpu_and_kernel_sources(monkeypatch, tmp_path, fake_smi):
    monkeypatch.delenv("POCKETVINA_KERNEL_SRC", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    exe = tmp_path / "engine"
    exe.write_bytes(b"engine v1")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(exe))
    key = engine.kernel_cache_key()
    assert engine.kernel_cache_key() == key

    fake_smi(["0, GPU-aaaa-1111, Tesla P40, 550.54.15"])  # driver update
    assert engine.kernel_cache_key() != key
    fake_smi(["0, GPU-aaaa-1111, NVIDIA A40, 545.23.08"])  # another GPU model
    assert engine.kernel_cache_key() != key
    fake_smi(None)  # no nvidia-smi output: still a valid key, from engine and sources only
    assert re.fullmatch(r"[0-9a-f]{16}", engine.kernel_cache_key())
    assert engine.kernel_cache_key() != key
    fake_smi(["0, GPU-aaaa-1111, Tesla P40, 545.23.08"])
    assert engine.kernel_cache_key() == key

    exe.write_bytes(b"engine v2, rebuilt")
    assert engine.kernel_cache_key() != key
    exe.write_bytes(b"engine v1")
    assert engine.kernel_cache_key() == key

    src = tmp_path / "src"
    for sub in engine.KERNEL_SOURCE_DIRS:
        shutil.copytree(os.path.join(engine.PACKAGE_DIR, sub), src / sub)
    monkeypatch.setenv("POCKETVINA_KERNEL_SRC", str(src))
    assert engine.kernel_cache_key() == key  # same sources, other place
    with open(src / "OpenCL" / "src" / "kernels" / "kernel2.cl", "a") as f:
        f.write("\n// edited\n")
    assert engine.kernel_cache_key() != key


def test_kernel_cache_key_with_an_unreadable_engine(monkeypatch, tmp_path, fake_smi, job):
    exe = tmp_path / "engine"
    exe.write_bytes(b"engine v1")
    exe.chmod(stat.S_IXUSR)  # execute-only installs exist on shared clusters
    if os.access(exe, os.R_OK):
        pytest.skip("running as root: every file is readable")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(exe))
    monkeypatch.delenv("POCKETVINA_KERNEL_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    key = engine.kernel_cache_key()
    assert re.fullmatch(r"[0-9a-f]{16}", key)
    cfg = _config(write_config(tmp_path / "run.txt", job["receptor"], job["ligand_dir"], None, job["center"],
                               job["size"], 8000, 1))
    assert cfg["opencl_binary_path"].endswith(key)
    exe.chmod(stat.S_IWUSR)
    exe.write_bytes(b"engine v2, rebuilt")
    exe.chmod(stat.S_IXUSR)
    assert engine.kernel_cache_key() != key  # a rebuilt engine still gets fresh kernels


def test_gpu_identity_follows_cuda_visible_devices(fake_smi):
    fake_smi(["0, GPU-aaaa-1111, Tesla P40, 545.23.08", "1, GPU-bbbb-2222, NVIDIA A40, 545.23.08"])
    both = ("NVIDIA A40|545.23.08", "Tesla P40|545.23.08")
    assert engine._gpu_identity(None) == both
    assert engine._gpu_identity("") == both
    assert engine._gpu_identity("1") == ("NVIDIA A40|545.23.08",)
    assert engine._gpu_identity("0,1") == ("Tesla P40|545.23.08",)
    assert engine._gpu_identity("GPU-bbbb") == ("NVIDIA A40|545.23.08",)
    assert engine._gpu_identity("7") == both  # not an index nvidia-smi knows: all GPUs count
    fake_smi(["0, GPU-cccc, NVIDIA RTX A6000, Ada, 560.28.03"])  # a comma in the name
    assert engine._gpu_identity(None) == ("NVIDIA RTX A6000, Ada|560.28.03",)


# --- write_config --------------------------------------------------------------------------------------------------

def test_write_config_contents(tmp_path, job):
    path = write_config(tmp_path / "cfg" / "run.txt", job["receptor"], job["ligand_dir"], tmp_path / "out",
                        job["center"], job["size"], job["thread"], job["pocket"], seed=42, num_modes=9,
                        energy_range=3.0, opencl_binary_path=tmp_path / "kernels")
    assert path == str(tmp_path / "cfg" / "run.txt")
    pairs = fe.parse_config(open(path).read())
    assert [k for k, _ in pairs] == ["receptor", "ligand_directory", "output_directory", "opencl_binary_path",
                                     "center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "thread",
                                     "pocket_rank", "seed", "num_modes", "energy_range"]
    cfg = dict(pairs)
    assert cfg["receptor"] == str(job["receptor"]) and cfg["ligand_directory"] == str(job["ligand_dir"])
    assert cfg["output_directory"] == str(tmp_path / "out")
    assert cfg["opencl_binary_path"] == str(tmp_path / "kernels")
    assert tuple(float(cfg[f"center_{a}"]) for a in "xyz") == job["center"]
    assert [cfg[f"size_{a}"] for a in "xyz"] == ["20", "20", "20"]
    assert (cfg["thread"], cfg["pocket_rank"], cfg["seed"], cfg["num_modes"]) == ("8000", "1", "42", "9")
    assert cfg["energy_range"] == "3.0"
    opts = fe.load_options(["--config", path])  # the fake engine checks options like the real one
    fe.validate(opts)
    assert (opts["seed"], opts["num_modes"], opts["energy_range"]) == (42, 9, 3.0)


def test_write_config_defaults(fake_engine_env, tmp_path, job):
    path = write_config(tmp_path / "run.txt", job["receptor"], job["ligand_dir"], None, job["center"], job["size"],
                        job["thread"], 3)
    cfg = _config(path)
    assert "output_directory" not in cfg  # engine default: <ligand_dir>_out
    assert not {"seed", "num_modes", "energy_range"} & cfg.keys()
    assert cfg["opencl_binary_path"] == kernel_cache_dir() == os.environ["POCKETVINA_KERNEL_CACHE"]
    assert cfg["pocket_rank"] == "3"


def test_write_config_makes_paths_absolute(tmp_path, monkeypatch, job):
    monkeypatch.chdir(tmp_path)
    path = write_config("run.txt", os.path.basename(job["receptor"]), "ligands", "out", job["center"], job["size"],
                        8000, 1, opencl_binary_path="kernels")
    assert path == str(tmp_path / "run.txt")
    cfg = _config(path)
    for key, name in (("receptor", job["receptor"].name), ("ligand_directory", "ligands"),
                      ("output_directory", "out"), ("opencl_binary_path", "kernels")):
        assert cfg[key] == str(tmp_path / name)


def test_write_config_numbers(tmp_path, job):
    path = write_config(tmp_path / "run.txt", job["receptor"], job["ligand_dir"], tmp_path / "out",
                        np.array([1, -2.5, 26.7673]), (np.float32(22.5), np.int64(20), 18.0), np.int32(1000),
                        np.int64(2), seed=np.int64(-7), energy_range=0)
    cfg = _config(path)
    assert [cfg[f"center_{a}"] for a in "xyz"] == ["1.0", "-2.5", "26.7673"]
    assert [cfg[f"size_{a}"] for a in "xyz"] == ["22.5", "20", "18.0"]
    assert (cfg["thread"], cfg["pocket_rank"], cfg["seed"], cfg["energy_range"]) == ("1000", "2", "-7", "0")


@pytest.mark.parametrize("change", [
    {"thread": 999}, {"thread": 8000.0}, {"thread": "8000"}, {"size": (20, 0, 20)}, {"size": (20, -1, 20)},
    {"size": (20, 20)}, {"center": (1.0, float("nan"), 2.0)}, {"center": (1.0, float("inf"), 2.0)},
    {"center": (1, 2, 3, 4)}, {"center": (True, 0, 0)}, {"pocket": 0}, {"pocket": True}, {"pocket": 1.5},
    {"seed": 2 ** 31}, {"seed": -2 ** 31 - 1}, {"seed": "42"}, {"num_modes": 0}, {"energy_range": -1},
    {"receptor": "/data/rec#1.pdbqt"}, {"out_dir": "/data/out\nseed = 1"}, {"out_dir": "/data/out "},
    {"ligand_dir": "/data/x.pdbqt_files/ligands"},
], ids=lambda change: next(iter(change)))
def test_write_config_rejects_what_the_engine_would(tmp_path, job, change):
    args = {"receptor": job["receptor"], "ligand_dir": job["ligand_dir"], "out_dir": tmp_path / "out",
            "center": job["center"], "size": job["size"], "thread": 8000, "pocket": 1,
            "opencl_binary_path": tmp_path / "k"}
    args.update(change)
    with pytest.raises(ValueError):
        write_config(tmp_path / "run.txt", **args)
    assert not (tmp_path / "run.txt").exists()


def test_write_config_accepts_extreme_seeds(tmp_path, job):
    for seed in (-2 ** 31, 0, 2 ** 31 - 1):
        path = write_config(tmp_path / "run.txt", job["receptor"], job["ligand_dir"], tmp_path / "out",
                            job["center"], job["size"], 8000, 1, seed=seed, opencl_binary_path=tmp_path / "k")
        assert _config(path)["seed"] == str(seed)


# --- run_engine + collect_outputs (fake engine) --------------------------------------------------------------------

def _run_job(tmp_path, job, *, seed=42, out="out", work="work", **kwargs):
    cfg = write_config(tmp_path / f"{out}.txt", job["receptor"], job["ligand_dir"], tmp_path / out, job["center"],
                       job["size"], job["thread"], job["pocket"], seed=seed)
    return run_engine(cfg, cwd=tmp_path / work, **kwargs), cfg


def test_run_engine_end_to_end(fake_engine_env, tmp_path, job, real_run, monkeypatch):
    monkeypatch.delenv("POCKETVINA_KERNEL_SRC", raising=False)
    res, cfg = _run_job(tmp_path, job)
    assert isinstance(res, EngineResult) and res.ok and res.returncode == 0 and not res.timed_out
    assert res.runtime_s > 0
    assert "QuickVina 2-GPU3 total runtime" in res.log_tail
    assert os.listdir(tmp_path / "work") == []  # created, and nothing left in the engine's cwd

    (call,) = fake_engine_env.calls()
    assert call["argv"] == ["--config", cfg]
    assert os.path.realpath(call["cwd"]) == os.path.realpath(tmp_path / "work")
    assert call["env"]["POCKETVINA_KERNEL_SRC"] == engine.PACKAGE_DIR
    assert call["env"]["OPENCL_ROOT"] == os.path.join(engine.PACKAGE_DIR, "OpenCL")
    assert call["options"]["opencl_binary_path"] == os.environ["POCKETVINA_KERNEL_CACHE"]

    outputs = collect_outputs(tmp_path / "out", job["receptor"], 1, STEMS)
    assert list(outputs) == list(STEMS)
    for stem in STEMS:
        raw = outputs[stem]
        assert raw.startswith(b"MODEL 1\n")
        got = [float(x) for x in re.findall(rb"REMARK VINA RESULT:\s+(\S+)", raw)]
        want = [m[0] for m in fe.score_modes(real_run.protein_id, 1, stem, 42)]
        assert got == want


def test_run_engine_environment(fake_engine_env, tmp_path, job, monkeypatch):
    monkeypatch.setenv("POCKETVINA_KERNEL_SRC", "relative/src")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/a::/opt/b:")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    res, _ = _run_job(tmp_path, job, env_extra={"CUDA_VISIBLE_DEVICES": 3, "OPENCL_ROOT": None})
    assert res.ok
    env = fake_engine_env.calls()[0]["env"]
    assert env["POCKETVINA_KERNEL_SRC"] == os.path.abspath("relative/src")  # the engine runs elsewhere
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["OPENCL_ROOT"] is None
    entries = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert "" not in entries and entries[-2:] == ["/opt/a", "/opt/b"]
    assert all(os.path.isdir(p) and p.startswith(engine.PACKAGE_DIR) for p in entries[:-2])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"  # the caller's environment is not touched


def test_engine_env_drops_an_all_empty_ld_library_path(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, "PACKAGE_DIR", str(tmp_path / "pkg"))  # no package boost lib dirs (e.g. CI)
    for value in ("", "::"):
        monkeypatch.setenv("LD_LIBRARY_PATH", value)
        assert "LD_LIBRARY_PATH" not in engine.engine_env()  # "" entries would mean the working directory
    monkeypatch.delenv("LD_LIBRARY_PATH")
    assert "LD_LIBRARY_PATH" not in engine.engine_env()
    monkeypatch.setenv("LD_LIBRARY_PATH", ":/opt/a:")
    assert engine.engine_env()["LD_LIBRARY_PATH"] == "/opt/a"


def test_run_engine_failing_ligand(fake_engine_env, tmp_path, job):
    with open(job["ligand_dir"] / "L0000002.pdbqt", "w") as f:
        f.write("REMARK FAIL\n")
    res, _ = _run_job(tmp_path, job)
    assert res.returncode == 1 and not res.ok and not res.timed_out
    outputs = collect_outputs(tmp_path / "out", job["receptor"], 1, STEMS + ("L0000002",))
    assert outputs["L0000002"] is None
    assert all(outputs[s] for s in STEMS)


def test_run_engine_log_path(fake_engine_env, tmp_path, job):
    log = tmp_path / "logs" / "pocket1.log"
    res, _ = _run_job(tmp_path, job, log_path=log)
    text = log.read_text()
    assert "Pocket rank: 1" in text and "Using random seed: 42" in text
    assert text.rstrip("\n").endswith(res.log_tail.rstrip("\n"))
    _run_job(tmp_path, job, out="out2", log_path=log)  # overwritten, not appended
    assert log.read_text().count("Pocket rank: 1") == 1


def test_run_engine_timeout(fake_engine_env, tmp_path, job, monkeypatch):
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "30")
    res, _ = _run_job(tmp_path, job, timeout=0.5)
    assert res.timed_out and not res.ok and res.returncode != 0
    assert res.runtime_s < 20
    assert "timeout" in res.log_tail
    assert collect_outputs(tmp_path / "out", job["receptor"], 1, STEMS) == {s: None for s in STEMS}


def _gone(pid, wait=10.0) -> bool:
    """True once ``pid`` has exited (a zombie waiting for its new parent to reap it counts as exited)."""
    deadline = time.monotonic() + wait
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            with open(f"/proc/{pid}/stat") as f:
                if f.read().rsplit(")", 1)[1].split()[0] == "Z":
                    return True
        except (OSError, IndexError):
            pass
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process groups")
def test_run_engine_timeout_kills_the_process_group(monkeypatch, tmp_path):
    """A timed-out engine is killed with everything it started (here a wrapper script's background process)."""
    exe = tmp_path / "group-engine"
    pids = tmp_path / "pids.txt"
    exe.write_text(f"#!/bin/sh\nsleep 60 &\necho \"$$ $!\" > '{pids}'\nsleep 60\n")
    exe.chmod(0o755)
    _skip_unless_executable(exe)
    monkeypatch.setenv("POCKETVINA_ENGINE", str(exe))
    cfg = tmp_path / "unused.txt"
    cfg.write_text("")
    t0 = time.monotonic()
    res = run_engine(cfg, cwd=tmp_path / "work", timeout=2)
    assert res.timed_out and not res.ok and res.returncode == -signal.SIGKILL
    assert time.monotonic() - t0 < 30
    assert res.log_tail.endswith("[pocketvina] engine killed (with its process group) after the 2 s timeout")
    shell, background = map(int, pids.read_text().split())
    assert _gone(shell) and _gone(background), "the engine's process group was left running"


@pytest.mark.skipif(not hasattr(os, "getsid"), reason="POSIX sessions")
def test_run_engine_starts_the_engine_in_its_own_session(monkeypatch, tmp_path):
    """Its own session and process group (so a timeout can kill the group, and a terminal's or notebook's signals go
    to Python, which kills it), and $POCKETVINA_DIE_WITH_PARENT names this process (main.cpp: die with it)."""
    cfg = _script_engine(tmp_path, monkeypatch, (
        "import os\n"
        "print(os.getpid(), os.getsid(0), os.getpgrp(), os.environ.get('POCKETVINA_DIE_WITH_PARENT'))"))
    res = run_engine(cfg, cwd=tmp_path / "work")
    assert res.ok, res.log_tail
    pid, sid, pgrp, parent = res.log_tail.split()
    assert pid == sid == pgrp and int(sid) != os.getsid(0)
    assert parent == str(os.getpid()) == engine.engine_env()[engine.DIE_WITH_PARENT_ENV]
    assert engine.engine_env({engine.DIE_WITH_PARENT_ENV: None}).get(engine.DIE_WITH_PARENT_ENV) is None


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "60"])
def test_run_engine_rejects_invalid_timeouts(fake_engine_env, tmp_path, job, bad):
    with pytest.raises(ValueError, match="timeout"):
        _run_job(tmp_path, job, timeout=bad)
    assert fake_engine_env.calls() == []
    assert engine.check_timeout("engine_timeout", 5) == 5.0 and engine.check_timeout("x", None) is None


def test_run_engine_kills_the_engine_on_interrupt(fake_engine_env, tmp_path, job, monkeypatch):
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "30")
    interrupted = []

    class InterruptedPopen(subprocess.Popen):
        def wait(self, timeout=None):
            if not interrupted:  # Ctrl-C while waiting for the engine
                interrupted.append(self)
                raise KeyboardInterrupt
            return super().wait(timeout)

    monkeypatch.setattr(engine.subprocess, "Popen", InterruptedPopen)
    with pytest.raises(KeyboardInterrupt):
        _run_job(tmp_path, job)
    (proc,) = interrupted
    assert proc.returncode == -signal.SIGKILL  # not left running on the GPU


EXIT_ON_SIGTERM = """\
import os, signal, time
from pocketvina.engine import exit_on_sigterm

with exit_on_sigterm():
    pass
assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL  # restored
signal.signal(signal.SIGTERM, lambda signum, frame: print("own handler", flush=True))
with exit_on_sigterm():  # an application's handler is left alone
    os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(0.5)
signal.signal(signal.SIGTERM, signal.SIG_DFL)
with exit_on_sigterm():
    try:
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(10)
        print("not interrupted", flush=True)
    finally:
        print("cleanup", flush=True)
print("survived", flush=True)
"""


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX signals")
def test_exit_on_sigterm(repo_root, tmp_path):
    """SIGTERM unwinds the block (``finally`` runs: engine killed, temp dirs removed), then the process still dies
    of SIGTERM. A subprocess, since it ends with that signal."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(repo_root), os.environ.get("PYTHONPATH")])))
    proc = subprocess.run([sys.executable, "-c", EXIT_ON_SIGTERM], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=120)
    assert proc.stdout.splitlines() == ["own handler", "cleanup"], proc.stdout + proc.stderr
    assert proc.returncode == -signal.SIGTERM, proc.stderr


def test_run_engine_never_runs_in_the_package_dir(fake_engine_env, tmp_path, job):
    cfg = write_config(tmp_path / "run.txt", job["receptor"], job["ligand_dir"], tmp_path / "out", job["center"],
                       job["size"], 8000, 1)
    for cwd in (engine.PACKAGE_DIR, os.path.join(engine.PACKAGE_DIR, "OpenCL"),
                os.path.join(engine.PACKAGE_DIR, "new_subdir")):
        with pytest.raises(ValueError, match="package directory"):
            run_engine(cfg, cwd=cwd)
    assert fake_engine_env.calls() == []
    assert not os.path.exists(os.path.join(engine.PACKAGE_DIR, "new_subdir"))


def test_run_engine_missing_or_not_executable_engine(monkeypatch, tmp_path, job):
    cfg = write_config(tmp_path / "run.txt", job["receptor"], job["ligand_dir"], tmp_path / "out", job["center"],
                       job["size"], 8000, 1, opencl_binary_path=tmp_path / "k")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(tmp_path / "no-such-engine"))
    with pytest.raises(FileNotFoundError, match="POCKETVINA_ENGINE"):
        run_engine(cfg, cwd=tmp_path / "work")
    exe = tmp_path / "not-executable"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setenv("POCKETVINA_ENGINE", str(exe))
    with pytest.raises(PermissionError):
        run_engine(cfg, cwd=tmp_path / "work")


def _script_engine(tmp_path, monkeypatch, body):
    exe = tmp_path / "script-engine"
    exe.write_text(f"#!{sys.executable}\nimport sys\n{body}\n")
    exe.chmod(0o755)
    _skip_unless_executable(exe)
    monkeypatch.setenv("POCKETVINA_ENGINE", str(exe))
    cfg = tmp_path / "unused.txt"
    cfg.write_text("")
    return cfg


def test_log_tail_shows_progress_bars_as_a_terminal_would(monkeypatch, tmp_path):
    cfg = _script_engine(tmp_path, monkeypatch, (
        "sys.stdout.write('GPU Device: Tesla P40\\n\\rPerform docking|===  |\\rPerform docking|  ===|'"
        " '\\rPerform docking|====done====|\\n')\n"
        "sys.stdout.flush()\n"
        "sys.stderr.write('Err-6:CL_OUT_OF_HOST_MEMORY\\n')\n"
        "sys.exit(255)"))
    res = run_engine(cfg, cwd=tmp_path / "work")
    assert res.returncode == 255 and not res.ok
    assert res.log_tail == "GPU Device: Tesla P40\nPerform docking|====done====|\nErr-6:CL_OUT_OF_HOST_MEMORY\n"


def test_log_tail_is_bounded_and_starts_at_a_line(monkeypatch, tmp_path):
    cfg = _script_engine(tmp_path, monkeypatch, (
        "for i in range(20000):\n"
        "    print(f'line {i:05d} ' + 'x' * 40)\n"
        "print('the end', end='')"))
    res = run_engine(cfg, cwd=tmp_path / "work")
    assert res.ok
    assert 0 < len(res.log_tail) <= engine.LOG_TAIL_CHARS
    assert res.log_tail.startswith("line ") and res.log_tail.endswith("line 19999 " + "x" * 40 + "\nthe end")


def test_collect_outputs_remove(fake_engine_env, tmp_path, job):
    res, _ = _run_job(tmp_path, job)
    assert res.ok and len(os.listdir(tmp_path / "out")) == 4  # pdbqt + json per ligand
    kept = collect_outputs(tmp_path / "out", job["receptor"], 1, STEMS)
    removed = collect_outputs(tmp_path / "out", job["receptor"], 1, STEMS + ("L9999999",), remove=True)
    assert removed == {**kept, "L9999999": None}
    assert os.listdir(tmp_path / "out") == []


def test_collect_outputs_uses_the_engine_names(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    receptor = "/data/structures/rec.v1.pdbqt"  # receptor base: up to the last dot
    for stem in ("L0000000", "lig.2"):
        pdbqt, json_path, _ = fe.output_paths(fe.directory_out_name("/lig", str(out), f"/lig/{stem}.pdbqt"),
                                              receptor, 3)
        assert os.path.basename(pdbqt) == f"rec.v1-pocket3-{stem}_out.pdbqt"
        with open(pdbqt, "wb") as f:
            f.write(hashlib.sha256(stem.encode()).digest())
    got = collect_outputs(str(out), receptor, 3, ["L0000000", "lig.2", "L0000001"])
    assert got == {"L0000000": hashlib.sha256(b"L0000000").digest(), "lig.2": hashlib.sha256(b"lig.2").digest(),
                   "L0000001": None}
    assert collect_outputs(out, receptor, 2, ["L0000000"]) == {"L0000000": None}  # other pocket


def test_engine_result_ok():
    assert EngineResult(0, 1.0, "").ok
    assert not EngineResult(1, 1.0, "").ok
    assert not EngineResult(0, 1.0, "", timed_out=True).ok


# --- real engine on a GPU ------------------------------------------------------------------------------------------

def _package_files():
    """{relative path: (size, mtime_ns)} of the package's non-Python files (other modules may add .py files)."""
    files = {}
    for dirpath, dirnames, filenames in os.walk(engine.PACKAGE_DIR):
        dirnames[:] = [d for d in dirnames if d not in ("boost_1_77_0", "p2rank_2.5", "__pycache__")]
        for name in filenames:
            if not name.endswith((".py", ".pyc")):
                st = os.stat(os.path.join(dirpath, name))
                files[os.path.relpath(os.path.join(dirpath, name), engine.PACKAGE_DIR)] = (st.st_size, st.st_mtime_ns)
    return files


def _digest(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


@pytest.mark.gpu
def test_real_engine_kernel_cache(tmp_path, monkeypatch, job):
    """Cold cache: the engine builds the kernels into the cache dir; warm cache: it loads them and gives the same
    poses for the same seed; neither run writes into its cwd or the package directory."""
    monkeypatch.delenv("POCKETVINA_KERNEL_SRC", raising=False)
    cache = tmp_path / "kernel_cache" / "key"  # parent missing too: the engine creates both
    monkeypatch.setenv("POCKETVINA_KERNEL_CACHE", str(cache))
    assert os.access(engine.engine_path(), os.X_OK), engine.engine_path()
    before = _package_files()

    cold, _ = _run_job(tmp_path, job, out="cold", work="work_cold", log_path=tmp_path / "cold.log", timeout=900)
    cold_log = (tmp_path / "cold.log").read_text(errors="replace")
    assert cold.ok, cold.log_tail
    assert "Build kernel 1 from source" in cold_log and f"Saved OpenCL kernels to {cache}" in cold_log
    kernels = {k: (_digest(cache / k), os.stat(cache / k).st_mtime_ns) for k in engine.KERNEL_FILES}
    assert sorted(os.listdir(cache)) == sorted(engine.KERNEL_FILES)  # no temp files left behind

    warm, _ = _run_job(tmp_path, job, out="warm", work="work_warm", log_path=tmp_path / "warm.log", timeout=900)
    warm_log = (tmp_path / "warm.log").read_text(errors="replace")
    assert warm.ok, warm.log_tail
    assert f"Loaded OpenCL kernels from {cache}" in warm_log and "from source" not in warm_log
    assert {k: (_digest(cache / k), os.stat(cache / k).st_mtime_ns) for k in engine.KERNEL_FILES} == kernels

    cold_out = collect_outputs(tmp_path / "cold", job["receptor"], 1, STEMS)
    warm_out = collect_outputs(tmp_path / "warm", job["receptor"], 1, STEMS)
    for stem in STEMS:
        affinities = [float(x) for x in re.findall(rb"REMARK VINA RESULT:\s+(\S+)", cold_out[stem] or b"")]
        assert affinities and affinities[0] < 0 and affinities == sorted(affinities)
    assert warm_out == cold_out  # same seed, same kernels: the same poses byte for byte
    assert os.listdir(tmp_path / "work_cold") == [] and os.listdir(tmp_path / "work_warm") == []

    # opencl_binary_path == the engine's cwd (as with the engine default "."): built in memory, never saved there
    work = tmp_path / "work_cwd"
    work.mkdir()
    cfg = write_config(tmp_path / "cwd.txt", job["receptor"], job["ligand_dir"], tmp_path / "cwd", job["center"],
                       job["size"], job["thread"], job["pocket"], seed=42, opencl_binary_path=work)
    res = run_engine(cfg, cwd=work, log_path=tmp_path / "cwd.log", timeout=900)
    assert res.ok, res.log_tail
    cwd_log = (tmp_path / "cwd.log").read_text(errors="replace")
    assert "not saved: opencl_binary_path is the working directory" in cwd_log
    assert os.listdir(work) == []
    assert collect_outputs(tmp_path / "cwd", job["receptor"], 1, STEMS) == cold_out
    assert _package_files() == before


@pytest.mark.gpu
def test_real_engine_never_writes_into_the_kernel_source_tree(tmp_path, job):
    """An opencl_binary_path inside the kernel sources (old configs name the package's QuickVina2-GPU-2-1) is only
    read: kernels the driver rejects, or that are missing there, are built in memory and not saved."""
    src = tmp_path / "src"
    for sub in engine.KERNEL_SOURCE_DIRS:
        shutil.copytree(os.path.join(engine.PACKAGE_DIR, sub), src / sub)
    rejected = src / "QuickVina2-GPU-2-1"
    rejected.mkdir()
    for name in engine.KERNEL_FILES:
        (rejected / name).write_bytes(b"not a kernel\n")
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in rejected.iterdir()}
    missing = src / "kernels"
    for i, kernel_dir in enumerate((rejected, missing)):
        cfg = write_config(tmp_path / f"run{i}.txt", job["receptor"], job["ligand_dir"], tmp_path / f"out{i}",
                           job["center"], job["size"], job["thread"], job["pocket"], seed=42,
                           opencl_binary_path=kernel_dir)
        log = tmp_path / f"run{i}.log"
        res = run_engine(cfg, cwd=tmp_path / f"work{i}", env_extra={"POCKETVINA_KERNEL_SRC": str(src)},
                         log_path=log, timeout=900)
        assert res.ok, res.log_tail
        assert "not saved: opencl_binary_path is inside the kernel source directory" in log.read_text(errors="replace")
        assert all(collect_outputs(tmp_path / f"out{i}", job["receptor"], 1, STEMS).values())
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in rejected.iterdir()} == before
    assert not missing.exists()


_WRITING_RE = re.compile(r"Writing ligand (\S+) output")


@pytest.mark.gpu
def test_real_engine_ligand_order_does_not_change_results(tmp_path, example_dir, real_run):
    """The ligands of one engine call share its random number generator, so their order changes the poses. The
    engine docks a ligand directory in file name order and skips what is not a regular .pdbqt file: two folders
    with the same ligands, created in opposite orders (and with a folder, a dangling link and a text file named
    like ligands next to them), give the same poses byte for byte for the same seed, on any file system."""
    from pocketvina.ligands import prepare_ligands, read_smiles_file, write_ligand_dir

    rows = [r for r in read_smiles_file(example_dir / "ligands.txt") if r[0] in ("molecule1", "molecule2",
                                                                                 "molecule3", "molecule5")]
    records = prepare_ligands(rows, seed=42, n_jobs=1)
    assert [r.status for r in records] == ["ok"] * 4
    stems = sorted(f"L{r.ligand_idx:07d}" for r in records)
    receptor = tmp_path / f"{real_run.protein_id}.pdbqt"
    shutil.copy(real_run.receptor, receptor)
    outputs, raw_orders = {}, {}
    for name, chosen in (("forward", records), ("reverse", records[::-1]), ("last_two", records[2:])):
        lig_dir = tmp_path / name
        lig_dir.mkdir()
        for rec in chosen:  # one file at a time, in this order
            write_ligand_dir([rec], lig_dir)
        (lig_dir / "Lfolder.pdbqt").mkdir()
        (lig_dir / "notes.txt").write_text("not a ligand\n")
        os.symlink(tmp_path / "missing.pdbqt", lig_dir / "Ldangling.pdbqt")
        raw_orders[name] = [n for n in os.listdir(lig_dir) if n.startswith("L000")]
        cfg = write_config(tmp_path / f"{name}.txt", receptor, lig_dir, tmp_path / f"{name}_out",
                           real_run.pocket_center(1), (20, 20, 20), 8000, 1, seed=42)
        log = tmp_path / f"{name}.log"
        res = run_engine(cfg, cwd=tmp_path / f"work_{name}", log_path=log, timeout=900)
        assert res.ok, res.log_tail
        written = [os.path.basename(p) for p in _WRITING_RE.findall(log.read_text(errors="replace"))]
        docked = stems if name != "last_two" else stems[2:]
        assert written == [f"{real_run.protein_id}-pocket1-{s}_out.pdbqt" for s in docked]  # file name order
        outputs[name] = collect_outputs(tmp_path / f"{name}_out", receptor, 1, docked)
        assert all(outputs[name].values())
    assert outputs["reverse"] == outputs["forward"]
    # the reason for the sort: the same ligand docked after other ones gets other poses
    assert outputs["last_two"] != {s: outputs["forward"][s] for s in stems[2:]}
    print("directory orders as created:", raw_orders)
