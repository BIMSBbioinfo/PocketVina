"""pocketvina.gpu (GPU list, engine slot planning) and the engine slot pool of pipeline.run_atlas, with the fake engine
and fake GPUs (a patched ``gpu.list_gpus``); the real-engine test is marked ``gpu``.

The fake engine logs the start and end time of every call and its ``CUDA_VISIBLE_DEVICES`` and slot, so the tests see
which calls ran at the same time and where. Its scores depend only on the call's ligands (and, with
``POCKETVINA_FAKE_ENGINE_SHARED_RNG=1``, their order in the call), as the real engine's, so any number of slots must
give the same results as one.
"""
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from pocketvina import PocketVinaConfig, PocketVinaGPU, engine, gpu, pipeline
from pocketvina.io.tables import read_table
from pocketvina.io.writer import ResultsDir
from test_pipeline import (
    FIXTURE_RECEPTOR,
    atlas_config,
    check_database_combined_csv,
    check_protein,
    crop_pdb,
    cropped_receptor,
    make_database,
    make_inputs,
    stub_predict_pockets,
)

TESTS_DIR = Path(__file__).resolve().parent
FIXTURE_ID = "AF-Q96LA5-F1-model_v4"
OOM_LINE = "Err-4:CL_MEM_OBJECT_ALLOCATION_FAILURE"


def fake_gpus(n, free=24000, name="Tesla P40"):
    return [gpu.GPU(i, f"GPU-{i}a{i}b{i}c-fake", name, 24576, free) for i in range(n)]


@pytest.fixture
def stubs(monkeypatch):
    """Pockets from the fixture predictions CSV (no P2Rank) and cropped receptors (fast OpenBabel)."""
    monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets())
    monkeypatch.setattr(pipeline, "receptor_pdbqt", cropped_receptor())


@pytest.fixture
def use_gpus(fake_engine_env, monkeypatch):
    """``use_gpus(list)``: what ``gpu.list_gpus`` returns in this test (``fake_engine_env`` makes it [])."""
    def install(gpus):
        monkeypatch.setattr(gpu, "list_gpus", lambda: list(gpus))
        return gpus
    return install


def two_proteins(tmp_path, example_dir, ligands=("molecule1", "molecule2", "molecule3", "molecule5")):
    return make_inputs(tmp_path, example_dir, ligands=ligands,
                       proteins=[FIXTURE_RECEPTOR, example_dir / "proteins" / "1fbl.pdb"])


def run(plist, smiles, out, tmp_path, **kw):
    pv = PocketVinaGPU(atlas_config(plist, smiles, out, tmp_path / "tmp", **kw))
    assert pipeline.run_atlas(pv) == str(out)
    assert pv.failed_proteins == {}
    assert os.listdir(tmp_path / "tmp") == []
    return pv


def results_of(out, pids) -> dict:
    """Pose files (bytes) and tables (pocket_runs without runtime_s) of these proteins."""
    got = {}
    for pid in pids:
        got[pid, "poses"] = (Path(out) / "poses" / f"{pid}.poses").read_bytes()
        for name in ("pairs", "modes", "proteins", "pockets", "pocket_runs"):
            df = read_table(Path(out) / "tables" / name / f"{pid}.parquet")
            got[pid, name] = df.drop(columns=["runtime_s"]) if name == "pocket_runs" else df
    return got


def assert_same_results(a, b, pids):
    ra, rb = results_of(a, pids), results_of(b, pids)
    for key, value in ra.items():
        if key[1] == "poses":
            assert value == rb[key], key
        else:
            pd.testing.assert_frame_equal(value, rb[key])


def max_overlap(calls) -> int:
    """Most calls running at the same time (from the fake engine's start/end times)."""
    events = sorted([(c["start"], 1) for c in calls] + [(c["end"], -1) for c in calls])
    best = current = 0
    for _, step in events:
        current += step
        best = max(best, current)
    return best


def is_kernel_build(call) -> bool:
    return os.path.basename(call["options"]["ligand_directory"]) == "kernel_build"


def wrapper_engine(tmp_path, fake_engine_env, body, name="wrapped_engine"):
    """An engine script: ``body`` (Python, with ``argv``, ``config`` (the config text) and ``env`` set), then the fake
    engine unless ``body`` exits."""
    script = tmp_path / f"{name}.py"
    script.write_text(f"import os, subprocess, sys\nargv = sys.argv[1:]\nconfig = open(argv[1]).read()\n"
                      f"env = os.environ\n{body}\n"
                      f"sys.exit(subprocess.call([{str(fake_engine_env)!r}] + argv))\n")
    exe = tmp_path / name
    exe.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(script))} \"$@\"\n")
    exe.chmod(0o755)
    if not os.access(exe, os.X_OK):
        pytest.skip(f"cannot execute {exe} (noexec temp dir?)")
    return exe


# --- gpu.py --------------------------------------------------------------------------------------------------------

SMI_ROWS = ["0, GPU-aaaa-1111, Tesla P40, 23040, 22912",
            "1, GPU-bbbb-2222, NVIDIA A40, 46068, 40000",
            "2, GPU-cccc-3333, NVIDIA RTX A6000, Ada, 49140, [N/A]"]


def test_parse_gpu_list():
    gpus = gpu.parse_gpu_list("\n".join(SMI_ROWS + ["", "No devices were found", "x, y, z, 1, 2"]))
    assert gpus == [gpu.GPU(0, "GPU-aaaa-1111", "Tesla P40", 23040, 22912),
                    gpu.GPU(1, "GPU-bbbb-2222", "NVIDIA A40", 46068, 40000),
                    gpu.GPU(2, "GPU-cccc-3333", "NVIDIA RTX A6000, Ada", 49140, None)]


def test_visible_gpus_follows_cuda_visible_devices(caplog):
    g0, g1, g2 = gpu.parse_gpu_list("\n".join(SMI_ROWS))

    def vis(gpus, visible_devices):  # CUDA numbers the GPUs as nvidia-smi does
        return gpu.visible_gpus(gpus, visible_devices, "PCI_BUS_ID")

    assert vis([g0, g1, g2], None) == [g0, g1, g2]
    assert vis([g0, g1, g2], "2,0") == [g2, g0]
    assert vis([g0, g1, g2], " 1 ") == [g1]
    assert vis([g0, g1, g2], "GPU-bbbb-2222,GPU-aaaa") == [g1, g0]  # UUIDs and UUID prefixes
    assert vis([g0, g1, g2], "") == []  # hides every GPU
    assert vis([g0, g1, g2], "-1") == [] and vis([g0, g1, g2], "NoDevFiles") == []
    assert vis([g0, g1, g2], "0,-1,1") == [g0]  # as in CUDA: the list ends at the first invalid entry
    assert vis([g0, g1, g2], "0,7,1") == [g0]
    assert vis([g0, g1, g2], "GPU-") == []  # a prefix of several UUIDs names no single GPU
    assert vis([g0, g1, g2], "1,1") == [g1]

    # CUDA's default order (FASTEST_FIRST) numbers GPUs of different models otherwise than nvidia-smi (PCI bus
    # order): "0" may be the A40, not nvidia-smi's GPU 0. The list ends before a number (the engines then get the
    # environment's CUDA_VISIBLE_DEVICES, as before), it used to name nvidia-smi's GPU 0 by its UUID
    for order in (None, "FASTEST_FIRST"):
        assert gpu.visible_gpus([g0, g1, g2], "0", order) == []
        assert gpu.visible_gpus([g0, g1, g2], "GPU-bbbb-2222,0", order) == [g1]
        assert gpu.visible_gpus([g0, g1, g2], "GPU-cccc,GPU-aaaa", order) == [g2, g0]  # UUIDs need no numbering
        assert gpu.visible_gpus([g0, g1, g2], None, order) == [g0, g1, g2]
    assert "CUDA_VISIBLE_DEVICES=0: CUDA's GPU 0 need not be nvidia-smi's" in caplog.text
    assert gpu.visible_gpus([g0, g1, g2], "1", " pci_bus_id ") == [g1]
    p40s = [g0, g0._replace(index=1, uuid="GPU-dddd-4444")]  # one model: the same numbering in either order
    assert gpu.visible_gpus(p40s, "1,0", None) == [p40s[1], p40s[0]]
    assert gpu.same_numbering(p40s, None) and gpu.same_numbering([], None)
    assert not gpu.same_numbering([g0, g1], "FASTEST_FIRST") and gpu.same_numbering([g0, g1], "PCI_BUS_ID")


@pytest.fixture
def fake_smi(tmp_path, monkeypatch):
    """A fake ``nvidia-smi`` first on PATH printing these rows for the memory query (None: exit 1) and the matching
    ``index, uuid, name, driver`` rows for the engine's kernel cache key."""
    bin_dir = tmp_path / "smi_bin"
    bin_dir.mkdir()
    exe = bin_dir / "nvidia-smi"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    def set_rows(rows):
        if rows is None:
            body = "exit 1\n"
        else:
            drivers = [",".join(r.split(",")[:3]) + ", 545.23.08" for r in rows]
            body = ('case "$1" in\n*driver_version*) cat <<\'EOF\'\n' + "".join(d + "\n" for d in drivers) +
                    "EOF\n;;\n*) cat <<'EOF'\n" + "".join(r + "\n" for r in rows) + "EOF\n;;\nesac\n")
        exe.write_text("#!/bin/sh\n" + body)
        exe.chmod(0o755)
        engine._gpu_identity.cache_clear()

    set_rows(SMI_ROWS[:2])
    if not os.access(exe, os.X_OK):
        pytest.skip(f"cannot execute {exe} (noexec temp dir?)")
    yield set_rows
    engine._gpu_identity.cache_clear()


def test_list_gpus_runs_nvidia_smi(fake_smi, monkeypatch, tmp_path):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    assert [(g.index, g.uuid, g.memory_free_mib) for g in gpu.list_gpus()] == [(0, "GPU-aaaa-1111", 22912),
                                                                            (1, "GPU-bbbb-2222", 40000)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")  # e.g. set by SLURM or the user
    assert gpu.list_gpus() == []  # a P40 and an A40: CUDA's GPU 1 need not be nvidia-smi's
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    assert [g.index for g in gpu.list_gpus()] == [1]
    fake_smi(["0, GPU-aaaa-1111, Tesla P40, 23040, 22912", "1, GPU-dddd-4444, Tesla P40, 23040, 20000"])
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    assert [g.uuid for g in gpu.list_gpus()] == ["GPU-dddd-4444"]  # one model: the same numbering
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu.list_gpus() == []
    fake_smi(None)  # driver problem
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert gpu.list_gpus() == []
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))  # no nvidia-smi at all
    assert gpu.list_gpus() == []


def test_check_gpu_spec_and_select_gpus():
    g0, g1, g2 = gpu.parse_gpu_list("\n".join(SMI_ROWS))
    check = gpu.check_gpu_spec
    assert check(None) is None
    assert check("0, 2") == "0,2" and check(1) == "1" and check([2, "GPU-aaaa"]) == "2,GPU-aaaa"
    for bad in ("", "0,", "a", "0;1", "GPU-", "-1", True, 1.5, [0, None], "0,0", {"0": 1}):
        with pytest.raises(ValueError):
            check(bad)
    assert gpu.select_gpus([g0, g1, g2], None) == [g0]  # one GPU per run: the first visible one
    assert gpu.select_gpus([g0, g1, g2], "all") == [g0, g1, g2] and check(" ALL ") == "all"
    assert gpu.select_gpus([g0, g1, g2], "2,GPU-aaaa") == [g2, g0]
    with pytest.raises(ValueError, match=r"'5' is not a visible GPU \(visible: 0 \(GPU-aaaa-1111\)"):
        gpu.select_gpus([g0, g1, g2], "0,5")
    with pytest.raises(ValueError, match="twice"):
        gpu.select_gpus([g0, g1, g2], "0,GPU-aaaa")
    with pytest.raises(ValueError, match="visible: none"):
        gpu.select_gpus([], "0")


def test_check_procs_per_gpu():
    assert gpu.check_procs_per_gpu(1) == 1 and gpu.check_procs_per_gpu(3) == 3
    assert gpu.check_procs_per_gpu("auto") == "auto" and gpu.check_procs_per_gpu(" AUTO ") == "auto"
    for bad in (0, -1, 1.0, "2", True, None, "max"):
        with pytest.raises(ValueError, match="procs_per_gpu"):
            gpu.check_procs_per_gpu(bad)


def test_plan_slots():
    p40 = gpu.GPU(0, "GPU-p40", "Tesla P40", 23040, 22912)
    a40 = gpu.GPU(1, "GPU-a40", "NVIDIA A40", 46068, 46000)
    busy = gpu.GPU(2, "GPU-busy", "Tesla P40", 23040, 3000)
    unknown = gpu.GPU(3, "GPU-unknown", "vGPU", None, None)
    plan = gpu.plan_slots
    assert plan([p40]) == ["GPU-p40"]  # procs_per_gpu=1
    assert plan([p40, a40], 2) == ["GPU-p40", "GPU-a40", "GPU-p40", "GPU-a40"]  # the GPUs alternate
    assert plan([p40], 5) == ["GPU-p40"] * 3  # (22912 - 1024) // 6600: capped by the free memory
    assert plan([busy], 2) == ["GPU-busy"]  # at least one, even if it does not fit
    assert plan([p40, a40], "auto") == ["GPU-p40", "GPU-a40"] * 3 + ["GPU-a40"] * 3  # 3 and 6 fit
    assert plan([p40], "auto", vram_per_proc_mib=10000) == ["GPU-p40"] * 2
    assert plan([p40], "auto", margin_mib=0) == ["GPU-p40"] * 3
    assert plan([busy, unknown], "auto") == ["GPU-busy", "GPU-unknown"]
    assert plan([unknown], 4) == ["GPU-unknown"] * 4  # no memory figure: not capped
    assert plan([gpu.GPU(0, "", "no uuid", 1000, 1000)]) == ["0"]  # the index then
    assert plan([], 2) == [] and plan([p40, p40], 1) == ["GPU-p40"]
    for kw in (dict(procs_per_gpu=0), dict(vram_per_proc_mib=0), dict(margin_mib=-1), dict(vram_per_proc_mib=1.5)):
        with pytest.raises(ValueError):
            plan([p40], **kw)
    assert gpu.DEFAULT_VRAM_PER_PROC_MIB == 6600


def test_is_out_of_memory():
    assert gpu.is_out_of_memory("Perform docking\n\nErr-4:CL_MEM_OBJECT_ALLOCATION_FAILURE\n")
    assert gpu.is_out_of_memory("\nErr-5:CL_OUT_OF_RESOURCES")
    assert not gpu.is_out_of_memory("Error: insufficient memory!") and not gpu.is_out_of_memory("")
    assert not gpu.is_out_of_memory("\nErr-11:CL_BUILD_PROGRAM_FAILURE")
    assert gpu.is_device_error("Analyzing the binding site ... done.\n\nErr-1001:CL_PLATFORM_NOT_FOUND_KHR\n")
    assert gpu.is_device_error("\nErr-2:CL_DEVICE_NOT_AVAILABLE") and gpu.is_device_error("\nErr-1:CL_DEVICE_NOT_FOUND")
    assert gpu.is_device_error("Fail to create context! Err(-5)\n\nErr-34:CL_INVALID_CONTEXT\n")
    assert not gpu.is_device_error("\nErr-4:CL_MEM_OBJECT_ALLOCATION_FAILURE") and not gpu.is_device_error("")
    assert not gpu.is_device_error("No valid ligands in the input directory")


def test_config_gpu_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = dict(protein_paths="p.txt", smiles_file="l.csv")
    cfg = PocketVinaConfig(**base)
    assert (cfg.gpus, cfg.procs_per_gpu, cfg.vram_per_proc_mib) == (None, "auto", 6600)
    cfg = PocketVinaConfig(**base, gpus="1, 0", procs_per_gpu="auto", vram_per_proc_mib=8000)
    assert (cfg.gpus, cfg.procs_per_gpu, cfg.vram_per_proc_mib) == ("1,0", "auto", 8000)
    assert PocketVinaConfig(**base, gpus=2).gpus == "2"
    for bad in (dict(gpus=""), dict(gpus="0,0"), dict(gpus="first"), dict(procs_per_gpu=0), dict(procs_per_gpu=1.5),
                dict(procs_per_gpu="many"), dict(vram_per_proc_mib=0), dict(vram_per_proc_mib=None),
                dict(vram_per_proc_mib=6.4)):
        with pytest.raises(ValueError):
            PocketVinaConfig(**base, **bad)


def test_plan_engine_slots(use_gpus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = dict(protein_paths="p.txt", smiles_file="l.csv", output_format="atlas")
    g0, g1 = use_gpus(fake_gpus(2))
    plan = pipeline.plan_engine_slots
    # default: the first visible GPU, as many engines as its free memory holds (24000 MiB: 3 of 6600 + 1024 margin)
    assert plan(PocketVinaConfig(**base)) == [g0.uuid] * 3
    assert plan(PocketVinaConfig(**base, gpus="all", procs_per_gpu=1)) == [g0.uuid, g1.uuid]
    assert plan(PocketVinaConfig(**base, gpus="1", procs_per_gpu=2)) == [g1.uuid] * 2
    assert plan(PocketVinaConfig(**base, gpus=g0.uuid, procs_per_gpu="auto")) == [g0.uuid] * 3
    with pytest.raises(ValueError, match="'2' is not a visible GPU"):
        plan(PocketVinaConfig(**base, gpus="2"))
    use_gpus([])  # no nvidia-smi: the default GPU
    assert plan(PocketVinaConfig(**base)) == [None]
    assert plan(PocketVinaConfig(**base, procs_per_gpu=3)) == [None] * 3
    assert plan(PocketVinaConfig(**base, procs_per_gpu="auto")) == [None]
    with pytest.raises(ValueError, match="nvidia-smi lists no GPU"):
        plan(PocketVinaConfig(**base, gpus="0"))


def test_engine_slots_keep_cuda_numbers_of_mixed_gpus(fake_smi, monkeypatch, tmp_path):
    """CUDA_VISIBLE_DEVICES=1 with a P40 and an A40 and CUDA's default device order: CUDA's GPU 1 need not be
    nvidia-smi's GPU 1, so the slots inherit the environment's value (the engines dock on the GPU it names, as
    before G2) instead of the UUID of nvidia-smi's GPU 1."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    base = dict(protein_paths="p.txt", smiles_file="l.csv", output_format="atlas", procs_per_gpu=2)
    assert pipeline.plan_engine_slots(PocketVinaConfig(**base)) == [None, None]
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    assert pipeline.plan_engine_slots(PocketVinaConfig(**base)) == ["GPU-bbbb-2222"] * 2
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb-2222")
    monkeypatch.delenv("CUDA_DEVICE_ORDER")
    assert pipeline.plan_engine_slots(PocketVinaConfig(**base)) == ["GPU-bbbb-2222"] * 2


def test_kernel_cache_per_gpu_model(fake_smi, monkeypatch, tmp_path):
    """Slots on GPUs of different models get kernel caches of their own; the caller's environment is restored."""
    monkeypatch.delenv("POCKETVINA_KERNEL_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_smi(SMI_ROWS[:2] + ["2, GPU-dddd-4444, Tesla P40, 23040, 22000"])
    p40, a40, other_p40 = (pipeline._kernel_cache_for(d) for d in ("GPU-aaaa-1111", "GPU-bbbb-2222",
                                                                    "GPU-dddd-4444"))
    assert p40 != a40 and p40 == other_p40
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert pipeline._kernel_cache_for(None) == engine.kernel_cache_dir() == p40  # the first visible GPU: a P40
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    pipeline._kernel_cache_for("GPU-bbbb-2222")
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    monkeypatch.setenv("POCKETVINA_KERNEL_CACHE", str(tmp_path / "kc"))
    assert pipeline._kernel_cache_for("GPU-aaaa-1111") == pipeline._kernel_cache_for("GPU-bbbb-2222") == str(
        tmp_path / "kc")


def test_kernel_cache_of_a_slot_never_changes_the_environment(fake_smi, monkeypatch, tmp_path):
    """A slot's kernel cache is keyed by its GPU without setting CUDA_VISIBLE_DEVICES in this process, even for a
    moment: another thread (another run in this process, or its engine slots copying the environment for their
    engines) would see the slot's GPU, or lose the variable while it is unset."""
    monkeypatch.delenv("POCKETVINA_KERNEL_CACHE", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)  # the cache under $HOME/.cache
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    fake_smi(SMI_ROWS[:2])
    expected = {d: os.path.join(str(tmp_path / "home"), ".cache", "pocketvina", "kernels", engine.kernel_cache_key(d))
                for d in ("GPU-aaaa-1111", "GPU-bbbb-2222")}
    assert expected["GPU-aaaa-1111"] != expected["GPU-bbbb-2222"]
    assert engine.kernel_cache_key() == engine.kernel_cache_key("0,1") != engine.kernel_cache_key(None)
    changes = []
    monkeypatch.setattr(os, "putenv", lambda *args: changes.append(("putenv",) + args))
    monkeypatch.setattr(os, "unsetenv", lambda *args: changes.append(("unsetenv",) + args))
    for device, path in expected.items():
        assert pipeline._kernel_cache_for(device) == engine.kernel_cache_dir(device) == path
    assert changes == [] and os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


# --- the slot pool -------------------------------------------------------------------------------------------------

def test_slots_dock_at_the_same_time_with_the_same_results(tmp_path, example_dir, fake_engine_env, stubs, use_gpus,
                                                           monkeypatch):
    """Two GPUs x two engines: calls run at the same time, each slot on its own GPU and in its own folder, the
    kernels are built once before the slots start, and the results are the same as with one engine on no GPU
    (sequential) and one engine on one GPU, byte for byte, although the scores depend on the calls' ligands."""
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SHARED_RNG", "1")
    plist, smiles = two_proteins(tmp_path, example_dir)
    pids = [FIXTURE_ID, "1fbl"]
    run(plist, smiles, tmp_path / "sequential", tmp_path, ligands_per_call=2)
    sequential = fake_engine_env.calls()
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")  # no GPU list: the engines get the environment's
    assert len(sequential) == 8 and {c["env"]["CUDA_VISIBLE_DEVICES"] for c in sequential} == {inherited}

    g0, g1 = use_gpus(fake_gpus(2))
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "0.5")
    run(plist, smiles, tmp_path / "shared", tmp_path, ligands_per_call=2, procs_per_gpu=2)
    calls = fake_engine_env.calls()[len(sequential):]
    builds = [c for c in calls if is_kernel_build(c)]
    docking = [c for c in calls if not is_kernel_build(c)]
    assert len(builds) == 1 and len(docking) == 8  # 4 slots share the (fake engine's empty) kernel cache
    assert len(builds[0]["outputs"]) == 1 and builds[0]["end"] <= min(c["start"] for c in docking)
    assert max_overlap(docking) >= 2
    assert {c["env"]["CUDA_VISIBLE_DEVICES"] for c in docking} == {g0.uuid, g1.uuid}
    slots = {c["env"]["POCKETVINA_ENGINE_SLOT"] for c in docking}
    assert len(slots) >= 2 and all(re.fullmatch(rf"{os.getpid()}-[0-9a-f]{{16}}/slot[1-4]", s) for s in slots)
    for c in docking:  # each slot works in its own folder
        slot = c["env"]["POCKETVINA_ENGINE_SLOT"].rsplit("/", 1)[1]
        assert Path(c["cwd"]).parent.name == slot and slot in Path(c["options"]["output_directory"]).parts
    assert sorted((c["options"]["receptor"].rsplit("/", 1)[1], c["options"]["pocket_rank"],
                   c["options"]["ligand_directory"].rsplit("/", 1)[1]) for c in docking) == sorted(
        (c["options"]["receptor"].rsplit("/", 1)[1], c["options"]["pocket_rank"],
         c["options"]["ligand_directory"].rsplit("/", 1)[1]) for c in sequential)
    assert_same_results(tmp_path / "shared", tmp_path / "sequential", pids)

    use_gpus(fake_gpus(1))
    monkeypatch.delenv("POCKETVINA_FAKE_ENGINE_SLEEP")
    n_before = len(fake_engine_env.calls())
    run(plist, smiles, tmp_path / "one_gpu", tmp_path, ligands_per_call=2)
    one = fake_engine_env.calls()[n_before:]
    assert len(one) == 8 and max_overlap(one) == 1  # one slot: no kernel build call either
    assert {c["env"]["CUDA_VISIBLE_DEVICES"] for c in one} == {fake_gpus(1)[0].uuid}
    assert_same_results(tmp_path / "one_gpu", tmp_path / "sequential", pids)
    ligands = read_table(tmp_path / "shared" / "tables" / "ligands.parquet")
    assert set(ligands.status) == {"ok"}


def test_protein_is_committed_while_the_next_one_runs_p2rank(tmp_path, example_dir, fake_engine_env, use_gpus,
                                                            monkeypatch):
    """The next protein is prepared while the last calls of the current one run; when those are back during the
    next protein's P2Rank run, the current protein is committed right away (a stop during P2Rank keeps it)."""
    plist, smiles = two_proteins(tmp_path, example_dir, ligands=("molecule1",))
    out = tmp_path / "results"
    use_gpus(fake_gpus(1))
    monkeypatch.setattr(pipeline, "receptor_pdbqt", cropped_receptor())
    write_pockets = stub_predict_pockets()
    committed_during_p2rank = []

    def predict_pockets(structure, work_dir, p2rank_config=None, *, while_running=None):
        if os.path.basename(structure) == "1fbl.pdb":  # the second protein: P2Rank runs until the first is committed
            deadline = time.monotonic() + 15
            while not ResultsDir(out).is_done(FIXTURE_ID) and time.monotonic() < deadline:
                if while_running is not None:
                    while_running()
                time.sleep(0.05)
            committed_during_p2rank.append(ResultsDir(out).is_done(FIXTURE_ID))
        return write_pockets(structure, work_dir, p2rank_config)

    monkeypatch.setattr(pipeline, "predict_pockets", predict_pockets)
    run(plist, smiles, out, tmp_path)
    assert committed_during_p2rank == [True]
    assert ResultsDir(out).done_proteins() == {FIXTURE_ID, "1fbl"}


def test_protein_is_committed_while_the_next_one_converts_its_receptor(tmp_path, example_dir, fake_engine_env,
                                                                     use_gpus, monkeypatch):
    """OpenBabel holds the GIL while it converts a receptor (seconds to tens of seconds): for the protein prepared
    while engine calls are due it runs in a child process, so the current protein's last call is recorded and the
    protein committed at once, not when the next protein's calls start."""
    big = crop_pdb(example_dir / "proteins" / "1fbl.pdb", tmp_path / "big.pdb", n_atoms=1200)  # not stubbed
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[FIXTURE_RECEPTOR, big])
    out = tmp_path / "results"
    use_gpus(fake_gpus(1))
    monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets())
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "0.3")
    commit, committed = ResultsDir.commit, {}

    def timed_commit(self, pid, *a, **kw):
        commit(self, pid, *a, **kw)
        committed[pid] = time.time()

    monkeypatch.setattr(ResultsDir, "commit", timed_commit)
    communicate, converted = pipeline._communicate, {}

    def timed_communicate(proc, while_running):  # the child process converting a receptor: when it ended
        stdout = communicate(proc, while_running)
        converted[Path(proc.args[3]).stem] = time.time()  # args: python -c code source dest mode
        return stdout

    monkeypatch.setattr(pipeline, "_communicate", timed_communicate)
    # a conversion that outlasts the fixture's engine calls on any machine (on a fast one it may take less than
    # 0.5 s, and end before them), polled often
    monkeypatch.setattr(pipeline, "_RECEPTOR_CODE", "import time\ntime.sleep(3)\n" + pipeline._RECEPTOR_CODE)
    monkeypatch.setattr(pipeline, "P2RANK_POLL_S", 0.05)
    run(plist, smiles, out, tmp_path)
    calls = fake_engine_env.calls()
    first_call_of_big = min(c["start"] for c in calls if c["options"]["receptor"].endswith("/big.pdbqt"))
    last_call_of_fixture = max(c["end"] for c in calls if c["options"]["receptor"].endswith(f"/{FIXTURE_ID}.pdbqt"))
    assert committed[FIXTURE_ID] - last_call_of_fixture < 2.0
    # committed while big's receptor was still converting (in the child process), so before big's calls started;
    # not a fixed margin: how long the conversion takes depends on the machine
    assert set(converted) == {"big"}
    assert committed[FIXTURE_ID] < converted["big"] <= first_call_of_big, (committed, converted, first_call_of_big)
    assert set(committed) == {FIXTURE_ID, "big"}  # the receptor bytes: test_receptor_pdbqt_in_a_child_process


def test_next_protein_is_prepared_while_the_slots_dock_all_calls_of_the_first(tmp_path, example_dir, fake_engine_env,
                                                                              use_gpus, monkeypatch):
    """As many slots as the first protein has calls: once the slots have taken them, none is queued, so the next
    protein is prepared while they dock (its P2Rank run starts before the first call is back), not after it, when
    every slot would wait for it."""
    plist, smiles = two_proteins(tmp_path, example_dir, ligands=("molecule1",))
    use_gpus(fake_gpus(1))
    monkeypatch.setattr(pipeline, "receptor_pdbqt", cropped_receptor())
    write_pockets, p2rank_started = stub_predict_pockets(), {}

    def predict_pockets(structure, work_dir, p2rank_config=None, *, while_running=None):
        p2rank_started[os.path.basename(structure)] = time.time()
        return write_pockets(structure, work_dir, p2rank_config)

    monkeypatch.setattr(pipeline, "predict_pockets", predict_pockets)
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "2.5")
    cache = Path(os.environ["POCKETVINA_KERNEL_CACHE"])
    cache.mkdir(parents=True)
    for name in engine.KERNEL_FILES:  # no kernel build call
        (cache / name).write_bytes(b"kernel")
    run(plist, smiles, tmp_path / "results", tmp_path, procs_per_gpu=2)  # 2 slots, 2 pockets per protein
    calls = fake_engine_env.calls()
    first = [c for c in calls if c["options"]["receptor"].endswith(f"/{FIXTURE_ID}.pdbqt")]
    assert len(first) == 2 and max_overlap(first) == 2
    assert p2rank_started["1fbl.pdb"] < min(c["end"] for c in first) - 0.5, (p2rank_started, first)
    assert ResultsDir(tmp_path / "results").done_proteins() == {FIXTURE_ID, "1fbl"}


def test_kernel_build_only_when_needed(tmp_path, example_dir, fake_engine_env, stubs, use_gpus):
    """No kernel build call when the shared cache holds both kernels, or when each slot has a cache of its own."""
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[FIXTURE_RECEPTOR])
    use_gpus(fake_gpus(1))
    cache = Path(os.environ["POCKETVINA_KERNEL_CACHE"])
    cache.mkdir(parents=True)
    for name in engine.KERNEL_FILES:
        (cache / name).write_bytes(b"kernel")
    run(plist, smiles, tmp_path / "cached", tmp_path, procs_per_gpu=2)
    calls = fake_engine_env.calls()
    assert len(calls) == 2 and not any(is_kernel_build(c) for c in calls)
    (cache / engine.KERNEL_FILES[1]).unlink()
    run(plist, smiles, tmp_path / "half", tmp_path, procs_per_gpu=2)
    assert [is_kernel_build(c) for c in fake_engine_env.calls()[2:]] == [True, False, False]


def test_kernel_build_skips_a_first_ligand_the_engine_rejects(tmp_path, example_dir, fake_engine_env, stubs, use_gpus,
                                                              monkeypatch):
    """The engine builds its kernels only after it has parsed a ligand ("No valid ligands" exits before): when it
    rejects the first prepared ligand, the kernel build goes on with the next one, so that the slots never build
    the kernels at the same time. The engine here builds them as the real one does (a call with a valid ligand and
    no kernels in its cache) and logs each build."""
    real_prepare = pipeline.prepare_ligands

    def first_without_torsdof(rows, **kw):  # the engine rejects a ligand without TORSDOF
        records = real_prepare(rows, **kw)
        records[0].pdbqt = "".join(line for line in records[0].pdbqt.splitlines(True) if "TORSDOF" not in line)
        return records

    monkeypatch.setattr(pipeline, "prepare_ligands", first_without_torsdof)
    builds = tmp_path / "builds.log"
    body = ("opts = dict(line.split(' = ', 1) for line in config.splitlines())\n"
            "rc = subprocess.call([" + repr(str(fake_engine_env)) + "] + argv)\n"
            "kernels = [os.path.join(opts['opencl_binary_path'], n) for n in ('Kernel1_Opt.bin', 'Kernel2_Opt.bin')]\n"
            "if os.listdir(opts['output_directory']) and not all(os.path.isfile(k) for k in kernels):\n"
            "    os.makedirs(opts['opencl_binary_path'], exist_ok=True)\n"
            "    for k in kernels:\n"
            "        open(k, 'wb').write(b'kernel')\n"
            f"    open({str(builds)!r}, 'a').write(os.path.basename(opts['ligand_directory']) + '\\n')\n"
            "sys.exit(rc)")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(wrapper_engine(tmp_path, fake_engine_env, body)))
    use_gpus(fake_gpus(1))
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2", "molecule3"),
                                proteins=[FIXTURE_RECEPTOR])
    run(plist, smiles, tmp_path / "results", tmp_path, procs_per_gpu=2)
    assert builds.read_text().splitlines() == ["kernel_build"]  # built once, before the slots started
    calls = fake_engine_env.calls()
    assert [len(c["outputs"]) for c in calls if is_kernel_build(c)] == [0, 1]  # rejected, then built
    runs = read_table(tmp_path / "results" / "tables" / "pocket_runs" / f"{FIXTURE_ID}.parquet")
    assert sorted(runs.status) == ["no_output"] * 2 + ["ok"] * 4


def test_out_of_memory_retires_a_slot_and_retries_the_call(tmp_path, example_dir, fake_engine_env, stubs, use_gpus,
                                                           monkeypatch, caplog):
    """A GPU that holds one engine only: the second slot's engine runs out of GPU memory, that slot retires, its call
    is docked by the other slot, and nothing is lost."""
    plist, smiles = two_proteins(tmp_path, example_dir)
    pids = [FIXTURE_ID, "1fbl"]
    run(plist, smiles, tmp_path / "reference", tmp_path)
    n_ref = len(fake_engine_env.calls())
    oom_log = tmp_path / "oom.log"
    body = (f"if env.get('POCKETVINA_ENGINE_SLOT', '').endswith('/slot2'):\n"
            f"    open({str(oom_log)!r}, 'a').write(config)\n"
            f"    print('\\n{OOM_LINE}', flush=True)\n"
            f"    sys.exit(255)")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(wrapper_engine(tmp_path, fake_engine_env, body)))
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "0.3")  # slot1 is busy when slot2 takes its first call
    (g0,) = use_gpus(fake_gpus(1))
    pv = run(plist, smiles, tmp_path / "oom", tmp_path, procs_per_gpu=2)
    assert oom_log.read_text().count("receptor = ") == 1  # slot2 ran (and lost) exactly one call
    calls = [c for c in fake_engine_env.calls()[n_ref:] if not is_kernel_build(c)]
    assert len(calls) == 4 and {c["env"]["POCKETVINA_ENGINE_SLOT"].rsplit("/", 1)[1] for c in calls} == {"slot1"}
    assert f"the engine ran out of GPU memory ({OOM_LINE}); GPU {g0.uuid} runs 1 engine from now on" in caplog.text
    assert_same_results(tmp_path / "oom", tmp_path / "reference", pids)
    for pid in pids:
        summary = json.loads((tmp_path / "oom" / "manifest" / f"{pid}.json").read_text())["summary"]
        assert (summary["engine_runs"], summary["engine_runs_failed"]) == (2, 0)
    assert pv.failed_proteins == {}


def test_out_of_memory_on_a_gpu_of_its_own_moves_its_calls_to_the_other_gpus(tmp_path, example_dir, fake_engine_env,
                                                                           stubs, use_gpus, monkeypatch, caplog):
    """Two GPUs, one engine each (the default), and the second one cannot hold an engine (too small, or busy with
    other work): its slot retires after its first call, which the first GPU docks, so nothing is lost."""
    plist, smiles = two_proteins(tmp_path, example_dir)
    pids = [FIXTURE_ID, "1fbl"]
    run(plist, smiles, tmp_path / "reference", tmp_path)
    n_ref = len(fake_engine_env.calls())
    g0, g1 = use_gpus(fake_gpus(2))
    oom_log = tmp_path / "oom.log"
    body = (f"if env.get('CUDA_VISIBLE_DEVICES') == {g1.uuid!r}:\n"
            f"    open({str(oom_log)!r}, 'a').write(config)\n"
            f"    print('\\n{OOM_LINE}', flush=True)\n"
            f"    sys.exit(255)")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(wrapper_engine(tmp_path, fake_engine_env, body)))
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "0.3")  # slot1 is busy when slot2 takes its first call
    pv = run(plist, smiles, tmp_path / "oom", tmp_path)
    assert oom_log.read_text().count("receptor = ") == 1  # the second GPU ran (and lost) exactly one call
    calls = [c for c in fake_engine_env.calls()[n_ref:] if not is_kernel_build(c)]
    assert len(calls) == 4 and {c["env"]["CUDA_VISIBLE_DEVICES"] for c in calls} == {g0.uuid}
    assert f"GPU {g1.uuid} is not used from now on, the call is retried on the other GPUs" in caplog.text
    assert_same_results(tmp_path / "oom", tmp_path / "reference", pids)
    for pid in pids:
        summary = json.loads((tmp_path / "oom" / "manifest" / f"{pid}.json").read_text())["summary"]
        assert (summary["engine_runs"], summary["engine_runs_failed"]) == (2, 0)
    assert pv.failed_proteins == {}


# what the real engine prints when OpenCL cannot use its GPU: no platform (CUDA_VISIBLE_DEVICES names no GPU it can
# see), or no context (wrapcl.cpp SetupContext prints the code, then fails on the NULL context)
DEVICE_LINES = {"no_platform": "Err-1001:CL_PLATFORM_NOT_FOUND_KHR",
                "no_context": "Fail to create context! Err(-5)\\n\\nErr-34:CL_INVALID_CONTEXT"}


@pytest.mark.parametrize("kind", sorted(DEVICE_LINES))
def test_gpu_the_engine_cannot_use_moves_its_calls_to_the_other_gpus(tmp_path, example_dir, fake_engine_env, stubs,
                                                                    use_gpus, monkeypatch, caplog, kind):
    """Two GPUs, one engine each (the default), and OpenCL cannot set up the second one (busy in an exclusive compute
    mode, no memory left for a context, a GPU it does not see): the engine fails before docking anything, whatever
    the call, so the slot retires as after an out-of-memory error and the other GPU docks its calls; nothing is
    committed with engine_error runs that a rerun would not retry."""
    plist, smiles = two_proteins(tmp_path, example_dir)
    pids = [FIXTURE_ID, "1fbl"]
    run(plist, smiles, tmp_path / "reference", tmp_path)
    n_ref = len(fake_engine_env.calls())
    g0, g1 = use_gpus(fake_gpus(2))
    failed_log = tmp_path / "failed.log"
    body = (f"if env.get('CUDA_VISIBLE_DEVICES') == {g1.uuid!r}:\n"
            f"    open({str(failed_log)!r}, 'a').write(config)\n"
            f"    print('\\n{DEVICE_LINES[kind]}', flush=True)\n"
            f"    sys.exit(255)")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(wrapper_engine(tmp_path, fake_engine_env, body)))
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "0.3")  # slot1 is busy when slot2 takes its first call
    run(plist, smiles, tmp_path / "results", tmp_path)
    assert failed_log.read_text().count("receptor = ") == 1
    calls = [c for c in fake_engine_env.calls()[n_ref:] if not is_kernel_build(c)]
    assert len(calls) == 4 and {c["env"]["CUDA_VISIBLE_DEVICES"] for c in calls} == {g0.uuid}
    assert "the engine could not use the GPU (" in caplog.text
    assert f"GPU {g1.uuid} is not used from now on, the call is retried on the other GPUs" in caplog.text
    assert_same_results(tmp_path / "results", tmp_path / "reference", pids)
    for pid in pids:
        summary = json.loads((tmp_path / "results" / "manifest" / f"{pid}.json").read_text())["summary"]
        assert (summary["engine_runs"], summary["engine_runs_failed"]) == (2, 0)


def test_kernel_build_moves_on_from_a_gpu_the_engine_cannot_use(tmp_path, example_dir, fake_engine_env, stubs,
                                                                 use_gpus, monkeypatch):
    """Two GPUs x two engines share a cold kernel cache and OpenCL cannot set up the first GPU: the kernel build
    call goes on on the second GPU, so the slots there do not build the kernels at the same time; the first GPU's
    slots retire on their first calls and the second GPU docks everything."""
    builds = tmp_path / "builds.log"
    g0, g1 = use_gpus(fake_gpus(2))
    body = (f"if env.get('CUDA_VISIBLE_DEVICES') == {g0.uuid!r}:\n"
            f"    print('\\n{DEVICE_LINES['no_context']}', flush=True)\n"
            "    sys.exit(255)\n"
            "opts = dict(line.split(' = ', 1) for line in config.splitlines())\n"
            "rc = subprocess.call([" + repr(str(fake_engine_env)) + "] + argv)\n"
            "kernels = [os.path.join(opts['opencl_binary_path'], n) for n in ('Kernel1_Opt.bin', 'Kernel2_Opt.bin')]\n"
            "if os.listdir(opts['output_directory']) and not all(os.path.isfile(k) for k in kernels):\n"
            "    os.makedirs(opts['opencl_binary_path'], exist_ok=True)\n"
            "    for k in kernels:\n"
            "        open(k, 'wb').write(b'kernel')\n"
            f"    open({str(builds)!r}, 'a').write(os.path.basename(opts['ligand_directory']) + ' ' + "
            "env['CUDA_VISIBLE_DEVICES'] + '\\n')\n"
            "sys.exit(rc)")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(wrapper_engine(tmp_path, fake_engine_env, body)))
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "0.3")
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"), proteins=[FIXTURE_RECEPTOR])
    run(plist, smiles, tmp_path / "results", tmp_path, procs_per_gpu=2)
    assert builds.read_text().splitlines() == [f"kernel_build {g1.uuid}"]  # once, before the slots started
    calls = [c for c in fake_engine_env.calls() if not is_kernel_build(c)]
    assert len(calls) == 2 and {c["env"]["CUDA_VISIBLE_DEVICES"] for c in calls} == {g1.uuid}
    runs = read_table(tmp_path / "results" / "tables" / "pocket_runs" / f"{FIXTURE_ID}.parquet")
    assert set(runs.status) == {"ok"} and len(runs) == 4


def test_gpu_the_engine_cannot_use_alone_fails_the_protein(tmp_path, example_dir, fake_engine_env, stubs, use_gpus,
                                                           monkeypatch):
    """With every slot unable to use its GPU the calls fail as before G2: no protein is committed, a rerun retries."""
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[FIXTURE_RECEPTOR])
    body = f"print('\\n{DEVICE_LINES['no_platform']}', flush=True)\nsys.exit(255)"
    monkeypatch.setenv("POCKETVINA_ENGINE", str(wrapper_engine(tmp_path, fake_engine_env, body)))
    use_gpus(fake_gpus(2))
    cache = Path(os.environ["POCKETVINA_KERNEL_CACHE"])
    cache.mkdir(parents=True)
    for name in engine.KERNEL_FILES:  # no kernel build call
        (cache / name).write_bytes(b"kernel")
    pv = PocketVinaGPU(atlas_config(plist, smiles, tmp_path / "results", tmp_path / "tmp"))
    pipeline.run_atlas(pv)
    assert list(pv.failed_proteins) == [FIXTURE_ID] and "CL_PLATFORM_NOT_FOUND_KHR" in pv.failed_proteins[FIXTURE_ID]
    assert ResultsDir(tmp_path / "results").done_proteins() == set()


@pytest.mark.parametrize("n_gpus", [2, 4])
def test_no_gpu_the_engine_can_use_fails_every_protein(tmp_path, example_dir, fake_engine_env, stubs, use_gpus,
                                                      monkeypatch, n_gpus):
    """No slot can use its GPU (e.g. an OpenCL loader that finds no platform), several GPUs, two proteins of two
    pockets: the run ends with both proteins failed, it does not keep retiring and restarting the slots."""
    plist, smiles = two_proteins(tmp_path, example_dir)
    body = f"print('\\n{DEVICE_LINES['no_platform']}', flush=True)\nsys.exit(255)"
    monkeypatch.setenv("POCKETVINA_ENGINE", str(wrapper_engine(tmp_path, fake_engine_env, body)))
    monkeypatch.setenv("OCL_ICD_VENDORS", "/etc/OpenCL/vendors")
    use_gpus(fake_gpus(n_gpus))
    cache = Path(os.environ["POCKETVINA_KERNEL_CACHE"])
    cache.mkdir(parents=True)
    for name in engine.KERNEL_FILES:  # no kernel build call
        (cache / name).write_bytes(b"kernel")
    pv = PocketVinaGPU(atlas_config(plist, smiles, tmp_path / "results", tmp_path / "tmp"))
    pipeline.run_atlas(pv)
    assert sorted(pv.failed_proteins) == sorted([FIXTURE_ID, "1fbl"])
    for reason in pv.failed_proteins.values():
        assert "CL_PLATFORM_NOT_FOUND_KHR" in reason and "OCL_ICD_VENDORS is set (/etc/OpenCL/vendors)" in reason
    assert ResultsDir(tmp_path / "results").done_proteins() == set()


def test_opencl_hint():
    log = "Analyzing the binding site ... done.\n\nErr-1001:CL_PLATFORM_NOT_FOUND_KHR\n"
    assert "unset OCL_ICD_VENDORS" in gpu.opencl_hint(log, {"OCL_ICD_VENDORS": "/etc/OpenCL/vendors"})
    assert gpu.opencl_hint(log, {}) is None
    assert gpu.opencl_hint("Err-4:CL_MEM_OBJECT_ALLOCATION_FAILURE", {"OCL_ICD_VENDORS": "/etc/OpenCL/vendors"}) is None


# --- OCL_ICD_VENDORS: engine.run_engine docks again without it when the engine finds no OpenCL platform with it ------

VENDORS = "/etc/OpenCL/vendors"


def vendors_engine(tmp_path, fake_engine_env, fail="env.get('OCL_ICD_VENDORS')", line=DEVICE_LINES["no_platform"]):
    """(engine, log): an engine that prints ``line`` and exits 255 when ``fail`` (Python) is true, by default while
    $OCL_ICD_VENDORS is set (the prebuilt engine with the CUDA toolkit's libOpenCL.so.1); the log gets the
    OCL_ICD_VENDORS of every call ("-": unset)."""
    log = tmp_path / "vendors.log"
    body = (f"open({str(log)!r}, 'a').write(env.get('OCL_ICD_VENDORS', '-') + '\\n')\n"
            f"if {fail}:\n"
            f"    print('\\n{line}', flush=True)\n"
            "    sys.exit(255)")
    return wrapper_engine(tmp_path, fake_engine_env, body, name="vendors_engine"), log


def engine_call(tmp_path, real_run, name):
    """Config of a call docking one fixture ligand into the fixture receptor's pocket 1, in ``tmp_path/name``."""
    lig_dir = tmp_path / name / "ligands"
    lig_dir.mkdir(parents=True)
    shutil.copy(real_run.ligand_pdbqt("molecule2"), lig_dir / "L0000000.pdbqt")
    return engine.write_config(tmp_path / name / "config.txt", real_run.receptor, lig_dir, tmp_path / name / "out",
                               real_run.pocket_center(1), (20, 20, 20), 8000, 1, seed=42)


def vendor_warnings(caplog) -> list:
    return [r.getMessage() for r in caplog.records
            if r.name == "pocketvina.engine" and r.levelno == logging.WARNING and "OCL_ICD_VENDORS" in r.getMessage()]


def test_engine_docks_without_ocl_icd_vendors_when_it_hides_the_gpu(tmp_path, real_run, fake_engine_env, monkeypatch,
                                                                   caplog):
    """The engine finds no OpenCL platform while OCL_ICD_VENDORS is set: run_engine runs the call again without it,
    later calls go without it at once, and one warning says so; the caller's environment is not changed."""
    exe, log = vendors_engine(tmp_path, fake_engine_env)
    monkeypatch.setenv("POCKETVINA_ENGINE", str(exe))
    monkeypatch.setenv("OCL_ICD_VENDORS", VENDORS)
    assert engine.engine_env()["OCL_ICD_VENDORS"] == VENDORS and not engine.ocl_vendors_dropped()
    first = engine.run_engine(engine_call(tmp_path, real_run, "a"), cwd=tmp_path / "cwd", log_path=tmp_path / "a.log")
    assert first.ok and "CL_PLATFORM_NOT_FOUND_KHR" not in first.log_tail
    assert "CL_PLATFORM_NOT_FOUND_KHR" not in (tmp_path / "a.log").read_text()  # the log of the call that worked
    assert log.read_text().splitlines() == [VENDORS, "-"]
    assert engine.collect_outputs(tmp_path / "a" / "out", real_run.receptor, 1, ["L0000000"])["L0000000"]
    assert engine.ocl_vendors_dropped() and "OCL_ICD_VENDORS" not in engine.engine_env()
    assert os.environ["OCL_ICD_VENDORS"] == VENDORS
    second = engine.run_engine(engine_call(tmp_path, real_run, "b"), cwd=tmp_path / "cwd")
    assert second.ok and log.read_text().splitlines() == [VENDORS, "-", "-"]
    (warning,) = vendor_warnings(caplog)
    assert f"OCL_ICD_VENDORS={VENDORS}" in warning and "left out of the engine's environment from now on" in warning
    # a later failure without a platform has another cause: no advice to unset what the engine does not get
    assert gpu.opencl_hint(DEVICE_LINES["no_platform"]) is None
    assert gpu.opencl_hint(DEVICE_LINES["no_platform"], {"OCL_ICD_VENDORS": VENDORS})  # explicit environ: as before


def test_engine_keeps_ocl_icd_vendors_when_the_retry_fails_too(tmp_path, real_run, fake_engine_env, monkeypatch,
                                                                caplog):
    """No platform without OCL_ICD_VENDORS either: each call is tried with and without it, fails, and the variable is
    kept (no warning); the hint to unset it still comes."""
    exe, log = vendors_engine(tmp_path, fake_engine_env, fail="True")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(exe))
    monkeypatch.setenv("OCL_ICD_VENDORS", VENDORS)
    for n, name in enumerate("ab", 1):
        res = engine.run_engine(engine_call(tmp_path, real_run, name), cwd=tmp_path / "cwd")
        assert res.returncode == 255 and "CL_PLATFORM_NOT_FOUND_KHR" in res.log_tail
        assert log.read_text().splitlines() == [VENDORS, "-"] * n
    assert not engine.ocl_vendors_dropped() and engine.engine_env()["OCL_ICD_VENDORS"] == VENDORS
    assert vendor_warnings(caplog) == [] and fake_engine_env.calls() == []
    assert "unset OCL_ICD_VENDORS" in gpu.opencl_hint(res.log_tail)


@pytest.mark.parametrize("case", ["unset", "other_error", "timeout"])
def test_engine_is_not_retried_for_other_failures(tmp_path, real_run, fake_engine_env, monkeypatch, caplog, case):
    """Only a call that found no platform with OCL_ICD_VENDORS set is tried again: not without the variable, not
    after another error (out of GPU memory), not after a timeout."""
    if case == "unset":
        exe, log = vendors_engine(tmp_path, fake_engine_env, fail="True")
    elif case == "other_error":
        exe, log = vendors_engine(tmp_path, fake_engine_env, fail="True", line=OOM_LINE)
    else:
        exe, log = vendors_engine(tmp_path, fake_engine_env, fail="False")
        monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "30")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(exe))
    if case != "unset":
        monkeypatch.setenv("OCL_ICD_VENDORS", VENDORS)
    res = engine.run_engine(engine_call(tmp_path, real_run, "a"), cwd=tmp_path / "cwd",
                            timeout=1 if case == "timeout" else None)
    assert not res.ok and res.timed_out == (case == "timeout")
    assert log.read_text().splitlines() == ["-" if case == "unset" else VENDORS]
    assert not engine.ocl_vendors_dropped() and vendor_warnings(caplog) == []


def test_atlas_run_docks_without_ocl_icd_vendors(tmp_path, example_dir, fake_engine_env, stubs, use_gpus, monkeypatch,
                                                 caplog):
    """An atlas run on two GPUs x two engines with OCL_ICD_VENDORS set and an engine that finds no platform with it:
    the engines that start at the same time all dock on their second try, one warning, the same results as without
    the variable, and no slot is retired."""
    plist, smiles = two_proteins(tmp_path, example_dir)
    pids = [FIXTURE_ID, "1fbl"]
    run(plist, smiles, tmp_path / "reference", tmp_path)
    exe, log = vendors_engine(tmp_path, fake_engine_env)
    monkeypatch.setenv("POCKETVINA_ENGINE", str(exe))
    monkeypatch.setenv("OCL_ICD_VENDORS", VENDORS)
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "0.3")
    use_gpus(fake_gpus(2))
    cache = Path(os.environ["POCKETVINA_KERNEL_CACHE"])
    cache.mkdir(parents=True, exist_ok=True)
    for name in engine.KERNEL_FILES:  # warm: no kernel build call first, the four slots start together
        (cache / name).write_bytes(b"kernel")
    run(plist, smiles, tmp_path / "results", tmp_path, procs_per_gpu=2)
    assert_same_results(tmp_path / "results", tmp_path / "reference", pids)
    tries = log.read_text().splitlines()
    assert tries.count("-") == 4 and 1 <= tries.count(VENDORS) <= 4  # every call docked once, without the variable
    assert len(vendor_warnings(caplog)) == 1 and "is not used from now on" not in caplog.text
    for pid in pids:
        summary = json.loads((tmp_path / "results" / "manifest" / f"{pid}.json").read_text())["summary"]
        assert (summary["engine_runs"], summary["engine_runs_failed"]) == (2, 0)


def test_files_format_docks_without_ocl_icd_vendors(tmp_path, example_dir, fake_engine_env, monkeypatch, caplog):
    """The "files" format runs the engine through run_engine too: the same retry."""
    db, protein_list = make_database(tmp_path)
    _, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"))
    exe, log = vendors_engine(tmp_path, fake_engine_env)
    monkeypatch.setenv("POCKETVINA_ENGINE", str(exe))
    monkeypatch.setenv("OCL_ICD_VENDORS", VENDORS)
    out = tmp_path / "out"
    PocketVinaGPU(PocketVinaConfig(database_path=str(db), protein_list=str(protein_list), smiles_file=str(smiles),
                                   output_dir=str(out))).execute_docking()
    check_database_combined_csv(out, ("molecule1", "molecule2"))
    assert log.read_text().splitlines() == [VENDORS, "-", "-"]  # two pockets: the first call is retried
    assert len(vendor_warnings(caplog)) == 1


def test_out_of_memory_alone_is_an_engine_error(tmp_path, example_dir, fake_engine_env, stubs, use_gpus, monkeypatch):
    """A call that runs out of memory with its GPU to itself is not retried again: its ligands are engine_error runs
    (the other pocket is docked), the protein is committed."""
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"), proteins=[FIXTURE_RECEPTOR])
    tries = tmp_path / "tries.log"
    body = (f"if 'pocket_rank = 2' in config:\n"
            f"    open({str(tries)!r}, 'a').write(env['POCKETVINA_ENGINE_SLOT'] + '\\n')\n"
            f"    print('\\n{OOM_LINE}', flush=True)\n"
            f"    sys.exit(255)")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(wrapper_engine(tmp_path, fake_engine_env, body)))
    use_gpus(fake_gpus(1))
    run(plist, smiles, tmp_path / "results", tmp_path, procs_per_gpu=2)
    assert len(tries.read_text().splitlines()) == 2  # with two engines (retried), then alone (not retried)
    runs = read_table(tmp_path / "results" / "tables" / "pocket_runs" / f"{FIXTURE_ID}.parquet")
    assert set(runs[runs.pocket == 1].status) == {"ok"}
    bad = runs[runs.pocket == 2]
    assert set(bad.status) == {"engine_error"} and all(OOM_LINE in e for e in bad.error)
    prot = read_table(tmp_path / "results" / "tables" / "proteins" / f"{FIXTURE_ID}.parquet").iloc[0]
    assert prot.error == "1 of 2 engine runs failed"


def test_call_out_of_memory_on_every_gpu_gives_the_slots_back(tmp_path, example_dir, fake_engine_env, stubs,
                                                              use_gpus, monkeypatch, caplog):
    """Three GPUs, one engine each, and calls that fail with CL_OUT_OF_RESOURCES wherever they run (NVIDIA's error
    for a kernel fault on some input, too): each such call retires the slots it ran in until it runs out of memory
    alone, which shows that the call is the problem, not the GPUs. Those slots then dock again: the next protein
    runs on all three GPUs (it ran on one only, for the rest of the run), with the same results."""
    plist, smiles = two_proteins(tmp_path, example_dir, ligands=("molecule1", "molecule2"))
    run(plist, smiles, tmp_path / "reference", tmp_path, ligands_per_call=1)
    n_ref = len(fake_engine_env.calls())
    body = (f"if 'pocket_rank = 2' in config and '/{FIXTURE_ID}.pdbqt' in config:\n"
            f"    print('\\nErr-5:CL_OUT_OF_RESOURCES', flush=True)\n"
            f"    sys.exit(255)")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(wrapper_engine(tmp_path, fake_engine_env, body)))
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_SLEEP", "0.3")
    gpus = use_gpus(fake_gpus(3))
    run(plist, smiles, tmp_path / "results", tmp_path, ligands_per_call=1)
    assert re.search(r"CL_OUT_OF_RESOURCES\) also alone, wherever it ran: the call fails, not the GPUs, so the [12] "
                     r"engine slots? it retired dock again", caplog.text)
    later = [c for c in fake_engine_env.calls()[n_ref:] if c["options"]["receptor"].endswith("/1fbl.pdbqt")]
    assert len(later) == 4 and {c["env"]["CUDA_VISIBLE_DEVICES"] for c in later} == {g.uuid for g in gpus}
    assert_same_results(tmp_path / "results", tmp_path / "reference", ["1fbl"])
    runs = read_table(tmp_path / "results" / "tables" / "pocket_runs" / f"{FIXTURE_ID}.parquet")
    assert set(runs[runs.pocket == 1].status) == {"ok"}
    bad = runs[runs.pocket == 2]
    assert set(bad.status) == {"engine_error"} and all("CL_OUT_OF_RESOURCES" in e for e in bad.error)


def _our_engines() -> list:
    """pids of live processes started by this process's slot pools (their ``SLOT_ENV`` names this pid)."""
    needle = f"{pipeline.SLOT_ENV}={os.getpid()}-".encode()
    found = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/environ", "rb") as f:
                if any(v.startswith(needle) for v in f.read().split(b"\0")):
                    found.append(int(p))
        except OSError:
            continue
    return found


@pytest.mark.skipif(not os.path.isdir("/proc/self"), reason="the slot pool kills engines through /proc (Linux)")
def test_interrupted_run_kills_its_engines_and_resumes(tmp_path, example_dir, fake_engine_env, stubs, use_gpus,
                                                      monkeypatch):
    """Ctrl-C while the slots dock the second protein: the run stops at once and kills the engines (here hanging for
    60 s), removes its temp files and keeps the first protein; a rerun with other slot settings docks the rest, with
    the same results as a run that was not stopped."""
    plist, smiles = two_proteins(tmp_path, example_dir, ligands=("molecule1", "molecule2"))
    run(plist, smiles, tmp_path / "reference", tmp_path)
    n_ref = len(fake_engine_env.calls())
    hang = wrapper_engine(tmp_path, fake_engine_env, "if '1fbl.pdbqt' in config:\n    import time; time.sleep(60)",
                          name="hanging_engine")
    monkeypatch.setenv("POCKETVINA_ENGINE", str(hang))
    use_gpus(fake_gpus(2))
    commit = ResultsDir.commit
    t_stop = []

    def interrupted_commit(self, pid, *a, **kw):
        commit(self, pid, *a, **kw)
        deadline = time.monotonic() + 30
        while len(_our_engines()) < 2 and time.monotonic() < deadline:  # the 1fbl calls are running
            time.sleep(0.05)
        t_stop.append(time.monotonic())
        raise KeyboardInterrupt

    monkeypatch.setattr(ResultsDir, "commit", interrupted_commit)
    pv = PocketVinaGPU(atlas_config(plist, smiles, tmp_path / "results", tmp_path / "tmp", procs_per_gpu=2))
    with pytest.raises(KeyboardInterrupt):
        pipeline.run_atlas(pv)
    assert time.monotonic() - t_stop[0] < 15
    assert _our_engines() == [] and pv.failed_proteins == {}
    assert os.listdir(tmp_path / "tmp") == []
    assert ResultsDir(tmp_path / "results").done_proteins() == {FIXTURE_ID}

    monkeypatch.setattr(ResultsDir, "commit", commit)
    monkeypatch.setenv("POCKETVINA_ENGINE", str(fake_engine_env))
    use_gpus(fake_gpus(1))
    n_before = len(fake_engine_env.calls())
    run(plist, smiles, tmp_path / "results", tmp_path)
    new = fake_engine_env.calls()[n_before:]
    assert len(new) == 2 and all(c["options"]["receptor"].endswith("/1fbl.pdbqt") for c in new)
    assert_same_results(tmp_path / "results", tmp_path / "reference", [FIXTURE_ID, "1fbl"])
    assert n_ref == 4  # 2 proteins x 2 pockets


def test_missing_engine_with_several_slots_aborts_the_run(tmp_path, example_dir, fake_engine_env, stubs, use_gpus,
                                                          monkeypatch):
    plist, smiles = two_proteins(tmp_path, example_dir, ligands=("molecule1",))
    monkeypatch.setenv("POCKETVINA_ENGINE", str(tmp_path / "no_such_engine"))
    use_gpus(fake_gpus(2))
    cache = Path(os.environ["POCKETVINA_KERNEL_CACHE"])
    cache.mkdir(parents=True)
    for name in engine.KERNEL_FILES:  # no kernel build call: the slots meet the missing engine
        (cache / name).write_bytes(b"kernel")
    pv = PocketVinaGPU(atlas_config(plist, smiles, tmp_path / "results", tmp_path / "tmp", procs_per_gpu=2))
    with pytest.raises(pipeline.RunAborted, match="no_such_engine"):
        pipeline.run_atlas(pv)
    assert ResultsDir(tmp_path / "results").done_proteins() == set()
    assert os.listdir(tmp_path / "tmp") == []


def test_invalid_gpus_fail_before_ligand_preparation(tmp_path, example_dir, fake_engine_env, stubs, use_gpus,
                                                     monkeypatch):
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[FIXTURE_RECEPTOR])
    use_gpus(fake_gpus(1))
    monkeypatch.setattr(pipeline, "prepare_ligands", lambda *a, **kw: pytest.fail("ligands prepared"))
    with pytest.raises(ValueError, match="'3' is not a visible GPU"):
        pipeline.run_atlas(atlas_config(plist, smiles, tmp_path / "results", tmp_path / "tmp", gpus="3"))
    assert not (tmp_path / "results").exists() or os.listdir(tmp_path / "results") == []


def test_slot_settings_are_not_run_parameters(tmp_path, example_dir, fake_engine_env, stubs, use_gpus):
    """procs_per_gpu, gpus and vram_per_proc_mib do not change the results: not in MANIFEST params, a run may be
    resumed with others."""
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",), proteins=[FIXTURE_RECEPTOR])
    use_gpus(fake_gpus(2))
    run(plist, smiles, tmp_path / "results", tmp_path, procs_per_gpu=2, gpus="1", vram_per_proc_mib=5000)
    params = ResultsDir(tmp_path / "results").run_manifest()["params"]
    assert not {"procs_per_gpu", "gpus", "vram_per_proc_mib"} & set(params)
    run(plist, smiles, tmp_path / "results", tmp_path)  # same run: nothing left to dock
    ligands = read_table(tmp_path / "results" / "tables" / "ligands.parquet")
    check_protein(tmp_path / "results", FIXTURE_ID, ligands, n_pockets=22, K=2, docked=(1, 2))


# --- real engine ---------------------------------------------------------------------------------------------------

def _sample_compute_apps(stop: threading.Event, samples: list):
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        samples.append([line for line in out.splitlines() if engine.ENGINE_NAME in line])
        stop.wait(0.3)


@pytest.mark.gpu
def test_real_engine_two_processes_on_one_gpu(tmp_path, example_dir, monkeypatch):
    """procs_per_gpu=2 on one GPU: two engines run at the same time (nvidia-smi lists both), every run is docked,
    and the poses are the same as with one engine."""
    monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets())
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=[f"molecule{i}" for i in (1, 2, 3, 5, 6, 7)],
                                proteins=[FIXTURE_RECEPTOR])
    assert gpu.list_gpus(), "no GPU visible"
    samples, stop = [], threading.Event()
    sampler = threading.Thread(target=_sample_compute_apps, args=(stop, samples), daemon=True)
    sampler.start()
    try:
        run(plist, smiles, tmp_path / "two", tmp_path, procs_per_gpu=2)
    finally:
        stop.set()
        sampler.join()
    assert max(len(s) for s in samples) >= 2, samples
    runs = read_table(tmp_path / "two" / "tables" / "pocket_runs" / f"{FIXTURE_ID}.parquet")
    assert len(runs) == 12 and set(runs.status) == {"ok"}
    run(plist, smiles, tmp_path / "one", tmp_path)
    assert_same_results(tmp_path / "two", tmp_path / "one", [FIXTURE_ID])


def test_memory_sampler_and_format(monkeypatch):
    """Peaks of host memory (this process) and of the GPU memory nvidia-smi lists for the engines, per GPU and on all
    GPUs; a record ends with untrack, the run record stays."""
    a, b = os.getpid() + 10 ** 6, os.getpid() + 10 ** 6 + 1  # engine pids nvidia-smi lists (no such processes)
    sampler = gpu.MemorySampler(lambda: [a, b])
    sampler._smi = True
    out = iter([f"GPU-0, {a}, 6000\nGPU-0, 123, 9999\n", f"GPU-0, {a}, 6500\nGPU-1, {b}, 6400\n"])
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, next(out), ""))
    job = sampler.track()
    sampler.sample()
    sampler.untrack(job)
    sampler.sample()
    assert (job["gpu_mib"], job["gpu_one_mib"], job["gpus"]) == (6000.0, 6000.0, 1) and job["host_mib"] > 0
    run = sampler.run
    assert (run["gpu_mib"], run["gpu_one_mib"], run["gpu_engine_mib"], run["gpus"], run["engines"]) == \
        (12900.0, 6500.0, 6500.0, 2, 2)
    assert gpu.format_memory(run).startswith("GPU 6.3 GiB on one GPU at most (2 GPUs, 2 engines of at most 6.3 GiB; "
                                             "12.6 GiB on all GPUs together), CPU ")
    assert gpu.format_memory(dict(gpu_mib=6500, gpu_one_mib=6500, gpu_engine_mib=6500, gpus=1, host_mib=None,
                                  engines=1)) == \
        "GPU 6.3 GiB on one GPU at most (1 GPU, 1 engine of at most 6.3 GiB), CPU not measured"
    assert gpu.format_memory(dict(gpu_mib=None, host_mib=1024, engines=0)) == "GPU not measured, CPU 1.0 GiB"


def test_pockets_option_and_htvs_mode_conflicts(tmp_path, example_dir, fake_engine_env, stubs, use_gpus):
    """``pockets`` docks only the given P2Rank ranks; htvs_mode with top_pockets or pockets, or both of those, is an
    error with a plain message."""
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1", "molecule2"), proteins=[FIXTURE_RECEPTOR])
    use_gpus(fake_gpus(1))
    run(plist, smiles, tmp_path / "results", tmp_path, top_pockets=None, pockets="2-3")
    runs = read_table(tmp_path / "results" / "tables" / "pocket_runs" / f"{FIXTURE_ID}.parquet")
    assert sorted(set(runs.pocket)) == [2, 3] and set(runs.status) == {"ok"}
    params = json.loads((tmp_path / "results" / "MANIFEST.json").read_text())["params"]
    assert params["pockets"] == [2, 3] and params["top_pockets"] is None

    cfg = atlas_config(plist, smiles, tmp_path / "other", tmp_path / "tmp", top_pockets=7)
    with pytest.raises(ValueError, match=r"htvs_mode=True docks the 5 best pockets .* top_pockets=7 is set too"):
        PocketVinaGPU(cfg, htvs_mode=True)
    cfg = atlas_config(plist, smiles, tmp_path / "other", tmp_path / "tmp", top_pockets=None, pockets=10)
    with pytest.raises(ValueError, match=r"pockets=\[10\] is set too"):
        PocketVinaGPU(cfg, htvs_mode=True)
    with pytest.raises(ValueError, match="are both set"):
        atlas_config(plist, smiles, tmp_path / "other", tmp_path / "tmp", top_pockets=2, pockets=[1, 2])
    cfg = atlas_config(plist, smiles, tmp_path / "other", tmp_path / "tmp", top_pockets=None)
    assert PocketVinaGPU(cfg, htvs_mode=True).top_pockets == 5


def test_pockets_dir_skips_p2rank(tmp_path, example_dir, fake_engine_env, use_gpus, monkeypatch):
    """P2Rank run beforehand (e.g. on a whole proteome): a protein whose predictions are in ``pockets_dir``
    (``<protein>.csv``, as P2RankProcessor names them) is docked into them; the others still run P2Rank."""
    from test_pipeline import POCKETS_CSV

    calls = []
    monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets(n_pockets=3, calls=calls))
    monkeypatch.setattr(pipeline, "receptor_pdbqt", cropped_receptor())
    plist, smiles = make_inputs(tmp_path, example_dir, ligands=("molecule1",),
                                proteins=[FIXTURE_RECEPTOR, example_dir / "proteins" / "1fbl.pdb"])
    pockets_dir = tmp_path / "pockets"
    pockets_dir.mkdir()
    (pockets_dir / f"{FIXTURE_ID}.csv").write_text(POCKETS_CSV.read_text())
    use_gpus(fake_gpus(1))
    run(plist, smiles, tmp_path / "results", tmp_path, top_pockets=None, pockets="1-2", pockets_dir=str(pockets_dir))
    assert [os.path.basename(c) for c in calls] == ["1fbl.pdb"]  # only the protein without predictions
    for pid in (FIXTURE_ID, "1fbl"):
        runs = read_table(tmp_path / "results" / "tables" / "pocket_runs" / f"{pid}.parquet")
        assert sorted(set(runs.pocket)) == [1, 2]
    params = json.loads((tmp_path / "results" / "MANIFEST.json").read_text())["params"]
    assert params["pockets_dir"] == str(pockets_dir)
