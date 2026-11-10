"""PoseBusters checks of the docked poses in a results directory: ``bust_results``, ``check_names``.

Per protein, the poses to check are picked from the results (``select``): the stored highest-affinity pose of every
ligand (section A SDF), or the highest-affinity modes overall or per pocket, converted from the engine output with
``pocketvina.convert``. The protein structure (``structures/<protein_id>.*.gz``) is decompressed in memory; PDBQT and
mmCIF structures are converted to PDB text first, since PoseBusters reads only PDB proteins, and crystal waters
(``HOH``) are left out as for docking (other HETATM records, e.g. metal ions and cofactors, stay). Worker processes
load the protein once each, exactly as PoseBusters loads a ``.pdb`` file (same loader options of the config), and
check one pose at a time, so that a pose which takes too long can be stopped: it is tried again with the next of
``timeouts`` in a fresh worker, and recorded as "timeout" after the last one. Every selected pose gets a row in
``tables/posebusters/<protein_id>.parquet`` (``io.tables.POSEBUSTERS``), written atomically once the protein is done.
The workers share the CPUs this process may use (PoseBusters' internal energy check would otherwise start a thread
per core of the machine in each of them).

PoseBusters (``pip install posebusters``) is an optional dependency, imported only when checks run. Workers are
forked on Linux; elsewhere they are spawned, which re-imports the calling script: guard it with
``if __name__ == "__main__":`` there.

PoseBusters is described in the `"PoseBusters: AI-based docking methods fail to generate physically valid poses or
generalise to novel sequences" <https://doi.org/10.1039/D3SC04185A>`_ paper.
"""
import copy
import hashlib
import inspect
import json
import logging
import math
import multiprocessing
import numbers
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, deque
from multiprocessing.connection import wait as wait_connections
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import pandas as pd
import pyarrow.parquet as pq

from . import pdbqt
from .convert import pose_to_sdf
from .engine import exit_on_sigterm
from .io import tables
from .io.results import Results
from .io.writer import atomic_place, resolve_tmp_dir

logger = logging.getLogger(__name__)

SELECT = ("best", "top_k", "per_pocket")
PB_STATUS = ("valid", "invalid", "timeout", "error", "convert_failed", "no_pose")
DEFAULT_TIMEOUTS = (30, 300)
MAX_DEFAULT_WORKERS = 16  # each worker holds RDKit, PoseBusters and the protein (~200 MB)
MAX_CHECKS = 31  # bits of the int32 masks (the sign bit is not used)
PROTEIN_LOAD_TIMEOUT = 600  # seconds a worker may take to load a protein
START_METHOD = None  # multiprocessing start method of the workers; None: "fork" on Linux, else "spawn"
# Parquet schema metadata of a posebusters part
META_CHECKS = "pocketvina.pb_checks"
META_CONFIG = "pocketvina.pb_config"
META_VERSION = "pocketvina.pb_version"
META_SELECT = "pocketvina.pb_select"
META_POSES = "pocketvina.poses_sha256"
_WATCHDOG_S = 5  # a worker checks this often that its parent is alive
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SERVE_CODE = "import sys; from pocketvina.bust import _serve; sys.exit(_serve(sys.argv[1]))"

ProteinArg = Optional[Union[str, Sequence[str]]]


# --- PoseBusters configuration -------------------------------------------------------------------------------------

def _posebusters():
    try:
        import posebusters
    except ImportError as e:
        raise ImportError("PoseBusters is not installed: pip install posebusters") from e
    return posebusters


def require_posebusters() -> str:
    """The PoseBusters version; ``ImportError`` if it is not installed."""
    return str(getattr(_posebusters(), "__version__", "unknown"))


def _config(config) -> Tuple[str, dict]:
    """(name, PoseBusters config dict) of a config name ("dock", "dock_fast", "mol", ...) or a config dict."""
    if isinstance(config, dict):
        cfg = copy.deepcopy(config)
        digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
        name = f"custom-{digest}"
    elif isinstance(config, str):
        path = Path(_posebusters().__file__).parent / "config" / f"{config}.yml"
        if "/" in config or "\\" in config or not path.is_file():
            known = sorted(p.stem for p in path.parent.glob("*.yml"))
            raise ValueError(f"unknown PoseBusters config {config!r}; use one of {known} or a config dict")
        import yaml  # a PoseBusters dependency

        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        name = config
    else:
        raise TypeError(f"config must be a PoseBusters config name or dict, not {type(config).__name__}")
    if not isinstance(cfg, dict) or not isinstance(cfg.get("modules"), list):
        raise ValueError(f"PoseBusters config {name!r} has no list of modules")
    return name, cfg


def _check_names(cfg: dict) -> List[str]:
    # the column names of PoseBusters.bust(): PoseBusters._make_table + _collect_in_table
    modules = cfg["modules"]
    columns = [(m["name"], out) for m in modules for out in m.get("chosen_binary_test_output", [])]
    renamed = {(m["name"], k): v for m in modules for k, v in m.get("rename_outputs", {}).items()}
    suffix = {m["name"]: m["rename_suffix"] for m in modules if "rename_suffix" in m}
    names = [renamed.get(c, c[-1] + suffix.get(c[0], "")).lower().replace(" ", "_") for c in columns]
    dup = sorted(n for n, k in Counter(names).items() if k > 1)
    if dup:
        raise ValueError(f"the PoseBusters config has several checks named {dup}")
    if len(names) > MAX_CHECKS:
        raise ValueError(f"the PoseBusters config has {len(names)} checks; the int32 masks hold at most {MAX_CHECKS}")
    return names


def check_names(config="dock") -> List[str]:
    """Names of the PoseBusters checks of ``config`` in mask bit order: bit ``b`` of ``pb_mask`` / ``pb_na_mask`` is
    check ``check_names(config)[b]``. The same names as the columns of ``PoseBusters(config).bust(...)``."""
    return _check_names(_config(config)[1])


def _new_posebusters(cfg: dict):
    from posebusters import PoseBusters

    pb = PoseBusters(copy.deepcopy(cfg), max_workers=0)
    pb.config["max_workers"] = 0  # one pose at a time, in this process
    return pb


# --- loading molecules as PoseBusters does ---------------------------------------------------------------------------

def _pb_process_mol():
    """PoseBusters' post-load processing (bond orders, cleanup, sanitize, hydrogens, stereo), or None if this
    PoseBusters version does not have it (then molecules are loaded from temp files)."""
    try:
        from posebusters.tools.loading import _process_mol
    except ImportError:
        return None
    return _process_mol


def _quiet():
    from rdkit import rdBase

    return rdBase.BlockLogs()


def _load_params(cfg: dict, which: str) -> dict:
    return dict((cfg.get("loading") or {}).get(which) or {})


def _load_protein(text: str, cfg: dict, tmp_dir: Optional[str] = None):
    """The protein (PDB text) as PoseBusters loads a ``.pdb`` file as ``mol_cond`` with the loading options of
    ``cfg`` (``safe_load_mol``), or None if it does not load."""
    from rdkit import Chem

    params = _load_params(cfg, "mol_cond")
    process = _pb_process_mol()
    if process is None:
        return _load_from_file(text, ".pdb", tmp_dir, lambda path: _pb_safe_load(path, params))
    params.pop("load_all", None)
    params.pop("index", None)
    params.pop("strictParsing", None)
    params.pop("cleanupSubstructures", None)
    sanitize = params.pop("sanitize", False)
    remove_hs = params.pop("removeHs", False)
    proximity = params.pop("proximityBonding", False)
    block = _quiet()
    try:
        # posebusters.tools.loading._load_mol for ".pdb": MolFromPDBFile(sanitize=False, ...), then _process_mol
        mol = Chem.MolFromPDBBlock(text, sanitize=False, removeHs=remove_hs, proximityBonding=proximity)
        return process(mol, sanitize=sanitize, **params)
    except Exception as e:  # safe_load_mol: any failure means "not loaded"
        logger.warning("PoseBusters cannot load the protein: %s", e)
        return None
    finally:
        del block


def _load_pose(sdf: Union[bytes, str], cfg: dict, tmp_dir: Optional[str] = None):
    """The first molecule of SDF text as PoseBusters loads an ``.sdf`` file as ``mol_pred`` with the loading options
    of ``cfg`` (``safe_supply_mols``). Raises like PoseBusters if it does not load."""
    from rdkit import Chem

    text = sdf.decode("utf-8", "replace") if isinstance(sdf, (bytes, bytearray)) else sdf
    params = _load_params(cfg, "mol_pred")
    process = _pb_process_mol()
    if process is None:
        return _load_from_file(text, ".sdf", tmp_dir, lambda path: _pb_supply_first(path, params))
    params.pop("load_all", None)
    params.pop("indices", None)
    sanitize = params.pop("sanitize", True)
    block = _quiet()
    try:
        supplier = Chem.SDMolSupplier()
        supplier.SetData(text, sanitize=False, strictParsing=True)  # as safe_supply_mols for ".sdf"
        mol = supplier[0] if len(supplier) else None
        return process(mol, sanitize=sanitize, **params)
    finally:
        del block


def _load_from_file(text: str, suffix: str, tmp_dir: Optional[str], load: Callable):
    folder = tempfile.mkdtemp(prefix="pocketvina_pb_", dir=tmp_dir)
    try:
        path = Path(folder) / f"molecule{suffix}"
        path.write_text(text)
        return load(path)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _pb_safe_load(path, params):
    from posebusters.tools.loading import safe_load_mol

    return safe_load_mol(path, **params)


def _pb_supply_first(path, params):
    from posebusters.tools.loading import safe_supply_mols

    for mol in safe_supply_mols(path, **params):
        if mol is None:
            raise ValueError("Could not load molecule.")
        return mol
    raise ValueError("Could not load molecule.")


def _flag(value) -> Optional[bool]:
    """A PoseBusters check result: True, False or None (not applicable / not run)."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral) and math.isnan(value):
        return None
    return bool(value)


def _check_pose(pb, names: List[str], payload: tuple, protein, tmp_dir: Optional[str]) -> dict:
    """PoseBusters on one pose: {"values": [True/False/None per check], "n_heavy": ...} or {"status", "error"}."""
    if payload[0] == "pdbqt":
        _, model, smiles, props = payload
        sdf, _ = pose_to_sdf(model, smiles, props)
        if sdf is None:
            return dict(status="convert_failed", error="the pose does not convert to the molecule of its SMILES")
    else:
        sdf = payload[1]
    mol = _load_pose(sdf, pb.config, tmp_dir)
    df = pb.bust([mol], mol_cond=protein)
    if len(df) != 1:
        raise RuntimeError(f"PoseBusters returned {len(df)} rows for one pose")
    row = df.iloc[0]
    return dict(values=[_flag(row[n]) for n in names], n_heavy=mol.GetNumHeavyAtoms())


# --- worker processes ----------------------------------------------------------------------------------------------

def _exit_with_parent(sig: Optional[int] = None) -> None:
    """Linux: the kernel sends this process ``sig`` (default SIGKILL) when its parent dies, also while it checks a
    pose (the parent can no longer stop a pose past its timeout once it is killed with SIGKILL)."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL if sig is None else sig)  # PR_SET_PDEATHSIG
    except Exception:  # not available: the watchdogs (_worker_main, _serve) still end an idle process
        pass


def _worker_main(conn, cfg: dict, parent_pid: Optional[int], tmp_dir: str, watchdog_s: float = _WATCHDOG_S):
    """Loop of a worker process: ("protein", key, pdb_text) loads a protein, ("pose", task, payload) checks a pose
    against it, ("stop",) or a closed pipe ends the loop; the worker also ends when its parent is gone.

    ``parent_pid``: the caller (fork, spawn); None with the "forkserver" start method, whose workers are children of
    the fork server (which ends with the caller): then that is watched. A forked worker (and its forked siblings)
    holds a copy of the caller's end of the pipe, so a closed pipe alone does not tell that the caller is gone."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C stops the parent, which stops the workers
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    _exit_with_parent()
    parent = os.getppid() if parent_pid is None else parent_pid
    logging.getLogger("rdkit").setLevel(logging.ERROR)  # PoseBusters sends RDKit's log to Python logging
    logging.getLogger("posebusters").setLevel(logging.WARNING)  # not "Using configuration dictionary ..."
    pb = _new_posebusters(cfg)
    names = _check_names(cfg)
    protein, protein_error = None, "no protein loaded"
    while True:
        try:
            if not conn.poll(watchdog_s):
                if os.getppid() != parent:
                    return
                continue
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if msg[0] == "stop":
            return
        t0 = time.perf_counter()
        if msg[0] == "protein":
            protein = _load_protein(msg[2], pb.config, tmp_dir)
            protein_error = None if protein is not None else "PoseBusters cannot load the protein structure"
            reply = ("protein", msg[1], protein_error)
        else:
            if protein_error:
                result = dict(status="error", error=protein_error)
            else:
                try:
                    result = _check_pose(pb, names, msg[2], protein, tmp_dir)
                except Exception as e:  # one pose must not stop the others
                    result = dict(status="error", error=f"{type(e).__name__}: {e}"[:1000])
            result["runtime_s"] = time.perf_counter() - t0
            reply = ("pose", msg[1], result)
        try:
            conn.send(reply)
        except (EOFError, OSError):
            return


def _context():
    method = START_METHOD or ("fork" if sys.platform.startswith("linux") else "spawn")
    return multiprocessing.get_context(method)


def _available_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def default_workers() -> int:
    """Worker processes when ``workers=None``: the CPUs this process may use, at most ``MAX_DEFAULT_WORKERS``."""
    return max(1, min(_available_cpus(), MAX_DEFAULT_WORKERS))


def _limit_threads(cfg: dict, threads: int) -> dict:
    """``cfg`` with the threads of PoseBusters' internal energy check (RDKit conformer embedding + UFF) set to
    ``threads``, unless the config sets them. Its default (0) starts one thread per core of the machine in every
    worker, also beyond the CPUs this process may use. The results do not change (the embedding is seeded)."""
    try:
        from posebusters.modules.energy_ratio import check_energy_ratio

        if "num_threads" not in inspect.signature(check_energy_ratio).parameters:
            return cfg
    except Exception:  # another PoseBusters version: leave its config as it is
        return cfg
    cfg = copy.deepcopy(cfg)
    for module in cfg.get("modules", []):
        if isinstance(module, dict) and module.get("function") == "energy_ratio":
            params = dict(module.get("parameters") or {})
            params.setdefault("num_threads", threads)
            module["parameters"] = params
    return cfg


class _Worker:
    def __init__(self, ctx, cfg: dict, tmp_dir: str):
        self.conn, child = ctx.Pipe()
        parent_pid = None if ctx.get_start_method() == "forkserver" else os.getpid()
        self.proc = ctx.Process(target=_worker_main, args=(child, cfg, parent_pid, tmp_dir, _WATCHDOG_S),
                                name="pocketvina-posebusters", daemon=True)
        self.proc.start()
        child.close()
        self.protein = None  # key of the protein sent to it
        self.loaded = False  # that protein is loaded
        self.task = None     # (task index, attempt) being checked
        self.deadline = None
        self.started = None

    def send(self, msg) -> bool:
        try:
            self.conn.send(msg)
            return True
        except (OSError, ValueError):
            return False

    def kill(self):
        if self.proc.is_alive():
            self.proc.kill()
        self.proc.join(5)
        self.conn.close()

    def exit_code(self):
        self.proc.join(1)
        return self.proc.exitcode


class _Pool:
    """Worker processes that check one pose at a time with a per-pose timeout."""

    def __init__(self, cfg: dict, n_workers: int, timeouts: Sequence[Optional[float]], tmp_dir: str):
        # the workers share the CPUs this process may use
        cfg = _limit_threads(cfg, max(1, _available_cpus() // max(1, n_workers)))
        self.cfg, self.n_workers, self.timeouts, self.tmp_dir = cfg, n_workers, tuple(timeouts), tmp_dir
        self.ctx = _context()
        self.workers: List[_Worker] = []
        self._key = 0

    def close(self):
        for w in self.workers:
            w.send(("stop",))
        deadline = time.monotonic() + 5
        for w in self.workers:
            w.proc.join(max(0.0, deadline - time.monotonic()))
            w.kill()
        self.workers = []

    def _discard(self, w: _Worker):
        w.kill()
        self.workers.remove(w)

    def _idle(self) -> Optional[_Worker]:
        for w in self.workers:
            if w.task is None:
                return w
        if len(self.workers) < self.n_workers:
            w = _Worker(self.ctx, self.cfg, self.tmp_dir)
            self.workers.append(w)
            return w
        return None

    def _timeout(self, attempt: int) -> Optional[float]:
        return self.timeouts[attempt]

    def _receive(self, w: _Worker, key: int) -> Tuple[Optional[dict], bool]:
        """Read what worker ``w`` (busy, readable) sent: (result of its pose, False), (None, False) if the pose is
        still being checked (the protein is loaded now), or (an "error" result, True) if the worker died."""
        while True:
            try:
                msg = w.conn.recv()
            except (EOFError, OSError):
                code = w.exit_code()
                return dict(status="error", error=f"the PoseBusters worker process died (exit code {code})"), True
            if msg[0] == "protein":
                if msg[1] != key:
                    raise RuntimeError(f"PoseBusters worker loaded protein {msg[1]}, expected {key}")
                now = time.monotonic()
                timeout = self._timeout(w.task[1])
                w.loaded, w.started = True, now
                w.deadline = None if timeout is None else now + timeout
                if not w.conn.poll(0):
                    return None, False
                continue
            if msg[1] != w.task[0]:
                raise RuntimeError(f"PoseBusters worker answered for pose {msg[1]}, expected {w.task[0]}")
            return msg[2], False

    def check(self, protein_text: str, payloads: Sequence[Callable[[], tuple]], label: str = "") -> List[dict]:
        """Check the poses (``payloads[i]()`` gives the message payload of pose ``i``) against the protein; returns a
        result dict per pose, in order. Every pose gets a result, also when it times out or kills its worker, or
        when its payload cannot be made (an "error" result)."""
        self._key += 1
        try:
            return self._check(self._key, protein_text, payloads, label)
        finally:
            # left busy only by an exception (or Ctrl-C): such a worker would answer for this protein's poses during
            # the next check (the pool of the background process goes on with the next protein)
            for w in [w for w in self.workers if w.task is not None]:
                self._discard(w)

    def _check(self, key: int, protein_text: str, payloads: Sequence[Callable[[], tuple]], label: str) -> List[dict]:
        n = len(payloads)
        results: List[Optional[dict]] = [None] * n
        queue = deque((i, 0) for i in range(n))
        spent = [0.0] * n
        left = n
        send_failures = 0
        while left:
            while queue:
                w = self._idle()
                if w is None:
                    break
                i, attempt = queue.popleft()
                try:
                    payload = payloads[i]()
                except Exception as e:  # e.g. an unreadable pose block: this pose only
                    results[i] = dict(status="error", error=f"cannot read the pose: {type(e).__name__}: {e}"[:1000])
                    logger.warning("%s: pose %d: %s", label, i, results[i]["error"])
                    left -= 1
                    continue
                ok = True
                if w.protein != key:
                    ok = w.send(("protein", key, protein_text))
                    w.protein, w.loaded = key, False
                ok = ok and w.send(("pose", i, payload))
                if not ok:
                    self._discard(w)
                    queue.appendleft((i, attempt))
                    send_failures += 1
                    if send_failures > 3 + 2 * self.n_workers:
                        raise RuntimeError("PoseBusters worker processes do not start")
                    continue
                now = time.monotonic()
                w.task, w.started = (i, attempt), now
                timeout = self._timeout(attempt)
                if not w.loaded:
                    w.deadline = now + PROTEIN_LOAD_TIMEOUT
                else:
                    w.deadline = None if timeout is None else now + timeout
            busy = [w for w in self.workers if w.task is not None]
            if not busy:
                continue
            deadlines = [w.deadline for w in busy if w.deadline is not None]
            wait_s = max(0.0, min(deadlines) - time.monotonic()) if deadlines else None
            ready = wait_connections([w.conn for w in busy], wait_s)
            for w in [w for w in busy if w.conn in ready]:
                result, died = self._receive(w, key)
                if result is None:
                    continue
                i = w.task[0]
                if died:
                    result["runtime_s"] = spent[i] + time.monotonic() - w.started
                    logger.warning("%s: pose %d: %s", label, i, result["error"])
                    self._discard(w)
                else:
                    w.task = w.deadline = None
                results[i] = result
                left -= 1
            now = time.monotonic()
            for w in [w for w in self.workers if w.task is not None and w.deadline is not None]:
                if now < w.deadline:
                    continue
                i, attempt = w.task
                spent[i] += now - w.started
                what = "loading the protein" if not w.loaded else "checking the pose"
                self._discard(w)
                if attempt + 1 < len(self.timeouts):
                    logger.info("%s: pose %d: %s took longer than %g s, trying again with %g s", label, i, what,
                                PROTEIN_LOAD_TIMEOUT if not w.loaded else self.timeouts[attempt],
                                self.timeouts[attempt + 1])
                    queue.append((i, attempt + 1))
                else:
                    limits = ", ".join(f"{t:g} s" for t in self.timeouts if t is not None)
                    results[i] = dict(status="timeout", error=f"PoseBusters timed out ({what}; limits {limits})",
                                      runtime_s=spent[i])
                    left -= 1
        return results


# --- poses of a protein --------------------------------------------------------------------------------------------

class _Pose(NamedTuple):
    ligand_idx: int
    ligand_id: str
    pocket: int
    mode: int
    status: Optional[str]  # outcome known without PoseBusters ("no_pose", "convert_failed"); None: check it
    error: Optional[str]
    source: Optional[tuple]  # ("top",) section A SDF, or ("mode", smiles) a mode converted from the engine output


def _encode(text: str) -> bytes:
    return text.encode("utf-8", "surrogateescape")  # the bytes Results decoded


def _select_poses(res: Results, pid: str, select: str, top_k: int, smiles: Dict[int, Optional[str]]) -> List[_Pose]:
    pairs = res.pairs(pid, columns=["ligand_idx", "ligand_id", "best_pocket", "best_mode", "sdf_status",
                                    "top_length"])
    poses = []
    if select == "best":
        for i, lig, k, mode, sdf_status, top_length in pairs.itertuples(index=False):
            i, k, mode = int(i), int(k), int(mode)
            if k < 1:
                poses.append(_Pose(i, lig, -1, -1, "no_pose", "the ligand has no docked pose", None))
            elif sdf_status != "ok" or not top_length:
                poses.append(_Pose(i, lig, k, mode, "convert_failed",
                                   f"no SDF of the highest-affinity pose (sdf_status {sdf_status})", None))
            else:
                poses.append(_Pose(i, lig, k, mode, None, None, ("top",)))
        return poses

    best = {int(i): (int(k), int(m)) for i, k, m, st, ln in
            pairs[["ligand_idx", "best_pocket", "best_mode", "sdf_status", "top_length"]].itertuples(index=False)
            if k >= 1 and st == "ok" and ln}
    modes = res.modes(pid, columns=["ligand_idx", "pocket", "mode", "affinity"])
    modes = modes.sort_values(["ligand_idx", "affinity", "pocket", "mode"], kind="stable")
    ids = dict(zip(pairs["ligand_idx"].astype(int), pairs["ligand_id"]))

    def pose(i, k, mode):
        if best.get(i) == (k, mode):  # the stored SDF of the best pose, as select="best" checks it
            return _Pose(i, ids[i], k, mode, None, None, ("top",))
        smi = smiles.get(i)
        if not smi:
            return _Pose(i, ids[i], k, mode, "convert_failed", "the ligand has no SMILES", None)
        return _Pose(i, ids[i], k, mode, None, None, ("mode", smi))

    if select == "top_k":
        by_ligand = {int(i): g for i, g in modes.groupby("ligand_idx", sort=True)}
        for i in sorted(ids):
            g = by_ligand.get(i)
            if g is None:
                poses.append(_Pose(i, ids[i], -1, -1, "no_pose", "the ligand has no docked pose", None))
                continue
            for k, mode in g[["pocket", "mode"]].head(top_k).itertuples(index=False):
                poses.append(pose(i, int(k), int(mode)))
        return poses

    runs = res.pocket_runs(pid, columns=["ligand_idx", "pocket", "status"])
    by_run = {(int(i), int(k)): g for (i, k), g in modes.groupby(["ligand_idx", "pocket"], sort=True)}
    for i, k, status in runs.sort_values(["ligand_idx", "pocket"]).itertuples(index=False):
        i, k = int(i), int(k)
        g = by_run.get((i, k))
        if g is None:
            poses.append(_Pose(i, ids[i], k, -1, "no_pose",
                               f"no docked pose in this pocket (run status {status})", None))
            continue
        for mode in g["mode"].head(top_k):
            poses.append(pose(i, k, int(mode)))
    return poses


def _without_waters(text: str) -> str:
    """PDB text without the water records that the docking receptor leaves out (``main.receptor_pdbqt``: ATOM /
    HETATM lines with ``HOH``), so that PoseBusters does not count contacts with waters the docking did not see."""
    return "".join(line for line in text.splitlines(True)
                   if not (line.startswith(("ATOM", "HETATM", "ANISOU")) and "HOH" in line))


def _protein_pdb(res: Results, pid: str) -> str:
    """PDB text of the stored input structure (PDBQT and mmCIF converted, as PoseBusters reads only PDB), without
    crystal waters (as docked)."""
    from .main import structure_kind

    path = res.structure_path(pid)
    text = res.structure(pid).decode("utf-8", "replace")
    kind = structure_kind(path)[0]
    if kind == "pdbqt":
        from .pipeline import _pdb_from_pdbqt

        return _pdb_from_pdbqt(text.replace("\r\n", "\n"))
    if kind == "cif":
        from openbabel import pybel

        text = pybel.readstring("mmcif", text).write("pdb")
    return _without_waters(text)


# --- results -------------------------------------------------------------------------------------------------------

def _table_meta(path) -> Dict[str, str]:
    meta = pq.read_schema(path).metadata or {}
    return {k.decode("utf-8", "replace"): v.decode("utf-8", "replace") for k, v in meta.items()}


def _poses_sha256(res: Results, pid: str) -> Optional[str]:
    files = (res.manifest(pid) or {}).get("files", {})
    return (files.get(f"poses/{pid}.poses") or {}).get("sha256")


class _Session:
    """Settings, Results and worker pool of a ``bust_results`` call (or of the background process of a run)."""

    def __init__(self, results_dir, *, select="best", top_k=1, config="dock", workers=None,
                 timeouts=DEFAULT_TIMEOUTS, overwrite=False, tmp_dir=None):
        if select not in SELECT:
            raise ValueError(f"select must be one of {SELECT}, got {select!r}")
        if isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral) or top_k < 1:
            raise ValueError(f"top_k must be an integer >= 1, got {top_k!r}")
        if workers is not None and (isinstance(workers, bool) or not isinstance(workers, numbers.Integral)
                                    or workers < 1):
            raise ValueError(f"workers must be an integer >= 1 or None, got {workers!r}")
        self.timeouts = _check_timeouts(timeouts)
        self.select, self.top_k, self.overwrite = select, int(top_k), bool(overwrite)
        self.pb_config, self.cfg = _config(config)
        self.names = _check_names(self.cfg)
        self.pb_version = require_posebusters()
        self.n_workers = int(workers) if workers is not None else default_workers()
        self.select_meta = json.dumps(dict(select=select, top_k=self.top_k), sort_keys=True)
        self.schema = tables.POSEBUSTERS.with_metadata({
            META_CHECKS: json.dumps(self.names), META_CONFIG: self.pb_config, META_VERSION: self.pb_version,
            META_SELECT: self.select_meta})
        self.all_bits = (1 << len(self.names)) - 1
        self.res = Results(results_dir)
        self.tmp = resolve_tmp_dir(tmp_dir)
        self.pool = _Pool(self.cfg, self.n_workers, self.timeouts, self.tmp)
        self._smiles = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self.pool.close()
        finally:
            self.res.close()
            shutil.rmtree(self.tmp, ignore_errors=True)

    def smiles(self) -> Dict[int, Optional[str]]:
        if self._smiles is None:
            lig = self.res.ligands(columns=["ligand_idx", "smiles_input", "smiles_docked"])
            self._smiles = {int(i): (d if isinstance(d, str) and d else s if isinstance(s, str) and s else None)
                            for i, s, d in lig.itertuples(index=False)}
        return self._smiles

    def target(self, pid: str) -> str:
        return os.path.join(self.res.path, "tables", "posebusters", f"{pid}.parquet")

    def needs_checks(self, pid: str, poses_sha: Optional[str]) -> bool:
        dst = self.target(pid)
        if self.overwrite or not os.path.isfile(dst):
            return True
        try:
            meta = _table_meta(dst)
        except Exception as e:  # unreadable: check again
            logger.warning("%s: unreadable PoseBusters table %s (%s), checking again", pid, dst, e)
            return True
        if meta.get(META_POSES) not in (None, poses_sha):
            logger.info("%s: docked again since its PoseBusters checks, checking again", pid)
            return True
        if (meta.get(META_CONFIG), meta.get(META_SELECT)) != (self.pb_config, self.select_meta):
            logger.warning("%s: keeping the PoseBusters results made with config %s, %s (overwrite=True replaces "
                           "them)", pid, meta.get(META_CONFIG), meta.get(META_SELECT))
        else:
            logger.info("%s: PoseBusters results already there, skipped", pid)
        return False

    def bust(self, pid: str) -> bool:
        """Check the poses of one protein and write its table; False if it was skipped (results already there)."""
        res = self.res
        poses_sha = _poses_sha256(res, pid)  # KeyError for a protein that is not done
        if not self.needs_checks(pid, poses_sha):
            return False
        t0 = time.monotonic()
        poses = _select_poses(res, pid, self.select, self.top_k, self.smiles())
        uniprot = res.pairs(pid, columns=["uniprot"])["uniprot"].dropna()
        uniprot = str(uniprot.iloc[0]) if len(uniprot) else None
        todo = [n for n, p in enumerate(poses) if p.status is None]
        outcome: Dict[int, dict] = {}
        if todo:
            try:
                text = _protein_pdb(res, pid)
            except Exception as e:
                error = f"cannot read the protein structure: {type(e).__name__}: {e}"[:1000]
                logger.error("%s: %s", pid, error)
                outcome = {n: dict(status="error", error=error) for n in todo}
            else:
                payload = self._payloads(pid, poses)
                found = self.pool.check(text, [lambda n=n: payload(n) for n in todo], label=pid)
                outcome = dict(zip(todo, found))
        rows = [self._row(pid, uniprot, p, outcome.get(n)) for n, p in enumerate(poses)]
        rows.sort(key=lambda r: (r["ligand_idx"], r["pocket"], r["mode"]))
        schema = self.schema
        if poses_sha:  # lets a later call (and Results) tell that the protein was docked again since
            schema = schema.with_metadata({**schema.metadata, META_POSES.encode(): poses_sha.encode()})
        local = os.path.join(self.tmp, f"{pid}.parquet")
        tables.write_table(rows, schema, local)
        atomic_place(local, self.target(pid))
        dt = time.monotonic() - t0
        counts = Counter(r["pb_status"] for r in rows)
        logger.info("%s: PoseBusters %s on %d poses: %s (%.1f s, %.2f s per checked pose)", pid, self.pb_config,
                    len(rows), dict(counts), dt, dt / len(todo) if todo else 0.0)
        return True

    def _payloads(self, pid: str, poses: List[_Pose]) -> Callable[[int], tuple]:
        res = self.res
        last = {}  # engine output of the last (ligand, pocket): poses come grouped by ligand

        def payload(n: int) -> tuple:
            p = poses[n]
            if p.source[0] == "top":
                return ("sdf", _encode(res.top_sdf(pid, p.ligand_idx)))
            key = (p.ligand_idx, p.pocket)
            if key not in last:
                last.clear()
                last[key] = _encode(res.pose_block(pid, p.ligand_idx, p.pocket))
            model = pdbqt.model(last[key], p.mode)
            modes = pdbqt.parse_modes(model)
            # the fields of the best-pose SDFs (pipeline.run_atlas)
            props = {"_Name": p.ligand_id, "protein_id": pid, "ligand_idx": p.ligand_idx, "pocket": p.pocket,
                     "mode": p.mode, "affinity": modes[0].affinity if modes else None}
            return ("pdbqt", model, p.source[1], props)

        return payload

    def _row(self, pid: str, uniprot: Optional[str], pose: _Pose, result: Optional[dict]) -> dict:
        row = dict(protein_id=pid, uniprot=uniprot, ligand_idx=pose.ligand_idx, ligand_id=pose.ligand_id,
                   pocket=pose.pocket, mode=pose.mode, pb_config=self.pb_config, pb_version=self.pb_version,
                   pb_status=pose.status, pb_valid=None, pb_mask=0, pb_na_mask=self.all_bits, failed_checks=None,
                   n_heavy_docked=None, runtime_s=None, error=pose.error)
        if result is None:
            return row
        row["runtime_s"] = result.get("runtime_s")
        if "values" not in result:
            row.update(pb_status=result["status"], error=result.get("error"))
            return row
        values = result["values"]
        failed = [name for name, v in zip(self.names, values) if v is False]
        row.update(pb_status="invalid" if failed else "valid", pb_valid=not failed,
                   pb_mask=sum(1 << b for b, v in enumerate(values) if v is True),
                   pb_na_mask=sum(1 << b for b, v in enumerate(values) if v is None),
                   failed_checks=",".join(failed) or None, n_heavy_docked=result.get("n_heavy"), error=None)
        return row


def _check_timeouts(timeouts) -> Tuple[Optional[float], ...]:
    if timeouts is None:
        return (None,)
    if isinstance(timeouts, numbers.Real) and not isinstance(timeouts, bool):
        timeouts = (timeouts,)
    out = []
    for t in timeouts:
        if isinstance(t, bool) or not isinstance(t, numbers.Real) or not 0 < t < float("inf"):
            raise ValueError(f"timeouts must be positive numbers of seconds (or None), got {timeouts!r}")
        out.append(float(t))
    if not out:
        raise ValueError("timeouts must hold at least one number of seconds (or be None)")
    return tuple(out)


@exit_on_sigterm()
def bust_results(results_dir, *, select: str = "best", top_k: int = 1, config="dock", workers: Optional[int] = None,
                 timeouts=DEFAULT_TIMEOUTS, proteins: ProteinArg = None, overwrite: bool = False,
                 tmp_dir: Optional[str] = None) -> List[str]:
    """Run PoseBusters on the docked poses of a results directory; returns the ids of the proteins checked.

    ``select``: "best" the stored highest-affinity pose of each ligand; "top_k" the ``top_k`` highest-affinity modes
    of each ligand over all pockets; "per_pocket" the ``top_k`` highest-affinity modes of each ligand in each pocket.
    Modes other than the stored highest-affinity pose are converted from the engine output with ``pocketvina.convert``
    (a failure is a "convert_failed" row).
    ``config``: a PoseBusters config name ("dock", "dock_fast", ...) or dict. ``workers``: worker processes (None:
    ``default_workers()``). ``timeouts``: seconds per pose for each attempt; a pose still running after the last one
    is a "timeout" row (None: no limit). ``proteins``: protein ids (None: every committed protein).
    A protein that already has a table is skipped unless ``overwrite`` (or it was docked again since). Temp files go
    to ``io.writer.resolve_tmp_dir(tmp_dir)`` and are removed at the end.

    Rows (``io.tables.POSEBUSTERS``): ``pb_status`` "valid" (no check failed), "invalid", "timeout", "error",
    "convert_failed" or "no_pose" (pocket and mode are -1 when there is no pose at all); bit ``b`` of ``pb_mask`` is
    set if check ``check_names(config)[b]`` passed, of ``pb_na_mask`` if it gave no result (all bits for poses that
    were not checked). The check names are also in the Parquet schema metadata (``pocketvina.pb_checks``).
    """
    t0 = time.monotonic()
    done = []
    with _Session(results_dir, select=select, top_k=top_k, config=config, workers=workers, timeouts=timeouts,
                  overwrite=overwrite, tmp_dir=tmp_dir) as session:
        if proteins is None:
            pids = session.res.proteins()
        elif isinstance(proteins, str):
            pids = [proteins]
        else:
            pids = list(proteins)
        for pid in pids:
            session.res.manifest(pid)  # KeyError for unknown proteins, before any work
        for pid in pids:
            if session.bust(pid):
                done.append(pid)
    logger.info("PoseBusters: %d of %d proteins checked (%.1f s)", len(done), len(pids), time.monotonic() - t0)
    return done


# --- background checks during a docking run ------------------------------------------------------------------------

def _child_env() -> Dict[str, str]:
    """Environment of a fresh Python process that imports this ``pocketvina``: its parent folder goes on
    ``PYTHONPATH`` unless it is a site-packages folder (an installed package; there it would come before the
    standard library)."""
    import site

    env = dict(os.environ)
    sites = list(getattr(site, "getsitepackages", list)()) + [site.getusersitepackages()]
    if os.path.realpath(_PACKAGE_PARENT) not in {os.path.realpath(p) for p in sites}:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [_PACKAGE_PARENT, env.get("PYTHONPATH")]))
    return env


class BackgroundBuster:
    """``bust_results`` for proteins handed over one at a time (``submit``), in a separate Python process, so that
    docking goes on while their poses are checked (``pipeline.run_atlas`` with ``posebusters=True``).

    A fresh interpreter (not a fork of the docking process, and without re-running the calling script). Its temp
    files and working directory are in ``resolve_tmp_dir(tmp_dir)``; ``close()`` waits for the proteins submitted,
    ``terminate()`` stops it. ``submit`` does not wait for the process: ids it has not read yet (more than the pipe
    holds, e.g. the many proteins found done when a large run is resumed) are kept here and passed on later.
    The process also stops when the one that made it is gone (on Linux: when the thread that made it ends).
    """

    def __init__(self, results_dir, *, workers: Optional[int] = None, tmp_dir: Optional[str] = None,
                 **options):
        self.work = resolve_tmp_dir(tmp_dir)
        settings = dict(options, results_dir=os.path.abspath(os.fspath(results_dir)), workers=workers,
                        tmp_dir=self.work, log_level=logging.getLogger("pocketvina").getEffectiveLevel(),
                        parent_pid=os.getpid())
        self.submitted = 0
        self._broken = False
        self._pending = bytearray()  # submitted ids not in the pipe yet ("<id>\n" each)
        try:
            # cwd: an empty folder, so that "-c" finds no other "pocketvina" in the caller's working directory
            self.proc = subprocess.Popen([sys.executable, "-c", _SERVE_CODE, json.dumps(settings)],
                                         stdin=subprocess.PIPE, bufsize=0, cwd=self.work, env=_child_env())
        except BaseException:
            shutil.rmtree(self.work, ignore_errors=True)
            raise
        self._set_blocking(False)

    def _set_blocking(self, blocking: bool) -> None:
        try:
            os.set_blocking(self.proc.stdin.fileno(), blocking)
        except (AttributeError, OSError, ValueError):  # Windows before Python 3.12: writes block
            pass

    def _write(self, wait: bool = False) -> None:
        """Pass pending ids into the pipe: those that fit now, or all of them (``wait``)."""
        if wait:
            self._set_blocking(True)
        while self._pending:
            try:
                n = os.write(self.proc.stdin.fileno(), self._pending)
            except BlockingIOError:  # the pipe is full: the rest goes with a later submit() or close()
                return
            del self._pending[:n]

    def _stopped(self) -> None:
        self._broken = True
        self._pending.clear()
        logger.error("the PoseBusters process stopped (exit code %s): no more proteins are checked in this run",
                     self.proc.poll())

    def submit(self, protein_id: str) -> None:
        """Hand a committed protein over to the process (returns at once)."""
        if self._broken:
            return
        try:
            if self.proc.stdin.closed:
                raise ValueError("closed")
            self._pending += (protein_id + "\n").encode()
            self._write()
            self.submitted += 1
        except (OSError, ValueError):
            self._stopped()

    def _close_stdin(self):
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass

    def close(self) -> int:
        """No more proteins: wait until those submitted are checked; returns the exit code (0: no failures)."""
        try:
            if not self._broken:
                try:
                    self._write(wait=True)
                except (OSError, ValueError):
                    self._stopped()
            self._close_stdin()
            return self.proc.wait()
        finally:
            if self.proc.poll() is None:
                self.terminate()
            shutil.rmtree(self.work, ignore_errors=True)

    def terminate(self, grace: float = 10) -> None:
        """Stop the checks now (SIGTERM, then SIGKILL after ``grace`` seconds); a protein being checked keeps no
        table, so a rerun checks it again."""
        self._close_stdin()
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(grace)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        shutil.rmtree(self.work, ignore_errors=True)


def _serve(settings_json: str) -> int:
    """Main of the ``BackgroundBuster`` process: check each protein id read from stdin, until stdin is closed or
    the docking process is gone (then the proteins not checked yet are checked when the run is resumed)."""
    settings = json.loads(settings_json)
    logging.basicConfig(level=settings.pop("log_level", logging.WARNING),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C stops the docking process, which stops this one
    results_dir = settings.pop("results_dir")
    parent = settings.pop("parent_pid", None)
    failed = 0
    with exit_on_sigterm(), _Session(results_dir, **settings) as session:
        # a docking process killed with SIGKILL (e.g. by the OOM killer) cannot terminate() this one: on Linux the
        # kernel sends SIGTERM, which stops the checks as terminate() does; elsewhere it stops between proteins
        _exit_with_parent(signal.SIGTERM)
        for line in sys.stdin:
            if parent is not None and os.getppid() != parent:
                logger.warning("the docking process is gone: PoseBusters checks stopped")
                break
            pid = line.strip()
            if not pid:
                continue
            try:
                session.bust(pid)
            except Exception:
                failed += 1
                logger.exception("%s: PoseBusters checks failed", pid)
    return 1 if failed else 0
