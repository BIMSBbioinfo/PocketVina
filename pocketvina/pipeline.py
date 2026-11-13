"""Docking runs that write the "atlas" results layout (docs/output-format.md): ``run_atlas``.

Ligands are prepared once per run (read back from ``tables/ligands.parquet`` when a run is resumed) and written to a
local temp dir. Each protein is prepared in its own temp folder: pockets (P2Rank, or the database's predictions CSV)
and receptor PDBQT. Its engine calls, one per pocket and chunk of ``ligands_per_call`` ligands (1000 by default, 0:
all in one call; see ``main.ligand_chunk_size``), each limited to ``engine_timeout`` seconds, go to a pool of engine
slots: worker threads that each run one engine at a time, on their own GPU (``CUDA_VISIBLE_DEVICES``; config ``gpus``
and ``procs_per_gpu``, see ``pocketvina.gpu`` and ``plan_engine_slots``) and in their own temp folder. The main thread
prepares the next protein while the slots dock, so the GPUs do not wait at protein boundaries, and records the engine
outputs in the protein's ``ProteinWriter`` as they come back, also while P2Rank and the OpenBabel receptor conversion
(then in a child process: it holds the GIL) run for the next protein (the output files are deleted).
Once all calls of a protein are back, its pose file, Parquet parts, gzipped input structure and P2Rank's SAS points
(``io.sas``; not for pockets read from a CSV) are committed to the results directory with ``manifest/<protein_id>.json``
last. A call docks the same ligands with the same settings
whatever slot runs it, so the results do not depend on the number of slots. Proteins with a manifest are skipped, so a
stopped run resumes where it was, and several processes may write different proteins into one results directory.
"""
import builtins
import dataclasses
import glob
import hashlib
import itertools
import json
import logging
import math
import numbers
import os
import queue
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import zlib
from collections import Counter, defaultdict
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import gpu, p2rank
from .convert import AD_ELEMENTS, pose_to_sdf
from .engine import (
    KERNEL_FILES,
    EngineResult,
    collect_outputs,
    exit_on_sigterm,
    kernel_cache_dir,
    run_engine,
    running_engines,
    write_config,
)
from .io import posefile, sas, tables
from .io.writer import MAX_POCKET, ProteinWriter, ResultsDir, protein_id_from_path, resolve_tmp_dir
from .ligands import LigandRecord, prepare_ligands, read_smiles_file, write_ligand_dir
from .main import (  # receptor_pdbqt is in main.py so that the "files" format needs no pyarrow
    P2RankConfig,
    PocketVinaConfig,
    PocketVinaGPU,
    check_ligand_names,
    ligand_chunk_size,
    read_database_list,
    read_protein_list,
    read_structure,
    receptor_pdbqt,
    structure_kind,
)
from .pdbqt import parse_modes

logger = logging.getLogger(__name__)

POCKET_CENTER = ("center_x", "center_y", "center_z")
STORED_LIGANDS_WAIT_S = 30  # another process may be writing ligands.parquet of the same run
_LOG_PROGRESS = ("Writing ligand", "JSON file successfully saved", "Docking ligand")
_LOG_STEM_RE = re.compile(r"(L\d{7,})(?:\.pdbqt|_out)")  # ligands.write_ligand_dir file names
_LOG_SEED_RE = re.compile(r"Using random seed:\s*(-?\d+)")
_LOG_PARSE_ERROR = "Parse error"  # main.cpp: the engine skips a ligand it cannot parse
_MODEL_RE = re.compile(rb"^MODEL\s+(\d+)")
_UNIPROT_RE = re.compile(r"^AF-([A-Z0-9]{6,10})-F\d+")
_P2RANK_FATAL = ("UnsupportedClassVersionError",)  # Java older than P2Rank needs
SLOT_ENV = "POCKETVINA_ENGINE_SLOT"  # set for the engines of a slot pool ("<pool marker><slot>"), see _kill_engines
# while the main thread waits for engine calls it checks this often that the slots are alive and whether they have
# taken the queued calls (the next protein is then prepared)
RESULT_POLL_S = 1.0
P2RANK_POLL_S = 0.5  # and records the engine calls that are back this often while P2Rank runs for the next protein
KERNEL_BUILD_TRIES = 10  # ligands tried one at a time until the engine parses one and builds its kernels
_PROCESS_GROUPS = hasattr(os, "killpg") and hasattr(os, "setsid")  # POSIX
_MAIN_RECEPTOR_PDBQT = receptor_pdbqt  # the one a child process runs (see _receptor_pdbqt)
_RECEPTOR_EXIT_ERROR = 3  # the child's exit code when receptor_pdbqt raised (the error is its last stdout line)
_RECEPTOR_CODE = """\
import json, sys
from pocketvina.main import receptor_pdbqt
try:
    data = receptor_pdbqt(sys.argv[1], legacy=sys.argv[3] == "legacy").encode("utf-8")
except Exception as e:
    print(json.dumps([type(e).__name__, str(e)]))
    sys.exit(3)
with open(sys.argv[2], "wb") as f:
    f.write(data)
"""


class RunAborted(RuntimeError):
    """A problem that stops the whole run (the engine or P2Rank cannot be started), not just one protein."""


class ProteinInput(NamedTuple):
    protein_id: str
    source_path: str             # input structure (absolute)
    uniprot: Optional[str]
    pockets_csv: Optional[str]   # precomputed P2Rank predictions (database workflow); None: run P2Rank


class Pocket(NamedTuple):
    rank: int
    name: str
    score: float
    probability: Optional[float]
    center: Tuple[float, float, float]
    residue_ids: Optional[str]


# --- inputs --------------------------------------------------------------------------------------------------------

def uniprot_from_id(protein_id: str) -> Optional[str]:
    """UniProt accession of an AlphaFold DB model id (``AF-<accession>-F<n>...``), else None."""
    m = _UNIPROT_RE.match(protein_id)
    return m.group(1) if m else None


def protein_inputs(pv: PocketVinaGPU) -> List[ProteinInput]:
    """The proteins of a run in list order; ``ValueError`` if two of them have the same ``protein_id``."""
    cfg = pv.config
    items = []
    if cfg.protein_paths:
        for path in read_protein_list(cfg.protein_paths):
            pid = protein_id_from_path(path)
            items.append(ProteinInput(pid, path, uniprot_from_id(pid), _pockets_in_dir(cfg.pockets_dir, path, pid)))
    else:
        for protein_db, accession in read_database_list(cfg.protein_list):
            path = os.path.abspath(pv.database_protein_file(protein_db, accession))
            pid = protein_id_from_path(path)
            uniprot = accession if protein_db == "UniProt" else uniprot_from_id(pid)
            items.append(ProteinInput(pid, path, uniprot, pv.database_pockets_csv(protein_db, accession)))
    paths = defaultdict(list)
    for item in items:
        paths[item.protein_id].append(item.source_path)
    dup = {pid: p for pid, p in paths.items() if len(p) > 1}
    if dup:
        raise ValueError(f"proteins with the same protein_id (file name without extension): {dup}")
    return items


def _pockets_in_dir(pockets_dir: Optional[str], path: str, pid: str) -> Optional[str]:
    """The P2Rank predictions of a protein in ``pockets_dir`` (``config.pockets_dir``), or None: ``<pid>.csv`` or, as
    ``P2RankProcessor`` names them, the file name without its last extension + ``.csv`` (``1abc.pdb.gz`` ->
    ``1abc.pdb.csv``)."""
    if not pockets_dir:
        return None
    for name in (pid, os.path.splitext(os.path.basename(path))[0]):
        csv_path = os.path.join(pockets_dir, f"{name}.csv")
        if os.path.isfile(csv_path):
            return csv_path
    return None


def _pdb_from_pdbqt(text: str) -> str:
    """PDB coordinate records of a PDBQT receptor (for P2Rank, which does not read PDBQT).

    Atoms are regrouped by residue: OpenBabel's PDBQT writer reorders them along its torsion tree, and P2Rank would
    read a residue split in several places as several residues.
    """
    chains = {}
    atoms = []
    for line in text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            ad_type = line[77:].strip()
            element = AD_ELEMENTS.get(ad_type, ad_type[:1])
            try:
                resseq = int(line[22:26])
            except ValueError:
                resseq = 0
            key = (chains.setdefault(line[21:22], len(chains)), resseq, line[26:27])
            atoms.append((key, f"{line[:66]:<66}          {element.upper():>2}\n"))
    atoms.sort(key=lambda a: a[0])  # stable: atom order within a residue is kept
    return "".join(line for _, line in atoms) + "END\n"


def _kill_process_group(proc: subprocess.Popen) -> None:
    """SIGKILL ``proc`` (not reaped yet) and, on POSIX, its process group (it was started in a session of its own)."""
    if _PROCESS_GROUPS and proc.returncode is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            return
        except OSError:  # e.g. the group is gone already
            pass
    proc.kill()


def _communicate(proc: subprocess.Popen, while_running: Optional[Callable[[], None]]) -> Optional[str]:
    """``proc.communicate()``'s stdout, calling ``while_running`` every ``P2RANK_POLL_S`` seconds until ``proc``
    ends; if it raises, or the wait is interrupted, ``proc`` is killed with its process group."""
    try:
        while True:
            try:
                return proc.communicate(timeout=None if while_running is None else P2RANK_POLL_S)[0]
            except subprocess.TimeoutExpired:  # no output is lost: communicate() goes on where it was
                while_running()
    except BaseException:  # Ctrl-C, SIGTERM (exit_on_sigterm) or while_running raised: do not leave it running
        _kill_process_group(proc)
        raise


def predict_pockets(structure, work_dir, p2rank_config: Optional[P2RankConfig] = None, *,
                    while_running: Optional[Callable[[], None]] = None) -> str:
    """Run P2Rank on ``structure`` with ``work_dir`` as its working and output folder; returns the predictions CSV.

    Uses ``threads`` and ``model`` of ``p2rank_config``. P2Rank also writes its visualizations there, for their SAS
    points file (``io.sas.points_file``), without a copy of the protein. ``while_running`` is called
    every ``P2RANK_POLL_S`` seconds while P2Rank runs (``run_atlas`` records the engine calls that are back meanwhile);
    if it raises, or the wait is interrupted, P2Rank is killed (with its Java process). P2Rank is the one of
    ``p2rank.ensure_p2rank()``, downloaded into the user cache first if there is none. ``RunAborted`` if P2Rank or
    Java cannot be started, ``RuntimeError`` if the prediction fails.
    """
    cfg = p2rank_config or P2RankConfig()
    work_dir = os.path.abspath(os.fspath(work_dir))
    out_dir = os.path.join(work_dir, "out")
    os.makedirs(out_dir, exist_ok=True)
    source = os.path.abspath(os.fspath(structure))
    if structure_kind(source)[0] == "pdbqt":
        pdb = os.path.join(work_dir, protein_id_from_path(source) + ".pdb")
        with open(pdb, "w") as f:
            f.write(_pdb_from_pdbqt(read_structure(source)[1]))
        source = pdb
    try:
        prank = p2rank.ensure_p2rank()
    except p2rank.P2RankError as e:
        raise RunAborted(str(e)) from e
    cmd = [prank, "predict", "-f", source, "-o", out_dir, "-visualizations", "true", "-threads", str(cfg.threads),
           "-vis_copy_proteins", "false", "-model", cfg.model]
    try:
        # a session of its own (POSIX): the prank launcher runs Java as a child, which a stop must kill too
        proc = subprocess.Popen(cmd, cwd=work_dir, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace",
                                start_new_session=_PROCESS_GROUPS)
    except OSError as e:
        raise RunAborted(f"cannot run P2Rank ({prank}): {e}") from e
    with proc:
        stdout = _communicate(proc, while_running)
    tail = stdout[-2000:].strip()
    if proc.returncode in (126, 127) or any(s in stdout for s in _P2RANK_FATAL):
        java = p2rank.java_problem() or f"it needs Java {p2rank.JAVA_MIN} or later"
        raise RunAborted(f"P2Rank cannot run (exit code {proc.returncode}; {java}): {tail}")
    if proc.returncode != 0:
        raise RuntimeError(f"P2Rank failed with exit code {proc.returncode}: {tail}")
    csv_path = os.path.join(out_dir, os.path.basename(source) + "_predictions.csv")
    if not os.path.isfile(csv_path):
        found = glob.glob(os.path.join(glob.escape(out_dir), "*_predictions.csv"))
        if len(found) != 1:
            raise RuntimeError(f"P2Rank wrote no predictions file into {out_dir}: {tail}")
        csv_path = found[0]
    return csv_path


def _text(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and value != value):
        return None
    return str(value).strip() or None


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_pockets_csv(path) -> List[Pocket]:
    """Pockets of a P2Rank predictions CSV (padded columns as P2Rank writes them), sorted by rank.

    Needs ``rank`` and ``center_x/y/z``; ``name``, ``score``, ``probability`` and ``residue_ids`` are optional.
    """
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = {"rank", *POCKET_CENTER} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing pocket columns {sorted(missing)}")
    pockets = []
    for row in df.to_dict("records"):
        rank = _number(row["rank"])
        if not (rank >= 1 and rank == int(rank)):
            raise ValueError(f"{path}: invalid pocket rank {row['rank']!r}")
        rank = int(rank)
        center = tuple(_number(row[c]) for c in POCKET_CENTER)
        if any(c != c or c in (float("inf"), float("-inf")) for c in center):
            raise ValueError(f"{path}: pocket {rank} has no valid center")
        probability = _number(row.get("probability"))
        pockets.append(Pocket(rank, _text(row.get("name")) or f"pocket{rank}", _number(row.get("score")),
                              None if probability != probability else probability, center,
                              _text(row.get("residue_ids"))))
    pockets.sort(key=lambda p: p.rank)
    ranks = [p.rank for p in pockets]
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{path}: duplicate pocket ranks")
    return pockets


def _plain(value):
    """Python int/float for numpy scalars, which ``json.dumps`` cannot write (e.g. a box size read with pandas)."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return value


def run_params(pv: PocketVinaGPU, n_ligands: int) -> dict:
    """Settings that change the docking results; a run directory can only be resumed with the same ones.
    ``n_ligands``: the rows of the ligand file (``ligands_per_call`` is stored as ``main.ligand_chunk_size``)."""
    cfg, pc = pv.config, pv.pocket_config
    params = dict(output_format="atlas", ligand_prep=cfg.ligand_prep, seed=cfg.seed, num_modes=cfg.num_modes,
                  energy_range=_plain(cfg.energy_range),
                  box_size=[_plain(v) for v in (pc.size_x, pc.size_y, pc.size_z)], thread=_plain(pc.thread),
                  top_pockets=pv.top_pockets, ligands_per_call=ligand_chunk_size(cfg, n_ligands))
    if cfg.pockets is not None:  # only when set: older run directories resume as before
        params["pockets"] = list(cfg.pockets)
    if cfg.protein_paths:
        params.update(pocket_source="p2rank", p2rank_model=(cfg.p2rank_config or P2RankConfig()).model)
        if cfg.pockets_dir:  # only when set: older run directories resume as before
            params["pockets_dir"] = cfg.pockets_dir
    else:
        params.update(pocket_source="database")
    return params


# --- ligands -------------------------------------------------------------------------------------------------------

def _stored_ligands(rd: ResultsDir, rows, params: dict, wait: float = 0) -> Optional[pa.Table]:
    """The run's ``ligands.parquet`` if the run has these params and was made from the same (name, SMILES) rows."""
    manifest = rd.run_manifest()
    if manifest is None or manifest.get("params") != json.loads(json.dumps(params, sort_keys=True)):
        return None
    deadline = time.monotonic() + wait
    while not os.path.isfile(rd.ligands_file):
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.5)
    try:
        table = tables.to_arrow(pq.read_table(rd.ligands_file, partitioning=None), tables.LIGANDS)
    except (OSError, ValueError, pa.ArrowException) as e:
        logger.warning("cannot reuse %s: %s", rd.ligands_file, e)
        return None
    if (table.column("ligand_id").to_pylist() != [name for name, _ in rows]
            or table.column("smiles_input").to_pylist() != [smiles for _, smiles in rows]):
        return None
    return table


def _ligand_table(rd: ResultsDir, rows, params: dict, cfg: PocketVinaConfig, force: bool) -> pa.Table:
    """Ligands of the run: read back when resuming (no new preparation, and unseeded preparations stay the same),
    else prepared now; registers the run with ``ResultsDir.init_run``."""
    stored = _stored_ligands(rd, rows, params)
    if stored is not None:
        rd.init_run(params, stored, force=force, tmp_dir=cfg.tmp_dir)
        return stored
    records = prepare_ligands(rows, method=cfg.ligand_prep, seed=cfg.seed)
    if not any(r.status == "ok" for r in records):
        # registering the run would commit every protein with nothing docked (and a resume would keep the ligands)
        first = next((r for r in records if r.error), records[0])
        raise RuntimeError(f"none of the {len(records)} ligands could be prepared "
                           f"({dict(Counter(r.status for r in records))}); {first.ligand_id}: {first.error}")
    table = tables.to_arrow([dataclasses.asdict(r) for r in records], tables.LIGANDS)
    try:
        rd.init_run(params, table, force=force, tmp_dir=cfg.tmp_dir)
    except ValueError:
        # another process may have started the same run a moment earlier (with an unseeded preparation)
        stored = None if force else _stored_ligands(rd, rows, params, wait=STORED_LIGANDS_WAIT_S)
        if stored is None:
            raise
        logger.info("using the ligands of the run started by another process in %s", rd.path)
        rd.init_run(params, stored, tmp_dir=cfg.tmp_dir)
        return stored
    return table


class _Ligands:
    """Columns of the ligand table that docking needs, as lists indexed by ``ligand_idx``."""

    def __init__(self, table: pa.Table):
        self.table = table
        self.n = table.num_rows
        self.ids = table.column("ligand_id").to_pylist()
        docked = table.column("smiles_docked").to_pylist()
        self.smiles = [d or s for d, s in zip(docked, table.column("smiles_input").to_pylist())]
        status = table.column("status").to_pylist()
        errors = table.column("error").to_pylist()
        self.ok = [i for i, s in enumerate(status) if s == "ok"]
        self.skipped = {i: f"ligand not prepared: {s}: {errors[i]}" for i, s in enumerate(status) if s != "ok"}


def _write_batches(lig: _Ligands, root: str, ligands_per_call: Optional[int]) -> List[Tuple[str, Dict[str, int]]]:
    """Write the prepared ligands (``L<ligand_idx>.pdbqt``) into one folder per engine call, ``ligands_per_call``
    each in ``ligand_idx`` order (0 or None: all in one): ``[(folder, {stem: ligand_idx})]``. The engine docks a
    folder in file name order, so a call always docks the same ligands in the same order."""
    if ".pdbqt" in root:
        raise ValueError(f"temp folder {root!r}: the engine cuts ligand paths at the first '.pdbqt', use another "
                         "tmp_dir")
    size = ligands_per_call or max(len(lig.ok), 1)
    batches = []
    for n, start in enumerate(range(0, len(lig.ok), size), 1):
        folder = os.path.join(root, f"batch_{n}")
        batches.append((folder, write_ligand_dir(_ligand_records(lig, lig.ok[start:start + size]), folder)))
    return batches


def _ligand_records(lig: _Ligands, idx: Sequence[int]) -> List[LigandRecord]:
    """``LigandRecord``s with the engine input PDBQT of the prepared ligands ``idx`` (for ``write_ligand_dir``)."""
    texts = lig.table.column("pdbqt").take(pa.array(idx, type=pa.int64())).to_pylist()
    return [LigandRecord(ligand_idx=i, ligand_id=lig.ids[i], smiles_input="", pdbqt=text)
            for i, text in zip(idx, texts)]


# --- engine calls --------------------------------------------------------------------------------------------------

def _unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:500] if lines else ""


def _scan_engine_log(path, stems) -> Tuple[Dict[str, str], Optional[int]]:
    """({stem: last log line about that ligand}, seed the engine reported) from an engine log."""
    messages, seed = {}, None
    try:
        with open(path, "rb") as f:
            for raw in f:
                parts = [p.strip() for p in raw.decode("utf-8", "replace").split("\r") if p.strip()]
                if not parts:
                    continue
                line = parts[-1]
                if seed is None:
                    m = _LOG_SEED_RE.search(line)
                    if m:
                        seed = int(m.group(1))
                        continue
                if line.startswith(_LOG_PROGRESS):
                    continue
                for stem in _LOG_STEM_RE.findall(line):
                    if stem in stems:
                        messages[stem] = line[:500]
    except OSError:
        pass
    return messages, seed


class _Engine(NamedTuple):
    """Engine settings of a run: the same for every call."""

    size: Tuple[float, float, float]
    thread: int
    seed: Optional[int]
    num_modes: Optional[int]
    energy_range: Optional[float]
    timeout: Optional[float] = None  # seconds per engine call


class _Call(NamedTuple):
    """One engine call: a ligand batch docked into one pocket of a protein."""

    protein_id: str
    receptor: str      # receptor PDBQT (in the protein's temp folder)
    pocket: Pocket
    batch_no: int      # 1-based; 0 for the kernel build call
    lig_dir: str
    stems: Dict[str, int]  # {ligand file stem: ligand_idx}


class _CallOutput(NamedTuple):
    result: EngineResult
    outputs: Dict[str, Optional[bytes]]  # {stem: engine output PDBQT, None if there is none}
    messages: Dict[str, str]             # {stem: last engine log line about that ligand}
    seed: Optional[int]                  # the seed the engine reported


class _Slot(NamedTuple):
    """Where one engine runs at a time: its GPU, its own temp folder and its kernel cache."""

    name: str
    device: Optional[str]  # CUDA_VISIBLE_DEVICES of its engines (a GPU UUID); None: the environment's
    work: str              # the engines' working dir, configs, outputs and logs
    kernel_cache: str      # opencl_binary_path

    def env(self, marker: Optional[str] = None) -> Dict[str, str]:
        """``env_extra`` for ``run_engine``; ``marker`` (``SLOT_ENV``) lets ``_kill_engines`` find the engine."""
        env = {} if self.device is None else {"CUDA_VISIBLE_DEVICES": self.device}
        if marker:
            env[SLOT_ENV] = marker
        return env


def _write_call_config(eng: _Engine, call: _Call, slot: _Slot) -> Tuple[str, str, str]:
    """Engine config of ``call`` in the slot's folder: (config path, output dir, log path). ``RunAborted`` for
    settings the engine cannot take (box size, thread count, a temp or kernel cache path: the same for every call)."""
    tag = f"pocket{call.pocket.rank}_batch{call.batch_no}"  # a slot runs one call at a time
    out_dir = os.path.join(slot.work, "out", tag)
    try:
        config = write_config(os.path.join(slot.work, "configs", f"{tag}.txt"), call.receptor, call.lig_dir, out_dir,
                              call.pocket.center, eng.size, eng.thread, call.pocket.rank, seed=eng.seed,
                              num_modes=eng.num_modes, energy_range=eng.energy_range,
                              opencl_binary_path=slot.kernel_cache)
    except ValueError as e:
        raise RunAborted(f"invalid engine settings: {e}") from e
    return config, out_dir, os.path.join(slot.work, "logs", f"{tag}.log")


def _run_call(eng: _Engine, call: _Call, slot: _Slot, env_extra: Optional[Dict[str, str]] = None) -> _CallOutput:
    """Run one engine call in the slot's folder and read what it wrote (the files are removed). Touches nothing
    outside the slot's folder, so the slots of a pool run calls at the same time."""
    os.makedirs(os.path.join(slot.work, "out"), exist_ok=True)
    config, out_dir, log_path = _write_call_config(eng, call, slot)
    try:
        result = run_engine(config, cwd=os.path.join(slot.work, "cwd"), env_extra=env_extra, log_path=log_path,
                            timeout=eng.timeout)
    except OSError as e:
        _unlink(config)
        raise RunAborted(str(e)) from e
    try:
        if result.timed_out:  # killed: an output it was writing may be cut short, so none is used
            outputs = {stem: None for stem in call.stems}
            messages, seed = {}, None
        else:
            outputs = collect_outputs(out_dir, call.receptor, call.pocket.rank, call.stems, remove=True)
            messages, seed = _scan_engine_log(log_path, call.stems)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
        _unlink(log_path)
        _unlink(config)
    return _CallOutput(result, outputs, messages, seed)


def _all_rejected(out: _CallOutput, stems: Dict[str, int]) -> bool:
    """The engine failed because it could parse none of the call's ligands ("No valid ligands", exit 1)."""
    return (not out.result.ok and not out.result.timed_out and bool(stems) and not any(out.outputs.values())
            and all(out.messages.get(stem, "").startswith(_LOG_PARSE_ERROR) for stem in stems))


def _record_call(writer: ProteinWriter, lig: _Ligands, eng: _Engine, call: _Call,
                 out: _CallOutput) -> Tuple[bool, int, Optional[str]]:
    """Add a run per ligand of an engine call to the protein's writer. Returns (ok, n outputs, failure)."""
    result, outputs, messages, stems = out.result, out.outputs, out.messages, call.stems
    seed = eng.seed if eng.seed is not None else out.seed
    failure = None
    # every ligand of the batch failed to parse ("No valid ligands", exit 1): as for an invalid ligand batched with
    # valid ones, its runs are "no_output" and the call is no engine failure (a rerun would not change it)
    rejected = _all_rejected(out, stems)
    if result.timed_out:
        failure = (f"engine timed out after {eng.timeout:g} s (engine_timeout) and was killed; none of the "
                   f"{len(stems)} ligands of this call was docked")
    elif not result.ok and not rejected:
        failure = f"engine exited with code {result.returncode}"
        last = _last_line(result.log_tail)
        if last:
            failure += f": {last}"
    runtime_s = result.runtime_s / len(stems) if stems else None
    n_outputs = 0
    for stem, i in stems.items():
        raw = outputs.get(stem)
        if raw:
            status, error = "ok", None
            n_outputs += 1
        elif failure:
            status, error = "engine_error", messages.get(stem) or failure
        else:
            status, error = "no_output", messages.get(stem) or "the engine wrote no output"
        writer.add_run(i, lig.ids[i], call.pocket.rank, raw, status, runtime_s=runtime_s, seed=seed, error=error)
    return failure is None, n_outputs, failure


# --- engine slots --------------------------------------------------------------------------------------------------

def plan_engine_slots(config: PocketVinaConfig) -> List[Optional[str]]:
    """``CUDA_VISIBLE_DEVICES`` of each engine slot of an atlas run: ``gpu.plan_slots`` over the GPUs that
    ``config.gpus`` names (None: the first visible GPU; "all": every one) with ``config.procs_per_gpu`` (default
    "auto": as many as the GPU's free memory holds) and ``config.vram_per_proc_mib``.

    Without a GPU list (no ``nvidia-smi``, e.g. in a container with the driver's OpenCL library only, or a
    ``CUDA_VISIBLE_DEVICES`` whose GPUs ``gpu.visible_gpus`` cannot match) the engines use the environment's default
    GPU (the first of NVIDIA's OpenCL platform): ``procs_per_gpu`` slots (1 for "auto") of value None.
    ``ValueError`` if ``config.gpus`` names a GPU that is not visible, or is set while ``nvidia-smi`` lists none.
    """
    gpus = gpu.list_gpus()
    if config.gpus not in (None, gpu.ALL_GPUS):
        if not gpus:
            raise ValueError(f"gpus={config.gpus!r}: nvidia-smi lists no GPU here (not installed, no NVIDIA driver, "
                             "or CUDA_VISIBLE_DEVICES hides them or names them by CUDA numbers nvidia-smi does not "
                             "use)")
    gpus = gpu.select_gpus(gpus, config.gpus)
    if not gpus:
        n = config.procs_per_gpu if isinstance(config.procs_per_gpu, int) else 1
        logger.info("no GPU list from nvidia-smi: %d engine process%s on the default GPU", n, "es" * (n > 1))
        return [None] * n
    devices = gpu.plan_slots(gpus, config.procs_per_gpu, config.vram_per_proc_mib)
    per_gpu = Counter(devices)
    logger.info("%d engine slots: %s", len(devices), ", ".join(
        f"{per_gpu[gpu.device_id(g)]} on GPU {g.index} ({g.name}, {g.memory_free_mib} MiB free)" for g in gpus))
    return list(devices)


def _kernel_cache_for(device: Optional[str]) -> str:
    """``engine.kernel_cache_dir()`` as an engine with ``CUDA_VISIBLE_DEVICES=device`` (None: the environment's)
    sees it: its key then names that GPU's model and driver, so GPUs of different models never share kernels. The
    environment of this process is left alone (other threads, e.g. another run's engine slots, read it)."""
    return kernel_cache_dir() if device is None else kernel_cache_dir(device)


def _has_kernels(cache: str) -> bool:
    return all(os.path.isfile(os.path.join(cache, name)) for name in KERNEL_FILES)


def _gpu_problem(result: EngineResult) -> Optional[str]:
    """What kept a failed engine call from docking on its GPU, or None (the call failed otherwise): it ran out of
    GPU memory (``gpu.is_out_of_memory``), or OpenCL could not set up the GPU (``gpu.is_device_error``)."""
    if result.ok or result.timed_out:
        return None
    if gpu.is_out_of_memory(result.log_tail):
        return "ran out of GPU memory"  # NVIDIA's OpenCL reports a full GPU as CL_OUT_OF_HOST_MEMORY too
    if gpu.is_device_error(result.log_tail):
        return "could not use the GPU"
    return None


def _warm_kernel_caches(slots: Sequence[_Slot], eng: _Engine, first: _Call, lig: _Ligands, tmp_root: str) -> None:
    """Build the OpenCL kernels of each kernel cache that several slots share and that holds none yet, with one
    engine call (the first prepared ligand into the first call's pocket; its poses are not used), before the slots
    start: several engines must not build the same kernels at the same time. The engine builds them only after it
    has parsed a ligand, so a ligand it rejects is followed by the next one (``KERNEL_BUILD_TRIES`` at most), and
    on a GPU it cannot use or that has no memory left for it (``_gpu_problem``), the next GPU of those slots builds
    them."""
    by_cache = defaultdict(list)
    for slot in slots:
        by_cache[slot.kernel_cache].append(slot)
    for cache, shared in by_cache.items():
        if len(shared) < 2 or _has_kernels(cache):
            continue
        folder = os.path.join(tmp_root, "ligands", "kernel_build")
        logger.info("building the OpenCL kernels in %s once, for the %d engine slots that share them", cache,
                    len(shared))
        builders = [s for n, s in enumerate(shared) if s.device not in {t.device for t in shared[:n]}]  # one per GPU
        tries = list(lig.ok[:KERNEL_BUILD_TRIES])
        n_tries = len(tries)
        while tries:
            i = tries[0]
            shutil.rmtree(folder, ignore_errors=True)
            call = first._replace(batch_no=0, lig_dir=folder,
                                  stems=write_ligand_dir(_ligand_records(lig, [i]), folder))
            out = _run_call(eng, call, builders[0], builders[0].env())
            if out.result.ok or _has_kernels(cache):
                break
            problem = _gpu_problem(out.result)
            if problem and len(builders) > 1:
                logger.warning("the kernel build call failed: the engine %s (%s) on GPU %s, building the kernels on "
                               "GPU %s", problem, _last_line(out.result.log_tail), builders[0].device,
                               builders[1].device)
                builders.pop(0)
                continue
            if not _all_rejected(out, call.stems):
                logger.warning("the kernel build call failed (exit code %d): %s", out.result.returncode,
                               _last_line(out.result.log_tail))
                break
            logger.info("the engine rejected ligand %s (%s), building the kernels with the next one", lig.ids[i],
                        out.messages.get(next(iter(call.stems)), ""))
            tries.pop(0)
        else:
            logger.warning("the engine rejected the first %d prepared ligands: the engine slots build the kernels "
                           "themselves", n_tries)
        shutil.rmtree(folder, ignore_errors=True)


def _kill_engines(marker: str) -> int:
    """SIGKILL the process group of each engine of this process whose ``SLOT_ENV`` starts with ``marker``
    (``run_engine`` makes every engine a session leader). Linux (``/proc``) only; the number of engines killed."""
    needle = f"{SLOT_ENV}={marker}".encode()
    me = os.getpid()
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return 0
    killed = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                stat = f.read()
            fields = stat[stat.rindex(b")") + 2:].split()  # after "<pid> (<comm>) ": state ppid pgrp session ...
            if int(fields[1]) != me or int(fields[3]) != pid:
                continue
            with open(f"/proc/{pid}/environ", "rb") as f:
                if not any(v.startswith(needle) for v in f.read().split(b"\0")):
                    continue
            os.killpg(pid, signal.SIGKILL)
            killed += 1
        except (OSError, ValueError, IndexError):  # gone meanwhile, or not ours to read
            continue
    return killed


class _Done(NamedTuple):
    call: _Call
    output: Optional[_CallOutput]  # None: the call raised ``error``, or was dropped
    error: Optional[BaseException] = None


class _SlotPool:
    """Worker threads, one per engine slot, that run engine calls (``_run_call``) from a shared queue.

    Calls are taken in (protein order, submission) order, so the proteins of a run finish about in turn; the results
    come back in ``results`` for the main thread, the only one that records them. An engine that fails with an OpenCL
    out-of-memory error while other slots are in use retires its own slot (its GPU runs one engine less from then on,
    none if it was the GPU's last) and the call is queued again; so is a call that ran out of memory while another
    slot of its GPU was retired. A call that ran out of memory in the last slot is a failed call like any other.
    An engine that cannot set up its GPU at all (``gpu.is_device_error``: e.g. a GPU busy in an exclusive compute
    mode, or no memory left for an OpenCL context) counts as out of memory: it fails before docking, whatever the
    call, and would otherwise turn every call its slot takes into ``engine_error`` runs of committed proteins.
    When a call that retired slots so fails in the last slot too, it ran out of memory wherever it ran, also alone:
    the call is the problem (e.g. ``CL_OUT_OF_RESOURCES`` is also NVIDIA's error for a kernel fault on some input),
    not those GPUs, so the slots it retired run engines again (one such call would otherwise take all GPUs but one
    out of the run).
    """

    def __init__(self, slots: Sequence[_Slot], eng: _Engine):
        self.eng = eng
        self.results: "queue.Queue[_Done]" = queue.Queue()
        self._tasks: "queue.PriorityQueue" = queue.PriorityQueue()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._marker = f"{os.getpid()}-{secrets.token_hex(8)}/"
        self._active = Counter()        # slots in use per GPU
        self._retired = Counter()       # slots retired after out-of-memory errors per GPU (never decreases)
        self._retired_by: Dict[tuple, List[_Slot]] = {}  # queue key of a call -> the slots it retired
        self._dropped = set()           # protein ids whose calls are not wanted any more
        self._threads: List[threading.Thread] = []
        with self._lock:
            for slot in slots:
                self._start(slot)

    def _start(self, slot: _Slot) -> None:
        """A worker thread for ``slot`` (``_lock`` held)."""
        t = threading.Thread(target=self._work, args=(slot,), name=f"pocketvina-{slot.name}", daemon=True)
        self._threads.append(t)
        self._active[slot.device] += 1
        t.start()

    def _settled(self, key: tuple) -> None:
        """The call is done (docked, failed otherwise, or dropped): the slots it retired stay retired."""
        with self._lock:
            self._retired_by.pop(key, None)

    @property
    def n_slots(self) -> int:
        """Slots in use (fewer than at the start after out-of-memory errors)."""
        with self._lock:
            return sum(self._active.values())

    @property
    def n_queued(self) -> int:
        """Calls waiting for a slot."""
        return self._tasks.qsize()

    def alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def submit(self, call: _Call, order: int) -> None:
        self._tasks.put(((order, next(self._seq)), call))

    def drop(self, protein_id: str) -> None:
        """The calls of this protein that have not started come back without running."""
        self._dropped.add(protein_id)

    def _work(self, slot: _Slot) -> None:
        env = slot.env(self._marker + slot.name)
        while True:
            key, call = self._tasks.get()
            if call is None or self._stop.is_set():
                return
            if call.protein_id in self._dropped:
                self._settled(key)
                self.results.put(_Done(call, None))
                continue
            with self._lock:
                retired_before = self._retired[slot.device]
            try:
                out = _run_call(self.eng, call, slot, env)
            except BaseException as e:  # for the main thread to handle
                self._settled(key)
                self.results.put(_Done(call, None, e))
                continue
            retry, retire = self._gpu_failed(slot, key, call, out.result, retired_before)
            if retry:
                self._tasks.put((key, call))  # before the calls of later proteins
                if retire:
                    return
                continue
            self.results.put(_Done(call, out))

    def _gpu_failed(self, slot: _Slot, key: tuple, call: _Call, result: EngineResult,
                    retired_before: int) -> Tuple[bool, bool]:
        """(retry the call, retire the slot) after an engine call of ``slot`` (``_gpu_problem``: out of memory, or
        a GPU the engine cannot use); ``key``: the call's queue key, ``retired_before``: slots retired on its GPU
        when the call started."""
        problem = None if self._stop.is_set() else _gpu_problem(result)
        if problem is None:
            self._settled(key)
            return False, False
        with self._lock:
            # also the last slot of a GPU while other GPUs dock: they take over its calls (a GPU too small for the
            # engine, or busy with other work, would otherwise lose every call it gets)
            retire = sum(self._active.values()) > 1
            if retire:
                self._active[slot.device] -= 1
                self._retired[slot.device] += 1
                self._retired_by.setdefault(key, []).append(slot)
            retry = retire or self._retired[slot.device] > retired_before
            back = [] if retry else self._retired_by.pop(key, [])
            if not self._closed:
                for other in back:  # its old thread has ended (or only returns): the slot's folder is free
                    self._start(other)
            left = self._active[slot.device]
        what = f"{call.protein_id} pocket {call.pocket.rank} batch {call.batch_no}"
        where = "the GPU" if slot.device is None else f"GPU {slot.device}"
        last = _last_line(result.log_tail)
        if back:
            gpus = sorted({"the GPU" if s.device is None else f"GPU {s.device}" for s in back})
            logger.warning("%s: the engine %s (%s) also alone, wherever it ran: the call fails%s, so the %d engine "
                           "slot%s it retired dock again (%s)", what, problem, last,
                           ", not the GPUs" if gpu.is_out_of_memory(result.log_tail) else "", len(back),
                           "s" * (len(back) > 1), ", ".join(gpus))
        elif retire and not left:
            logger.warning("%s: the engine %s (%s); %s is not used from now on, the call is retried on the other "
                           "GPUs", what, problem, last, where)
        elif retire:
            logger.warning("%s: the engine %s (%s); %s runs %d engine%s from now on, the call is retried", what,
                           problem, last, where, left, "s" * (left > 1))
        elif retry:
            logger.warning("%s: the engine %s (%s) while %s ran more engines; the call is retried", what, problem,
                           last, where)
        return retry, retire

    def close(self, cancel: bool = False) -> None:
        """Stop the workers once the queued calls are done, or (``cancel``) at once: calls not started are dropped
        and the running engines killed (Linux; elsewhere the running calls are waited for)."""
        with self._lock:  # no slot is started again from now on (see _gpu_failed)
            if self._closed:
                return
            self._closed = True
            if cancel:
                self._stop.set()
            threads = list(self._threads)
        if cancel:
            while True:
                try:
                    self._tasks.get_nowait()
                except queue.Empty:
                    break
        for _ in threads:
            self._tasks.put(((-math.inf if cancel else math.inf, next(self._seq)), None))
        told = False
        for t in threads:
            while t.is_alive():
                if cancel:
                    _kill_engines(self._marker)
                t.join(timeout=0.25)
                if t.is_alive() and not told:
                    logger.info("waiting for the running engine calls to end")
                    told = True


# --- proteins ------------------------------------------------------------------------------------------------------

def _pose_sdf(lig: _Ligands, protein_id: str, i: int, k: int, model: bytes):
    m = _MODEL_RE.match(model)
    modes = parse_modes(model)
    props = {"_Name": lig.ids[i], "protein_id": protein_id, "ligand_idx": i, "pocket": k,
             "mode": int(m.group(1)) if m else None, "affinity": modes[0].affinity if modes else None}
    return pose_to_sdf(model, lig.smiles[i], props)


def _child_error(stdout: str) -> Exception:
    """The exception that ``receptor_pdbqt`` raised in a ``_RECEPTOR_CODE`` process (its last stdout line), as the
    same built-in exception type when there is one."""
    try:
        name, message = json.loads(stdout.strip().splitlines()[-1])
    except (ValueError, IndexError, TypeError):
        return RuntimeError(f"receptor conversion failed: {stdout.strip()[-500:]}")
    cls = getattr(builtins, str(name), None)
    if isinstance(cls, type) and issubclass(cls, Exception):
        try:
            return cls(message)
        except TypeError:  # needs other arguments (e.g. UnicodeDecodeError)
            pass
    return RuntimeError(f"{name}: {message}")


def _receptor_pdbqt(source: str, dest: str, while_running: Optional[Callable[[], None]] = None,
                    legacy: bool = False) -> bytes:
    """``receptor_pdbqt(source, legacy)`` as UTF-8 bytes, also written to ``dest``.

    With ``while_running`` (engine calls are under way), a structure that needs OpenBabel is converted in a fresh
    Python process while ``while_running`` is called every ``P2RANK_POLL_S`` seconds (``run_atlas`` records the
    engine calls that are back): OpenBabel holds the GIL for the whole conversion, tens of seconds for a large
    protein with ``legacy``, which would also stop the engine slots from handing back their calls and starting the
    next ones. If ``while_running`` raises, or the wait is interrupted, that process is killed. A ``receptor_pdbqt``
    replaced in this module (a stub), a PDBQT input, or a child process that cannot run (e.g. no usable
    ``sys.executable``) converts in this process; a child killed by a signal (e.g. a crash in OpenBabel) raises
    ``RuntimeError``, since converting here would end the run with it.
    """
    # sys.executable may be None or "" (an embedded interpreter): Popen would raise TypeError for None
    if (while_running is None or receptor_pdbqt is not _MAIN_RECEPTOR_PDBQT or not sys.executable
            or structure_kind(source)[0] == "pdbqt"):
        data = receptor_pdbqt(source, legacy=legacy).encode("utf-8")
        with open(dest, "wb") as f:
            f.write(data)
        return data
    from .bust import _child_env

    cmd = [sys.executable, "-c", _RECEPTOR_CODE, source, dest, "legacy" if legacy else "rigid"]
    try:
        # cwd: dest's folder (a temp folder), so that "-c" finds no other "pocketvina" in the caller's working dir
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(dest), env=_child_env(), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, text=True, errors="replace", start_new_session=_PROCESS_GROUPS)
    except OSError as e:
        logger.warning("cannot convert the receptor in a child process (%s), converting it here", e)
        return _receptor_pdbqt(source, dest, legacy=legacy)
    with proc:
        stdout = _communicate(proc, while_running)
    if proc.returncode == 0:
        with open(dest, "rb") as f:
            return f.read()
    if proc.returncode == _RECEPTOR_EXIT_ERROR:
        raise _child_error(stdout)
    if proc.returncode < 0:  # killed by a signal: a crash (or the OOM killer) here would end the run, calls and all
        try:
            name = signal.Signals(-proc.returncode).name
        except ValueError:
            name = f"signal {-proc.returncode}"
        raise RuntimeError(f"the receptor conversion process was killed by {name} (OpenBabel crashed, or it ran out "
                           f"of memory) {_last_line(stdout)}".strip())
    logger.warning("the receptor conversion process failed (exit code %d: %s), converting it here", proc.returncode,
                   _last_line(stdout))
    return _receptor_pdbqt(source, dest, legacy=legacy)


def _mib(record: Optional[dict], key: str) -> Optional[float]:
    value = record.get(key) if record else None
    return None if value is None else round(value, 1)


def _stage_structure(source: str, protein_id: str, stage: str, gzip_level: int) -> Tuple[str, str]:
    """Gzipped copy of the input structure (a gzipped input is copied as it is): (path in results, temp path)."""
    kind, gz = structure_kind(source)
    rel = f"structures/{protein_id}.{kind}.gz"
    dst = os.path.join(stage, *rel.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if gz:
        shutil.copyfile(source, dst)
    else:
        with open(source, "rb") as f:
            data = f.read()
        with open(dst, "wb") as f:
            f.write(posefile.gz_member(data, gzip_level))
    return rel, dst


_no_points_logged = False  # "P2Rank wrote no SAS points file" is logged once (_stage_sas)


def _stage_sas(predictions_csv: str, protein_id: str, docked: Sequence[int], stage: str,
               gzip_level: int) -> Optional[Tuple[str, str]]:
    """P2Rank's SAS points of the protein (its points file next to ``predictions_csv``) as ``sas/<protein_id>.json.gz``
    (``io.sas``, the docked pockets' points in "pockets"): (path in results, temp path); None if P2Rank wrote no
    points file (a warning, once), or it cannot be read or has no points (a warning: the protein is docked all the
    same)."""
    global _no_points_logged
    path = sas.points_file(predictions_csv)
    if path is None:
        if not _no_points_logged:
            _no_points_logged = True
            logger.warning("%s: P2Rank wrote no SAS points file (visualizations/data/*_points.pdb.gz next to %s), so "
                           "the results keep no SAS points for it (said once: other proteins may have none either)",
                           protein_id, os.path.basename(predictions_csv))
        return None
    try:
        points = sas.read_points(path)
    except (OSError, ValueError, EOFError, zlib.error) as e:
        logger.warning("%s: P2Rank's SAS points %s cannot be read, none are kept: %s", protein_id, path, e)
        return None
    if not points:
        logger.warning("%s: P2Rank's SAS points file %s has no points, none are kept", protein_id, path)
        return None
    rel = f"sas/{protein_id}.json.gz"
    dst = os.path.join(stage, *rel.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(sas.encode(sas.build(points, docked, protein_id), gzip_level))
    return rel, dst


@dataclasses.dataclass
class _ProteinJob:
    """A protein whose engine calls are under way (main thread only)."""

    n: int                      # 1-based position in the protein list
    prot: ProteinInput
    work: str                   # its temp folder
    writer: ProteinWriter
    pockets: List[Pocket]       # all predicted pockets
    docked: List[Pocket]        # the pockets docked
    notes: List[str]
    receptor_sha256: bytes
    structure: Tuple[str, str]  # (path in results, staged temp path)
    calls: List[_Call]
    t0: float
    pending: int = 0            # calls not back yet
    n_runs: int = 0
    n_failed: int = 0
    n_outputs: int = 0
    last_failure: Optional[str] = None
    error: Optional[str] = None  # set when the protein failed: nothing is committed
    p2rank_s: float = 0.0       # how long P2Rank took (0 for pockets given as a CSV)
    receptor_s: float = 0.0     # how long the receptor PDBQT took
    t_docking: float = 0.0      # time.monotonic() when its engine calls were queued
    memory: Optional[dict] = None  # peak memory while it docked (gpu.MemorySampler record)
    sas: Optional[Tuple[str, str]] = None  # P2Rank's SAS points (path in results, staged temp path); None: none

    def cleanup(self):
        self.writer.cleanup()
        shutil.rmtree(self.work, ignore_errors=True)


def _prepare_protein(pv: PocketVinaGPU, n: int, prot: ProteinInput, lig: _Ligands, batches, work: str,
                     params: dict, ligands_sha256: Optional[str],
                     while_running: Optional[Callable[[], None]] = None) -> _ProteinJob:
    """Pockets, receptor PDBQT, staged structure and writer of one protein (in ``work``) and its engine calls; the
    runs of the ligands that were not prepared are added at once. ``while_running``: see ``predict_pockets`` and
    ``_receptor_pdbqt``."""
    cfg = pv.config
    pid = prot.protein_id
    t0 = time.monotonic()
    if not os.path.isfile(prot.source_path):
        raise FileNotFoundError(f"structure not found: {prot.source_path}")
    pockets_csv = prot.pockets_csv or predict_pockets(prot.source_path, os.path.join(work, "p2rank"),
                                                      cfg.p2rank_config, while_running=while_running)
    p2rank_s = 0.0 if prot.pockets_csv else time.monotonic() - t0
    pockets = read_pockets_csv(pockets_csv)
    if prot.pockets_csv:
        logger.info("%s: %d pockets read from %s", pid, len(pockets), prot.pockets_csv)
    else:
        logger.info("%s: pocket prediction (P2Rank) took %.1f s: %d pockets", pid, p2rank_s, len(pockets))
    receptor = os.path.join(work, f"{pid}.pdbqt")
    t_receptor = time.monotonic()
    receptor_bytes = _receptor_pdbqt(prot.source_path, receptor, while_running,
                                    legacy=cfg.ligand_prep == "legacy")
    receptor_sha256 = hashlib.sha256(receptor_bytes).digest()
    receptor_s = time.monotonic() - t_receptor
    logger.info("%s: receptor preparation took %.1f s", pid, receptor_s)

    notes = []
    top, ranks = pv.top_pockets, cfg.pockets
    docked = [p for p in pockets if (top is None or p.rank <= top) and (ranks is None or p.rank in ranks)]
    if any(p.rank > MAX_POCKET for p in docked):
        n_high = sum(p.rank > MAX_POCKET for p in docked)
        notes.append(f"{n_high} pockets ranked above {MAX_POCKET} were not docked (int8 pocket columns)")
        logger.warning("%s: %s", pid, notes[-1])
        docked = [p for p in docked if p.rank <= MAX_POCKET]
    K = top if top is not None else max(ranks) if ranks else max((p.rank for p in docked), default=0)

    structure = _stage_structure(prot.source_path, pid, os.path.join(work, "stage"), cfg.gzip_level)
    # pockets of a CSV (given, or the database's) come without points
    sas_file = None if prot.pockets_csv else _stage_sas(pockets_csv, pid, [p.rank for p in docked],
                                                        os.path.join(work, "stage"), cfg.gzip_level)
    meta = dict(params=params, source=os.path.basename(prot.source_path), structure_file=structure[0],
                receptor_sha256=receptor_sha256.hex(), ligands_sha256=ligands_sha256)
    writer = ProteinWriter(pid, lig.n, [p.rank for p in docked], K, work, meta, uniprot=prot.uniprot,
                           gzip_level=cfg.gzip_level)
    try:
        for pocket in docked:
            for i, reason in lig.skipped.items():
                writer.add_run(i, lig.ids[i], pocket.rank, None, "skipped", error=reason)
    except BaseException:
        writer.cleanup()
        raise
    calls = [_Call(pid, receptor, pocket, batch_no, lig_dir, stems)
             for pocket in docked for batch_no, (lig_dir, stems) in enumerate(batches, 1)]
    return _ProteinJob(n, prot, work, writer, pockets, docked, notes, receptor_sha256, structure, calls, t0,
                       pending=len(calls), p2rank_s=p2rank_s, receptor_s=receptor_s, t_docking=time.monotonic(),
                       sas=sas_file)


def _finish_protein(pv: PocketVinaGPU, rd: ResultsDir, job: _ProteinJob, lig: _Ligands) -> dict:
    """Finalize and commit a protein whose engine calls are all recorded; returns its summary. Raises if every
    engine call failed (e.g. no GPU or a broken OpenCL setup: the protein is left for a rerun)."""
    pc = pv.pocket_config
    pid = job.prot.protein_id
    if job.n_runs and job.n_failed == job.n_runs and not job.n_outputs:
        hint = gpu.opencl_hint(job.last_failure or "")
        raise RuntimeError(f"all {job.n_runs} engine runs failed, the last one: {job.last_failure}"
                           + (f" ({hint})" if hint else ""))
    notes = list(job.notes)
    if job.n_failed:
        notes.append(f"{job.n_failed} of {job.n_runs} engine runs failed")
    structure_rel, structure_tmp = job.structure
    stage = os.path.join(job.work, "stage")
    staged = job.writer.finalize(lambda i, k, model: _pose_sdf(lig, pid, i, k, model))
    staged[structure_rel] = structure_tmp
    if job.sas is not None:
        staged[job.sas[0]] = job.sas[1]
    status = "ok" if job.docked else "no_pockets"
    protein_row = dict(protein_id=pid, uniprot=job.prot.uniprot, source_path=job.prot.source_path,
                       structure_file=structure_rel, receptor_sha256=job.receptor_sha256,
                       n_pockets=len(job.pockets), n_pockets_docked=len(job.docked), status=status,
                       error="; ".join(notes) or None)
    pocket_rows = [dict(protein_id=pid, pocket=p.rank, name=p.name, score=p.score, probability=p.probability,
                        center_x=p.center[0], center_y=p.center[1], center_z=p.center[2], size_x=pc.size_x,
                        size_y=pc.size_y, size_z=pc.size_z, residue_ids=p.residue_ids) for p in job.pockets]
    for name, rows, schema in (("proteins", [protein_row], tables.PROTEINS), ("pockets", pocket_rows, tables.POCKETS)):
        rel = f"tables/{name}/{pid}.parquet"
        path = os.path.join(stage, *rel.split("/"))
        tables.write_table(rows, schema, path)
        staged[rel] = path
    summary = dict(job.writer.summary, status=status, n_pockets=len(job.pockets), n_pockets_docked=len(job.docked),
                   engine_runs=job.n_runs, engine_runs_failed=job.n_failed,
                   runtime_s=round(time.monotonic() - job.t0, 3), p2rank_s=round(job.p2rank_s, 3),
                   receptor_s=round(job.receptor_s, 3), docking_s=round(time.monotonic() - job.t_docking, 3),
                   gpu_mem_peak_mib=_mib(job.memory, "gpu_mib"),
                   gpu_mem_one_gpu_peak_mib=_mib(job.memory, "gpu_one_mib"),
                   host_mem_peak_mib=_mib(job.memory, "host_mib"))
    rd.commit(pid, staged, summary)
    return summary


class _Docking:
    """The docking part of ``run_atlas`` (main thread): proteins are prepared in list order, their engine calls run
    in a ``_SlotPool``, and each protein is committed as soon as all its calls are back. The next protein is
    prepared while fewer calls wait than there are slots, so that no slot waits for it."""

    def __init__(self, pv: PocketVinaGPU, rd: ResultsDir, lig: _Ligands, params: dict, ligands_sha256: Optional[str],
                 tmp_root: str, devices: Sequence[Optional[str]], failed: dict, buster=None):
        cfg, pc = pv.config, pv.pocket_config
        self.pv, self.rd, self.lig, self.params, self.ligands_sha256 = pv, rd, lig, params, ligands_sha256
        self.tmp_root, self.devices, self.failed, self.buster = tmp_root, list(devices), failed, buster
        self.eng = _Engine((pc.size_x, pc.size_y, pc.size_z), pc.thread, cfg.seed, cfg.num_modes, cfg.energy_range,
                           cfg.engine_timeout)
        self.batches = None
        self.pool: Optional[_SlotPool] = None
        self.jobs: Dict[str, _ProteinJob] = {}
        self.total = 0
        self.n_docked = self.n_done_before = 0
        self.memory: Optional[gpu.MemorySampler] = None  # started with the slots

    def run(self, proteins: Sequence[ProteinInput]) -> None:
        self.total = len(proteins)
        todo = iter(enumerate(proteins, 1))
        more = True
        try:
            while True:
                self._collect(block=False)
                if more and (self.pool is None or self.pool.n_queued < self.pool.n_slots):
                    more = self._start_next(todo)
                    continue
                if not self.jobs:
                    break
                self._collect(block=True)
            if self.pool is not None:
                self.pool.close()
        finally:
            if self.pool is not None:  # stopped early (an error, Ctrl-C, SIGTERM): kill the engines still running
                self.pool.close(cancel=True)
            for job in self.jobs.values():
                job.cleanup()
            self.jobs.clear()
            if self.memory is not None:
                self.memory.stop()

    def _start_next(self, todo) -> bool:
        """Prepare the next protein that is not done yet and queue its engine calls; False when none is left."""
        for n, prot in todo:
            pid = prot.protein_id
            if self.rd.is_done(pid):
                self.n_done_before += 1
                logger.info("%s (%d/%d): already in %s, skipped", pid, n, self.total, self.rd.path)
                if self.buster is not None:  # skipped there if its PoseBusters results are in place
                    self.buster.submit(pid)
                continue
            if self.batches is None:  # only when something is left to dock
                self.batches = _write_batches(self.lig, os.path.join(self.tmp_root, "ligands"),
                                              self.params["ligands_per_call"])
                logger.info("%d ligands, at most %d per engine call: %d engine call%s per pocket", len(self.lig.ok),
                            self.params["ligands_per_call"], len(self.batches), "s" * (len(self.batches) != 1))
            work = os.path.join(self.tmp_root, "proteins", pid)
            try:
                os.makedirs(work)
                # while engine calls are due, the calls that come back meanwhile are recorded, so that a protein
                # whose calls are all done is committed at once, not after the next protein's P2Rank run and
                # receptor conversion (a stop then would lose it), and the slots go on with the queued calls
                job = _prepare_protein(self.pv, n, prot, self.lig, self.batches, work, self.params,
                                       self.ligands_sha256,
                                       while_running=(lambda: self._collect(block=False)) if self.jobs else None)
            except Exception as e:
                shutil.rmtree(work, ignore_errors=True)
                if isinstance(e, RunAborted):
                    raise
                self._failed(n, pid, f"{type(e).__name__}: {e}")
                return True
            self.jobs[pid] = job
            if not job.calls:
                self._finish(job)
                return True
            if self.pool is None:
                self.pool = self._start_pool(job.calls[0])
            for call in job.calls:
                self.pool.submit(call, n)
            job.memory = self.memory.track() if self.memory is not None else None
            return True
        return False

    def _start_pool(self, first: _Call) -> _SlotPool:
        """The engine slots, started with the first engine call: the settings are checked on it in this thread
        (they are the same for every call, so a problem stops the run before any engine starts), and kernel caches
        that several slots share are built first (``_warm_kernel_caches``)."""
        slots = [_Slot(f"slot{n}", device, os.path.join(self.tmp_root, "slots", f"slot{n}"),
                       _kernel_cache_for(device)) for n, device in enumerate(self.devices, 1)]
        config, _, _ = _write_call_config(self.eng, first, slots[0])
        _unlink(config)
        _warm_kernel_caches(slots, self.eng, first, self.lig, self.tmp_root)
        pool = _SlotPool(slots, self.eng)
        self.memory = gpu.MemorySampler(running_engines).start()
        return pool

    def _collect(self, block: bool) -> None:
        """Record the engine calls that are back (``block``: wait up to ``RESULT_POLL_S`` for one first)."""
        if self.pool is None:
            return
        while True:
            try:
                done = self.pool.results.get(timeout=RESULT_POLL_S) if block else self.pool.results.get_nowait()
            except queue.Empty:
                if block and not self.pool.alive():
                    raise RuntimeError("the engine slots stopped while engine calls were due") from None
                # back to run() also when nothing came back: the slots may have taken the queued calls meanwhile
                # (they take a protein's calls only after it is queued), and the next protein is then due
                return
            self._record(done)
            block = False

    def _record(self, done: _Done) -> None:
        call = done.call
        job = self.jobs.get(call.protein_id)
        if job is None:
            return
        job.pending -= 1
        if job.error is None:
            try:
                if done.error is not None:
                    raise done.error
                if done.output is not None:
                    ok, n_out, failure = _record_call(job.writer, self.lig, self.eng, call, done.output)
                    job.n_runs += 1
                    job.n_outputs += n_out
                    if not ok:
                        job.n_failed += 1
                        job.last_failure = failure
                        logger.warning("%s pocket %d batch %d: %s", call.protein_id, call.pocket.rank, call.batch_no,
                                       failure)
            except RunAborted:
                raise
            except Exception as e:
                job.error = f"{type(e).__name__}: {e}"
                self.pool.drop(call.protein_id)
        if job.pending == 0:
            self._finish(job)

    def _finish(self, job: _ProteinJob) -> None:
        """Commit a protein whose calls are all back, or record why it failed; its temp files are removed."""
        pid = job.prot.protein_id
        del self.jobs[pid]
        if job.memory is not None:
            self.memory.untrack(job.memory)
        summary = None
        try:
            if job.error is None:
                summary = _finish_protein(self.pv, self.rd, job, self.lig)
        except RunAborted:
            raise
        except Exception as e:
            job.error = f"{type(e).__name__}: {e}"
        finally:
            job.cleanup()
        if summary is None:
            self._failed(job.n, pid, job.error)
            return
        self.n_docked += 1
        if self.buster is not None:
            self.buster.submit(pid)
        logger.info("%s (%d/%d): docking took %.1f s: %d pockets x %d ligands, %d runs %s, %d highest-affinity poses",
                    pid, job.n, self.total, summary["docking_s"], summary["n_pockets_docked"], len(self.lig.ok),
                    summary["n_runs"], summary["runs_by_status"], summary["n_with_pose"])
        if job.memory is not None:
            logger.info("%s (%d/%d): peak memory while docking: %s", pid, job.n, self.total,
                        gpu.format_memory(job.memory))


    def _failed(self, n: int, pid: str, reason: str) -> None:
        self.failed[pid] = reason
        logger.error("%s (%d/%d): failed, not committed (a rerun retries it): %s", pid, n, self.total, reason)


# --- run -----------------------------------------------------------------------------------------------------------

@exit_on_sigterm()
def run_atlas(pv, force: bool = False) -> str:
    """Dock every protein of ``pv`` (a ``PocketVinaGPU`` or a ``PocketVinaConfig``) into the atlas results directory
    ``config.output_dir`` and return its path.

    Proteins already committed there are skipped. A protein that fails (missing file, P2Rank error, every engine run
    failed or timed out) is logged, left out and retried by the next run; the ids are in ``pv.failed_proteins``.
    An engine call that fails or takes longer than ``config.engine_timeout`` while other calls of the protein work
    leaves its ligands as ``engine_error`` runs of the committed protein. A results
    directory that holds a run with other settings or ligands raises ``ValueError`` unless ``force=True`` (proteins
    done stay done, unless the ligand list changed: see ``ResultsDir.init_run``).
    The engine calls run on the slots of ``plan_engine_slots`` (config ``gpus``, ``procs_per_gpu``,
    ``vram_per_proc_mib``), several at a time when there are several slots; they are not run parameters, so a run
    may be resumed with others. All temp files live in one folder from ``config.tmp_dir`` (see
    ``io.writer.resolve_tmp_dir``), removed at the end, also when the run is stopped with Ctrl-C or SIGTERM
    (``engine.exit_on_sigterm``; the running engines are killed).
    With ``config.posebusters``, every protein committed (or found done) is handed to a background process that runs
    ``bust.bust_results`` on it while docking goes on; the run waits for it at the end. A protein whose checks did
    not finish (the run was stopped) is checked when the run is resumed.
    """
    if isinstance(pv, PocketVinaConfig):
        if pv.output_format != "atlas":  # re-validated (num_modes must fit the atlas tables)
            pv = dataclasses.replace(pv, output_format="atlas")
        pv = PocketVinaGPU(pv)
    elif pv.config.output_format != "atlas":
        # a "files" PocketVinaGPU has already put P2Rank folders and logs into output_dir
        raise ValueError('run_atlas needs a PocketVinaGPU made with output_format="atlas" (or pass the config)')
    cfg = pv.config
    failed = pv.failed_proteins = {}  # filled as the run goes (also seen after RunAborted or an interrupt)
    rd = ResultsDir(cfg.output_dir)
    proteins = protein_inputs(pv)
    rows = read_smiles_file(cfg.smiles_file)
    if not proteins:
        raise ValueError(f"no proteins in {cfg.protein_paths or cfg.protein_list}")
    if not rows:
        raise ValueError(f"no ligands in {cfg.smiles_file}")
    check_ligand_names([name for name, _ in rows], files=False)
    if cfg.posebusters:
        from . import bust

        bust.require_posebusters()  # before docking, not after it
    params = run_params(pv, len(rows))
    if cfg.ligands_per_call is not None and cfg.batch_size is not None:
        logger.warning("batch_size=%d is ignored: ligands_per_call=%d sets the ligands per engine call",
                       cfg.batch_size, cfg.ligands_per_call)
    devices = plan_engine_slots(cfg)
    t0 = time.monotonic()
    tmp_root = resolve_tmp_dir(cfg.tmp_dir)
    buster = docking = None
    try:
        lig = _Ligands(_ligand_table(rd, rows, params, cfg, force))
        ligands_sha256 = (rd.run_manifest() or {}).get("ligands_sha256")
        if cfg.posebusters:
            buster = bust.BackgroundBuster(rd.path, workers=cfg.posebusters_workers,
                                           tmp_dir=os.path.join(tmp_root, "posebusters"))
        docking = _Docking(pv, rd, lig, params, ligands_sha256, tmp_root, devices, failed, buster)
        docking.run(proteins)
        if buster is not None:
            logger.info("waiting for the PoseBusters checks (%d proteins)", buster.submitted)
            code, buster = buster.close(), None
            if code:
                logger.error("PoseBusters checks failed for some proteins (see the log); run bust.bust_results to "
                             "retry them")
    finally:
        if buster is not None:  # the run was stopped: proteins not checked yet are checked when it is resumed
            buster.terminate()
        shutil.rmtree(tmp_root, ignore_errors=True)
    n_docked, n_done_before = (docking.n_docked, docking.n_done_before) if docking else (0, 0)
    logger.info("run finished in %.1f s: %d proteins docked, %d done before, %d failed (results in %s; the steps of "
                "consecutive proteins overlap, the next protein's pockets are predicted while the GPUs dock)",
                time.monotonic() - t0, n_docked, n_done_before, len(failed), rd.path)
    if docking is not None and docking.memory is not None:
        logger.info("peak memory of the run: %s", gpu.format_memory(docking.memory.run))
    if failed:
        logger.error("failed proteins (rerun to retry): %s", ", ".join(failed))
    return rd.path
