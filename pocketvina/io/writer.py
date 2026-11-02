"""Write a results directory in the "atlas" layout (docs/dev/interfaces.md, "Results directory layout").

``ResultsDir``     run-level ``MANIFEST.json`` + ``tables/ligands.parquet``; per-protein ``commit`` and resume
``ProteinWriter``  collects the engine outputs of one protein, writes its pose file and Parquet parts into a temp dir

Several processes (e.g. SLURM array jobs) may write different proteins into the same results directory at once.
Every file is written elsewhere first and moved into place atomically; a protein's ``manifest/<protein_id>.json`` is
written last, so its presence means the protein is done.
"""
import errno
import hashlib
import json
import logging
import os
import posixpath
import re
import shutil
import sys
import tempfile
import time
import uuid
from array import array
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Callable, Dict, NamedTuple, Optional, Tuple

import numpy as np

# resolve_tmp_dir lives in the (pyarrow-free) engine module, so the "files" format can use it; part of this API too
from ..engine import TMP_ENV_VARS, resolve_tmp_dir  # noqa: F401
from ..pdbqt import check_modes
from . import posefile, tables

logger = logging.getLogger(__name__)

FORMAT = "pocketvina-atlas"
FORMAT_VERSION = 1
MAX_POCKET = 127  # pocket is int8 in pocket_runs, modes and pairs
# fixed spool write buffer: open()'s default is st_blksize, which is 1 MiB or more on some network filesystems
SPOOL_BUFFER = 1 << 16
_PROTEIN_ID_RE = re.compile(r"[A-Za-z0-9._-]+")
_STRUCTURE_EXTS = (".pdbqt", ".pdb", ".ent", ".cif", ".mmcif")  # main.STRUCTURE_KINDS


# --- helpers -------------------------------------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _version() -> str:
    version = getattr(sys.modules.get("pocketvina"), "__version__", None)
    if version:
        return str(version)
    try:
        from importlib.metadata import version as dist_version
        return dist_version("pocketvina")
    except Exception:
        return "unknown"


def _json_default(obj):
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        try:
            return sorted(obj)
        except TypeError:
            return sorted(obj, key=repr)
    return str(obj)


def _jsonable(obj):
    """``obj`` as it reads back from JSON (tuples become lists, paths and other objects strings)."""
    return json.loads(json.dumps(obj, sort_keys=True, default=_json_default))


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, default=_json_default) + "\n").encode()


def _tmp_name(path: str) -> str:
    return f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


def _write_synced(path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _write_json(path, obj):
    """Write JSON atomically: temp file in the same folder, then rename."""
    tmp = _tmp_name(path)
    try:
        _write_synced(tmp, _json_bytes(obj))
        os.replace(tmp, path)
    except BaseException:
        _unlink(tmp)
        raise


def _create_exclusive(path, obj) -> bool:
    """Write JSON ``obj`` to ``path`` unless it already exists; True if this call created it.

    A finished temp file is hard-linked to ``path``, so readers never see it half written. Where the filesystem
    has no hard links (some FUSE mounts) the file is created with ``O_EXCL`` instead.
    """
    data = _json_bytes(obj)
    tmp = _tmp_name(path)
    try:
        _write_synced(tmp, data)
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            # NFS may report EEXIST when the reply to a successful link was lost: then tmp has two links
            return os.stat(tmp).st_nlink == 2
        except OSError as e:
            logger.debug("no hard links for %s (%s), creating it with O_EXCL", path, e)
    finally:
        _unlink(tmp)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    except FileExistsError:
        return False
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return True


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_protein_id(protein_id) -> str:
    """Return ``protein_id`` if it is a valid id (``[A-Za-z0-9._-]+``, not "." or ".."), else raise ``ValueError``."""
    if not isinstance(protein_id, str) or not _PROTEIN_ID_RE.fullmatch(protein_id) or protein_id in (".", ".."):
        raise ValueError(f"invalid protein_id {protein_id!r}: only [A-Za-z0-9._-] (see protein_id_from_path)")
    return protein_id


def protein_id_from_path(path) -> str:
    """``protein_id`` of an input structure: its file name without ``.pdb``, ``.ent``, ``.cif``, ``.mmcif`` or
    ``.pdbqt`` (also with ``.gz``), characters other than ``[A-Za-z0-9._-]`` replaced by ``_``."""
    name = os.path.basename(os.fspath(path))
    low = name.lower()
    if low.endswith(".gz") and low[:-3].endswith(_STRUCTURE_EXTS):
        name, low = name[:-3], low[:-3]
    for ext in _STRUCTURE_EXTS:
        if low.endswith(ext):
            name = name[:-len(ext)]
            break
    return check_protein_id(re.sub(r"[^A-Za-z0-9._-]", "_", name))


def _check_relpath(rel) -> str:
    if not isinstance(rel, str) or not rel:
        raise ValueError(f"staged path must be a non-empty string: {rel!r}")
    norm = posixpath.normpath(rel)
    if norm.startswith("/") or norm in (".", "..") or norm.startswith("../"):
        raise ValueError(f"staged path must be inside the results directory: {rel!r}")
    if norm in ("MANIFEST.json", "manifest") or norm.startswith("manifest/"):
        raise ValueError(f"{rel!r} is written by ResultsDir itself")
    return norm


def ligands_sha256(table) -> str:
    """Content digest of a ligand table (``tables.LIGANDS`` columns, row by row), independent of the Parquet
    encoding, so a resumed run can check that it uses the same ligands."""
    table = tables.to_arrow(table, tables.LIGANDS)
    h = hashlib.sha256(json.dumps(table.schema.names).encode())
    for batch in table.to_batches(max_chunksize=8192):
        for row in zip(*(col.to_pylist() for col in batch.columns)):
            h.update(json.dumps(row, separators=(",", ":")).encode())
            h.update(b"\n")
    return h.hexdigest()


# --- atomic moves --------------------------------------------------------------------------------------------------

def atomic_place(src: str, dst: str) -> None:
    """Move file ``src`` to ``dst`` atomically, creating the parent folders and replacing an existing ``dst``.

    Same filesystem: ``os.replace``. Different filesystem: copy to ``<dst>.tmp-<pid>-<random>`` next to ``dst``,
    fsync, ``os.replace`` over ``dst``, then remove ``src``. Readers see the old file or the new one, never a part.
    """
    src, dst = os.fspath(src), os.fspath(dst)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    try:
        os.replace(src, dst)
        return
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
    tmp = _tmp_name(dst)
    try:
        with open(src, "rb") as fi, open(tmp, "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 22)
            fo.flush()
            os.fsync(fo.fileno())
        os.replace(tmp, dst)
    except BaseException:
        _unlink(tmp)
        raise
    os.unlink(src)


# --- results directory ---------------------------------------------------------------------------------------------

class ResultsDir:
    """A results directory (created on first write). Processes writing different proteins may share it."""

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(os.fspath(path)))

    def __repr__(self):
        return f"ResultsDir({self.path!r})"

    def __fspath__(self):
        return self.path

    @property
    def manifest_file(self) -> str:
        return os.path.join(self.path, "MANIFEST.json")

    @property
    def ligands_file(self) -> str:
        return os.path.join(self.path, "tables", "ligands.parquet")

    def protein_manifest_file(self, protein_id) -> str:
        return os.path.join(self.path, "manifest", f"{check_protein_id(protein_id)}.json")

    def run_manifest(self) -> Optional[dict]:
        """``MANIFEST.json``, or None if the run was not initialised."""
        deadline = time.monotonic() + 10
        while True:
            try:
                with open(self.manifest_file, "rb") as f:
                    return json.loads(f.read())
            except FileNotFoundError:
                return None
            except json.JSONDecodeError:
                # being written by another process (only where O_EXCL replaces hard links)
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    def protein_manifest(self, protein_id) -> Optional[dict]:
        """``manifest/<protein_id>.json``, or None if the protein is not done."""
        try:
            with open(self.protein_manifest_file(protein_id), "rb") as f:
                return json.loads(f.read())
        except FileNotFoundError:
            return None

    def init_run(self, params: dict, ligands_df, force: bool = False, tmp_dir: Optional[str] = None) -> None:
        """Start or resume a run: write ``MANIFEST.json`` and ``tables/ligands.parquet`` if absent.

        ``ligands_df`` has the ``tables.LIGANDS`` columns, ``ligand_idx`` = 0..N-1 in row order and unique
        ``ligand_id``. If the directory already holds a run with other ``params`` (compared as JSON) or other ligands,
        raise ``ValueError``, or with ``force=True`` replace the run definition. Proteins already done are kept,
        unless the ligand list (``ligand_id`` and ``smiles_input`` by ``ligand_idx``) changed: their tables and pose
        files index the old list, so they are no longer done and a run docks them again.
        Several processes may call this at once: one of them writes ``MANIFEST.json``, the others compare with it.
        ``ligands.parquet`` is staged in ``resolve_tmp_dir(tmp_dir)``.
        """
        if not isinstance(params, Mapping):
            raise TypeError(f"params must be a dict, not {type(params).__name__}")
        table = tables.to_arrow(ligands_df, tables.LIGANDS)
        idx = table.column("ligand_idx").to_numpy()
        if not np.array_equal(idx, np.arange(len(idx))):
            raise ValueError("ligand_idx must be 0, 1, ..., N-1 in row order")
        ids = table.column("ligand_id").to_pylist()
        if len(set(ids)) != len(ids):
            dup = sorted(k for k, n in Counter(ids).items() if n > 1)
            raise ValueError(f"duplicate ligand_id: {dup[:5]}{' ...' if len(dup) > 5 else ''}")
        params = _jsonable(dict(params))
        digest = ligands_sha256(table)
        manifest = dict(format=FORMAT, format_version=FORMAT_VERSION, pose_format=posefile.MAGIC.decode(),
                        pose_format_version=posefile.VERSION, pocketvina_version=_version(), created=_now(),
                        params=params, n_ligands=table.num_rows, ligands_sha256=digest)
        os.makedirs(self.path, exist_ok=True)
        existing = self.run_manifest()
        if existing is None:
            if _create_exclusive(self.manifest_file, manifest):
                self._place_ligands(table, tmp_dir)
                logger.info("new run in %s (%d ligands)", self.path, table.num_rows)
                return
            existing = self.run_manifest()
            if existing is None:
                raise RuntimeError(f"{self.manifest_file} disappeared while starting the run")
        problems = _run_differences(existing, params, digest)
        if problems:
            if not force:
                raise ValueError(f"{self.path} holds another run ({'; '.join(problems)}): use a new results "
                                 "directory, or force=True to replace the run definition")
            logger.warning("replacing the run definition of %s (%s)", self.path, "; ".join(problems))
            stale = sorted(self.done_proteins()) if not self._lists_ligands(table) else []
            if stale:
                # before the new ligands: a crash in between must not leave these done against the new list
                logger.warning("%d protein(s) done with another ligand list are no longer done and will be docked "
                               "again: %s", len(stale), ", ".join(stale[:10]) + (" ..." if len(stale) > 10 else ""))
                for pid in stale:
                    _unlink(self.protein_manifest_file(pid))
            self._place_ligands(table, tmp_dir)
            _write_json(self.manifest_file, manifest)
            return
        if not os.path.isfile(self.ligands_file):
            self._place_ligands(table, tmp_dir)  # a process stopped between MANIFEST.json and ligands.parquet
        logger.info("resuming the run in %s (%d proteins done)", self.path, len(self.done_proteins()))

    def _lists_ligands(self, table) -> bool:
        """True if ``ligands.parquet`` holds the ligands of ``table``: same ``ligand_id``, ``smiles_input`` per
        ``ligand_idx``. Other columns (e.g. conformers of another seed) do not change what ``ligand_idx`` means."""
        try:
            old = tables.read_table(self.ligands_file, columns=["ligand_id", "smiles_input"])
        except (OSError, ValueError):  # missing or unreadable
            return False
        return (old["ligand_id"].tolist() == table.column("ligand_id").to_pylist()
                and old["smiles_input"].tolist() == table.column("smiles_input").to_pylist())

    def _place_ligands(self, table, tmp_dir=None):
        tmp = resolve_tmp_dir(tmp_dir)
        try:
            local = os.path.join(tmp, "ligands.parquet")
            tables.write_table(table, tables.LIGANDS, local)
            atomic_place(local, self.ligands_file)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def is_done(self, protein_id) -> bool:
        return os.path.isfile(self.protein_manifest_file(protein_id))

    def done_proteins(self) -> set:
        try:
            names = os.listdir(os.path.join(self.path, "manifest"))
        except FileNotFoundError:
            return set()
        return {n[:-5] for n in names
                if n.endswith(".json") and _PROTEIN_ID_RE.fullmatch(n[:-5]) and n[:-5] not in (".", "..")}

    def commit(self, protein_id, staged: Dict[str, str], summary: dict) -> dict:
        """Move the staged files (``{path relative to the results dir: temp path}``) into place, then write
        ``manifest/<protein_id>.json`` (sha256 and size of every file, ``summary``) last. Committing a done protein
        again replaces its files (it is not done while they are replaced). Returns the manifest."""
        check_protein_id(protein_id)
        entries = {}
        for rel, src in staged.items():
            norm = _check_relpath(rel)
            if norm in entries:
                raise ValueError(f"two staged files for {norm!r}")
            src = os.fspath(src)
            if not os.path.isfile(src):
                raise FileNotFoundError(f"staged file for {rel!r} not found: {src}")
            entries[norm] = (src, _file_sha256(src), os.path.getsize(src))
        manifest_file = self.protein_manifest_file(protein_id)
        os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
        if os.path.exists(manifest_file):
            logger.info("%s: replacing the committed results", protein_id)
            try:
                os.unlink(manifest_file)
            except FileNotFoundError:
                pass
        for rel, (src, _, _) in entries.items():
            atomic_place(src, os.path.join(self.path, *rel.split("/")))
        manifest = dict(format=FORMAT, format_version=FORMAT_VERSION, protein_id=protein_id,
                        pocketvina_version=_version(), committed=_now(),
                        files={rel: dict(sha256=h, bytes=n) for rel, (_, h, n) in sorted(entries.items())},
                        summary=_jsonable(dict(summary or {})))
        _write_json(manifest_file, manifest)
        return manifest

    def verify(self, protein_id) -> list:
        """Problems with a committed protein (missing manifest, missing or changed files); empty if all is well."""
        manifest = self.protein_manifest(protein_id)
        if manifest is None:
            return [f"{protein_id}: no manifest"]
        problems = []
        for rel, info in manifest.get("files", {}).items():
            path = os.path.join(self.path, *rel.split("/"))
            if not os.path.isfile(path):
                problems.append(f"{rel}: missing")
            elif os.path.getsize(path) != info["bytes"] or _file_sha256(path) != info["sha256"]:
                problems.append(f"{rel}: changed")
        return problems


def _run_differences(existing: dict, params: dict, digest: str) -> list:
    out = []
    if existing.get("format") != FORMAT:
        out.append(f"format {existing.get('format')!r}, not {FORMAT!r}")
    old = existing.get("params")
    if old != params:
        if isinstance(old, dict):
            missing = object()
            keys = sorted(k for k in set(old) | set(params) if old.get(k, missing) != params.get(k, missing))
            out.append("other params: " + ", ".join(keys))
        else:
            out.append("other params")
    if existing.get("ligands_sha256") != digest:
        out.append("other ligands")
    return out


# --- one protein ---------------------------------------------------------------------------------------------------

class _Run(NamedTuple):
    ligand_idx: int
    ligand_id: str
    pocket: int
    status: str
    n_modes: int
    best_affinity: Optional[float]
    spool_offset: int
    gz_length: int
    raw_length: int
    sha256: Optional[bytes]
    seed: Optional[int]
    runtime_s: Optional[float]
    error: Optional[str]
    first_mode: int  # index of its first mode in the mode columns


class ProteinWriter:
    """Collects the engine outputs of one protein; ``finalize`` writes ``poses/<protein_id>.poses`` and the
    ``pocket_runs``, ``modes`` and ``pairs`` Parquet parts into a private folder under ``tmp_dir``.

    Outputs are gzip-compressed and spooled to that folder as they are added, so memory holds only the scores.
    ``K`` is the number of pocket slots in the pose file; ``pocket_ranks`` the ranks (1..K) that may be added.
    Use it as a context manager, or call ``cleanup()``, to remove the folder once the staged files are committed.
    """

    def __init__(self, protein_id, n_ligands, pocket_ranks, K, tmp_dir, meta, uniprot=None, gzip_level=6):
        self.protein_id = check_protein_id(protein_id)
        self.n_ligands, self.K = int(n_ligands), int(K)
        ranks = [int(k) for k in pocket_ranks]
        if self.n_ligands < 0 or self.K < 0:
            raise ValueError(f"n_ligands and K must be >= 0, got {n_ligands}, {K}")
        if len(set(ranks)) != len(ranks) or any(not 1 <= k <= self.K for k in ranks):
            raise ValueError(f"pocket_ranks must be distinct ranks in 1..K={self.K}, got {ranks}")
        if any(k > MAX_POCKET for k in ranks):
            raise ValueError(f"pocket ranks above {MAX_POCKET} do not fit the int8 pocket columns: {ranks}")
        self.pocket_ranks = sorted(ranks)
        self._rank_set = frozenset(ranks)
        self.uniprot = uniprot
        self.gzip_level = int(gzip_level)
        if not -1 <= self.gzip_level <= 9:  # zlib's range; fail now, not at the first add_run after docking
            raise ValueError(f"gzip_level must be -1..9, got {gzip_level!r}")
        self.meta = dict(meta or {})
        self.meta.update(format=posefile.MAGIC.decode(), protein_id=self.protein_id, uniprot=uniprot, K=self.K,
                         n_ligands=self.n_ligands, pockets=self.pocket_ranks,
                         ligands="tables/ligands.parquet (ligand_idx)",
                         section_a="best pose per ligand, SDF", section_a_order="affinity",
                         section_b="engine output PDBQT per ligand x pocket, original bytes",
                         block_index="j = ligand_idx*K + pocket-1")
        json.dumps(self.meta, sort_keys=True)  # fail now, not after docking (posefile.Writer sorts the keys)
        if tmp_dir is None:
            self.dir = resolve_tmp_dir()
        else:
            tmp_dir = os.path.abspath(os.path.expanduser(os.fspath(tmp_dir)))
            os.makedirs(tmp_dir, exist_ok=True)
            self.dir = tempfile.mkdtemp(prefix=f"{self.protein_id}_", dir=tmp_dir)
        self._spool_path = os.path.join(self.dir, "blocks.spool")
        self._spool = open(self._spool_path, "w+b", buffering=SPOOL_BUFFER)
        self._spool_size = 0
        self._runs = []
        self._by_j = {}          # block index j -> index in _runs
        self._ligand_ids = {}    # ligand_idx -> ligand_id
        self._best = {}          # ligand_idx -> (affinity, pocket, mode, run index)
        self._modes = dict(mode=array("b"), affinity=array("d"), rmsd_lb=array("d"), rmsd_ub=array("d"),
                           raw_offset=array("I"), raw_length=array("I"), run=array("I"))
        self._finalized = False
        self.summary = None  # set by finalize()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()

    def cleanup(self):
        """Remove the temp folder, including staged files that were not committed."""
        if not self._spool.closed:
            self._spool.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    @property
    def n_runs(self) -> int:
        return len(self._runs)

    def add_run(self, ligand_idx, ligand_id, pocket, pdbqt: Optional[bytes], status, runtime_s=None, seed=None,
                error=None) -> str:
        """Add the engine output of one ligand x pocket (``pdbqt`` None if there is none) with the caller's
        ``status`` (``tables.RUN_STATUS``). The bytes are always kept; modes are read only for "ok" runs. An "ok"
        run whose output does not parse (``pdbqt.check_modes``) becomes "parse_error", one without output
        "no_output". Returns the status recorded."""
        if self._finalized:
            raise RuntimeError("finalize() was already called")
        i, k = int(ligand_idx), int(pocket)
        if not 0 <= i < self.n_ligands:
            raise ValueError(f"ligand_idx {i} out of range 0..{self.n_ligands - 1}")
        if k not in self._rank_set:
            raise ValueError(f"pocket {k} is not one of the pocket ranks {self.pocket_ranks}")
        if status not in tables.RUN_STATUS:
            raise ValueError(f"unknown status {status!r}, expected one of {tables.RUN_STATUS}")
        if ligand_id is None:
            raise ValueError("ligand_id is required")
        ligand_id = str(ligand_id)
        if self._ligand_ids.get(i, ligand_id) != ligand_id:
            raise ValueError(f"ligand {i} was added as {self._ligand_ids[i]!r}, now as {ligand_id!r}")
        j = i * self.K + (k - 1)
        if j in self._by_j:
            raise ValueError(f"ligand {i} pocket {k} was already added")
        if isinstance(pdbqt, str):
            raise TypeError("pdbqt must be bytes, not str")
        raw = bytes(pdbqt) if pdbqt else b""
        # convert before any state changes: a rejected call must leave the spool and mode columns as they were
        seed = None if seed is None else int(seed)
        runtime_s = None if runtime_s is None else float(runtime_s)
        error = None if error is None else str(error)

        modes, sha, offset, gz_length = [], None, self._spool_size, 0
        if raw:
            if status == "ok":
                try:
                    modes = check_modes(raw)
                except ValueError as e:
                    logger.warning("%s: ligand %d pocket %d: unreadable engine output: %s", self.protein_id, i, k, e)
                    status, error = "parse_error", f"{error}; {e}" if error else str(e)
            gz = posefile.gz_member(raw, self.gzip_level)
            self._spool.write(gz)
            self._spool_size += len(gz)
            gz_length, sha = len(gz), posefile.sha256(raw)
        elif status == "ok":
            status = "no_output"

        r = len(self._runs)
        first = len(self._modes["mode"])
        best_affinity = None
        if modes:
            cols = self._modes
            for m in modes:
                cols["mode"].append(m.mode)
                cols["affinity"].append(m.affinity)
                cols["rmsd_lb"].append(m.rmsd_lb)
                cols["rmsd_ub"].append(m.rmsd_ub)
                cols["raw_offset"].append(m.raw_offset)
                cols["raw_length"].append(m.raw_length)
                cols["run"].append(r)
            top = min(modes, key=lambda m: (m.affinity, m.mode))
            best_affinity = top.affinity
            current = self._best.get(i)
            if current is None or (top.affinity, k, top.mode) < current[:3]:
                self._best[i] = (top.affinity, k, top.mode, r)
        self._runs.append(_Run(i, ligand_id, k, status, len(modes), best_affinity, offset, gz_length, len(raw), sha,
                               seed, runtime_s, error, first))
        self._by_j[j] = r
        self._ligand_ids[i] = ligand_id
        return status

    def _read_spool(self, run: _Run) -> bytes:
        self._spool.seek(run.spool_offset)
        data = self._spool.read(run.gz_length)
        if len(data) != run.gz_length:
            raise OSError(f"spool file {self._spool_path} is truncated")
        return data

    def _convert(self, sdf_for, i, k, model):
        try:
            sdf, status = sdf_for(i, k, model)
        except Exception as e:  # one molecule must not cost the whole protein
            logger.warning("%s: SDF conversion of ligand %d (pocket %d) failed: %r", self.protein_id, i, k, e)
            return None, "convert_failed"
        if status not in tables.SDF_STATUS:
            raise ValueError(f"sdf_for returned sdf_status {status!r}, expected one of {tables.SDF_STATUS}")
        if isinstance(sdf, str):
            sdf = sdf.encode()
        if status != "ok":
            return None, status
        return (bytes(sdf), "ok") if sdf else (None, "convert_failed")

    def finalize(self, sdf_for: Callable[[int, int, bytes], Tuple[Optional[bytes], str]]) -> Dict[str, str]:
        """Write the pose file and the Parquet parts; returns ``{path relative to the results dir: temp path}``.

        Best pose of a ligand: the mode with the lowest affinity over its "ok" runs; ties go to the lower pocket,
        then the lower mode. ``sdf_for(ligand_idx, pocket, model_bytes)`` converts it to ``(sdf bytes or None,
        sdf_status)``; if it raises, the status is "convert_failed". Section A holds the SDFs in affinity order
        (then ligand_idx), section B every engine output in block order. ``pairs`` has a row for every ligand with at
        least one run; one without a pose has sdf_status "no_pose", NaN scores and best_pocket = best_mode = -1.
        """
        if self._finalized:
            raise RuntimeError("finalize() was already called")
        self._finalized = True
        pid, K = self.protein_id, self.K
        cols = {name: np.frombuffer(arr, dtype=arr.typecode) for name, arr in self._modes.items()}
        poses_path = os.path.join(self.dir, "poses", f"{pid}.poses")
        os.makedirs(os.path.dirname(poses_path), exist_ok=True)
        top, sdf_status = {}, {}
        try:
            self._spool.flush()
            w = posefile.Writer(poses_path, self.n_ligands, K, self.meta)
            try:
                for i in sorted(self._best, key=lambda i: (self._best[i][0], i)):
                    _, k, mode, r = self._best[i]
                    q = self._runs[r].first_mode + mode - 1
                    raw = posefile.gunzip_member(self._read_spool(self._runs[r]))
                    off, ln = int(cols["raw_offset"][q]), int(cols["raw_length"][q])
                    sdf, sdf_status[i] = self._convert(sdf_for, i, k, raw[off:off + ln])
                    if sdf:
                        gz = posefile.gz_member(sdf, self.gzip_level)
                        top[i] = (w.pos, len(gz), len(sdf), posefile.sha256(sdf))
                        w.add_top(i, gz, len(sdf))
                w.begin_modes()
                for j in sorted(self._by_j):
                    run = self._runs[self._by_j[j]]
                    if run.gz_length:
                        w.add_block(j, self._read_spool(run))
                info = w.close()
            except BaseException:
                w.f.close()
                raise
        finally:
            self._spool.close()
            _unlink(self._spool_path)

        B = np.asarray(w.B, dtype=np.uint64)
        order = sorted(self._by_j)
        runs = [self._runs[self._by_j[j]] for j in order]
        js = np.asarray(order, dtype=np.int64)
        n = len(runs)
        pocket_runs = dict(
            protein_id=[pid] * n, ligand_idx=[r.ligand_idx for r in runs], ligand_id=[r.ligand_id for r in runs],
            pocket=[r.pocket for r in runs], status=[r.status for r in runs], n_modes=[r.n_modes for r in runs],
            best_affinity=[r.best_affinity for r in runs], offset=B[js], length=(B[js + 1] - B[js]).astype(np.uint32),
            raw_length=[r.raw_length for r in runs], sha256=[r.sha256 for r in runs], seed=[r.seed for r in runs],
            runtime_s=[r.runtime_s for r in runs], error=[r.error for r in runs])

        j_of_run = np.empty(len(self._runs), dtype=np.int64)
        for j, r in self._by_j.items():
            j_of_run[r] = j
        mode_j = j_of_run[cols["run"]] if len(cols["run"]) else np.zeros(0, dtype=np.int64)
        sel = np.argsort(mode_j, kind="stable")  # block order; modes of one run stay in mode order
        mj = mode_j[sel]
        m = len(sel)
        modes = dict(
            protein_id=[pid] * m, uniprot=[self.uniprot] * m, ligand_idx=(mj // K).astype(np.int32) if K else mj,
            pocket=(mj % K + 1).astype(np.int8) if K else mj, mode=cols["mode"][sel],
            affinity=cols["affinity"][sel].astype(np.float32), rmsd_lb=cols["rmsd_lb"][sel].astype(np.float32),
            rmsd_ub=cols["rmsd_ub"][sel].astype(np.float32), block_offset=B[mj], block_length=(B[mj + 1] - B[mj]),
            raw_offset=cols["raw_offset"][sel], raw_length=cols["raw_length"][sel])
        modes["block_length"] = modes["block_length"].astype(np.uint32)

        rows = []
        nan = float("nan")
        for i in sorted(self._ligand_ids):
            row = dict(protein_id=pid, uniprot=self.uniprot, ligand_idx=i, ligand_id=self._ligand_ids[i],
                       pb_status=None)
            if i in self._best:
                aff, k, mode, r = self._best[i]
                q = self._runs[r].first_mode + mode - 1
                off, gz_len, raw_len, sha = top.get(i, (0, 0, 0, None))
                row.update(vina=aff, best_pocket=k, best_mode=mode, rmsd_lb=float(cols["rmsd_lb"][q]),
                           rmsd_ub=float(cols["rmsd_ub"][q]), top_offset=off, top_length=gz_len,
                           top_raw_length=raw_len, top_sha256=sha, sdf_status=sdf_status[i])
            else:
                row.update(vina=nan, best_pocket=-1, best_mode=-1, rmsd_lb=nan, rmsd_ub=nan, top_offset=0,
                           top_length=0, top_raw_length=0, top_sha256=None, sdf_status="no_pose")
            rows.append(row)

        staged = {f"poses/{pid}.poses": poses_path}
        for name, data, schema in (("pocket_runs", pocket_runs, tables.POCKET_RUNS), ("modes", modes, tables.MODES),
                                   ("pairs", rows, tables.PAIRS)):
            path = os.path.join(self.dir, "tables", name, f"{pid}.parquet")
            tables.write_table(data, schema, path)
            staged[f"tables/{name}/{pid}.parquet"] = path

        self.summary = dict(
            protein_id=pid, n_ligands=self.n_ligands, K=K, pockets=self.pocket_ranks, n_runs=n,
            runs_by_status=dict(Counter(r.status for r in runs)), n_modes=m, n_pairs=len(rows),
            n_with_pose=len(self._best), sdf_by_status=dict(Counter(row["sdf_status"] for row in rows)),
            poses_bytes=info["file_size"], section_a_bytes=info["section_a_bytes"],
            section_b_bytes=info["section_b_bytes"], raw_pdbqt_bytes=sum(r.raw_length for r in runs),
            raw_sdf_bytes=sum(t[2] for t in top.values()))
        logger.info("%s: %d runs, %d modes, %d best poses, pose file %d bytes", pid, n, m, len(top),
                    info["file_size"])
        return staged
