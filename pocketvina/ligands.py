"""Ligand preparation: SMILES -> engine input PDBQT, one ``LigandRecord`` (a ``LIGANDS`` table row) per ligand.

Methods:
  standard  largest organic fragment, seeded ETKDGv3 embedding (no minimisation), OpenBabel mol -> mol2 -> PDBQT
            in memory, first molecule up to ``TORSDOF``.
  legacy    the pre-v0.2 ``PocketVinaGPU.smiles_to_pdbqt`` steps (whole SMILES, unseeded ETKDG, mol2 file +
            ``obabel`` CLI, truncation after the first ``TORSDOF``), kept so published results can be regenerated.

The 3D embedding is RDKit's ETKDGv3 (`"Improving Conformer Generation for Small Rings and Macrocycles Based on Distance
Geometry and Experimental Torsional-Angle Preferences" <https://doi.org/10.1021/acs.jcim.0c00025>`_); the PDBQT
conversion is Open Babel's (`"Open Babel: An open chemical toolbox" <https://doi.org/10.1186/1758-2946-3-33>`_).
"""
import contextlib
import csv
import functools
import io
import logging
import operator
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

try:
    from rdkit.Chem import rdinchi
except ImportError:  # RDKit built without InChI
    rdinchi = None

logger = logging.getLogger(__name__)

METHODS = ("standard", "legacy")
# "error" = an unexpected exception; the other failures name the step that failed
STATUSES = ("ok", "invalid_smiles", "unsupported_element", "no_heavy_atoms", "embed_failed", "conversion_failed",
            "error")
SUPPORTED_ELEMENTS = frozenset({"C", "N", "O", "S", "P", "F", "Cl", "Br", "I"})

_SEED_MAX = 2**31 - 1  # RDKit's randomSeed is a C int; negative values mean "random"
_OBABEL_TIMEOUT_S = 300
_HYDROGEN_TYPES = frozenset({"H", "HD", "HS"})


@dataclass
class LigandRecord:
    """One row of the ``LIGANDS`` table (``pocketvina.io.tables``); ``pdbqt`` is the engine input text."""

    ligand_idx: int
    ligand_id: str
    smiles_input: str
    smiles_docked: Optional[str] = None
    inchikey: Optional[str] = None
    n_heavy: Optional[int] = None
    prep_method: str = "standard"
    seed: Optional[int] = None
    status: str = "ok"
    error: Optional[str] = None
    pdbqt: Optional[str] = None


class _Failed(Exception):
    def __init__(self, status: str, error: str):
        super().__init__(error)
        self.status = status
        self.error = error


def read_smiles_file(path) -> List[Tuple[str, str]]:
    """``[(molecule_name, smiles), ...]`` from a CSV ligand table with a header row (``.txt`` holds the same CSV).

    Columns are taken by name when the header has ``molecule_name`` and ``smiles``, else the first two columns are
    used (as before). Fields are stripped and blank rows skipped.
    """
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return rows
        names = [h.strip().lower() for h in header]
        if "molecule_name" in names and "smiles" in names:
            i_name, i_smiles = names.index("molecule_name"), names.index("smiles")
        else:
            i_name, i_smiles = 0, 1
        need = max(i_name, i_smiles) + 1
        for row in reader:
            if not any(field.strip() for field in row):
                continue
            if len(row) < need:
                raise ValueError(f"{path}, line {reader.line_num}: expected molecule_name,smiles, got {row!r}")
            rows.append((row[i_name].strip(), row[i_smiles].strip()))
    return rows


def _parse_smiles(smiles: str) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """(mol, None) or (None, reason). The final parse is a plain ``MolFromSmiles``, as in the legacy code."""
    if not smiles or not smiles.strip():
        return None, "empty SMILES"
    raw = Chem.MolFromSmiles(smiles, sanitize=False)
    if raw is None:
        return None, "SMILES syntax error"
    try:
        Chem.SanitizeMol(raw)
    except Exception as e:
        return None, str(e) or type(e).__name__
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, "RDKit could not parse the SMILES"
    return mol, None


@contextlib.contextmanager
def _keep_sigint():
    """Put Python's SIGINT handler back afterwards (main thread only): RDKit's embedding (2024.09) installs its own
    handler and leaves it installed, which swallows Ctrl-C (a docking run went on; the next Ctrl-C killed the
    process without cleanup). A Ctrl-C during the embedding itself is still RDKit's."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    handler = signal.getsignal(signal.SIGINT)
    try:
        yield
    finally:
        if handler is not None:  # None: not set from Python, nothing to restore
            signal.signal(signal.SIGINT, handler)


def _largest_fragment_mol(mol: Chem.Mol) -> Chem.Mol:
    # the chooser writes info log lines, which RDKit prints (to stdout) while another thread is inside
    # EmbedMolecule (it switches the global log state); most ligands have one fragment and do not need it
    if len(Chem.GetMolFrags(mol)) == 1:
        return mol
    return rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(mol)


def _standard_fragment(smiles: str) -> Chem.Mol:
    mol, error = _parse_smiles(smiles)
    if mol is None:
        raise _Failed("invalid_smiles", error)
    return _largest_fragment_mol(mol)


def _prechoose_fragment(smiles) -> Optional[Chem.Mol]:
    """Largest fragment of a multi-fragment SMILES, computed by ``prepare_ligands`` while no worker is embedding
    (see ``_largest_fragment_mol``); None leaves the step to the worker."""
    if not isinstance(smiles, str) or "." not in smiles:
        return None
    try:
        return _standard_fragment(smiles)
    except Exception:
        return None  # the worker repeats the step and records the failure


def largest_fragment(smiles: str) -> str:
    """Canonical SMILES of the largest fragment, organic fragments first (the parent of a salt)."""
    mol, error = _parse_smiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES {smiles!r}: {error}")
    return Chem.MolToSmiles(_largest_fragment_mol(mol))


def truncate_after_torsdof(text: str) -> Optional[str]:
    """Lines up to and including the first ``TORSDOF`` line, or None if there is none.

    OpenBabel writes one PDBQT molecule per fragment; only the first one is kept (``main.py`` legacy behaviour).
    """
    return _truncate_lines(io.StringIO(text, newline=None).readlines())


def _truncate_lines(lines: Iterable[str]) -> Optional[str]:
    kept = []
    for line in lines:
        kept.append(line)
        if line.startswith("TORSDOF"):
            return "".join(kept)
    return None


def _inchikey(mol: Chem.Mol) -> Optional[str]:
    if rdinchi is None:
        return None
    try:
        # the low-level call: Chem.MolToInchiKey logs every InChI warning (e.g. each undefined stereocentre)
        value = rdinchi.MolToInchi(mol, "/AuxNone")[0]
        key = rdinchi.InchiToInchiKey(value) if value else None
    except Exception:
        return None
    return key or None


def _describe(rec: LigandRecord, mol: Chem.Mol) -> None:
    rec.smiles_docked = Chem.MolToSmiles(mol)
    rec.inchikey = _inchikey(mol)
    rec.n_heavy = mol.GetNumHeavyAtoms()


def _unsupported_elements(mol: Chem.Mol) -> List[str]:
    return sorted({a.GetSymbol() for a in mol.GetAtoms()
                   if a.GetAtomicNum() != 1 and a.GetSymbol() not in SUPPORTED_ELEMENTS})


def _check_seed(seed) -> Optional[int]:
    if seed is None:
        return None
    seed = operator.index(seed)
    if not 0 <= seed <= _SEED_MAX:
        raise ValueError(f"seed must be in [0, {_SEED_MAX}] or None, got {seed}")
    return seed


def _check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"unknown ligand preparation method {method!r}; expected one of {METHODS}")


def _prepare_standard(rec: LigandRecord, seed: Optional[int], fragment: Optional[Chem.Mol] = None) -> None:
    if fragment is None:
        fragment = _standard_fragment(rec.smiles_input)
    # embed the canonical form, so one molecule written differently (salt, atom order) gives the same PDBQT
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(fragment)) or fragment
    _describe(rec, mol)
    bad = _unsupported_elements(mol)
    if bad:
        raise _Failed("unsupported_element", f"unsupported element(s): {', '.join(bad)}")
    if mol.GetNumHeavyAtoms() == 0:  # e.g. [H+]: the PDBQT would hold only hydrogens
        raise _Failed("no_heavy_atoms", "no heavy atoms to dock")
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = -1 if seed is None else seed
        embedded = AllChem.EmbedMolecule(mol, params) == 0
        if not embedded:
            params.useRandomCoords = True  # still seeded; helps large or strained molecules
            embedded = AllChem.EmbedMolecule(mol, params) == 0
    except Exception as e:
        raise _Failed("embed_failed", f"ETKDGv3 embedding failed: {e}") from e
    if not embedded:
        raise _Failed("embed_failed", "ETKDGv3 embedding failed")
    try:
        # via mol2 in memory, as the legacy obabel CLI step: the output is byte-identical to it for the same
        # conformer, and a direct mol -> pdbqt write leaves every partial charge at 0.000
        mol2 = pybel.readstring("mol", Chem.MolToMolBlock(mol)).write("mol2")
        text = pybel.readstring("mol2", mol2).write("pdbqt")
    except Exception as e:
        raise _Failed("conversion_failed", f"OpenBabel PDBQT conversion failed: {e}") from e
    rec.pdbqt = truncate_after_torsdof(text)
    if rec.pdbqt is None:
        raise _Failed("conversion_failed", "OpenBabel output has no TORSDOF line")


def _pdbqt_heavy_atoms(pdbqt: str) -> int:
    return sum(1 for line in pdbqt.splitlines()
               if line.startswith(("ATOM", "HETATM")) and line.split()[-1] not in _HYDROGEN_TYPES)


def _prepare_legacy(rec: LigandRecord, obabel: str) -> None:
    mol, error = _parse_smiles(rec.smiles_input)
    if mol is None:
        raise _Failed("invalid_smiles", error)
    _describe(rec, mol)
    try:
        mol_h = Chem.AddHs(mol)
        embedded = AllChem.EmbedMolecule(mol_h, AllChem.ETKDG()) == 0
    except Exception as e:
        raise _Failed("embed_failed", f"ETKDG embedding failed: {e}") from e
    if not embedded:
        raise _Failed("embed_failed", "ETKDG embedding failed")
    with tempfile.TemporaryDirectory(prefix="pocketvina_lig_") as tmp:
        mol2_file = os.path.join(tmp, "ligand.mol2")
        pdbqt_file = os.path.join(tmp, "ligand.pdbqt")
        try:
            pybel.readstring("mol", Chem.MolToMolBlock(mol_h)).write("mol2", mol2_file, overwrite=True)
            proc = subprocess.run([obabel, mol2_file, "-O", pdbqt_file], cwd=tmp, capture_output=True, text=True,
                                  timeout=_OBABEL_TIMEOUT_S)
        except Exception as e:
            raise _Failed("conversion_failed", f"mol2/obabel conversion failed: {e}") from e
        if proc.returncode != 0:
            raise _Failed("conversion_failed",
                          f"obabel exited with code {proc.returncode}: {proc.stderr.strip()[-500:]}")
        try:
            with open(pdbqt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as e:
            raise _Failed("conversion_failed", f"obabel wrote no PDBQT: {e}") from e
    rec.pdbqt = _truncate_lines(lines)
    if rec.pdbqt is None:
        raise _Failed("conversion_failed", "obabel output has no TORSDOF line")
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        # OpenBabel writes the fragments in atom order and only the first survives the truncation
        n_heavy = _pdbqt_heavy_atoms(rec.pdbqt)
        _describe(rec, next((f for f in frags if f.GetNumHeavyAtoms() == n_heavy), frags[0]))


def _prepare(idx: int, name: str, smiles: str, method: str, seed: Optional[int],
             obabel: Optional[str], fragment: Optional[Chem.Mol] = None) -> LigandRecord:
    rec = LigandRecord(ligand_idx=idx, ligand_id=name, smiles_input=smiles, prep_method=method,
                       seed=seed if method == "standard" else None)
    try:
        if method == "legacy":
            _prepare_legacy(rec, obabel)
        else:
            _prepare_standard(rec, seed, fragment)
    except _Failed as e:
        rec.status, rec.error, rec.pdbqt = e.status, e.error, None
    except Exception as e:
        rec.status, rec.error, rec.pdbqt = "error", f"{type(e).__name__}: {e}", None
    if rec.status != "ok":
        logger.debug("ligand %d (%s): %s: %s", idx, name, rec.status, rec.error)
    return rec


def find_obabel() -> Optional[str]:
    """Path of a working ``obabel`` executable (needed by ``method="legacy"``), or None.

    ``$POCKETVINA_OBABEL`` if set; otherwise the ``obabel`` next to the running Python (the same environment as
    pybel), then the one on ``PATH``.
    """
    explicit = os.environ.get("POCKETVINA_OBABEL")
    if explicit:
        candidates = (shutil.which(os.path.expanduser(explicit)),)
    else:
        here = os.path.dirname(sys.executable) if sys.executable else None
        candidates = (shutil.which("obabel", path=here) if here else None, shutil.which("obabel"))
    # absolute: the legacy conversion runs obabel with its cwd set to a temp dir
    return _first_working(tuple(dict.fromkeys(os.path.abspath(c) for c in candidates if c)))


@functools.lru_cache(maxsize=None)
def _first_working(candidates: Tuple[str, ...]) -> Optional[str]:
    for exe in candidates:
        try:
            if subprocess.run([exe, "-V"], capture_output=True, timeout=60).returncode == 0:
                return exe
        except (OSError, subprocess.SubprocessError):
            pass
        logger.debug("obabel candidate %s does not run", exe)
    return None


def _require_obabel() -> str:
    obabel = find_obabel()
    if obabel is None:
        raise RuntimeError("ligand_prep='legacy' needs a working 'obabel' executable (OpenBabel CLI): none found "
                           "next to the Python interpreter or on PATH; set $POCKETVINA_OBABEL to its path")
    return obabel


def prepare_ligand(idx: int, name: str, smiles: str, method: str = "standard", seed: Optional[int] = 42
                   ) -> LigandRecord:
    """Prepare one ligand; failures are reported in ``status``/``error``, not raised.

    ``seed`` seeds the ETKDGv3 embedding (``None`` = random); the legacy method ignores it and records ``None``.
    """
    _check_method(method)
    if method == "legacy":
        obabel = _require_obabel()
        with _keep_sigint():
            return _prepare(idx, name, smiles, method, None, obabel)
    seed = _check_seed(seed)
    with _keep_sigint():
        return _prepare(idx, name, smiles, method, seed, None)


def _default_workers() -> int:
    try:
        n = len(os.sched_getaffinity(0))  # honours CPU binding (cgroups, taskset, batch schedulers)
    except AttributeError:
        n = os.cpu_count() or 1
    return min(32, n)


def prepare_ligands(rows: Sequence[Tuple[str, str]], method: str = "standard", seed: Optional[int] = 42,
                    n_jobs: Optional[int] = None) -> List[LigandRecord]:
    """Prepare ``[(name, smiles), ...]``; ``ligand_idx`` is the row position and the output keeps the input order.

    Uses a thread pool (RDKit embedding releases the GIL and dominates the run time) of ``n_jobs`` workers, by
    default the usable CPUs (at most 32); ``n_jobs=1`` runs serially. Results do not depend on ``n_jobs``.
    """
    _check_method(method)
    seed = _check_seed(seed) if method == "standard" else None
    obabel = _require_obabel() if method == "legacy" else None
    items = [(idx, name, smiles) for idx, (name, smiles) in enumerate(rows)]
    if n_jobs is None:
        n_jobs = _default_workers()
    n_jobs = operator.index(n_jobs)
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be >= 1 or None, got {n_jobs}")

    def one(item, fragment=None):
        return _prepare(item[0], item[1], item[2], method, seed, obabel, fragment)

    n_jobs = min(n_jobs, len(items))
    t0 = time.monotonic()
    with _keep_sigint():
        if n_jobs <= 1:
            records = [one(item) for item in items]
        else:
            records = []
            window = n_jobs * 64  # bounds the number of pending futures for very large libraries
            with ThreadPoolExecutor(max_workers=n_jobs) as pool:
                for start in range(0, len(items), window):
                    chunk = items[start:start + window]
                    # salts are reduced here, between windows, while every worker is idle (see _largest_fragment_mol)
                    fragments = [_prechoose_fragment(item[2]) if method == "standard" else None for item in chunk]
                    records.extend(pool.map(one, chunk, fragments))

    counts = Counter(rec.status for rec in records)
    n_failed = len(records) - counts.get("ok", 0)
    logger.info("ligand preparation: %d of %d ligands prepared in %.1f s (method=%s, seed=%s)",
                len(records) - n_failed, len(records), time.monotonic() - t0, method, seed)
    if n_failed:
        logger.warning("%d ligand(s) not prepared: %s", n_failed,
                       ", ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k != "ok"))
    return records


def write_ligand_dir(records: Iterable[LigandRecord], dir) -> Dict[str, int]:
    """Write ``L{ligand_idx:07d}.pdbqt`` for every record with status ``ok``; returns ``{file_stem: ligand_idx}``.

    A negative or duplicate ``ligand_idx``, or an ``ok`` record without PDBQT text, raises ``ValueError`` before
    anything is written (an ``ok`` ligand is never silently left out of the engine run).
    """
    files = {}
    for rec in records:
        if rec.status != "ok":
            continue
        idx = int(rec.ligand_idx)
        if idx < 0:
            raise ValueError(f"negative ligand_idx {idx}")
        stem = f"L{idx:07d}"
        if stem in files:
            raise ValueError(f"duplicate ligand_idx {idx}")
        if not rec.pdbqt:
            raise ValueError(f"ligand_idx {idx} ({rec.ligand_id!r}) has status 'ok' but no PDBQT text")
        files[stem] = (idx, rec.pdbqt)
    os.makedirs(dir, exist_ok=True)
    for stem, (_, pdbqt) in files.items():
        with open(os.path.join(dir, stem + ".pdbqt"), "w", encoding="utf-8", newline="") as f:
            f.write(pdbqt)
    return {stem: idx for stem, (idx, _) in files.items()}
