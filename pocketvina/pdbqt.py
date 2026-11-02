"""Engine output PDBQT on bytes (no files): MODEL records, single models and output file names.

``parse_modes`` gives the same results as ``modes_of`` in the PocketVina Atlas importer
(``db_prototype/scripts/import_batch.py``), so ``raw_offset``/``raw_length`` in the modes table mean the same thing in
both: the byte span of ``MODEL ... ENDMDL\\n`` inside the engine output.
"""
import math
import os
import re
from typing import List, NamedTuple

# A literal "\n" prefix lets the regex engine use a fast literal search; line 1 is handled by prefixing b"\n".
VINA_RE = re.compile(rb"\nREMARK VINA RESULT:[ \t]+(\S+)[ \t]+(\S+)[ \t]+(\S+)")
MODEL_RE = re.compile(rb"\nMODEL(?![A-Za-z0-9_])")
ENDMDL_RE = re.compile(rb"\nENDMDL[^\n]*\n?")
MAX_MODES = 127  # n_modes and mode are int8 in the tables


class ModeRecord(NamedTuple):
    """One MODEL of an engine output; ``raw[raw_offset:raw_offset + raw_length]`` is its ``MODEL ... ENDMDL`` text."""

    mode: int
    affinity: float
    rmsd_lb: float
    rmsd_ub: float
    raw_offset: int
    raw_length: int


def _scan(raw: bytes):
    r = b"\n" + raw
    starts = [m.start() for m in MODEL_RE.finditer(r)]  # index in r of the "\n" == offset in raw of "MODEL"
    ends = [m.end() - 1 for m in ENDMDL_RE.finditer(r)]
    remarks = list(VINA_RE.finditer(r))
    return starts, ends, remarks


def _records(raw, starts, ends, remarks) -> List[ModeRecord]:
    out = []
    for q, s in enumerate(starts):
        e = ends[q] if q < len(ends) else len(raw)
        a, b, c = remarks[q].groups() if q < len(remarks) else (b"nan", b"nan", b"nan")
        out.append(ModeRecord(q + 1, float(a), float(b), float(c), s, e - s))
    return out


def parse_modes(raw: bytes) -> List[ModeRecord]:
    """MODEL records of an engine output, exactly as the Atlas ``modes_of`` reads them.

    The q-th MODEL is paired with the q-th ENDMDL (or the end of ``raw``) and the q-th ``REMARK VINA RESULT`` line
    (NaN scores if there is none). Raises ``ValueError`` if a score is not a number.
    """
    return _records(raw, *_scan(raw))


def check_modes(raw: bytes) -> List[ModeRecord]:
    """``parse_modes``, but raise ``ValueError`` unless the output is well formed: at least one MODEL, each with one
    ``REMARK VINA RESULT`` line with finite scores and an ENDMDL before the next MODEL, at most 127 MODELs."""
    starts, ends, remarks = _scan(raw)
    if not starts:
        raise ValueError("no MODEL record")
    if len(starts) > MAX_MODES:
        raise ValueError(f"{len(starts)} MODEL records, at most {MAX_MODES} are supported")
    if len(ends) != len(starts):
        raise ValueError(f"{len(starts)} MODEL but {len(ends)} ENDMDL records")
    if len(remarks) != len(starts):
        raise ValueError(f"{len(starts)} MODEL but {len(remarks)} REMARK VINA RESULT lines")
    bounds = starts[1:] + [len(raw)]
    for q, s in enumerate(starts):
        if not s < remarks[q].start() < ends[q] <= bounds[q]:
            raise ValueError(f"MODEL {q + 1} is not closed by ENDMDL before the next MODEL, or has no score")
    records = _records(raw, starts, ends, remarks)
    for rec in records:
        if not all(math.isfinite(v) for v in (rec.affinity, rec.rmsd_lb, rec.rmsd_ub)):
            raise ValueError(f"MODEL {rec.mode} has a non-finite score")
    return records


def model(raw: bytes, mode: int) -> bytes:
    """The bytes of MODEL ``mode`` (1-based, as in the modes table); ``IndexError`` if there is no such mode."""
    records = parse_modes(raw)
    if not 1 <= mode <= len(records):
        raise IndexError(f"mode {mode} out of range: the output has {len(records)} MODEL records")
    rec = records[mode - 1]
    return bytes(raw[rec.raw_offset:rec.raw_offset + rec.raw_length])


def _boost_basename(path: str) -> str:
    """``boost::filesystem::basename`` (filesystem v3, as the engine is built): the file name up to its last dot."""
    name = path.rsplit("/", 1)[-1]
    if not name and path.strip("/"):
        name = "."  # v3: the file name of "dir/" is "."
    if name in (".", ".."):
        return name
    cut = name.rfind(".")
    return name if cut < 0 else name[:cut]


def engine_output_name(receptor_pdbqt: str, pocket: int, ligand_file: str) -> str:
    """File name the engine gives the output of ``ligand_file`` docked into ``pocket`` of ``receptor_pdbqt``:
    ``<receptor_base>-pocket<k>-<ligand_stem>_out.pdbqt`` (``pocketvina/main/main.cpp:405-416, 826-846``).

    ``<receptor_base>`` is the receptor file name up to its last dot; ``<ligand_stem>`` the ligand file name up to its
    first ``.pdbqt`` (all of it if there is none). Only the file names are used: the engine also cuts a ligand path
    at the first ``.pdbqt`` in its directory names, so ligand directories should not contain one.
    """
    ligand_name = os.fspath(ligand_file).rsplit("/", 1)[-1]
    cut = ligand_name.find(".pdbqt")
    stem = ligand_name[:cut] if cut >= 0 else ligand_name
    return f"{_boost_basename(os.fspath(receptor_pdbqt))}-pocket{int(pocket)}-{stem}_out.pdbqt"
