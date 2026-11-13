"""P2Rank's solvent accessible surface (SAS) points of a protein, as PocketVina keeps them for the 3D page: one
gzip-compressed JSON file per protein, ``sas/<protein_id>.json.gz`` in a results folder (written by
``pipeline.run_atlas``, read by ``Results.sas``) and ``sas/<UniProt>.json.gz`` in the PocketVina Atlas::

    {"protein": "1fbl", "pockets": {"1": [[x, y, z, score], ...], "2": [...]},
     "other_pockets": {"3": [[x, y, z, score], ...]}, "rest": [[x, y, z, score], ...]}

"pockets" holds the points of the docked pockets (by P2Rank rank), with coordinates at 2 decimals and the score at 3;
"other_pockets" the points of P2Rank's other pockets and "rest" the points in no pocket, at 1 and 2 decimals.
Coordinates are in the frame of the structure P2Rank read. A file of the Atlas names its protein "uniprot" instead of
"protein"; ``decode`` reads both.

The points come from P2Rank's points file, ``visualizations/data/<structure file>_points.pdb.gz`` in its output folder
(``points_file``; P2Rank writes it with ``-visualizations true``): one HETATM "STP" line per point, with the rank of
the point's pocket as the residue number (0: in no pocket) and the point's score as the B-factor.
"""
import gzip
import json
import os
import re
from typing import Iterable, List, Optional, Tuple

from .posefile import gz_member

POINTS_DIR = ("visualizations", "data")  # P2Rank's folder of the points file, in its output folder
PREDICTIONS_SUFFIX = "_predictions.csv"
POINTS_SUFFIX = "_points.pdb.gz"
POINTS_SUFFIXES = (POINTS_SUFFIX, "_points.pdb")  # the points file as P2Rank 2.5 writes it, else not gzipped
FIELDS = ("pockets", "other_pockets", "rest")
# P2Rank writes a point as "HETATM%5d H    STP 1  %2d    %8.3f%8.3f%8.3f  0.50%6.3f": a rank of 100 or more shifts the
# columns after it, and a coordinate of -100 or less, or 1000 or more, touches the one before. So the rank is the
# number after "STP 1", and x, y, z, occupancy and score are the numbers after it, as they come, each with at most 3
# decimals ("68.7931000.500" is 68.793 and 1000.5).
_POINT = re.compile(r"^HETATM\s*\d+\s+\S+\s+STP\s+\S+\s+(-?\d+)\s+(.*)$")
_NUMBER = re.compile(r"-?\d+\.\d{1,3}")

Point = Tuple[int, float, float, float, float]  # (pocket rank, x, y, z, score)


def points_file(predictions_csv) -> Optional[str]:
    """P2Rank's points file that goes with its predictions CSV (``<out>/<name>_predictions.csv`` ->
    ``<out>/visualizations/data/<name>_points.pdb.gz``, or ``_points.pdb`` not gzipped), or None if P2Rank wrote
    none."""
    folder, name = os.path.split(os.path.abspath(os.fspath(predictions_csv)))
    if not name.endswith(PREDICTIONS_SUFFIX):
        return None
    for suffix in POINTS_SUFFIXES:
        path = os.path.join(folder, *POINTS_DIR, name[:-len(PREDICTIONS_SUFFIX)] + suffix)
        if os.path.isfile(path):
            return path
    return None


def read_points(path) -> List[Point]:
    """The points of a P2Rank points file (gzip-compressed or not), in file order: (pocket rank, x, y, z, score).
    ValueError if a point's line does not have these numbers."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    points = []
    for line in data.decode("ascii", "replace").splitlines():
        m = _POINT.match(line)
        if not m:
            continue
        numbers = _NUMBER.findall(m.group(2))
        if len(numbers) < 5:
            raise ValueError(f"{path}: not a point: {line.strip()!r}")
        x, y, z, _occupancy, score = (float(v) for v in numbers[:5])
        points.append((int(m.group(1)), x, y, z, score))
    return points


def build(points: Iterable[Point], docked: Iterable[int], protein: str) -> dict:
    """The content of a protein's SAS file (see above): ``points`` as ``read_points`` gives them, ``docked`` the ranks
    of its docked pockets, ``protein`` its id."""
    docked = set(docked)
    pockets, other, rest = {}, {}, []
    for rank, x, y, z, score in points:
        if rank in docked:
            pockets.setdefault(rank, []).append([round(x, 2), round(y, 2), round(z, 2), round(score, 3)])
            continue
        point = [round(x, 1), round(y, 1), round(z, 1), round(score, 2)]
        if rank > 0:
            other.setdefault(rank, []).append(point)
        else:
            rest.append(point)
    return {"protein": protein, "pockets": {str(k): pockets[k] for k in sorted(pockets)},
            "other_pockets": {str(k): other[k] for k in sorted(other)}, "rest": rest}


def encode(data: dict, level: int = 9) -> bytes:
    """The bytes of a SAS file: compact JSON in one gzip member without a timestamp, so the same points give the same
    bytes."""
    return gz_member(json.dumps(data, separators=(",", ":"), allow_nan=False).encode("utf-8"), level)


def decode(data: bytes) -> dict:
    """The points of a SAS file's bytes (gzip-compressed or not), with "protein" (the Atlas's "uniprot"), "pockets",
    "other_pockets" and "rest" (empty when the file has none). ValueError if it is not a SAS file."""
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    obj = json.loads(data)
    if not (isinstance(obj, dict) and isinstance(obj.get("pockets"), dict)
            and isinstance(obj.get("other_pockets", {}), dict) and isinstance(obj.get("rest", []), list)):
        raise ValueError('not a file of SAS points: it needs a "pockets" object')
    return {"protein": obj.get("protein", obj.get("uniprot")), "pockets": obj["pockets"],
            "other_pockets": obj.get("other_pockets") or {}, "rest": obj.get("rest") or []}
