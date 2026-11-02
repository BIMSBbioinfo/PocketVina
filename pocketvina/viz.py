"""3D views of docked poses for notebooks (Mol* through MolViewSpec, or py3Dmol) and a table of the best pairs.

    from pocketvina import viz
    viz.table("results")                                   # best pose per protein x ligand, best first
    viz.show("results", "AF-P00533-F1-model_v4", "gefitinib")          # best pose (the stored SDF)
    viz.show("results", "AF-P00533-F1-model_v4", "gefitinib", pocket=2, mode=3)
    viz.show_modes("results", "AF-P00533-F1-model_v4", "gefitinib", pocket=2)
    viz.to_html("results", "AF-P00533-F1-model_v4", "gefitinib", path="pose.html")   # a file to share

The scene follows the PocketVina Atlas viewer (light theme): grey cartoon, the residues within ``pocket_radius`` A of
the pose as thin ball-and-stick with grey carbons under a faint surface in the pocket's colour, the pose as
ball-and-stick with gold carbons, coloured by element, hydrogens hidden, the camera on the pose (+ 8 A), and a label
with protein, ligand, pocket, mode and affinity.

Backends: ``"molstar"`` needs ``molviewspec`` (Python >= 3.10), ``"py3dmol"`` needs ``py3Dmol``; ``"auto"`` takes
Mol* when molviewspec imports, else py3Dmol. Both are optional and imported only when a view is made.

Offline: the views hold the structures, but not the viewers' JavaScript. Mol* loads it from a CDN (jsDelivr) when
the view is drawn; without internet use ``backend="py3dmol"`` -- py3Dmol also needs its 3Dmol.js, from a CDN by
default, but a small script that the browser often has cached. ``to_html`` writes a standalone page (structures
embedded) to share; it, too, loads the viewer's JavaScript when opened.

Mol* version: ``MOLSTAR_VERSION`` (5.5.0: released with molviewspec 1.8.1, it reads the scenes molviewspec 1.8
writes), not molviewspec's default "latest", so a view draws the same way later. To use another one, pass
``molstar_version=`` (e.g. ``"latest"``) or set ``$POCKETVINA_MOLSTAR_VERSION`` (also for ``pocketvina show``).

Mol* is described in the `"Mol* Viewer: modern web app for 3D visualization and analysis of large biomolecular
structures" <https://doi.org/10.1093/nar/gkab314>`_ paper, MolViewSpec in the `"Describing and Sharing Molecular
Visualizations Using the MolViewSpec Toolkit" <https://doi.org/10.1002/cpz1.1099>`_ paper and 3Dmol.js in the
`"3Dmol.js: molecular visualization with WebGL" <https://doi.org/10.1093/bioinformatics/btu829>`_ paper.
"""
import contextlib
import dataclasses
import html
import io
import logging
import operator
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem

from . import convert, pdbqt
from .io.results import Results, short_floats
from .main import structure_kind

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "molstar", "py3dmol")
MOLSTAR_UIS = ("viewer", "stories")
# Mol* (npm package molstar, from jsDelivr) that draws the MolViewSpec views: 5.5.0 came out with molviewspec 1.8.1
# (2025-12-22) and reads the MolViewSpec 1.8 scenes it writes (here: sdf/pdbqt files, surface_type, label background
# and attachment, snapshots; its page's viewportShowToggleFullscreen option). A view uses its molstar_version
# argument, else $POCKETVINA_MOLSTAR_VERSION, else this.
MOLSTAR_VERSION = "5.5.0"
MOLSTAR_VERSION_ENV = "POCKETVINA_MOLSTAR_VERSION"
_MOLSTAR_VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]*")  # npm version or tag (it goes into the page's URLs)
# PocketVina Atlas viewer, light theme (site/public/viewer/viewer.js)
BACKGROUND = "#F4F4F6"
CARTOON_COLOR = "#C9CFD6"
RESIDUE_CARBON = "#A9B1BB"
LIGAND_CARBON = "#E3B35A"
LABEL_COLOR = "#1B1B1F"
LABEL_BACKGROUND = "#FFFFFF"
POCKET_COLORS = ("#5C93E6", "#6A47B3", "#E65CE0", "#B34761", "#E69E5C")  # P2Rank rank 1..5, then again
SURFACE_OPACITY = 0.15
PY3DMOL_SURFACE_OPACITY = 0.3  # 3Dmol.js draws translucent surfaces lighter than Mol*
FOCUS_EXTENT = 8.0  # A around the pose
RESIDUE_SIZE = 0.5  # ball-and-stick size factor of the residue sticks (the pose: 1)
LABEL_SIZE = 1.0  # A (Mol* text height)
# element colours of Mol*'s element-symbol theme (Jmol); carbon is the representation's own colour
ELEMENT_COLORS = {
    "H": "#FFFFFF", "Li": "#CC80FF", "B": "#FFB5B5", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050",
    "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
    "Cl": "#1FF01F", "K": "#8F40D4", "Ca": "#3DFF00", "V": "#A6A6AB", "Cr": "#8A99C7", "Mn": "#9C7AC7",
    "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F",
    "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929", "Sr": "#00FF00", "Mo": "#54B5B5", "Ru": "#248F8F",
    "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F", "Sn": "#668080", "Sb": "#9E63B5", "I": "#940094",
    "Ba": "#00C900", "Gd": "#45FFC7", "W": "#2194D6", "Os": "#266696", "Ir": "#175487", "Pt": "#D0D0E0",
    "Au": "#FFD123", "Hg": "#B8B8D0", "Pb": "#575961",
}
_ELEMENTS = frozenset(Chem.GetPeriodicTable().GetElementSymbol(z) for z in range(1, 119))
WATERS = frozenset({"HOH", "WAT", "DOD", "H2O", "TIP", "TIP3", "SOL"})
_CIF_TOKEN = re.compile(r"'(?:[^']|'(?=\S))*'|\"(?:[^\"]|\"(?=\S))*\"|\S+")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

ResultsArg = Union[str, "os.PathLike[str]", Results]


# --- structures --------------------------------------------------------------------------------------------------

class Residue(NamedTuple):
    """A protein residue near a pose: chain (auth_asym_id, may be ""), number, insertion code, name, distance (A)."""

    chain: str
    seq: int
    icode: str
    name: str
    distance: float

    @property
    def label(self) -> str:
        res = f"{self.name}{self.seq}{self.icode}"
        return f"{self.chain}:{res}" if self.chain else res


@dataclass
class _Atoms:
    """Heavy, non-water atoms of a structure (first model)."""

    chain: np.ndarray
    seq: np.ndarray
    icode: np.ndarray
    name: np.ndarray
    element: np.ndarray
    xyz: np.ndarray

    @classmethod
    def from_rows(cls, rows: List[tuple]) -> "_Atoms":
        if not rows:
            empty = np.empty(0, dtype=object)
            return cls(empty, np.empty(0, dtype=np.int64), empty, empty, empty, np.empty((0, 3)))
        chain, seq, icode, name, element, x, y, z = zip(*rows)
        return cls(np.array(chain, dtype=object), np.array(seq, dtype=np.int64), np.array(icode, dtype=object),
                   np.array(name, dtype=object), np.array(element, dtype=object),
                   np.column_stack([x, y, z]).astype(np.float64))


@dataclass
class _Receptor:
    protein_id: str
    data: bytes  # PDB or mmCIF text, as given to the viewer
    fmt: str  # MolViewSpec parse format: "pdb" or "mmcif"
    atoms: _Atoms

    @property
    def file_name(self) -> str:
        return "protein.cif" if self.fmt == "mmcif" else "protein.pdb"

    @property
    def py3dmol_format(self) -> str:
        return "cif" if self.fmt == "mmcif" else "pdb"


@dataclass(eq=False)
class _Pose:
    protein_id: str
    ligand_idx: int
    ligand_id: str
    pocket: int
    mode: int
    affinity: Optional[float]
    data: bytes  # SDF record, or the PDBQT model when it did not convert
    fmt: str  # "sdf" or "pdbqt"
    model: bytes  # the PDBQT model (MODEL ... ENDMDL) as the engine wrote it
    elements: List[str] = field(init=False)  # heavy-atom elements present
    xyz: np.ndarray = field(init=False, repr=False)  # heavy-atom coordinates

    def __post_init__(self):
        self.elements, self.xyz = _ligand_atoms(self.data, self.fmt)

    def file_name(self, stem: str = "ligand") -> str:
        return f"{stem}.{self.fmt}"

    def as_pdbqt(self) -> "_Pose":
        """This pose drawn from its PDBQT model."""
        return self if self.fmt == "pdbqt" else dataclasses.replace(self, data=self.model, fmt="pdbqt")

    @property
    def affinity_text(self) -> str:
        return "affinity n/a" if self.affinity is None else f"{self.affinity:.1f} kcal/mol"

    @property
    def label(self) -> str:
        """``<ligand> | pocket <k> | mode <m> | <affinity>`` (the 3D label)."""
        return f"{self.ligand_id} | pocket {self.pocket} | mode {self.mode} | {self.affinity_text}"

    @property
    def title(self) -> str:
        """``<protein> | <ligand> | pocket <k> | mode <m> | <affinity>``."""
        return f"{self.protein_id} | {self.label}"


def _ad_element(ad_type: str, name: str) -> str:
    """Element of an AutoDock atom type, else guessed from the atom name ("" if unknown).

    Elements outside the AutoDock types (Hg, Pt, Cd, As, ...) are written by OpenBabel as their symbol."""
    if ad_type in convert.AD_ELEMENTS:
        return convert.AD_ELEMENTS[ad_type]
    if ad_type.capitalize() in _ELEMENTS:
        return ad_type.capitalize()
    letters = re.sub(r"[^A-Za-z]", "", name)
    return letters[:1].upper()


def _pdb_element(line: str) -> str:
    element = line[76:78].strip()
    if element:
        return element.capitalize()
    letters = re.sub(r"[^A-Za-z]", "", line[12:16])  # "1HG1" -> H, "CA" -> C (no element column: proteins)
    return letters[:1].upper()


def pdbqt_to_pdb(text: str) -> str:
    """PDB text of a PDBQT receptor: ATOM/HETATM lines up to the B-factor, element from the AutoDock type."""
    out = []
    for line in text.splitlines():
        record = line[:6]
        if record in ("ATOM  ", "HETATM"):
            tokens = line[54:].split()  # occupancy, B-factor, charge, AutoDock type
            ad_type = tokens[-1] if tokens and tokens[-1][0].isalpha() else ""
            element = _ad_element(ad_type, line[12:16])
            out.append(f"{line[:66]:<66}          {element.upper():>2}")
        elif record.startswith(("TER", "MODEL", "ENDMDL")):
            out.append(line.rstrip())
    out.append("END")
    return "\n".join(out) + "\n"


def _pdb_atoms(text: str) -> _Atoms:
    rows = []
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break  # first model only, as the viewers show
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[17:20].strip()
        element = _pdb_element(line)
        if element in ("H", "D") or name in WATERS:
            continue
        try:
            seq = int(line[22:26])
            xyz = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        rows.append((line[21:22].strip(), seq, line[26:27].strip(), name, element, *xyz))
    return _Atoms.from_rows(rows)


def _cif_value(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


def _cif_atoms(text: str) -> _Atoms:
    """Heavy atoms of the ``_atom_site`` loop of an mmCIF file (first model; author chain/residue numbers)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_" and i + 1 < len(lines) and lines[i + 1].startswith("_atom_site."):
            break
        i += 1
    else:
        return _Atoms.from_rows([])
    keys = []
    i += 1
    while i < len(lines) and lines[i].startswith("_atom_site."):
        keys.append(lines[i].split()[0][len("_atom_site."):])
        i += 1
    tokens: List[str] = []
    while i < len(lines):
        line = lines[i]
        if line.startswith(("loop_", "_", "data_", "#")):
            break
        tokens.extend(_CIF_TOKEN.findall(line))
        i += 1
    col = {k: j for j, k in enumerate(keys)}

    def pick(*names):
        return next((col[n] for n in names if n in col), None)

    c_chain, c_seq = pick("auth_asym_id", "label_asym_id"), pick("auth_seq_id", "label_seq_id")
    c_icode, c_name = pick("pdbx_PDB_ins_code"), pick("auth_comp_id", "label_comp_id")
    c_el, c_model = pick("type_symbol"), pick("pdbx_PDB_model_num")
    c_xyz = [pick(f"Cartn_{a}") for a in "xyz"]
    if c_seq is None or None in c_xyz:
        return _Atoms.from_rows([])
    rows, first_model = [], None
    n = len(keys)
    for r in range(len(tokens) // n):
        row = [_cif_value(t) for t in tokens[r * n:(r + 1) * n]]
        if c_model is not None:
            first_model = row[c_model] if first_model is None else first_model
            if row[c_model] != first_model:
                continue
        name = row[c_name] if c_name is not None else ""
        element = row[c_el].capitalize() if c_el is not None else ""
        if element in ("H", "D") or name in WATERS:
            continue
        try:
            seq = int(row[c_seq])
            xyz = [float(row[c]) for c in c_xyz]
        except ValueError:
            continue
        chain = row[c_chain] if c_chain is not None else ""
        icode = row[c_icode] if c_icode is not None else ""
        rows.append((chain if chain not in ("?", ".") else "", seq, icode if icode not in ("?", ".") else "", name,
                     element, *xyz))
    return _Atoms.from_rows(rows)


def _receptor(res: Results, protein: str) -> _Receptor:
    """The stored input structure as the viewers get it: PDB (a PDBQT receptor is turned into PDB) or mmCIF."""
    kind, _ = structure_kind(res.structure_path(protein))
    raw = res.structure(protein)
    text = raw.decode("utf-8", "replace")
    if kind == "cif":
        return _Receptor(protein, raw, "mmcif", _cif_atoms(text))
    if kind == "pdbqt":
        text = pdbqt_to_pdb(text)
        raw = text.encode()
    return _Receptor(protein, raw, "pdb", _pdb_atoms(text))


def near_residues(atoms: _Atoms, xyz: np.ndarray, radius: float) -> List[Residue]:
    """Residues with a heavy atom within ``radius`` A of a heavy atom of ``xyz``, in chain and number order."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if not len(atoms.xyz) or not len(xyz):
        return []
    lo, hi = xyz.min(axis=0) - radius, xyz.max(axis=0) + radius
    idx = np.nonzero(np.all((atoms.xyz >= lo) & (atoms.xyz <= hi), axis=1))[0]
    if not len(idx):
        return []
    d2 = ((atoms.xyz[idx, None, :] - xyz[None, :, :]) ** 2).sum(axis=-1).min(axis=1)
    best: Dict[tuple, Residue] = {}
    for i, d in zip(idx[d2 <= radius * radius], np.sqrt(d2[d2 <= radius * radius])):
        key = (atoms.chain[i], int(atoms.seq[i]), atoms.icode[i])
        if key not in best or d < best[key].distance:
            best[key] = Residue(key[0], key[1], key[2], atoms.name[i], round(float(d), 2))
    return [best[k] for k in sorted(best)]


def _ligand_atoms(data: bytes, fmt: str) -> Tuple[List[str], np.ndarray]:
    """(heavy-atom elements present, heavy-atom coordinates) of an SDF record or a PDBQT model."""
    elements, xyz = [], []
    text = data.decode("utf-8", "replace")
    if fmt == "sdf":
        mol = Chem.MolFromMolBlock(text, sanitize=False, removeHs=False)
        if mol is not None and mol.GetNumConformers():
            pos = mol.GetConformer().GetPositions()
            for atom, p in zip(mol.GetAtoms(), pos):
                if atom.GetAtomicNum() > 1:
                    elements.append(atom.GetSymbol())
                    xyz.append(p)
    else:
        for line in text.splitlines():
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            tokens = line[54:].split()
            element = _ad_element(tokens[-1] if tokens and tokens[-1][0].isalpha() else "", line[12:16])
            if element in ("H", ""):
                continue
            try:
                xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
            elements.append(element)
    return sorted(set(elements)), np.array(xyz, dtype=np.float64).reshape(-1, 3)


# --- poses -------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _results(results: ResultsArg) -> Iterator[Results]:
    """``results`` as an open ``Results`` (opened and closed here if it is a path)."""
    if isinstance(results, Results):
        yield results
    else:
        with Results(results) as res:
            yield res


def _text_bytes(text: str) -> bytes:
    return text.encode("utf-8", "surrogateescape")  # the bytes ``Results`` decoded


def _smiles_docked(res: Results, i: int) -> Optional[str]:
    lig = res.ligands(columns=["ligand_idx", "smiles_docked"])
    row = lig.loc[lig["ligand_idx"] == i, "smiles_docked"]
    smi = row.iloc[0] if len(row) else None
    return smi if isinstance(smi, str) and smi else None


def _convert(model: bytes, smiles: Optional[str], props: dict) -> Tuple[bytes, str]:
    """(SDF, "sdf") of a PDBQT model, or (the model, "pdbqt") if it does not convert."""
    if smiles:
        sdf, status = convert.pose_to_sdf(model, smiles, props)
        if sdf:
            return sdf, "sdf"
        logger.info("pose %s pocket %s mode %s: %s, showing the PDBQT model", props.get("_Name"),
                    props.get("pocket"), props.get("mode"), status)
    return model, "pdbqt"


def _best(res: Results, protein: str, i: int, lid: str) -> Tuple[int, int, float]:
    """(pocket, mode, Vina score) of the ligand's best pose in the protein (the pairs table)."""
    pairs = res.pairs(protein, columns=["ligand_idx", "vina", "best_pocket", "best_mode"])
    row = pairs[pairs["ligand_idx"] == i]
    if row.empty or int(row["best_pocket"].iloc[0]) < 1:
        raise KeyError(f"{protein}: ligand {lid!r} has no docked pose")
    return (int(row["best_pocket"].iloc[0]), int(row["best_mode"].iloc[0]),
            float(short_floats(row["vina"].to_numpy())[0]))


def _best_pose(res: Results, protein: str, i: int, lid: str) -> _Pose:
    k, m, affinity = _best(res, protein, i, lid)
    model = _text_bytes(res.pose_model(protein, i, k, m))
    sdf = res.top_sdf(protein, i)
    if sdf:
        return _Pose(protein, i, lid, k, m, affinity, _text_bytes(sdf), "sdf", model)
    logger.info("%s: no stored SDF for %r, showing the PDBQT model", protein, lid)
    return _Pose(protein, i, lid, k, m, affinity, model, "pdbqt", model)


def _pocket_poses(res: Results, protein: str, i: int, lid: str, pocket, modes: Optional[Sequence[int]]) \
        -> List[_Pose]:
    """Poses of the modes ``modes`` (None: all) of ligand ``i`` in ``pocket``, converted to SDF where possible."""
    k = operator.index(pocket)
    block = _text_bytes(res.pose_block(protein, i, k))
    records = pdbqt.parse_modes(block)
    if not records:
        raise KeyError(f"{protein}: ligand {lid!r} has no modes in pocket {k}")
    by_mode = {r.mode: r for r in records}
    if modes is None:
        modes = [r.mode for r in records]
    smiles = _smiles_docked(res, i)
    poses = []
    for m in modes:
        m = operator.index(m)
        rec = by_mode.get(m)
        if rec is None:
            raise KeyError(f"{protein}: ligand {lid!r} in pocket {k} has no mode {m} "
                           f"(modes {records[0].mode}..{records[-1].mode})")
        model = block[rec.raw_offset:rec.raw_offset + rec.raw_length]
        # the fields of the best-pose SDFs (pipeline.run_atlas)
        props = {"_Name": lid, "protein_id": protein, "ligand_idx": i, "pocket": k, "mode": m,
                 "affinity": rec.affinity}
        data, fmt = _convert(model, smiles, props)
        poses.append(_Pose(protein, i, lid, k, m, rec.affinity, data, fmt, model))
    return poses


def _load_pose(res: Results, protein: str, ligand, pocket, mode) -> _Pose:
    i = res.ligand_idx(ligand)
    lid = res.ligand_id(i)
    if pocket is None:
        if operator.index(mode) != 1:
            raise ValueError("mode needs a pocket: pass pocket=<rank> (pocket=None shows the best pose)")
        return _best_pose(res, protein, i, lid)
    return _pocket_poses(res, protein, i, lid, pocket, [mode])[0]


def _check_protein(res: Results, protein) -> str:
    if not isinstance(protein, str):
        raise TypeError(f"protein must be a protein_id string, not {type(protein).__name__}")
    known = res.proteins()
    if protein not in known:
        hint = f" (proteins: {', '.join(known[:5])}{', ...' if len(known) > 5 else ''})" if known else ""
        raise KeyError(f"protein {protein!r} is not in {res.path} (not docked, or not finished){hint}")
    return protein


def _check_radius(pocket_radius) -> float:
    r = float(pocket_radius)
    if not r >= 0:
        raise ValueError(f"pocket_radius must be >= 0, got {pocket_radius!r}")
    return r


# --- backends ----------------------------------------------------------------------------------------------------

def _molviewspec():
    try:
        import molviewspec
    except Exception as e:  # not installed, or Python < 3.10 (its modules use `X | Y` annotations at import)
        raise ImportError(f"backend 'molstar' needs molviewspec (pip install molviewspec; Python >= 3.10): {e}") \
            from e
    return molviewspec


def _py3dmol():
    try:
        import py3Dmol
    except Exception as e:
        raise ImportError(f"backend 'py3dmol' needs py3Dmol (pip install py3Dmol): {e}") from e
    return py3Dmol


def _backend(backend: str) -> str:
    """The backend to use, checked to be importable (before any data is read)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "molstar":
        _molviewspec()
    elif backend == "py3dmol":
        _py3dmol()
    else:
        try:
            _molviewspec()
            return "molstar"
        except ImportError as e:
            logger.debug("%s; using py3Dmol", e)
            try:
                _py3dmol()
            except ImportError:
                raise ImportError("pocketvina.viz needs molviewspec (Mol*, Python >= 3.10) or py3Dmol: "
                                  "pip install molviewspec py3Dmol") from e
            return "py3dmol"
    return backend


def _download_name(*parts) -> str:
    return _UNSAFE_CHARS.sub("_", "_".join(str(p) for p in parts)).strip("_") or "pocketvina"


def _molstar_version(version: Optional[str] = None) -> str:
    """The Mol* version of a view: ``version``, else ``$POCKETVINA_MOLSTAR_VERSION``, else ``MOLSTAR_VERSION``."""
    if version is None:
        version = os.environ.get(MOLSTAR_VERSION_ENV) or MOLSTAR_VERSION
        where = f"${MOLSTAR_VERSION_ENV}" if os.environ.get(MOLSTAR_VERSION_ENV) else "MOLSTAR_VERSION"
    else:
        where = "molstar_version"
    if not isinstance(version, str) or not _MOLSTAR_VERSION_RE.fullmatch(version):
        raise ValueError(f"{where} must be a Mol* (npm molstar) version or tag such as {MOLSTAR_VERSION!r} or "
                         f"'latest', got {version!r}")
    return version


class MolstarView:
    """A MolViewSpec scene (``state``) and the files it loads by name (``data``), drawn by Mol*.

    The result of ``show``/``show_modes`` with the Mol* backend: Jupyter and Colab draw it when it is a cell's result
    (through ``molviewspec.molstar_notebook``); ``to_html()`` gives a standalone page, ``save(path)`` writes it.
    ``info`` holds what is shown (protein, ligand, pocket, mode(s), affinity, residues, files).
    """

    def __init__(self, state, data: Dict[str, bytes], *, info: dict, width=950, height=600, ui: str = "viewer",
                 molstar_version: Optional[str] = None):
        self.state = state
        self.data = data
        self.info = info
        self.width = width
        self.height = height
        self.ui = ui
        self.molstar_version = _molstar_version(molstar_version)

    def __repr__(self):
        what = ", ".join(f"{k}={self.info[k]!r}" for k in ("protein_id", "ligand_id", "pocket", "mode", "modes")
                         if k in self.info)
        return f"<MolstarView {what} files={sorted(self.data)}>"

    def state_json(self, indent: Optional[int] = None) -> str:
        """The MolViewSpec state (MVSJ text)."""
        return self.state.dumps(indent=indent)

    def to_html(self) -> str:
        """Standalone HTML page: the state and its files as an MVSX archive (base64), Mol* ``molstar_version`` from
        the CDN."""
        mvs = _molviewspec()
        return mvs.molstar_html(self.state, data=self.data, ui=self.ui, molstar_version=self.molstar_version)

    def save(self, path) -> str:
        """Write ``to_html()`` to ``path``; returns the absolute path."""
        return _write_text(path, self.to_html())

    def display(self) -> None:
        """Show the view in Jupyter / Colab now (``molviewspec.molstar_notebook``)."""
        mvs = _molviewspec()
        mvs.molstar_notebook(self.state, data=self.data, width=self.width, height=self.height,
                             download_filename=self.info.get("download_name", "pocketvina"), ui=self.ui,
                             molstar_version=self.molstar_version)

    def _ipython_display_(self):
        self.display()

    def _repr_html_(self) -> str:
        # for HTML renderers without IPython's display hook (IPython uses _ipython_display_)
        w = self.width if isinstance(self.width, str) else f"{self.width}px"
        h = self.height if isinstance(self.height, str) else f"{self.height}px"
        return (f'<iframe srcdoc="{html.escape(self.to_html(), quote=True)}" style="border: 0; width: {w}; '
                f'height: {h}" allow="fullscreen" allowfullscreen></iframe>')


def _type_symbols(element: str) -> list:
    """Component expressions of an element as PDB (upper case) and SDF/mmCIF (capitalised) files spell it."""
    mvs = _molviewspec()
    forms = sorted({element.capitalize(), element.upper()})
    return [mvs.ComponentExpression(type_symbol=f) for f in forms]


def _color_by_element(rep, carbon: str, elements: Sequence[str]) -> None:
    rep.color(color=carbon)
    for el in sorted(set(elements) - {"C", "H"}):
        color = ELEMENT_COLORS.get(el.capitalize())
        if color is not None:
            rep.color(color=color, selector=_type_symbols(el))


def _residue_expression(r: Residue):
    mvs = _molviewspec()
    kw = {"auth_seq_id": r.seq}
    if r.chain:
        kw["auth_asym_id"] = r.chain
    if r.icode:
        kw["pdbx_PDB_ins_code"] = r.icode
    return mvs.ComponentExpression(**kw)


def _residue_elements(atoms: _Atoms, residues: Sequence[Residue]) -> List[str]:
    keys = {(r.chain, r.seq, r.icode) for r in residues}
    return sorted({e for c, s, i, e in zip(atoms.chain, atoms.seq, atoms.icode, atoms.element)
                   if (c, int(s), i) in keys})


def _pocket_color(pocket: int) -> str:
    return POCKET_COLORS[(pocket - 1) % len(POCKET_COLORS)]


def _label_position(xyz: np.ndarray) -> List[float]:
    """Above the pose (+y, the screen's up in the default view along -z)."""
    center = xyz.mean(axis=0)
    radius = float(np.sqrt(((xyz - center) ** 2).sum(axis=1)).max()) if len(xyz) else 0.0
    return [round(float(center[0]), 3), round(float(center[1] + radius + 1.5), 3), round(float(center[2]), 3)]


def _molstar_scene(builder, receptor: _Receptor, pose: _Pose, ligand_file: str, residues: Sequence[Residue], *,
                   surface: bool) -> None:
    """The Atlas scene on a MolViewSpec builder: protein (cartoon, pocket residues) and one pose."""
    builder.canvas(background_color=BACKGROUND)
    protein = builder.download(url=receptor.file_name).parse(format=receptor.fmt).model_structure(ref="protein")
    protein.component(selector="polymer").representation(type="cartoon").color(color=CARTOON_COLOR)
    if residues:
        site = protein.component(selector=[_residue_expression(r) for r in residues], ref="pocket_residues")
        sticks = site.representation(type="ball_and_stick", ignore_hydrogens=True, size_factor=RESIDUE_SIZE)
        _color_by_element(sticks, RESIDUE_CARBON, _residue_elements(receptor.atoms, residues))
        if surface:
            (site.representation(type="surface", surface_type="molecular", ignore_hydrogens=True)
             .color(color=_pocket_color(pose.pocket)).opacity(opacity=SURFACE_OPACITY))
    ligand = builder.download(url=ligand_file).parse(format=pose.fmt).model_structure(ref="ligand")
    lig = ligand.component(selector="all")
    _color_by_element(lig.representation(type="ball_and_stick", ignore_hydrogens=True), LIGAND_CARBON,
                      pose.elements)
    lig.tooltip(text=pose.title)
    lig.focus(radius_extent=FOCUS_EXTENT)
    if len(pose.xyz):
        (ligand.primitives(label_color=LABEL_COLOR, label_background_color=LABEL_BACKGROUND,
                           label_attachment="bottom-center")
         .label(position=_label_position(pose.xyz), text=pose.label, label_size=LABEL_SIZE))


def _summary(pose: _Pose) -> str:
    """Markdown line of a single-pose view (the Mol* viewer shows it over the scene)."""
    return (f"**{pose.ligand_id}** in **{pose.protein_id}**, pocket {pose.pocket}, mode {pose.mode}: "
            f"{pose.affinity_text}")


def _description(receptor: _Receptor, poses: Sequence[_Pose], current: Optional[_Pose],
                 residues: Sequence[Residue], radius: float) -> str:
    """Markdown of a ``show_modes`` snapshot: the modes (current one in bold) and the residues near the pose."""
    p = current or poses[0]
    lines = [f"**{p.ligand_id}** in **{receptor.protein_id}**, pocket {p.pocket}", "",
             "| mode | affinity (kcal/mol) | ligand file |", "|---:|---:|:---|"]
    for q in poses:
        aff = "n/a" if q.affinity is None else f"{q.affinity:.1f}"
        row = f"| {q.mode} | {aff} | {q.fmt} |"
        lines.append(row.replace(f"| {q.mode} |", f"| **{q.mode}** |", 1) if q is current else row)
    if residues:
        lines += ["", f"Residues within {radius:g} \u00c5: " + ", ".join(r.label for r in residues)]
    return "\n".join(lines)


def _molstar_view(receptor: _Receptor, pose: _Pose, residues, *, surface, width, height,
                  molstar_version) -> MolstarView:
    mvs = _molviewspec()
    builder = mvs.create_builder()
    ligand_file = pose.file_name()
    _molstar_scene(builder, receptor, pose, ligand_file, residues, surface=surface)
    state = builder.get_state(title=pose.title, description=_summary(pose), description_format="markdown")
    data = {receptor.file_name: receptor.data, ligand_file: pose.data}
    return MolstarView(state, data, info=_info(pose, residues, data), width=width, height=height,
                       ui="viewer", molstar_version=molstar_version)


def _molstar_modes(receptor: _Receptor, poses: List[_Pose], residues_per_pose, *, radius, surface, width, height,
                   ui, molstar_version) -> MolstarView:
    mvs = _molviewspec()
    data = {receptor.file_name: receptor.data}
    snapshots = []
    for pose, residues in zip(poses, residues_per_pose):
        builder = mvs.create_builder()
        ligand_file = pose.file_name(f"mode{pose.mode}")
        data[ligand_file] = pose.data
        _molstar_scene(builder, receptor, pose, ligand_file, residues, surface=surface)
        snapshots.append(builder.get_snapshot(
            title=pose.title, description=_description(receptor, poses, pose, residues, radius),
            description_format="markdown", key=f"mode{pose.mode}", linger_duration_ms=2000,
            transition_duration_ms=300))
    first = poses[0]
    title = f"{first.protein_id} | {first.ligand_id} | pocket {first.pocket} | {len(poses)} modes"
    state = mvs.States(snapshots=snapshots, metadata=mvs.GlobalMetadata(
        title=title, description=_description(receptor, poses, None, [], radius), description_format="markdown"))
    near = {(r.chain, r.seq, r.icode): r for rs in residues_per_pose for r in rs}
    info = _modes_info(poses, [near[k].label for k in sorted(near)], data)
    return MolstarView(state, data, info=info, width=width, height=height, ui=ui, molstar_version=molstar_version)


def _residue_groups(residues: Sequence[Residue]) -> List[dict]:
    """py3Dmol selections of the residues, one per chain."""
    by_chain: Dict[str, List[int]] = {}
    for r in residues:
        by_chain.setdefault(r.chain, [])
        if r.seq not in by_chain[r.chain]:
            by_chain[r.chain].append(r.seq)
    groups = []
    for chain, seqs in by_chain.items():
        sel = {"model": 0, "resi": seqs}
        if chain:
            sel["chain"] = chain
        groups.append(sel)
    return groups


def _py3dmol_scheme(carbon: str) -> dict:
    """3Dmol.js colour scheme: element colours with ``carbon`` for C (keys as PDB and SDF files spell them)."""
    colors = dict(ELEMENT_COLORS, C=carbon)
    scheme = {}
    for el, color in colors.items():
        for key in {el, el.upper()}:
            scheme[key] = int(color.lstrip("#"), 16)
    return {"prop": "elem", "map": scheme}


def _pdbqt_models_to_pdb(models: Sequence[bytes]) -> str:
    """PDB text of PDBQT models for 3Dmol.js, one MODEL each when there are several.

    3Dmol.js reads a PDBQT model only up to its first ``END...`` record (``ENDROOT``), so it would draw the root
    fragment of the pose; as PDB (HETATM lines with elements, no ROOT/BRANCH records) it gets every atom. HETATM, not
    ATOM: 3Dmol.js bonds ATOM records with its protein rules, which miss bonds of a ligand.
    """
    out = []
    for n, model in enumerate(models, 1):
        atoms = ["HETATM" + line[6:] for line in pdbqt_to_pdb(model.decode("utf-8", "replace")).splitlines()
                 if line.startswith(("ATOM  ", "HETATM"))]
        out += [f"MODEL     {n:>4}", *atoms, "ENDMDL"] if len(models) > 1 else atoms
    return "\n".join(out + ["END"]) + "\n"


def _py3dmol_ligand(poses: Sequence[_Pose]) -> Tuple[str, str]:
    """(text, 3Dmol.js format) of the poses: their SDF records, or PDB made from the PDBQT models if any pose did not
    convert (the frames of one model are all of one format)."""
    if all(p.fmt == "sdf" for p in poses):
        texts = [p.data.decode("utf-8", "replace") for p in poses]
        return "".join(t if t.endswith("\n") else t + "\n" for t in texts), "sdf"
    return _pdbqt_models_to_pdb([p.model for p in poses]), "pdb"


def _py3dmol_view(receptor: _Receptor, poses: List[_Pose], residues: Sequence[Residue], label: str, *, surface,
                  width, height, animate: bool):
    py3Dmol = _py3dmol()
    view = py3Dmol.view(width=width, height=height)
    view.setBackgroundColor(BACKGROUND)
    view.addModel(receptor.data.decode("utf-8", "replace"), receptor.py3dmol_format)
    view.setStyle({"model": 0}, {"cartoon": {"color": CARTOON_COLOR}})
    pose = poses[0]
    for sel in _residue_groups(residues):
        view.addStyle(sel, {"stick": {"radius": 0.12, "colorscheme": _py3dmol_scheme(RESIDUE_CARBON)}})
        if surface:
            view.addSurface(py3Dmol.MS, {"opacity": PY3DMOL_SURFACE_OPACITY, "color": _pocket_color(pose.pocket)},
                            sel)
    view.setStyle({"model": 0, "elem": "H"}, {})
    text, fmt = _py3dmol_ligand(poses)
    if len(poses) > 1:
        view.addModelsAsFrames(text, fmt)
    else:
        view.addModel(text, fmt)
    scheme = _py3dmol_scheme(LIGAND_CARBON)
    view.setStyle({"model": 1}, {"stick": {"radius": 0.15, "colorscheme": scheme},
                                 "sphere": {"scale": 0.22, "colorscheme": scheme}})
    view.setStyle({"model": 1, "elem": "H"}, {})
    # a caption in the viewer's top left corner (screen coordinates, so it stays put while the view turns)
    view.addLabel(label, {"useScreen": True, "position": {"x": 10, "y": 10, "z": 0}, "alignment": "topLeft",
                          "fontSize": 13, "fontColor": LABEL_COLOR, "backgroundColor": LABEL_BACKGROUND,
                          "backgroundOpacity": 0.85, "borderThickness": 0})
    view.zoomTo({"within": {"distance": FOCUS_EXTENT, "sel": {"model": 1}}})
    view.zoom(0.8)
    if animate:
        view.animate({"loop": "forward", "reps": 0, "interval": 1000})
    return view


def _info(pose: _Pose, residues: Sequence[Residue], data: Optional[dict]) -> dict:
    """What a view shows (``MolstarView.info``, ``py3Dmol.view.pocketvina_info``)."""
    return {"protein_id": pose.protein_id, "ligand_id": pose.ligand_id, "ligand_idx": pose.ligand_idx,
            "pocket": pose.pocket, "mode": pose.mode, "affinity": pose.affinity, "ligand_format": pose.fmt,
            "residues": [r.label for r in residues], "files": sorted(data) if data else [],
            "download_name": _download_name(pose.protein_id, pose.ligand_id, f"pocket{pose.pocket}",
                                            f"mode{pose.mode}")}


def _modes_info(poses: Sequence[_Pose], residues: List[str], data: Optional[dict]) -> dict:
    p = poses[0]
    return {"protein_id": p.protein_id, "ligand_id": p.ligand_id, "ligand_idx": p.ligand_idx, "pocket": p.pocket,
            "modes": [q.mode for q in poses], "affinities": [q.affinity for q in poses],
            "ligand_formats": [q.fmt for q in poses], "residues": residues,
            "files": sorted(data) if data else [],
            "download_name": _download_name(p.protein_id, p.ligand_id, f"pocket{p.pocket}", "modes")}


def _write_text(path, text: str) -> str:
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return path


def _py3dmol_html(view) -> str:
    out = io.StringIO()
    view.write_html(out, fullpage=True)
    return out.getvalue()


# --- public API --------------------------------------------------------------------------------------------------

def _scene(results, protein, ligand, pocket, mode, backend, width, height, pocket_radius, surface, molstar_version):
    radius = _check_radius(pocket_radius)
    backend = _backend(backend)
    with _results(results) as res:
        protein = _check_protein(res, protein)
        pose = _load_pose(res, protein, ligand, pocket, mode)
        receptor = _receptor(res, protein)
    residues = near_residues(receptor.atoms, pose.xyz, radius)
    if backend == "molstar":
        return _molstar_view(receptor, pose, residues, surface=surface, width=width, height=height,
                             molstar_version=molstar_version)
    view = _py3dmol_view(receptor, [pose], residues, pose.title, surface=surface, width=width, height=height,
                         animate=False)
    view.pocketvina_info = _info(pose, residues, None)
    return view


def show(results: ResultsArg, protein: str, ligand, pocket: Optional[int] = None, mode: int = 1, *,
         backend: str = "auto", width=950, height=600, pocket_radius: float = 5.0, surface: bool = True,
         molstar_version: Optional[str] = None):
    """3D view of a ligand's pose in a protein; the notebook draws it when it is the cell's result.

    ``results``: a results directory (path) or ``io.results.Results``; ``ligand``: ligand_id or ligand_idx.
    ``pocket=None``: the best pose (the stored SDF of section A; the raw PDBQT model if there is none); else mode
    ``mode`` of that pocket (P2Rank rank), converted to SDF in memory (``convert.pose_to_sdf``) or, if that fails,
    the raw PDBQT model. Residues within ``pocket_radius`` A of the pose are drawn as sticks (and a faint surface if
    ``surface``).

    ``backend``: ``"molstar"`` returns a ``MolstarView`` (a MolViewSpec scene with ``data={"protein.pdb": ...,
    "ligand.sdf": ...}``, shown with ``molviewspec.molstar_notebook``), ``"py3dmol"`` a ``py3Dmol.view``,
    ``"auto"`` Mol* when molviewspec imports, else py3Dmol. Outside a notebook the object is returned without
    drawing anything. Mol* loads its JavaScript from a CDN: version ``molstar_version`` (None:
    ``$POCKETVINA_MOLSTAR_VERSION``, else ``MOLSTAR_VERSION``, 5.5.0; any molstar npm version or tag, e.g.
    ``"latest"``); without internet use ``backend="py3dmol"`` (py3Dmol also needs its 3Dmol.js, from a CDN by
    default). To share a view, see ``to_html``.

    Unknown proteins, ligands, pockets or modes, and ligands without a pose, raise ``KeyError``.
    """
    return _scene(results, protein, ligand, pocket, mode, backend, width, height, pocket_radius, surface,
                  molstar_version)


def show_modes(results: ResultsArg, protein: str, ligand, pocket: Optional[int], *, backend: str = "auto",
               width=950, height=600, pocket_radius: float = 5.0, surface: bool = True,
               molstar_version: Optional[str] = None, ui: str = "stories"):
    """All modes of a ligand in one pocket (``pocket=None``: the pocket of its best pose).

    Mol*: one MolViewSpec snapshot per mode (residues near that mode, the mode table in the description), in the
    MVS Stories page (``ui="stories"``, with previous/next buttons) or the plain viewer (``ui="viewer"``, Mol*'s
    snapshot controls). py3Dmol: the modes as frames of one model, animated (residues near any mode).
    Each mode is converted to SDF in memory, else drawn from its PDBQT model (py3Dmol: if any mode does not
    convert, all frames are the PDBQT models). See ``show`` for the arguments and the offline note.
    """
    radius = _check_radius(pocket_radius)
    backend = _backend(backend)
    if ui not in MOLSTAR_UIS:
        raise ValueError(f"ui must be one of {MOLSTAR_UIS}, got {ui!r}")
    with _results(results) as res:
        protein = _check_protein(res, protein)
        i = res.ligand_idx(ligand)
        lid = res.ligand_id(i)
        if pocket is None:
            pocket = _best(res, protein, i, lid)[0]
        poses = _pocket_poses(res, protein, i, lid, pocket, None)
        receptor = _receptor(res, protein)
    if backend == "molstar":
        per_pose = [near_residues(receptor.atoms, p.xyz, radius) for p in poses]
        return _molstar_modes(receptor, poses, per_pose, radius=radius, surface=surface, width=width,
                              height=height, ui=ui, molstar_version=molstar_version)
    residues = near_residues(receptor.atoms, np.concatenate([p.xyz for p in poses]), radius)
    best = min((p for p in poses if p.affinity is not None), key=lambda p: p.affinity, default=poses[0])
    label = (f"{protein} | {lid} | pocket {poses[0].pocket} | modes {poses[0].mode}-{poses[-1].mode} | "
             f"best {best.affinity_text}")
    view = _py3dmol_view(receptor, poses, residues, label, surface=surface, width=width, height=height,
                         animate=len(poses) > 1)
    if len(poses) > 1 and any(p.fmt != poses[0].fmt for p in poses):
        poses = [p.as_pdbqt() for p in poses]
    view.pocketvina_info = _modes_info(poses, [r.label for r in residues], None)
    return view


def to_html(results: ResultsArg, protein: str, ligand, pocket: Optional[int] = None, mode: int = 1,
            path=None, *, backend: str = "auto", width=950, height=600, pocket_radius: float = 5.0,
            surface: bool = True, molstar_version: Optional[str] = None) -> str:
    """Standalone HTML page of the ``show`` view (structures embedded); written to ``path`` if given.

    Mol*: ``molviewspec.molstar_html`` (the scene and its files as a base64 MVSX archive); py3Dmol: its full-page
    HTML. The page loads the viewer's JavaScript from a CDN when opened, so it needs internet to draw.
    """
    view = _scene(results, protein, ligand, pocket, mode, backend, width, height, pocket_radius, surface,
                  molstar_version)
    text = view.to_html() if isinstance(view, MolstarView) else _py3dmol_html(view)
    if path is not None:
        _write_text(path, text)
    return text


def table(results: ResultsArg, top: Optional[int] = 50, protein=None) -> pd.DataFrame:
    """Best pose per protein x ligand, best (lowest) Vina score first; ``top=None`` for all rows.

    Columns: protein_id, ligand_id, smiles_docked, vina, best_pocket, best_mode, ligand_efficiency (``-vina /
    n_heavy`` of the docked fragment), and pb_status / pb_valid when PoseBusters results are there. Ligands without
    a pose are left out. ``protein``: one protein id or a list (default: all).
    """
    if top is not None and (isinstance(top, bool) or operator.index(top) < 0):
        raise ValueError(f"top must be a non-negative int or None, got {top!r}")
    with _results(results) as res:
        pairs = res.pairs(protein)
        ligands = res.ligands(columns=["ligand_idx", "smiles_docked", "n_heavy"])
    df = pairs[(pairs["best_pocket"] >= 1) & pairs["vina"].notna()]
    df = df.merge(ligands, on="ligand_idx", how="left", suffixes=("", "_ligand"))
    if "ligand_efficiency" not in df.columns:  # Results.pairs without the PoseBusters-phase columns
        heavy = pd.to_numeric(df["n_heavy"], errors="coerce").to_numpy(dtype=np.float64)
        heavy[heavy <= 0] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            df["ligand_efficiency"] = -short_floats(df["vina"].to_numpy()) / heavy
    df = df.sort_values(["vina", "protein_id", "ligand_idx"], kind="stable")
    cols = ["protein_id", "ligand_id", "smiles_docked", "vina", "best_pocket", "best_mode", "ligand_efficiency"]
    cols += [c for c in ("pb_status", "pb_valid") if c in df.columns and df[c].notna().any()]
    if top is not None:
        df = df.head(operator.index(top))
    return df[cols].reset_index(drop=True)


__all__ = ["show", "show_modes", "to_html", "table", "MolstarView", "near_residues", "pdbqt_to_pdb", "Residue"]
