import os
import csv
import gzip
import json
import time
import shutil
import sys
import logging
import numbers
import subprocess
from tqdm import tqdm
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Sequence, Union

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel

from . import engine, gpu, p2rank
from .ligands import METHODS as LIGAND_PREP_METHODS, prepare_ligands, read_smiles_file
from .pdbqt import MAX_MODES, engine_output_name

OUTPUT_FORMATS = ("files", "atlas")
COMBINED_CSV_COLUMNS = ("protein", "molecule_name", "affinity (kcal/mol)", "dist from rmsd l.b.", "dist from rmsd u.b.",
                        "mode", "pocket rank")
HTVS_TOP_POCKETS = 5
DEFAULT_BATCH_SIZE = 10  # ligands per batch folder (and engine call) of the "files" format
# "atlas": ligands per engine call when neither ligands_per_call nor batch_size is set; the engine holds every ligand
# of a call in host memory (about 0.8 MB each), so this bounds an engine process to about 0.8 GB
DEFAULT_LIGANDS_PER_CALL = 1000
MAX_TOP_POCKETS = 127  # pocket is int8 in the atlas tables
# printed at the start of every docking run (the engine prints it too, but its output goes to log files now)
CITATION = """\
################################################################################
# If you use PocketVina in your work, please cite:                             #
#                                                                              #
#   Sarigün, Ahmet, et al. “PocketVina Enables Scalable and Highly Accurate    #
#   Physically Valid Docking through Multi-Pocket Conditioning.” arXiv (2025). #
#                                                                              #
# Please also cite the P2Rank and AutoDock Vina family papers. Their BibTeX    #
# entries are available at:                                                    #
#   https://bioinformatics.mdc-berlin.de/PocketVina/citing.html                #
#                                                                              #
################################################################################"""
# rows of the best scores an "atlas" execute_docking() shows at its end: all of them if there are no more, else
# the best ones and a line saying how many there are
BEST_SCORES_SHOWN = 50
_SEED_MAX = 2**31 - 1
# structure file extension (after an optional .gz) -> kind; anything else is read as PDB, as before v0.2
STRUCTURE_KINDS = {".pdb": "pdb", ".ent": "pdb", ".cif": "cif", ".mmcif": "cif", ".pdbqt": "pdbqt"}


def _abspath(path) -> Optional[str]:
    return None if path is None else os.path.abspath(os.path.expanduser(os.fspath(path)))


def read_protein_list(path) -> List[str]:
    """Absolute structure paths from a protein list file (one path per line, blank lines skipped).

    Relative entries are relative to the folder of the list file. An entry that only exists relative to the working
    directory (how lists were read before v0.2) is still taken from there, with a warning.
    """
    base = os.path.dirname(os.path.abspath(path))
    with open(path, 'r') as f:
        entries = [os.path.expanduser(line.strip()) for line in f if line.strip()]
    paths = []
    for entry in entries:
        resolved = os.path.normpath(os.path.join(base, entry))
        if not os.path.isabs(entry) and not os.path.exists(resolved) and os.path.exists(entry):
            logging.getLogger(__name__).warning(
                f"{path}: {entry!r} is not in the list's folder, using {os.path.abspath(entry)} (relative to the "
                "working directory); write paths relative to the list file or absolute")
            resolved = os.path.abspath(entry)
        paths.append(resolved)
    return paths


def read_database_list(path) -> List[Tuple[str, str]]:
    """``[(protein_db, protein_id), ...]`` from the database protein list (CSV with a header row)."""
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        return [(row[0].strip(), row[1].strip()) for row in reader if any(cell.strip() for cell in row)]


def structure_kind(path) -> Tuple[str, bool]:
    """(kind, gzipped) of a structure file: kind is "pdb", "cif" or "pdbqt" (see ``STRUCTURE_KINDS``)."""
    name = os.path.basename(os.fspath(path)).lower()
    gz = name.endswith(".gz")
    if gz:
        name = name[:-3]
    return STRUCTURE_KINDS.get(os.path.splitext(name)[1], "pdb"), gz


def read_structure(path) -> Tuple[str, str]:
    """(kind, text) of a structure file, gunzipped, with universal newlines (as a text-mode ``open``)."""
    kind, gz = structure_kind(path)
    with open(path, "rb") as f:
        data = f.read()
    if gz:
        data = gzip.decompress(data)
    return kind, data.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")


def receptor_pdbqt(structure, legacy: bool = False) -> str:
    """Receptor PDBQT text for the engine, made from a PDB, mmCIF or PDBQT file (optionally gzipped).

    PDB: the pre-v0.2 steps, in memory: ATOM/HETATM lines without ``HOH`` -> OpenBabel PDBQT -> ATOM lines only.
    The receptor is rigid, so it is written as one (``obabel -xr``): the same atoms, types, charges and coordinates
    as before, in input order, in well under a second; ``legacy=True`` writes the pre-v0.2 output (atoms in the order
    of OpenBabel's torsion tree, byte-identical to ``obabel cleaned.pdb -O x.pdbqt``), which takes tens of seconds for
    a large protein. mmCIF is first converted to PDB by OpenBabel; a PDBQT file is used as it is.
    """
    kind, text = read_structure(structure)
    if kind == "pdbqt":
        return text
    if kind == "cif":
        text = pybel.readstring("mmcif", text).write("pdb")
    cleaned = "".join(line for line in text.splitlines(True)
                      if line.startswith(("ATOM", "HETATM")) and "HOH" not in line)
    if not cleaned:
        raise ValueError(f"{structure}: no ATOM/HETATM records")
    log = pybel.ob.obErrorLog
    level = log.GetOutputLevel()
    log.SetOutputLevel(pybel.ob.obError)  # no "Failed to kekulize" warnings for every protein
    try:
        pdbqt = pybel.readstring("pdb", cleaned).write("pdbqt", opt=None if legacy else {"r": None})
    finally:
        log.SetOutputLevel(level)
    atoms = "".join(line for line in pdbqt.splitlines(True) if line.startswith("ATOM"))
    if not atoms:
        raise ValueError(f"{structure}: OpenBabel wrote no ATOM records")
    return atoms


def _engine_stem(file_name: str) -> str:
    cut = file_name.find(".pdbqt")
    return file_name[:cut] if cut >= 0 else file_name


def check_ligand_names(names: Sequence[str], files: bool = True) -> None:
    """Raise ``ValueError`` if two ligands would get the same name, file or engine output.

    Always: duplicate names. With ``files`` (the "files" layout, ``<name>_cleaned.pdbqt``): names that are not plain
    file names, and names whose engine outputs would overwrite each other (the engine cuts at the first ".pdbqt").
    """
    names = list(names)
    problems = []
    dup = sorted(n for n, c in Counter(names).items() if c > 1)
    if dup:
        problems.append(f"duplicate names {dup[:5]}{' ...' if len(dup) > 5 else ''}")
    if files:
        bad = sorted({n for n in names if not n or n in (".", "..") or "/" in n or os.sep in n or "\0" in n})
        if bad:
            problems.append(f"names that are not file names {bad[:5]}")
        by_stem = defaultdict(set)
        for name in set(names):
            by_stem[_engine_stem(f"{name}_cleaned.pdbqt")].add(name)
        clash = sorted(sorted(v) for v in by_stem.values() if len(v) > 1)
        if clash:
            problems.append(f"names with the same engine output name (cut at '.pdbqt') {clash[:5]}")
    if problems:
        raise ValueError("ligand names collide: " + "; ".join(problems))


def effective_top_pockets(config: "PocketVinaConfig", htvs_mode: bool = False) -> Optional[int]:
    """Number of best-ranked pockets to dock: ``config.top_pockets``, else 5 in HTVS mode, else None (all, or the
    ranks of ``config.pockets``). ``ValueError`` if ``htvs_mode`` is combined with ``top_pockets`` or ``pockets``:
    they choose the pockets in different ways."""
    if htvs_mode and (config.top_pockets is not None or config.pockets is not None):
        given = (f"top_pockets={config.top_pockets}" if config.top_pockets is not None
                 else f"pockets={list(config.pockets)}")
        raise ValueError(f"htvs_mode=True docks the {HTVS_TOP_POCKETS} best pockets of each protein, and {given} is set "
                         f"too: use one of them (PocketVinaGPU(config) without htvs_mode to dock {given})")
    if config.top_pockets is not None:
        return config.top_pockets
    return HTVS_TOP_POCKETS if htvs_mode and config.pockets is None else None


def parse_pockets(value) -> Optional[Tuple[int, ...]]:
    """``PocketVinaConfig.pockets`` as sorted distinct ranks: an int (10), ints ([10, 11, 12], range(10, 13)) or text
    ("10", "10-12", "1,3,10-12"); None stays None. ``ValueError`` for anything else or ranks outside
    1..MAX_TOP_POCKETS."""
    if value is None:
        return None
    ranks = set()
    if isinstance(value, str):
        for part in value.replace(" ", "").split(","):
            first, sep, last = part.partition("-")
            if not first.isdigit() or (sep and not last.isdigit()):
                raise ValueError(f"pockets must look like 10, 10-12 or 1,3,10-12, got {value!r}")
            ranks.update(range(int(first), int(last if sep else first) + 1))
    else:
        items = [value] if isinstance(value, numbers.Integral) else value
        try:
            for r in items:
                if isinstance(r, bool) or not isinstance(r, numbers.Integral):
                    raise TypeError
                ranks.add(int(r))
        except TypeError:
            raise ValueError(f"pockets must be a pocket rank, a list or range of ranks, or text such as \"10-12\", "
                             f"got {value!r}") from None
    if not ranks or any(not 1 <= r <= MAX_TOP_POCKETS for r in ranks):
        raise ValueError(f"pockets must name ranks in 1..{MAX_TOP_POCKETS}, got {value!r}")
    return tuple(sorted(ranks))


def effective_ligands_per_call(config: "PocketVinaConfig") -> int:
    """Ligands per engine call of an "atlas" run as configured: ``config.ligands_per_call``, else ``config.batch_size``
    when the user set it (as batch_size was used before), else ``DEFAULT_LIGANDS_PER_CALL`` (1000). 0 means all
    ligands of a protein pocket in one engine call (see ``ligand_chunk_size``)."""
    if config.ligands_per_call is not None:
        return config.ligands_per_call
    if config.batch_size is not None:
        return config.batch_size
    return DEFAULT_LIGANDS_PER_CALL


def ligand_chunk_size(config: "PocketVinaConfig", n_ligands: int) -> int:
    """Ligands per engine call of an "atlas" run with ``n_ligands`` ligands: ``effective_ligands_per_call``, or
    ``n_ligands`` (all in one call) when that is 0 or not smaller. It is the run parameter, so 0, ``n_ligands`` and
    any larger value (e.g. the default 1000 for a small library) are the same run and can resume each other."""
    n = max(int(n_ligands), 1)
    per_call = effective_ligands_per_call(config)
    return n if per_call == 0 or per_call >= n else per_call


def _check_optional_int(name, value, minimum, maximum=None):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer or None, got {value!r}")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ValueError(f"{name} must be {bound}, got {value}")
    return value


@dataclass
class P2RankConfig:
    """How P2Rank finds the pockets of each protein.

    P2Rank is described in the `"P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding
    sites from protein structure" <https://doi.org/10.1186/s13321-018-0285-8>`_ paper.
    """
    threads: int = 4  # CPU threads P2Rank uses
    visualizations: bool = False  # also write P2Rank's PyMOL visualisations ("files" format)
    vis_copy_proteins: bool = False  # copy the proteins into those visualisations
    output_dir: str = "output"  # P2Rank's folder ("files" format; set to the run's output_dir)
    # P2Rank model: "alphafold" for AlphaFold models, "default" for crystal structures, "alphafold_conservation_hmm"
    # (the default) and "conservation_hmm" also use conservation; see P2Rank's models/ folder for all of them
    model: str = "alphafold_conservation_hmm"

@dataclass
class PocketConfig:
    """The docking box around each pocket centre, and the GPU threads per docking."""
    size_x: int = 20  # box size along x, in Angstrom
    size_y: int = 20  # box size along y, in Angstrom
    size_z: int = 20  # box size along z, in Angstrom
    thread: int = 8000  # GPU threads of the engine (at least 1000; more threads search more, and need more memory)

@dataclass
class Paths:
    database_path: Optional[str] = None
    protein_paths: Optional[str] = None
    output_dir: str = "output"  # Add this
    pdb_structures: Optional[str] = None
    uniprot_structures: Optional[str] = None
    pdb_pockets: Optional[str] = None
    uniprot_pockets: Optional[str] = None
    structures_dir: Optional[str] = None
    pockets_dir: Optional[str] = None

    def __post_init__(self):
        if self.database_path:
            # Database path initialization
            self.pdb_structures = f"{self.database_path}/PDB_conservation/structures"
            self.uniprot_structures = f"{self.database_path}/AF_conservation/structures"
            self.pdb_pockets = f"{self.database_path}/PDB_conservation/pockets"
            self.uniprot_pockets = f"{self.database_path}/AF_conservation/pockets"
        elif self.protein_paths:
            # Direct protein paths initialization
            self.structures_dir = os.path.join(self.output_dir, "proteins", "structures")
            self.pockets_dir = os.path.join(self.output_dir, "proteins", "pockets")

@dataclass
class PocketVinaConfig:
    """Settings of a docking run. Only ``smiles_file`` and ``protein_paths`` (or ``database_path`` with
    ``protein_list``) are required.

    - protein_paths: text file with one protein structure per line (.pdb, .cif or .pdbqt, may be gzipped).
    - smiles_file: CSV file with the header ``molecule_name,smiles`` and one ligand per line (.csv or .txt).
    - output_dir: folder the results are written to (default "output").
    - output_format: "files" (PDBQT files and combined_results.csv, as in v0.1) or "atlas" (tables, resumable).
    - batch_size: molecules per docking batch (default 10; "atlas": 1000 per engine call, see ligands_per_call).
    - p2rank_config: how P2Rank finds the pockets, a ``P2RankConfig`` (default ``P2RankConfig()``).
    - pocket_config: docking box size and GPU threads, a ``PocketConfig`` (default ``PocketConfig()``).
    - top_pockets: dock only the N best-ranked pockets of each protein (default all; ``htvs_mode=True``: 5).
    - pockets: dock only these P2Rank ranks: 10, [10, 11, 12], range(10, 13) or "10-12" (not with top_pockets or
      htvs_mode).
    - pockets_dir: "atlas" only: a folder of P2Rank predictions, <protein file name>.csv, e.g. from P2RankProcessor
      (its <output_dir>/proteins/pockets): proteins with one there are docked into those pockets, P2Rank is not run.
    - seed: random seed; the same seed gives the same results (default 42; None: random).

    Advanced:

    - database_path, protein_list: dock the proteins of a prepared database folder instead of protein_paths.
    - tmp_dir: folder for temp files (default $POCKETVINA_TMPDIR, $SLURM_TMPDIR, $TMPDIR, the system temp folder).
    - num_modes, energy_range: poses kept per ligand and pocket, and their score range (default 20 within 20 kcal/mol).
    - ligand_prep: "standard" (largest fragment, seeded 3D) or "legacy" (as before v0.2, to regenerate old results).
    - gzip_level: compression level of the "atlas" pose files (-1 to 9, default 6).
    - posebusters: "atlas" only: check the highest-affinity pose of each ligand with PoseBusters while docking
      (default False).
    - posebusters_workers: processes for the PoseBusters checks (default the CPUs available, at most 16).
    - ligands_per_call: "atlas" only: ligands per engine call (default batch_size if set, else 1000; 0: all at once).
    - engine_timeout: seconds one engine call may take; a slower call is stopped (default no limit).
    - gpus: "atlas" only: the GPU to dock on, an nvidia-smi index or UUID (default the first visible one, e.g. the
      one SLURM gives the job); "all" or "0,2" for several (usually one job per GPU is simpler).
    - procs_per_gpu: "atlas" only: pockets docked at the same time on the GPU, a number or "auto" (default: as many
      as its free memory holds at vram_per_proc_mib each, e.g. 3 on a 24 GB P40, 5 on a 40 GB A100).
    - vram_per_proc_mib: GPU memory one engine needs, in MiB, for procs_per_gpu (default 6600).
    """
    database_path: Optional[str] = None
    protein_paths: Optional[str] = None
    protein_list: Optional[str] = None  # For database workflow
    smiles_file: Optional[str] = None  # Can be .csv or .txt
    output_dir: Optional[str] = None   # Will default to "output"
    batch_size: Optional[int] = None   # "files": None means 10; "atlas": see ligands_per_call
    p2rank_config: Optional[P2RankConfig] = None
    pocket_config: PocketConfig = field(default_factory=PocketConfig)
    output_format: str = "files"
    tmp_dir: Optional[str] = None
    top_pockets: Optional[int] = None
    pockets: Optional[Union[int, Sequence[int], str]] = None
    pockets_dir: Optional[str] = None
    seed: Optional[int] = 42
    num_modes: Optional[int] = None
    energy_range: Optional[float] = None
    ligand_prep: str = "standard"
    gzip_level: int = 6
    posebusters: bool = False
    posebusters_workers: Optional[int] = None
    ligands_per_call: Optional[int] = None
    engine_timeout: Optional[float] = None
    gpus: Optional[str] = None
    procs_per_gpu: Union[int, str] = gpu.AUTO
    vram_per_proc_mib: int = gpu.DEFAULT_VRAM_PER_PROC_MIB

    def __post_init__(self):
        if not self.smiles_file:
            raise ValueError("SMILES file is required")
        if not (self.database_path or self.protein_paths):
            raise ValueError("Either database_path or protein_paths must be provided")
        if self.database_path and not self.protein_list:
            raise ValueError("protein_list is required when using database_path")
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, got {self.output_format!r}")
        if self.ligand_prep not in LIGAND_PREP_METHODS:
            raise ValueError(f"ligand_prep must be one of {LIGAND_PREP_METHODS}, got {self.ligand_prep!r}")
        if self.output_dir is None:
            self.output_dir = "output"
        # batch_size stays None when not given: the "files" format then uses DEFAULT_BATCH_SIZE, while "atlas"
        # honours only a batch_size that was set (see effective_ligands_per_call)
        # absolute, so nothing depends on the working directory of later calls (or of the engine)
        self.smiles_file = _abspath(self.smiles_file)
        self.output_dir = _abspath(self.output_dir)
        self.database_path = _abspath(self.database_path) if self.database_path else self.database_path
        self.protein_paths = _abspath(self.protein_paths) if self.protein_paths else self.protein_paths
        self.protein_list = _abspath(self.protein_list) if self.protein_list else self.protein_list
        self.tmp_dir = _abspath(self.tmp_dir) if self.tmp_dir else None
        self.batch_size = _check_optional_int("batch_size", self.batch_size, 1)
        self.ligands_per_call = _check_optional_int("ligands_per_call", self.ligands_per_call, 0)  # 0: all in one
        self.engine_timeout = engine.check_timeout("engine_timeout", self.engine_timeout)
        self.top_pockets = _check_optional_int("top_pockets", self.top_pockets, 1, MAX_TOP_POCKETS)
        self.pockets = parse_pockets(self.pockets)
        self.pockets_dir = _abspath(self.pockets_dir) if self.pockets_dir else None
        if self.pockets is not None and self.top_pockets is not None:
            raise ValueError(f"top_pockets={self.top_pockets} and pockets={list(self.pockets)} are both set: use one "
                             "of them")
        self.seed = _check_optional_int("seed", self.seed, 0, _SEED_MAX)
        max_modes = MAX_MODES if self.output_format == "atlas" else None  # n_modes is int8 in the atlas tables
        self.num_modes = _check_optional_int("num_modes", self.num_modes, 1, max_modes)
        self.gzip_level = _check_optional_int("gzip_level", self.gzip_level, -1, 9)
        if self.gzip_level is None:
            raise ValueError("gzip_level must be an integer in [-1, 9], got None")
        if not isinstance(self.posebusters, bool):
            raise ValueError(f"posebusters must be True or False, got {self.posebusters!r}")
        self.posebusters_workers = _check_optional_int("posebusters_workers", self.posebusters_workers, 1)
        self.gpus = gpu.check_gpu_spec(self.gpus)
        self.procs_per_gpu = gpu.check_procs_per_gpu(self.procs_per_gpu)
        self.vram_per_proc_mib = _check_optional_int("vram_per_proc_mib", self.vram_per_proc_mib, 1)
        if self.vram_per_proc_mib is None:
            raise ValueError("vram_per_proc_mib must be an integer >= 1 (MiB), got None")
        if self.energy_range is not None:
            if (isinstance(self.energy_range, bool) or not isinstance(self.energy_range, numbers.Real)
                    or not self.energy_range >= 0 or self.energy_range == float("inf")):
                raise ValueError(f"energy_range must be a finite number >= 0 or None, got {self.energy_range!r}")


class P2RankProcessor:
    """Runs P2Rank on a list of proteins and saves their pockets, for docking later.

    P2Rank is described in the `"P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding
    sites from protein structure" <https://doi.org/10.1186/s13321-018-0285-8>`_ paper.
    """
    def __init__(self, config: P2RankConfig = None):
        """Initialize P2Rank processor with configuration"""
        self.config = config or P2RankConfig()
        self.config.output_dir = _abspath(self.config.output_dir)
        self._p2rank_error = None  # why P2Rank could not be installed (see _prank)
        self.legacy_receptor = False  # receptor_pdbqt(legacy=True): PocketVinaConfig(ligand_prep="legacy")
        self.setup_directories()
        self.setup_logging()

    def _prank(self) -> str:
        """P2Rank's ``prank`` launcher (``p2rank.ensure_p2rank()``: downloaded into the user cache first if there is
        no P2Rank). Once installing it failed, the same ``P2RankError`` again: not one more download for every
        protein of a list."""
        if self._p2rank_error is None:
            try:
                return p2rank.ensure_p2rank()
            except p2rank.P2RankError as e:
                self._p2rank_error = str(e)
        raise p2rank.P2RankError(self._p2rank_error)

    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('P2RankProcessor')
        self.logger.setLevel(logging.INFO)

        # One processor's handlers at a time: each new processor used to add another pair (repeated log lines)
        for handler in list(self.logger.handlers):
            if getattr(handler, "_pocketvina_p2rank", False):
                self.logger.removeHandler(handler)
                handler.close()

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(f'{self.config.output_dir}/p2rank_processor.log')

        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)

        # Add handlers
        for handler in (c_handler, f_handler):
            handler._pocketvina_p2rank = True
            self.logger.addHandler(handler)

    def setup_directories(self):
        """Setup required directories"""

        os.makedirs(self.config.output_dir, exist_ok=True)

        self.pockets_dir = os.path.join(self.config.output_dir, "proteins", "pockets")
        self.structures_dir = os.path.join(self.config.output_dir, "proteins", "structures")
        os.makedirs(self.pockets_dir, exist_ok=True)
        os.makedirs(self.structures_dir, exist_ok=True)

    def run_p2rank(self, pdb_file: str) -> bool:
        """Run P2Rank prediction on a single PDB file"""
        self.logger.info(f"Running P2Rank on {pdb_file}")

        try:
            # Run P2Rank ($P2RANK_HOME, prank on PATH, p2rank_2.5 in the package or the user cache, downloaded there
            # once if there is none; see pocketvina.p2rank)
            result = subprocess.run([
                self._prank(), 'predict',
                '-f', pdb_file,
                '-o', self.config.output_dir,
                '-visualizations', str(self.config.visualizations).lower(),
                '-threads', str(self.config.threads),
                '-vis_copy_proteins', str(self.config.vis_copy_proteins).lower(),
                '-model', self.config.model,  # Ensure this model exists in your setup
            ], check=True, capture_output=True, text=True)

            # Handle prediction file
            input_filename = os.path.basename(pdb_file)
            base_name = os.path.splitext(input_filename)[0]
            predictions_file = os.path.join(self.config.output_dir, f"{input_filename}_predictions.csv")
            target_file = os.path.join(self.pockets_dir, f"{base_name}.csv")

            if os.path.exists(predictions_file):
                shutil.move(predictions_file, target_file)
                self._cleanup_temp_files(self.config.output_dir, input_filename)
                self.logger.info(f"Successfully processed {pdb_file}")
                return True

            self.logger.error(f"No predictions file found at {predictions_file}")
            return False

        except Exception as e:
            detail = ""
            if isinstance(e, subprocess.CalledProcessError):  # P2Rank's own message, and a Java that cannot run it
                output = f"{e.stderr or ''}{e.stdout or ''}".strip()
                detail = f": {output[-2000:]}" if output else ""
                java = p2rank.java_problem()
                detail += f" ({java})" if java else ""
            self.logger.error(f"Error in P2Rank execution: {str(e)}{detail}")
            return False

    def process_pdb_file(self, input_file: str) -> Optional[str]:
        """Process PDB file and convert to PDBQT"""
        self.logger.info(f"Processing PDB file: {input_file}")

        try:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            final_pdbqt = os.path.join(self.structures_dir, f"{base_name}.pdbqt")

            # ATOM/HETATM lines without HOH -> OpenBabel PDBQT -> ATOM lines (same output as the former obabel CLI
            # steps, see receptor_pdbqt); written under a temp name so a stopped run leaves no partial file
            text = receptor_pdbqt(input_file, legacy=self.legacy_receptor)
            tmp_pdbqt = f"{final_pdbqt}.tmp-{os.getpid()}"
            with open(tmp_pdbqt, 'w') as outfile:
                outfile.write(text)
            os.replace(tmp_pdbqt, final_pdbqt)

            if os.path.exists(final_pdbqt):
                self.logger.info(f"Successfully converted {input_file} to PDBQT")
                return final_pdbqt

            self.logger.error(f"PDBQT file was not created at {final_pdbqt}")
            return None

        except Exception as e:
            self.logger.error(f"Error processing PDB file: {str(e)}")
            return None

    def process_file_list(self, file_list_path: Union[str, Sequence[str]]) -> Dict[str, bool]:
        """Process multiple PDB files from a list file (or a list of paths)"""
        self.logger.info(f"Processing file list from: {file_list_path}")
        results = {}

        try:
            # Read file paths; relative entries are relative to the list file's folder
            if isinstance(file_list_path, (list, tuple)):
                pdb_files = [_abspath(p) for p in file_list_path]
            else:
                pdb_files = read_protein_list(file_list_path)

            # Process each file
            for pdb_file in pdb_files:
                base_name = os.path.splitext(os.path.basename(pdb_file))[0]
                pdbqt_file = os.path.join(self.structures_dir, f"{base_name}.pdbqt")
                pockets_csv = os.path.join(self.pockets_dir, f"{base_name}.csv")

                # Check if both files already exist
                if os.path.exists(pdbqt_file) and os.path.exists(pockets_csv):
                    self.logger.info(f"Skipping {base_name}: files already exist")
                    results[pdb_file] = True
                    continue

                self.logger.info(f"Processing: {pdb_file}")

                if not self.run_p2rank(pdb_file):
                    results[pdb_file] = False
                    continue

                if not self.process_pdb_file(pdb_file):
                    results[pdb_file] = False
                    continue

                results[pdb_file] = True

        except Exception as e:
            self.logger.error(f"Error processing file list: {str(e)}")

        self._print_summary(results)
        return results

    def _cleanup_temp_files(self, output_dir: str, input_filename: str):
        """Clean up temporary files after processing"""
        temp_files = [
            f"{input_filename}_residues.csv",
            "params.txt",
            "run.log"
        ]
        for temp_file in temp_files:
            temp_path = os.path.join(output_dir, temp_file)
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _print_summary(self, results: Dict[str, bool]):
        """Print processing summary"""
        successful = sum(1 for success in results.values() if success)
        total = len(results)

        self.logger.info("\nProcessing Summary:")
        self.logger.info("-" * 50)
        self.logger.info(f"Successfully processed: {successful}/{total} files")

        if successful != total:
            self.logger.info("\nFailed files:")
            for file_path, success in results.items():
                if not success:
                    self.logger.info(f"- {file_path}")


class PocketVinaGPU:
    """Docking with PocketVina, from the `"PocketVina Enables Scalable and Highly Accurate Physically Valid Docking through
    Multi-Pocket Conditioning" <https://arxiv.org/abs/2506.20043>`_ paper.

    P2Rank finds the pockets of each protein (`"P2Rank: machine learning based tool for rapid and accurate prediction of
    ligand binding sites from protein structure" <https://doi.org/10.1186/s13321-018-0285-8>`_) and every ligand is
    docked into the best pockets on the GPU, with an engine based on QuickVina2-GPU 2.1 (`"Vina-GPU 2.1: towards further
    optimizing docking speed and precision of AutoDock Vina and its derivatives"
    <https://doi.org/10.1101/2023.11.04.565429>`_) and AutoDock Vina (`"AutoDock Vina: improving the speed and accuracy
    of docking with a new scoring function, efficient optimization, and multithreading"
    <https://doi.org/10.1002/jcc.21334>`_).
    """
    def __init__(self, config: PocketVinaConfig, htvs_mode = False):
        """Initialize PocketVina with configuration"""
        self.htvs_mode = htvs_mode
        self.config = config
        self.paths = Paths(
            database_path=config.database_path,
            protein_paths=config.protein_paths,
            output_dir=config.output_dir
        )
        self.pocket_config = config.pocket_config
        self.batch_size = config.batch_size if config.batch_size is not None else DEFAULT_BATCH_SIZE  # "files"
        self.top_pockets = effective_top_pockets(config, htvs_mode)
        self.output_dir = config.output_dir
        self.config_dir = os.path.join(self.output_dir, 'configs')
        self.ligand_output_dir = self.output_dir
        self.p2rank_processor = None
        self._run_json_files = None  # engine JSON outputs of the current execute_docking call
        self._engine_cwd = None

        # Initialize P2Rank if using direct protein paths ("atlas" runs P2Rank in a temp dir, see pipeline.py)
        if config.protein_paths and config.output_format == "files":
            p2rank_config = config.p2rank_config or P2RankConfig()
            p2rank_config.output_dir = config.output_dir
            self.p2rank_processor = P2RankProcessor(p2rank_config)
            self.p2rank_processor.legacy_receptor = config.ligand_prep == "legacy"

        # None: the engine's per-user kernel cache (engine.kernel_cache_dir()), never the package directory
        self.opencl_binary_path = None
        self._setup_logging()

    def process_proteins(self) -> Dict[str, Dict]:
        """Process proteins either from database or direct paths"""
        if self.config.protein_paths:
            return self._process_protein_paths()
        elif self.config.database_path:
            return self._process_database_proteins()
        else:
            self.logger.error("Neither protein paths nor database path provided")
            return {}

    def _get_p2rank_processor(self) -> P2RankProcessor:
        if self.p2rank_processor is None:
            p2rank_config = self.config.p2rank_config or P2RankConfig()
            p2rank_config.output_dir = self.config.output_dir
            self.p2rank_processor = P2RankProcessor(p2rank_config)
            self.p2rank_processor.legacy_receptor = self.config.ligand_prep == "legacy"
        return self.p2rank_processor

    def _process_protein_paths(self) -> Dict[str, Dict]:
        """Process proteins from direct paths using P2Rank"""
        results = {}

        # Read protein paths from file
        protein_paths = read_protein_list(self.config.protein_paths)

        for protein_path in protein_paths:
            base_name = os.path.splitext(os.path.basename(protein_path))[0]
            pdbqt_path = os.path.join(self.paths.structures_dir, f"{base_name}.pdbqt")
            pockets_csv = os.path.join(self.paths.pockets_dir, f"{base_name}.csv")

            # Check if both files already exist
            if os.path.exists(pdbqt_path) and os.path.exists(pockets_csv):
                self.logger.info(f"Found existing files for {base_name}, skipping P2Rank processing")
                try:
                    df = pd.read_csv(pockets_csv)
                    df.columns = df.columns.str.strip().str.lower()
                    results[protein_path] = {
                        'pdbqt_path': os.path.abspath(pdbqt_path),
                        'pockets_csv': os.path.abspath(pockets_csv),
                        'pockets': df.to_dict('records')
                    }
                    continue
                except Exception as e:
                    self.logger.error(f"Error reading existing files for {base_name}: {e}")

        # Process remaining proteins with P2Rank
        remaining_proteins = [p for p in protein_paths if p not in results]
        if remaining_proteins:
            p2rank_results = self._get_p2rank_processor().process_file_list(remaining_proteins)

            for protein_path, success in p2rank_results.items():
                if not success:
                    continue

                base_name = os.path.splitext(os.path.basename(protein_path))[0]
                protein_data = {
                    'pdbqt_path': os.path.abspath(os.path.join(self.paths.structures_dir, f"{base_name}.pdbqt")),
                    'pockets_csv': os.path.abspath(os.path.join(self.paths.pockets_dir, f"{base_name}.csv"))
                }

                try:
                    df = pd.read_csv(protein_data['pockets_csv'])
                    df.columns = df.columns.str.strip().str.lower()
                    protein_data['pockets'] = df.to_dict('records')
                    results[protein_path] = protein_data
                except Exception as e:
                    self.logger.error(f"Error reading pockets for {protein_path}: {e}")

        return results

    def _process_database_proteins(self) -> Dict[str, Dict]:
        """Process proteins from database"""
        results = {}

        try:
            protein_ids = read_database_list(self.config.protein_list)

            for protein_db, protein_id in protein_ids:
                self.protein_db = protein_db
                self.protein_id = protein_id
                protein_file = self.get_database_protein_file()
                if not protein_file:
                    continue

                pockets = self.read_database_pockets()
                if not pockets:
                    continue

                results[f"{protein_db}_{protein_id}"] = {
                    'pdbqt_path': protein_file,
                    'pockets': pockets
                }

        except Exception as e:
            self.logger.error(f"Error processing database proteins: {e}")

        return results

    def _setup_logging(self) -> None:
        """Setup logging configuration with performance considerations"""
        handlers = [logging.StreamHandler()]
        if self.config.output_format == "files":
            # "atlas" results directories hold only the results (several processes may share one)
            os.makedirs(self.ligand_output_dir, exist_ok=True)
            # Buffer writes to file for better performance
            handlers.insert(0, logging.FileHandler(f'{self.ligand_output_dir}/pocketvina_summary.log', 'a',
                                                   encoding='utf-8', delay=True))
        logging.basicConfig(
            level=logging.INFO,  # Only log INFO and above
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)

    def truncate_after_torsdof(self, file_path: str, output_path: str) -> None:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            truncated_lines = []
            found_torsdof = False

            for line in lines:
                truncated_lines.append(line)
                if line.startswith("TORSDOF"):
                    found_torsdof = True
                    break

            if found_torsdof:
                with open(output_path, 'w') as file:
                    file.writelines(truncated_lines)
            else:
                print(f"No TORSDOF line found in {file_path}")
        except Exception as e:
            self.logger.error(f"Error truncating file {file_path}: {e}")

    def _create_molecule(self, smiles: str) -> Chem.Mol:
        """Create RDKit molecule from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            raise ValueError(f"3D embedding failed for: {smiles}")
        return mol

    def _convert_and_save_molecule(self, mol: Chem.Mol, molecule_name: str, output_dir: str) -> None:
        """Convert molecule to PDBQT format and save"""
        mol2_file = os.path.join(output_dir, f"{molecule_name}.mol2")
        pdbqt_file = os.path.join(output_dir, f"{molecule_name}.pdbqt")
        cleaned_pdbqt = os.path.join(output_dir, f"{molecule_name}_cleaned.pdbqt")

        try:
            ob_mol = pybel.readstring("mol", Chem.MolToMolBlock(mol))
            ob_mol.write("mol2", mol2_file, overwrite=True)

            subprocess.run(['obabel', mol2_file, '-O', pdbqt_file], check=True, text=True)
            os.remove(mol2_file)
            self.truncate_after_torsdof(pdbqt_file, cleaned_pdbqt)
            os.remove(pdbqt_file)
        except Exception as e:
            raise RuntimeError(f"Error converting molecule: {e}")

    def process_smiles(self, row: Tuple[str, str], output_dir: str) -> None:
        """Process single SMILES entry"""
        molecule_name, smiles = row
        try:
            mol = self._create_molecule(smiles)
            self._convert_and_save_molecule(mol, molecule_name, output_dir)
        except Exception as e:
            self.logger.error(f"Error processing {molecule_name}: {e}")

    def _ligand_rows(self) -> List[Tuple[str, str]]:
        """``(molecule_name, smiles)`` rows of the SMILES file; raises if two names would collide."""
        rows = read_smiles_file(self.config.smiles_file)
        check_ligand_names([name for name, _ in rows], files=True)
        return rows

    def smiles_to_pdbqt(self) -> None:
        """Convert SMILES to PDBQT format (``<molecule_name>_cleaned.pdbqt``, see ``config.ligand_prep``)"""
        rows = self._ligand_rows()
        try:
            os.makedirs(self.ligand_output_dir, exist_ok=True)
            records = prepare_ligands(rows, method=self.config.ligand_prep, seed=self.config.seed)
            for rec in records:
                if rec.status != "ok":
                    self.logger.error(f"Error processing {rec.ligand_id}: {rec.status}: {rec.error}")
                    continue
                path = os.path.join(self.ligand_output_dir, f"{rec.ligand_id}_cleaned.pdbqt")
                # one ligand that cannot be written (e.g. a name too long for a file name) skips only that ligand
                try:
                    with open(path, 'w', encoding='utf-8', newline='') as f:
                        f.write(rec.pdbqt)
                except OSError as e:
                    self.logger.error(f"Error writing {rec.ligand_id}: {e}")
                    if os.path.isfile(path):
                        os.remove(path)
        except Exception as e:
            self.logger.error(f"Error in SMILES conversion: {e}")

    def split_files_into_batches(self, source_folder: str) -> List[Tuple[str, int]]:
        """Split files into batches for processing"""
        try:
            destination_folder = source_folder
            # sorted: the same ligands always form the same batches
            files = sorted(f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f)) and f.endswith('.pdbqt'))

            # Remove batch_size parameter since it's now an instance variable
            existing_batch_folders = [
                (os.path.join(destination_folder, folder), int(folder.split('_')[1]))
                for folder in os.listdir(destination_folder)
                if folder.startswith("batch_") and os.path.isdir(os.path.join(destination_folder, folder))
            ]

            if not files and existing_batch_folders:
                print("No `.pdbqt` files left to process. All files have already been moved to batch folders.")
                return sorted(existing_batch_folders, key=lambda x: x[1])

            if not files:
                print("No `.pdbqt` files found and no batch folders exist.")
                return []

            print(f"Found {len(files)} '.pdbqt' files in the source folder.")

            batch_folders = existing_batch_folders.copy()

            for i in range(0, len(files), self.batch_size):
                batch_number = len(batch_folders) + 1
                batch_folder = os.path.join(destination_folder, f"batch_{batch_number}")

                if os.path.exists(batch_folder):
                    print(f"Batch folder '{batch_folder}' already exists.")
                else:
                    print(f"Creating new batch folder: '{batch_folder}'")
                    os.makedirs(batch_folder, exist_ok=True)

                batch_folders.append((batch_folder, batch_number))

                batch_files = files[i:i + self.batch_size]
                print(f"Moving {len(batch_files)} files to '{batch_folder}'.")
                for file in batch_files:
                    shutil.move(os.path.join(source_folder, file), os.path.join(batch_folder, file))

            print("Files have been split into batches and moved.")
            return sorted(batch_folders, key=lambda x: x[1])
        except Exception as e:
            self.logger.error(f"Error splitting files into batches: {e}")
            return []

    def database_protein_file(self, protein_db: str, protein_id: str) -> str:
        """Receptor PDBQT of a database protein (``ValueError`` for an unknown database)."""
        if protein_db == "PDB":
            return os.path.join(self.paths.pdb_structures, f"pdb{protein_id}_filtered.pdbqt")
        elif protein_db == "UniProt":
            return os.path.join(self.paths.uniprot_structures, f"AF-{protein_id}-F1-model_v1_filtered.pdbqt")
        raise ValueError(f"Unsupported protein database: {protein_db}")

    def database_pockets_csv(self, protein_db: str, protein_id: str) -> str:
        """P2Rank predictions CSV of a database protein (``ValueError`` for an unknown database)."""
        if protein_db == "PDB":
            return os.path.join(self.paths.pdb_pockets, f"pdb{protein_id}.ent.gz_predictions.csv")
        elif protein_db == "UniProt":
            return os.path.join(self.paths.uniprot_pockets, f"AF-{protein_id}-F1-model_v1.pdb.gz_predictions.csv")
        raise ValueError(f"Unsupported protein database: {protein_db}")

    def get_database_protein_file(self) -> Optional[str]:
        """Get protein file path based on database type"""
        try:
            return self.database_protein_file(self.protein_db, self.protein_id)
        except Exception as e:
            self.logger.error(f"Error getting protein file: {e}")
            return None

    def read_database_pockets(self) -> Optional[List[Dict]]:
        """Read pocket information from CSV file"""
        try:
            csv_file = self.database_pockets_csv(self.protein_db, self.protein_id)

            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            df = df.rename(columns=lambda x: x.strip().lower())

            required_columns = {'rank', 'center_x', 'center_y', 'center_z'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Missing required columns in CSV file: {required_columns - set(df.columns)}")

            return df.to_dict('records')

        except Exception as e:
            self.logger.error(f"Error reading pockets: {e}")
            return None

    def generate_config(self, protein_file: str, ligand_directory: str,
                       pocket: Dict, batch: int) -> str:
        """Generate configuration file for docking"""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            filename_without_extension = os.path.splitext(os.path.basename(protein_file))[0]
            config_file = os.path.join(
                self.config_dir,
                f"{filename_without_extension}_pocketrank_{pocket['rank']}_config_batch_{batch}.txt"
            )
            # outputs go to the engine default <ligand_directory>_out
            return engine.write_config(
                config_file, protein_file, ligand_directory, None,
                center=(pocket['center_x'], pocket['center_y'], pocket['center_z']),
                size=(self.pocket_config.size_x, self.pocket_config.size_y, self.pocket_config.size_z),
                thread=self.pocket_config.thread, pocket=int(pocket['rank']), seed=self.config.seed,
                num_modes=self.config.num_modes, energy_range=self.config.energy_range,
                opencl_binary_path=self.opencl_binary_path)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    @engine.exit_on_sigterm()  # SIGTERM still stops the run, but kills the engine and removes temp dirs first
    def execute_docking(self) -> Optional[str]:
        """Main docking execution with better error handling

        With ``output_format="atlas"`` this is ``pipeline.run_atlas`` and returns the results directory; at the end
        it shows the lowest Vina score of each of the run's protein x ligand pairs (``best_scores()``, at most
        ``BEST_SCORES_SHOWN`` rows): a table in a notebook, else printed.
        """
        print(CITATION, flush=True)
        if self.config.output_format == "atlas":
            from .pipeline import run_atlas
            path = run_atlas(self)
            self._show_best_scores()
            return path
        if self.config.posebusters:
            self.logger.warning('posebusters=True needs output_format="atlas": no PoseBusters checks in this run')
        if self.config.ligands_per_call is not None:
            self.logger.warning(f'ligands_per_call applies to output_format="atlas" only: the "files" format docks '
                                f'batches of batch_size={self.batch_size} ligands')
        if self.config.gpus is not None or self.config.procs_per_gpu not in (1, gpu.AUTO):
            self.logger.warning('gpus and procs_per_gpu apply to output_format="atlas" only: the "files" format runs '
                                'one engine at a time on the default GPU')

        # before anything is written: colliding ligand names would overwrite each other's files
        self._ligand_rows()
        self._run_json_files = []
        self._engine_cwd = engine.resolve_tmp_dir(self.config.tmp_dir)
        try:

            # Prepare directories and convert SMILES if needed
            os.makedirs(self.ligand_output_dir, exist_ok=True)
            batch_folders = [f for f in os.listdir(self.ligand_output_dir)
                            if f.startswith('batch_') and os.path.isdir(os.path.join(self.ligand_output_dir, f))]

            if batch_folders:
                self.logger.info(f"Found {len(batch_folders)} existing batch folders.")
            else:
                self.logger.info("No batch folders found. Proceeding with conversion and batch splitting.")
                self.smiles_to_pdbqt()

            # Use self.batch_size here
            lig_batch_folders = self.split_files_into_batches(self.ligand_output_dir)
            lig_batch_folders = [(folder, number) for folder, number in lig_batch_folders
                                if not folder.endswith('_out')]

            start_time = time.time()
            self.logger.info("Starting docking execution...")
            if self.config.database_path:
                # Database workflow
                try:
                    protein_ids = read_database_list(self.config.protein_list)

                    for protein_db, protein_id in protein_ids:
                        try:
                            self.protein_db = protein_db
                            self.protein_id = protein_id

                            protein_file = self.get_database_protein_file()
                            if not protein_file:
                                self.logger.warning(f"Skipping {self.protein_db}_{self.protein_id}: protein file not found")
                                continue

                            pockets = self.read_database_pockets()
                            if not pockets:
                                self.logger.warning(f"Skipping {self.protein_db}_{self.protein_id}: no pockets found")
                                continue

                            self._process_protein_docking(protein_file, pockets, lig_batch_folders)
                        except Exception as e:
                            self.logger.error(f"Error processing {protein_db}_{protein_id}: {e}")
                            continue
                except Exception as e:
                    self.logger.error(f"Error processing database proteins: {e}")
                    raise
            else:
                # Direct protein paths workflow
                protein_results = self._get_p2rank_processor().process_file_list(self.config.protein_paths)
                for protein_path, success in protein_results.items():
                    if not success:
                        continue

                    base_name = os.path.splitext(os.path.basename(protein_path))[0]
                    pdbqt_path = os.path.join(self.paths.structures_dir, f"{base_name}.pdbqt")
                    pockets_csv = os.path.join(self.paths.pockets_dir, f"{base_name}.csv")

                    try:
                        df = pd.read_csv(pockets_csv)
                        df.columns = df.columns.str.strip().str.lower()
                        pockets = df.to_dict('records')
                        self._process_protein_docking(pdbqt_path, pockets, lig_batch_folders)
                    except Exception as e:
                        self.logger.error(f"Error processing {protein_path}: {e}")
                        continue


            self.logger.info(f"Docking execution completed in {time.time() - start_time:.2f} seconds")

            # Process results
            self._process_results()

        except Exception as e:
            self.logger.error(f"Docking execution failed: {e}")
            raise
        finally:
            shutil.rmtree(self._engine_cwd, ignore_errors=True)
            self._engine_cwd = None

    def results(self):
        """The results in ``output_dir`` (a ``pocketvina.io.results.Results``), also of an earlier session:
        ``PocketVinaGPU(config).results()``. Needs ``output_format="atlas"``."""
        if self.config.output_format != "atlas":
            raise ValueError(f'results need output_format="atlas" (this config has output_format='
                             f'{self.config.output_format!r}): set output_format="atlas" in PocketVinaConfig and '
                             f'run execute_docking()')
        from .io.results import Results

        if not os.path.isfile(os.path.join(self.output_dir, "MANIFEST.json")):
            raise FileNotFoundError(f"no results in {self.output_dir} yet (no MANIFEST.json): run execute_docking() "
                                    "first")
        return Results(self.output_dir)

    def best_scores(self, protein=None, top: Optional[int] = None) -> pd.DataFrame:
        """Lowest Vina score of each protein x ligand, highest-affinity first; see ``Results.best_scores``.

        protein: a protein id (the file name without extension), e.g. "1fbl", or a list, e.g. ["1fbl", "2W83"];
        None (default): all proteins.
        top: only the first N rows, e.g. 10; None (default): all rows.
        """
        with self.results() as res:
            return res.best_scores(protein=protein, top=top)

    def select(self, protein: str, ligand, pocket=None, mode=1):
        """One protein x ligand pair, to show, check or save; see ``Results.select``:
        ``pair = pocketvina.select(protein="1fbl", ligand="molecule1")``, then ``pair.show()``, ``pair.bust()``,
        ``pair.export("1fbl_molecule1.sdf")``.

        protein: a protein id as in best_scores(), e.g. "1fbl".
        ligand: a ligand id (its name in the SMILES file) as in best_scores(), e.g. "molecule1".
        pocket: a pocket number, e.g. 2; None (default): the pocket of the pair's highest-affinity pose; "all": every
            pocket the ligand was docked into.
        mode: 1 (default): the highest-affinity mode of the pocket; 2, 3, ...: that mode; "all": every mode of the
            pocket(s).
        """
        with self.results() as res:
            return res.select(protein, ligand, pocket=pocket, mode=mode)

    def show(self, protein=None, *, path=None, molstar: str = "inline", theme: str = "auto", height: int = 760,
             max_ligands: Optional[int] = 100):
        """The results as on the PocketVina Atlas website's protein page (3D view, Top hits, Selected pair, Pockets);
        in a notebook below the cell, elsewhere an HTML file whose path is returned. See ``Results.show``.

        protein: a protein id, e.g. "1fbl", or a list of them; None (default): all proteins.
        path: write the page to this HTML file instead, e.g. "results.html".
        molstar: "inline" (default, works offline) or "url" (the 3D viewer's code from the PocketVina Atlas website).
        theme: "auto" (default), "light" or "dark".
        height: height of the view in a notebook, in pixels, e.g. 900.
        max_ligands: ligands per protein in the page (highest-affinity first), e.g. 50; None: all.
        """
        with self.results() as res:
            return res.show(protein, path=path, molstar=molstar, theme=theme, height=height, max_ligands=max_ligands)

    def _show_best_scores(self) -> None:
        """The best scores of the proteins of this run (``best_scores``): all rows, or the best
        ``BEST_SCORES_SHOWN`` and a line saying how many there are (only those rows are read: a run may have
        millions); in IPython/Jupyter as a table, else printed. Never fails the run."""
        try:
            from .pipeline import protein_inputs

            listed = {p.protein_id for p in protein_inputs(self)}
            with self.results() as res:
                done = [pid for pid in res.proteins() if pid in listed]
                df, total = res._best_scores_head(done, BEST_SCORES_SHOWN) if done else (None, 0)
            if not total:
                print("No docked poses in this run.")
                return
            more = (f"The top {len(df)} of {total} protein x ligand pairs; pocketvina.best_scores() returns all "
                    "of them." if total > len(df) else None)
            # IPython imported and running (Jupyter, Colab, VSCode; get_ipython() is None in a plain script)
            shell = getattr(sys.modules.get("IPython"), "get_ipython", lambda: None)()
            if shell is not None:
                from IPython.display import display

                display(df)
            else:
                print(f"Best scores:\n{df.to_string(index=False)}")
            if more:
                print(more)
        except Exception as e:  # e.g. a display or stdout that fails: the run is done, its results are written
            self.logger.warning(f"cannot show the best scores: {type(e).__name__}: {e}")

    def _process_protein_docking(self, protein_file: str, pockets: List[Dict],
                            batch_folders: List[Tuple[str, int]]) -> None:
        """Process docking for a single protein"""
        top_pockets = getattr(self, "top_pockets", None)
        pockets_to_process = pockets[:top_pockets] if top_pockets else pockets
        if self.config.pockets is not None:
            pockets_to_process = [p for p in pockets_to_process if int(p['rank']) in self.config.pockets]
        own_cwd = self._engine_cwd is None
        cwd = self._engine_cwd
        if own_cwd:
            cwd = engine.resolve_tmp_dir(self.config.tmp_dir)

        try:
            for pocket in pockets_to_process:
                for folder, batch_num in batch_folders:
                    try:
                        config_file = self.generate_config(
                            protein_file=protein_file,
                            ligand_directory=folder,
                            pocket=pocket,
                            batch=batch_num
                        )
                        if not config_file:
                            continue

                        before = self._output_stats(protein_file, pocket, folder)
                        # The engine runs in a temp dir (never the package directory); it writes <folder>_out
                        result = engine.run_engine(config_file, cwd=cwd, timeout=self.config.engine_timeout)
                        if result.timed_out:
                            # killed: its outputs may be missing or cut short, none of them is merged
                            self.logger.error(f"Error in docking: engine timed out after {self.config.engine_timeout:g}"
                                              f" s (engine_timeout) and was killed, the outputs of this call are left "
                                              f"out ({config_file}):\n{result.log_tail[-1500:]}")
                            continue
                        if not result.ok:
                            self.logger.error(f"Error in docking: engine exited with code {result.returncode} "
                                              f"({config_file}):\n{result.log_tail[-1500:]}")
                        self._record_outputs(before)
                    except Exception as e:
                        self.logger.error(f"Error in docking: {e}")
        finally:
            if own_cwd:
                shutil.rmtree(cwd, ignore_errors=True)

    @staticmethod
    def _stat(path: str):
        try:
            st = os.stat(path)
        except OSError:
            return None
        return st.st_ino, st.st_size, st.st_mtime_ns

    def _output_stats(self, protein_file: str, pocket: Dict, folder: str) -> Dict[str, Optional[tuple]]:
        """``{JSON output path: stat or None}`` for every ligand in ``folder`` docked into ``pocket``."""
        rank = int(pocket['rank'])
        out_dir = f"{folder}_out"
        stats = {}
        for name in sorted(f for f in os.listdir(folder) if f.endswith('.pdbqt')):
            out_name = engine_output_name(protein_file, rank, name)
            path = os.path.join(out_dir, out_name[:-len(".pdbqt")] + ".json")
            stats[path] = self._stat(path)
        return stats

    def _record_outputs(self, before: Dict[str, Optional[tuple]]) -> None:
        """Remember the JSON outputs one engine call wrote (merged by ``_process_results``); a file it did not
        (re)write holds an older result and is left out."""
        if self._run_json_files is None:
            return
        for path, old in before.items():
            new = self._stat(path)
            if new is not None and new != old:
                self._run_json_files.append(path)

    def _process_results(self) -> None:
        """Process docking results"""
        try:
            output_json = os.path.join(self.ligand_output_dir, "combined_results.json")
            output_csv = os.path.join(self.ligand_output_dir, "combined_results.csv")

            # Initialize empty lists for combined data
            combined_data = []

            if self._run_json_files is not None:
                # Only this run's outputs: other *_out folders may hold results of earlier runs
                for file_path in dict.fromkeys(self._run_json_files):
                    if not os.path.isfile(file_path):
                        continue
                    with open(file_path, 'r') as f:
                        try:
                            combined_data.append(json.load(f))
                        except json.JSONDecodeError:
                            self.logger.warning(f"Skipping invalid JSON file: {file_path}")
            else:
                # Process each batch output folder
                for folder in os.listdir(self.ligand_output_dir):
                    folder_path = os.path.join(self.ligand_output_dir, folder)

                    if folder.endswith('_out') and os.path.isdir(folder_path):
                        self.logger.info(f"Processing folder: {folder}")

                        for file in os.listdir(folder_path):
                            if file.endswith('.json'):
                                file_path = os.path.join(folder_path, file)

                                with open(file_path, 'r') as f:
                                    try:
                                        data = json.load(f)
                                        combined_data.append(data)
                                    except json.JSONDecodeError:
                                        self.logger.warning(f"Skipping invalid JSON file: {file_path}")

            # Save combined JSON
            with open(output_json, 'w') as f:
                json.dump(combined_data, f, indent=4)
            self.logger.info(f"Combined JSON file created at: {output_json}")

            # Convert to CSV
            self.json_to_csv(output_json, output_csv)
        except Exception as e:
            self.logger.error(f"Error processing results: {e}")

    def combine_json_files(self, base_dir: str, output_file: str) -> None:
        """
        Combines all JSON files from folders ending with '_out' into a single JSON file.

        Args:
            base_dir (str): The base directory containing the folders.
            output_file (str): The path to the output JSON file.
        """
        combined_data = []

        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)

            if folder.endswith('_out') and os.path.isdir(folder_path):
                self.logger.info(f"Processing folder: {folder}")

                for file in os.listdir(folder_path):
                    if file.endswith('.json'):
                        file_path = os.path.join(folder_path, file)

                        with open(file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                combined_data.append(data)
                            except json.JSONDecodeError:
                                self.logger.warning(f"Skipping invalid JSON file: {file_path}")

        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=4)

        self.logger.info(f"Combined JSON file created at: {output_file}")

    def json_to_csv(self, json_file: str, csv_file: str) -> None:
        """
        Converts a nested JSON file into a flattened CSV format.

        Args:
            json_file (str): Path to the input JSON file.
            csv_file (str): Path to the output CSV file.
        """
        # Read the JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Prepare the CSV rows
        csv_rows = []
        for entry in tqdm(data):
            for complex_name, results in entry.items():
                for result in results:
                    # Add the complex name to each row
                    row = {
                        "complex": complex_name,
                        **result  # Merge the rest of the result fields
                    }
                    csv_rows.append(row)

        # Extract headers from the keys of the first row
        if not csv_rows:
            # still replace the file: a combined_results.csv left by an earlier run would pass for this run's
            self.logger.warning("No data to convert to CSV")
            with open(csv_file, 'w', newline='') as f:
                csv.writer(f).writerow(COMBINED_CSV_COLUMNS)
            return

        headers = csv_rows[0].keys()

        # Write to CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()  # Write the header row
            writer.writerows(csv_rows)  # Write all rows

        self.logger.info(f"CSV file created at: {csv_file}")
        self.split_complex_column(csv_file, csv_file)

    def split_complex_column(self, input_csv: str, output_csv: str) -> None:
        """Split the complex column into protein and molecule_name columns"""
        try:
            df = pd.read_csv(input_csv)

            # Handle cases where complex might be numeric or NaN
            df['complex'] = df['complex'].fillna('').astype(str)

            def split_complex(x):
                try:
                    # Remove '_cleaned_out' and '-results' if present
                    x = x.replace('_cleaned_out', '').replace('-results', '')

                    # Find the last occurrence of 'pocket' in the string
                    import re
                    pocket_match = re.search(r'(.*pocket\d+)-(.*)', x)
                    if pocket_match:
                        protein = pocket_match.group(1)  # Everything up to and including pocketX
                        molecule = pocket_match.group(2)  # Everything after the hyphen

                        # Remove '_filtered-pocket*' part from protein
                        protein = re.sub(r'_filtered-pocket\d+', '', protein)

                        return [protein, molecule]
                    return ['', x]
                except:
                    return ['', x]

            # Apply the split function
            split_results = df['complex'].apply(split_complex)
            df['protein'] = split_results.str[0]
            df['molecule_name'] = split_results.str[1]

            # Reorder columns
            columns = ['protein', 'molecule_name']
            for col in df.columns:
                if col not in ['protein', 'molecule_name', 'complex']:
                    columns.append(col)

            df = df[columns]

            # Save the modified DataFrame to a new CSV file
            df.to_csv(output_csv, index=False)

            self.logger.info(f"Modified CSV file saved at: {output_csv}")
        except Exception as e:
            self.logger.error(f"Error splitting complex column: {e}")

