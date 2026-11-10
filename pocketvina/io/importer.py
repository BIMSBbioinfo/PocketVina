"""Import a legacy PocketVina ``output_dir`` ("files" format) into a results directory in the atlas layout.

Legacy layout read here (``PocketVinaGPU.execute_docking`` with ``protein_paths``)::

  <output_dir>/
    batch_<n>/<name>_cleaned.pdbqt                                 engine input ligands
    batch_<n>_out/<receptor>-pocket<k>-<name>_cleaned_out.pdbqt   engine outputs (the .json next to them is not
                                                                   needed: it holds the same scores)
    proteins/structures/<receptor>.pdbqt                           receptors (one protein each)
    proteins/pockets/<receptor>.csv                                P2Rank predictions
    configs/<receptor>_pocketrank_<k>_config_batch_<n>.txt         engine configs, if kept (box size, docked pockets)

Engine outputs are stored byte for byte. The legacy output folder is only read.
"""
import csv
import hashlib
import logging
import os
import re
import shutil
from typing import Dict, List, Optional, Set, Tuple

from .. import pdbqt
from . import posefile, tables
from .writer import MAX_POCKET, ProteinWriter, ResultsDir, protein_id_from_path, resolve_tmp_dir

logger = logging.getLogger(__name__)

SOURCE = "import_legacy"
LEGACY_SUFFIX = "_cleaned"  # main.py writes the input of ligand <name> as <name>_cleaned.pdbqt
_HYDROGEN_TYPES = frozenset({"H", "HD", "HS"})
_BATCH_DIR_RE = re.compile(r"^batch_\d+$")
_POCKET_TAIL_RE = re.compile(r"^(\d+)-(.+)_out\.pdbqt$")  # what follows "<receptor>-pocket"
_MODEL_RE = re.compile(rb"^MODEL[ \t]+(\d+)")
_AF_RE = re.compile(r"^AF-([A-Z0-9]{6,10})-F\d+")  # AlphaFold DB model names carry the UniProt accession
_NO_OUTPUT = "no engine output in the legacy output_dir"


def _natural_key(name: str):
    """molecule2 before molecule10; ties such as mol1 / mol01 are broken by the name itself."""
    # odd items are the \d+ runs; str.isdigit() would also accept "²", which int() rejects
    return [int(t) if i % 2 else t for i, t in enumerate(re.split(r"(\d+)", name))], name


def _base(pdbqt_name: str) -> str:
    """The engine's ``<receptor_base>`` of a ``*.pdbqt`` file name (boost ``basename``: up to the last dot)."""
    return pdbqt_name[:-len(".pdbqt")]


def _ligand_name(stem: str) -> str:
    """Legacy molecule name of an engine ligand stem (``<name>_cleaned`` -> ``<name>``), as ``combined_results``."""
    if stem.endswith(LEGACY_SUFFIX) and len(stem) > len(LEGACY_SUFFIX):
        return stem[:-len(LEGACY_SUFFIX)]
    return stem


def _read(path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _listdir(path) -> List[str]:
    try:
        return sorted(os.listdir(path), key=_natural_key)
    except FileNotFoundError:
        return []


def _uniprot(protein_id: str) -> Optional[str]:
    m = _AF_RE.match(protein_id)
    return m.group(1) if m else None


# --- scanning the legacy folder -------------------------------------------------------------------------------------

def _find_receptors(src: str) -> Dict[str, str]:
    """``{receptor_base: path}`` of ``proteins/structures/*.pdbqt``."""
    folder = os.path.join(src, "proteins", "structures")
    return {_base(n): os.path.join(folder, n) for n in _listdir(folder)
            if n.endswith(".pdbqt") and len(n) > len(".pdbqt") and os.path.isfile(os.path.join(folder, n))}


def _find_inputs(src: str) -> Dict[str, str]:
    """``{ligand name: path}`` of the engine input ligands in ``batch_<n>/`` (and unbatched ``*_cleaned.pdbqt``)."""
    found = {}
    folders = [os.path.join(src, n) for n in _listdir(src) if _BATCH_DIR_RE.match(n)]
    for folder in [src] + folders:
        if not os.path.isdir(folder):
            continue
        for n in _listdir(folder):
            path = os.path.join(folder, n)
            if not n.endswith(".pdbqt") or not os.path.isfile(path):
                continue
            stem = n[:n.find(".pdbqt")]
            if folder == src and not stem.endswith(LEGACY_SUFFIX):
                continue  # only ligands that were never moved into a batch folder
            name = _ligand_name(stem)
            if name in found:
                if _read(found[name]) != _read(path):
                    logger.warning("ligand %s has two different input files, keeping %s (not %s)", name, found[name],
                                   path)
                continue
            found[name] = path
    return found


def _find_outputs(src: str, bases: Set[str]) -> Tuple[Dict[Tuple[str, int, str], str], List[str]]:
    """Engine outputs in the ``*_out`` folders: ``{(receptor_base, pocket, ligand name): path}`` and the names of
    output files that belong to no known receptor."""
    found, unknown = {}, []
    for d in _listdir(src):
        folder = os.path.join(src, d)
        if not d.endswith("_out") or not os.path.isdir(folder):
            continue
        for n in _listdir(folder):
            path = os.path.join(folder, n)
            if not n.endswith("_out.pdbqt") or not os.path.isfile(path):
                continue
            key = None
            # "<receptor_base>-pocket<k>-<ligand_stem>_out.pdbqt"; the longest known receptor base wins
            for m in re.finditer(r"-pocket", n):
                tail = _POCKET_TAIL_RE.match(n[m.end():])
                if n[:m.start()] in bases and tail:
                    key = (n[:m.start()], int(tail.group(1)), _ligand_name(tail.group(2)))
            if key is None:
                unknown.append(n)
                continue
            if key in found:
                if _read(found[key]) != _read(path):
                    raise ValueError(f"two different engine outputs for receptor {key[0]}, pocket {key[1]}, ligand "
                                     f"{key[2]}: {found[key]} and {path}")
                continue
            found[key] = path
    return found, unknown


def _read_configs(src: str) -> Dict[Tuple[str, int], Tuple[float, float, float]]:
    """``{(receptor_base, pocket): (size_x, size_y, size_z)}`` from the engine configs in ``configs/``."""
    folder = os.path.join(src, "configs")
    out = {}
    for n in _listdir(folder):
        path = os.path.join(folder, n)
        if not n.endswith(".txt") or not os.path.isfile(path):
            continue
        values = {}
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                key, sep, value = line.partition("=")
                if sep:
                    values[key.strip()] = value.strip()
        try:
            receptor = os.path.basename(values["receptor"].replace("\\", "/"))
            if not receptor.endswith(".pdbqt"):
                raise ValueError(f"receptor {receptor!r} is not a .pdbqt file")
            key = (_base(receptor), int(values["pocket_rank"]))
            out[key] = tuple(float(values[f"size_{c}"]) for c in "xyz")
        except (KeyError, ValueError) as e:
            logger.debug("%s: not a PocketVina engine config (%s)", path, e)
    return out


def _box_sizer(configs, base: str):
    """Box size of a pocket rank: its own config, else the one size used for this receptor (or the whole run; the
    legacy run had one box size), else NaN."""
    def single(sizes):
        sizes = set(sizes)
        return sizes.pop() if len(sizes) == 1 else None

    default = single(v for (b, _), v in configs.items() if b == base) or single(configs.values())
    default = default or (float("nan"),) * 3
    return lambda rank: configs.get((base, rank)) or default


def _number(row: dict, key: str, required: bool = True) -> Optional[float]:
    value = row.get(key, "")
    if value == "":
        if required:
            raise ValueError(f"no {key!r} value")
        return None
    return float(value)


def _read_pockets(path: str, protein_id: str, size_of) -> List[dict]:
    """``POCKETS`` rows from a P2Rank ``*_predictions.csv`` (space-padded columns)."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for n, raw in enumerate(csv.DictReader(f, skipinitialspace=True), 2):
            row = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if isinstance(k, str)}
            if not any(row.values()):
                continue
            try:
                rank = int(_number(row, "rank"))
                score = _number(row, "score", required=False)
                size = size_of(rank)
                rows.append(dict(protein_id=protein_id, pocket=rank, name=row.get("name") or f"pocket{rank}",
                                 score=float("nan") if score is None else score,
                                 probability=_number(row, "probability", required=False),
                                 center_x=_number(row, "center_x"), center_y=_number(row, "center_y"),
                                 center_z=_number(row, "center_z"), size_x=size[0], size_y=size[1], size_z=size[2],
                                 residue_ids=row.get("residue_ids") or None))
            except ValueError as e:
                raise ValueError(f"{path}, line {n}: {e}") from e
    return rows


# --- ligands --------------------------------------------------------------------------------------------------------

def _pdbqt_heavy_atoms(text: str) -> int:
    return sum(1 for line in text.splitlines()
               if line.startswith(("ATOM", "HETATM")) and line.split() and line.split()[-1] not in _HYDROGEN_TYPES)


def _describe(smiles: str, pdbqt_text: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """(docked SMILES, InChIKey, heavy atoms) as the legacy preparation docks it: the whole SMILES, or the fragment
    OpenBabel kept (the one with the input PDBQT's heavy atoms). All None for an unreadable SMILES."""
    from rdkit import Chem, rdBase

    block = rdBase.BlockLogs()  # silences RDKit's parser and InChI warnings until deleted
    try:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None or mol.GetNumAtoms() == 0:
            return None, None, None
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1 and pdbqt_text:
            n = _pdbqt_heavy_atoms(pdbqt_text)
            mol = next((f for f in frags if f.GetNumHeavyAtoms() == n), frags[0])
        try:
            key = Chem.MolToInchiKey(mol) or None
        except Exception:
            key = None
        return Chem.MolToSmiles(mol), key, mol.GetNumHeavyAtoms()
    finally:
        del block


def _ligand_rows(smiles_rows: Optional[List[Tuple[str, str]]], inputs: Dict[str, str], docked: Set[str],
                 smiles_file: Optional[str]) -> List[dict]:
    """``LIGANDS`` rows: the ``smiles_file`` order, then ligands of the run missing from it (by name)."""
    seen = set(inputs) | docked
    if smiles_rows is None:
        names, smiles = sorted(seen, key=_natural_key), {}
    else:
        names = [name for name, _ in smiles_rows]
        smiles = {}
        for name, smi in smiles_rows:
            if name in smiles:
                raise ValueError(f"{smiles_file}: ligand {name!r} is listed twice")
            smiles[name] = smi
        extra = sorted(seen - set(smiles), key=_natural_key)
        if extra:
            logger.warning("%d ligand(s) of the legacy run are not in %s and are added without SMILES: %s",
                           len(extra), smiles_file, ", ".join(extra[:10]) + (" ..." if len(extra) > 10 else ""))
        names += extra
    rows = []
    for idx, name in enumerate(names):
        text = None
        if name in inputs:
            text = _read(inputs[name]).decode("utf-8", errors="replace")
        smi = smiles.get(name)
        smiles_docked, inchikey, n_heavy = _describe(smi, text) if smi else (None, None, None)
        if n_heavy is None and text:
            n_heavy = _pdbqt_heavy_atoms(text)
        status, error = "ok", None
        if name not in seen:
            status = "error" if smiles_docked else "invalid_smiles"
            error = "no engine input or output in the legacy output_dir (ligand preparation failed)"
        rows.append(dict(ligand_idx=idx, ligand_id=name, smiles_input=smi or "", smiles_docked=smiles_docked,
                         inchikey=inchikey, n_heavy=n_heavy, prep_method="legacy", seed=None, status=status,
                         error=error, pdbqt=text))
    return rows


def _sdf_maker(protein_id: str, ligand_rows: List[dict]):
    """``sdf_for`` for ``ProteinWriter.finalize``: ``convert.pose_to_sdf`` with the docked SMILES, if there is one."""
    ids = [r["ligand_id"] for r in ligand_rows]
    smiles = [r["smiles_docked"] for r in ligand_rows]
    if any(smiles):
        from .. import convert
    else:
        convert = None

    def sdf_for(i, k, model):
        if not smiles[i]:
            return None, "convert_failed"
        props = {"_Name": ids[i], "protein_id": protein_id, "ligand_idx": i, "pocket": k}  # as pipeline.run_atlas
        m = _MODEL_RE.match(model)
        if m:
            props["mode"] = int(m.group(1))
        modes = pdbqt.parse_modes(model)
        if modes:
            props["affinity"] = modes[0].affinity
        return convert.pose_to_sdf(model, smiles[i], props)

    return sdf_for


# --- import ---------------------------------------------------------------------------------------------------------

def import_legacy(output_dir: str, results_dir: str, smiles_file: Optional[str] = None, *,
                  tmp_dir: Optional[str] = None, gzip_level: int = 6, force: bool = False) -> str:
    """Convert a legacy PocketVina ``output_dir`` into the atlas layout in ``results_dir``; returns its path.

    One protein per ``proteins/structures/<receptor>.pdbqt``; engine outputs of other receptors are skipped with a
    warning. Ligands are numbered in the order of ``smiles_file`` (``molecule_name,smiles``; ligands of the run that
    it lacks follow), else by name. The docked pockets are those with an engine output or config; a ligand without
    output in one of them gets a "no_output" run, a ligand of ``smiles_file`` that the legacy run never prepared a
    "skipped" one. Section A holds the highest-affinity pose of each ligand as SDF
    (``pocketvina.convert``) when its SMILES is known; without SMILES it stays empty and ``sdf_status`` is
    "convert_failed". Box sizes come from ``configs/`` when kept, else they are NaN. Docked pockets ranked above
    127 do not fit the int8 pocket columns: they are left out with a warning and noted in the proteins table.

    Importing into a results directory that holds this import already resumes it (proteins done are skipped).
    ``force=True`` replaces another run definition there and imports every protein again.
    """
    src = os.path.abspath(os.path.expanduser(os.fspath(output_dir)))
    if not os.path.isdir(src):
        raise FileNotFoundError(f"legacy output_dir not found: {output_dir}")
    receptors = _find_receptors(src)
    if not receptors:
        raise ValueError(f"{src} is not a legacy PocketVina output_dir: no proteins/structures/*.pdbqt")
    pids = {}
    for base, path in receptors.items():
        pid = protein_id_from_path(path)
        if pid in pids.values():
            other = next(b for b, p in pids.items() if p == pid)
            raise ValueError(f"receptors {other}.pdbqt and {base}.pdbqt both give protein_id {pid!r}")
        pids[base] = pid

    inputs = _find_inputs(src)
    outputs, unknown = _find_outputs(src, set(receptors))
    if unknown:
        logger.warning("%d engine output(s) of receptors not in %s are skipped: %s", len(unknown),
                       os.path.join(src, "proteins", "structures"),
                       ", ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else ""))
    configs = _read_configs(src)
    smiles_rows = None
    if smiles_file is not None:
        from ..ligands import read_smiles_file
        smiles_file = os.path.abspath(os.path.expanduser(os.fspath(smiles_file)))
        smiles_rows = read_smiles_file(smiles_file)
    ligand_rows = _ligand_rows(smiles_rows, inputs, {name for _, _, name in outputs}, smiles_file)

    results = ResultsDir(results_dir)
    results.init_run(dict(source=SOURCE, ligand_prep="legacy"), ligand_rows, force=force, tmp_dir=tmp_dir)
    base_tmp = resolve_tmp_dir(tmp_dir)
    try:
        for base in sorted(receptors, key=_natural_key):
            pid = pids[base]
            if results.is_done(pid) and not force:
                logger.info("%s: already imported, skipped", pid)
                continue
            mine = {(k, name): path for (b, k, name), path in outputs.items() if b == base}
            _import_protein(results, pid, src, receptors[base], mine, configs, ligand_rows, base_tmp, gzip_level)
    finally:
        shutil.rmtree(base_tmp, ignore_errors=True)
    return results.path


def _import_protein(results: ResultsDir, pid: str, src: str, receptor_path: str, outputs, configs, ligand_rows,
                    base_tmp: str, gzip_level: int) -> None:
    base = _base(os.path.basename(receptor_path))
    receptor = _read(receptor_path)
    ranks = sorted({k for k, _ in outputs} | {k for b, k in configs if b == base})
    notes = []
    high = [k for k in ranks if k > MAX_POCKET]
    if high:
        # the legacy run docked every P2Rank pocket; as pipeline.run_atlas, keep the rest of the protein
        shown = ", ".join(map(str, high[:10])) + (" ..." if len(high) > 10 else "")
        notes.append(f"{len(high)} docked pocket(s) ranked above {MAX_POCKET} not imported (int8 pocket columns): "
                     f"{shown}")
        logger.warning("%s: %s", pid, notes[-1])
        ranks = [k for k in ranks if k <= MAX_POCKET]
        outputs = {key: path for key, path in outputs.items() if key[0] <= MAX_POCKET}
    if not outputs:
        notes.append(_NO_OUTPUT)
    pockets_csv = os.path.join(src, "proteins", "pockets", f"{base}.csv")
    pocket_rows = []
    if os.path.isfile(pockets_csv):
        pocket_rows = _read_pockets(pockets_csv, pid, _box_sizer(configs, base))
    else:
        logger.warning("%s: no P2Rank predictions at %s, the pockets table stays empty", pid, pockets_csv)
    uniprot = _uniprot(pid)
    meta = dict(source=SOURCE, receptor=os.path.basename(receptor_path))
    with ProteinWriter(pid, len(ligand_rows), ranks, max(ranks, default=0), base_tmp, meta, uniprot=uniprot,
                       gzip_level=gzip_level) as w:
        for row in ligand_rows:
            for k in ranks:
                path = outputs.get((k, row["ligand_id"]))
                if row["status"] != "ok":  # as pipeline.run_atlas: a "skipped" run, so pairs lists every ligand
                    w.add_run(row["ligand_idx"], row["ligand_id"], k, None, "skipped",
                              error=f"ligand not prepared: {row['status']}: {row['error']}")
                elif path is None:
                    w.add_run(row["ligand_idx"], row["ligand_id"], k, None, "no_output", error=_NO_OUTPUT)
                else:
                    w.add_run(row["ligand_idx"], row["ligand_id"], k, _read(path), "ok")
        staged = w.finalize(_sdf_maker(pid, ligand_rows))

        structure_file = f"structures/{pid}.pdbqt.gz"
        structure = os.path.join(w.dir, *structure_file.split("/"))
        os.makedirs(os.path.dirname(structure), exist_ok=True)
        with open(structure, "wb") as f:
            f.write(posefile.gz_member(receptor, gzip_level))
        staged[structure_file] = structure
        protein = dict(protein_id=pid, uniprot=uniprot, source_path=receptor_path, structure_file=structure_file,
                       receptor_sha256=hashlib.sha256(receptor).digest(), n_pockets=len(pocket_rows),
                       n_pockets_docked=len(ranks), status="ok" if outputs else "no_output",
                       error="; ".join(notes) or None)
        for name, rows, schema in (("proteins", [protein], tables.PROTEINS), ("pockets", pocket_rows, tables.POCKETS)):
            path = os.path.join(w.dir, "tables", name, f"{pid}.parquet")
            tables.write_table(rows, schema, path)
            staged[f"tables/{name}/{pid}.parquet"] = path
        summary = dict(w.summary, source=SOURCE, legacy_output_dir=src, receptor=receptor_path,
                       n_pockets=len(pocket_rows), n_outputs=len(outputs))
        results.commit(pid, staged, summary)
    logger.info("%s: imported %d engine outputs in %d pocket(s)", pid, len(outputs), len(ranks))
