"""One protein x ligand pair of a results directory: pick it with ``select``, then show, check or save it.

    pair = pocketvina.select(protein="1fbl", ligand="molecule1")                      # its highest-affinity pose
    pair = pocketvina.select(protein="1fbl", ligand="molecule1", pocket=2, mode=3)    # one pose
    pair = pocketvina.select(protein="1fbl", ligand="molecule1", pocket="all")        # that pose in each pocket
    pair = pocketvina.select(protein="1fbl", ligand="molecule1", pocket=2, mode="all")  # every mode of pocket 2
    pair            # in a notebook: a short summary
    pair.show()     # 3D, like the PocketVina Atlas website (pocketvina.workspace)
    pair.bust()     # PoseBusters on the selected poses, in memory; returns a table
    pair.export()   # one SDF (or PDBQT) file of the selected poses, e.g. PV_1fbl_molecule1_top.sdf
    pair.export(protein=True)   # those poses and the protein as PDB (PV_1fbl.pdb), next to each other

``select`` is a method of ``Results`` and ``PocketVinaGPU``; ``Atlas.select`` gives the same ``Selection`` for a pair
of the PocketVina Atlas. The selection is fixed when it is made (the poses are read then); ``show``, ``bust`` and
``export`` take only output options. File names follow the PocketVina Atlas ID scheme with "PV_" in front
(``pose_name``).
"""
import html
import logging
import numbers
import os
import re
import tempfile
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import pdbqt
from .io.results import Results, short_floats

logger = logging.getLogger(__name__)

EXPORT_FORMATS = ("sdf", "pdbqt")
MAX_NAME_PART = 100  # characters of a protein or ligand id in a file name (as Results.export's MAX_STEM)
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_SHOWN_IDS = 5  # valid ids listed in an error message
_SCORE_HEADER = "lowest Vina score (kcal/mol)"  # header of the per-pocket table of a selection's summary

PocketArg = Union[None, int, str]
ModeArg = Union[int, str]


def name_part(text) -> str:
    """A protein or ligand id as the part of a file name: characters other than letters, digits and . _ - become _."""
    part = _UNSAFE.sub("_", str(text))[:MAX_NAME_PART].strip("._")
    return part or "x"


def pose_name(protein: str, ligand: str, *parts) -> str:
    """``PV_<protein>_<ligand>[_<part>...]``: the PocketVina Atlas ID of a pair (``lib/keys.ts``) with "PV_" in front,
    ids made file-name safe; e.g. ``pose_name("1fbl", "molecule1", "pk2", "m03")`` is ``PV_1fbl_molecule1_pk2_m03``."""
    return "_".join(["PV", name_part(protein), name_part(ligand), *map(str, parts)])


def _target(path, default: str) -> str:
    """The file to write: ``path``, or ``default`` in the folder ``path`` (None: the working folder). A path is a
    folder if it is one, ends in "/" or has no extension, as the CLI's --out ("my.poses" has one; "results_v2" and
    ".poses" none, as os.path.splitext); a missing folder is made when the file is written. A file without an
    extension, e.g. "poses" written by 0.3.0, is an error."""
    if path is None:
        return os.path.join(os.getcwd(), default)
    given = os.fspath(path)
    path = os.path.abspath(os.path.expanduser(given))
    if given.endswith(os.sep) or os.path.isdir(path):
        return os.path.join(path, default)
    if not os.path.splitext(path)[1]:
        if os.path.isfile(path):
            raise ValueError(f"{path} is a file, but a path without an extension is a folder: use e.g. "
                             f"{os.path.basename(path)}{os.path.splitext(default)[1]} for a file")
        return os.path.join(path, default)
    return path


def _few(values: List[str]) -> str:
    shown = ", ".join(repr(v) for v in values[:_SHOWN_IDS])
    return shown + (f", ... ({len(values)} in all)" if len(values) > _SHOWN_IDS else "")


def _check_protein(res: Results, protein) -> str:
    known = res.proteins()
    if not isinstance(protein, str):
        raise TypeError(f"protein must be a protein id (a string) such as {known[0] if known else '1fbl'!r}, not "
                        f"{type(protein).__name__}")
    if protein not in known:
        hint = f"; proteins in these results: {_few(known)}" if known else "; these results have no protein yet"
        raise KeyError(f"no protein {protein!r} in {res.path}{hint}")
    return protein


def _check_ligand(res: Results, ligand) -> Tuple[int, str]:
    try:
        i = res.ligand_idx(ligand)
    except KeyError:
        ids = list(res.ligands(columns=["ligand_id"])["ligand_id"])
        raise KeyError(f"no ligand {ligand!r} in {res.path}; ligands (as in best_scores()): {_few(ids)}") from None
    return i, res.ligand_id(i)


def _check_pocket(pocket) -> PocketArg:
    if pocket is None or pocket == "all":
        return pocket
    if isinstance(pocket, (bool, np.bool_)) or not isinstance(pocket, numbers.Integral) or pocket < 1:
        raise ValueError(f"pocket must be a pocket number (1, 2, ...), None (the pocket of the highest-affinity "
                         f"pose) or 'all', got {pocket!r}")
    return int(pocket)


def _check_mode(mode) -> ModeArg:
    if mode == "all" and isinstance(mode, str):
        return mode
    if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, numbers.Integral) or mode < 1:
        raise ValueError(f"mode must be a mode number (1 = the highest-affinity mode of the pocket, 2, 3, ...) or "
                         f"'all', got {mode!r}")
    return int(mode)


class Pose(NamedTuple):
    """A selected pose: pocket (P2Rank rank), mode (as the engine numbers them, best first), Vina score
    (kcal/mol) and RMSD bounds to the pocket's mode 1 (A), and its model (``MODEL ... ENDMDL``) as the engine wrote
    it."""

    pocket: int
    mode: int
    vina: float
    rmsd_lb: float
    rmsd_ub: float
    model: bytes


def _short(x: float) -> float:
    return float(short_floats([x])[0])  # the score as the engine printed it: -6.3, not -6.300000190734863


def _best_record(records: List[pdbqt.ModeRecord]) -> pdbqt.ModeRecord:
    """The best mode of a pocket (lowest score, then the lower mode number), as the pairs table picks it."""
    return min(records, key=lambda r: (r.affinity, r.mode))


def _pick_poses(block_of: Callable[[int], bytes], docked: List[int], pocket: PocketArg, mode: ModeArg,
                best_pocket: int, best_mode: int, protein: str, ligand: str) -> List[Pose]:
    """The poses ``select`` picks from one pair's engine output, for a results directory and for the PocketVina
    Atlas alike. ``block_of(k)``: the engine output of pocket k (PDBQT bytes, every mode); ``docked``: the pockets with
    poses; ``best_pocket`` / ``best_mode``: where the pair's best pose is; ``pocket`` and ``mode`` as ``select`` takes
    them (checked). Raises ``ValueError`` for a pocket without poses and for a mode out of range."""
    if pocket is None:
        pockets = [best_pocket]
    elif pocket == "all":
        pockets = docked
    elif pocket not in docked:
        raise ValueError(f"pocket {pocket} has no poses of ligand {ligand!r} in protein {protein!r}: it was not docked "
                         f"(or the docking gave no pose there); pockets with poses: "
                         f"{', '.join(map(str, docked))}")
    else:
        pockets = [pocket]

    poses: List[Pose] = []
    n_modes: Dict[int, int] = {}
    for k in pockets:
        block = block_of(k)
        records = pdbqt.parse_modes(block)
        n_modes[k] = len(records)
        if mode == "all":
            chosen = records
        elif mode == 1:
            best = [r for r in records if (k, r.mode) == (best_pocket, best_mode)]
            chosen = best or [_best_record(records)]
        elif mode <= len(records):
            chosen = [records[mode - 1]]
        elif pocket == "all":
            continue  # this pocket has fewer modes
        else:
            raise ValueError(f"mode {mode} is out of range: pocket {k} has modes 1 to {len(records)} for ligand "
                             f"{ligand!r} in protein {protein!r}")
        for r in chosen:
            poses.append(Pose(k, r.mode, _short(r.affinity), _short(r.rmsd_lb), _short(r.rmsd_ub),
                              bytes(block[r.raw_offset:r.raw_offset + r.raw_length])))
    if not poses:
        raise ValueError(f"mode {mode} is out of range: no pocket has that many modes for ligand {ligand!r} in protein "
                         f"{protein!r} (modes per pocket: {', '.join(f'{k}: {n}' for k, n in n_modes.items())})")
    return poses


def select(results: Results, protein: str, ligand, pocket: PocketArg = None, mode: ModeArg = 1) -> "Selection":
    """The poses of one protein x ligand pair (see ``Selection``); ``results``: an open ``Results``.

    protein: a protein id as in best_scores(), e.g. "1fbl".
    ligand: a ligand id (its name in the SMILES file) as in best_scores(), e.g. "molecule1".
    pocket: a pocket number (P2Rank rank), e.g. 2; None (default): the pocket of the pair's highest-affinity pose;
        "all": every pocket the ligand was docked into.
    mode: 1 (default): the highest-affinity mode of the pocket; 2, 3, ...: that mode (the engine numbers them
        highest-affinity first); "all": every mode of the pocket(s). With pocket="all" and a mode number, the pockets
        without that mode are left out.

    Unknown proteins and ligands raise ``KeyError`` (listing some valid ids); a pocket that was not docked, a mode
    out of range, or a ligand without any pose raise ``ValueError``.
    """
    pocket, mode = _check_pocket(pocket), _check_mode(mode)
    pid = _check_protein(results, protein)
    i, lid = _check_ligand(results, ligand)
    lig = results.ligands(columns=["ligand_idx", "smiles_input", "smiles_docked", "status", "error"])
    row = lig[lig["ligand_idx"] == i].iloc[0]
    pair = results.pairs(pid, columns=["ligand_idx", "vina", "best_pocket", "best_mode"])
    pair = pair[pair["ligand_idx"] == i]
    if pair.empty or int(pair["best_pocket"].iloc[0]) < 1:
        error = f", {row['error']}" if isinstance(row["error"], str) and row["error"] else ""
        why = ("the docking gave no pose in any pocket" if row["status"] == "ok" else
               f"it could not be prepared for docking: {row['status']}{error}")
        raise ValueError(f"ligand {lid!r} has no docked pose in protein {pid!r} ({why})")
    best_pocket, best_mode = int(pair["best_pocket"].iloc[0]), int(pair["best_mode"].iloc[0])
    runs = results.pocket_runs(pid, columns=["ligand_idx", "pocket", "status", "n_modes", "length"])
    runs = runs[(runs["ligand_idx"] == i) & (runs["status"] == "ok") & (runs["n_modes"] > 0) & (runs["length"] > 0)]
    docked = sorted(int(k) for k in runs["pocket"])
    poses = _pick_poses(lambda k: results.pose_block(pid, i, k).encode("utf-8", "surrogateescape"), docked, pocket,
                        mode, best_pocket, best_mode, pid, lid)
    smiles_docked = row["smiles_docked"] if isinstance(row["smiles_docked"], str) and row["smiles_docked"] else None
    has_best = any((p.pocket, p.mode) == (best_pocket, best_mode) for p in poses)
    return Selection(
        results_dir=results.path, protein=pid, ligand=lid, ligand_idx=i, smiles=row["smiles_input"],
        smiles_docked=smiles_docked or row["smiles_input"], pocket=pocket, mode=mode, best_pocket=best_pocket,
        best_mode=best_mode, vina=_short(pair["vina"].iloc[0]), docked_pockets=docked, poses=poses,
        top_sdf=results.top_sdf(pid, i) if has_best else None)


class _ResultsSource:
    """Where a ``Selection`` of a results directory gets its protein and its page (the source hook of ``Selection``;
    ``Atlas.select`` gives another one): the protein is the stored input structure, the page is ``Results.show``'s."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir

    def protein_pdb(self, protein: str) -> str:
        """PDB text of the protein the poses were docked into."""
        from .bust import _protein_pdb

        with Results(self.results_dir) as res:
            return _protein_pdb(res, protein)

    def show(self, selection: "Selection", *, path=None, molstar: str = "inline", theme: str = "auto",
             height: int = 760, max_ligands: Optional[int] = 100):
        """The page of the selection's protein with the selection selected (``Selection.show``)."""
        from . import workspace

        with Results(self.results_dir) as res:
            return workspace.show(res, [selection.protein], selection=selection, path=path, molstar=molstar,
                                  theme=theme, height=height, max_ligands=max_ligands)


class _Default:
    """An argument not given: its value depends on where the selection comes from (shown as "default")."""

    def __repr__(self):
        return "default"


_DEFAULT = _Default()


class Selection:
    """Poses of one protein x ligand pair, picked with ``select`` (``Results.select``, ``PocketVinaGPU.select``,
    ``Atlas.select``).

    Attributes: ``protein``, ``ligand`` (ids; for the Atlas the UniProt accession and the PVL ID), ``smiles``,
    ``pocket`` and ``mode`` (as selected), ``vina`` (the pair's lowest Vina score, kcal/mol), ``best_pocket`` /
    ``best_mode`` (where that pose is), ``poses`` (a table: pocket, mode, vina, rmsd_lb, rmsd_ub), ``name`` (the file
    name stem, e.g. ``PV_1fbl_molecule1_top``).
    """

    def __init__(self, *, results_dir: Optional[str] = None, protein: str, ligand: str, ligand_idx: int, smiles: str,
                 smiles_docked: str, pocket: PocketArg, mode: ModeArg, best_pocket: int, best_mode: int, vina: float,
                 docked_pockets: List[int], poses: List[Pose], top_sdf: Optional[str], source=None):
        # source: where show(), bust() and export_protein() get the protein (protein_pdb) and the page (show); by
        # default the results directory (results_dir; None for a pair of the Atlas)
        self.results_dir = results_dir
        self._source = source if source is not None else _ResultsSource(results_dir)
        self.protein, self.ligand, self.ligand_idx = protein, ligand, ligand_idx
        self.smiles, self.smiles_docked = smiles, smiles_docked
        self.pocket, self.mode = pocket, mode
        self.best_pocket, self.best_mode, self.vina = best_pocket, best_mode, vina
        self.docked_pockets = list(docked_pockets)
        self._poses = list(poses)
        self._top_sdf = top_sdf

    @property
    def poses(self) -> pd.DataFrame:
        """The selected poses: pocket, mode, vina (kcal/mol), rmsd_lb, rmsd_ub (A, to mode 1 of the pocket)."""
        return pd.DataFrame([p[:5] for p in self._poses], columns=list(Pose._fields[:5]))

    @property
    def name(self) -> str:
        """File name stem of the selection (Atlas IDs with "PV_"): ``PV_<protein>_<ligand>_top`` (the highest-affinity
        pose), ``..._pk2_m03`` (one pose), ``..._pk2_modes`` (all modes of a pocket), ``..._pocketall_top`` (the
        highest-affinity pose in each pocket), ``..._pocketall_modes``, ``..._pocketall_m03``."""
        if self.pocket is None and self.mode == 1:
            return pose_name(self.protein, self.ligand, "top")
        where = "pocketall" if self.pocket == "all" else f"pk{self._poses[0].pocket}"
        if self.mode == "all":
            what = "modes"
        elif self.pocket == "all" and self.mode == 1:
            what = "top"
        else:
            what = f"m{self._poses[0].mode:02d}" if self.pocket != "all" else f"m{self.mode:02d}"
        return pose_name(self.protein, self.ligand, where, what)

    # --- summary ---------------------------------------------------------------------------------------------------

    def _what(self) -> str:
        if self.pocket == "all":
            what = ("the highest-affinity pose in each pocket" if self.mode == 1 else
                    "every mode in each pocket" if self.mode == "all" else f"mode {self.mode} in each pocket")
        else:
            where = f"pocket {self.best_pocket if self.pocket is None else self.pocket}"
            what = (f"{where}, the highest-affinity pose" if self.mode == 1 else
                    f"{where}, every mode" if self.mode == "all" else f"{where}, mode {self.mode}")
        return f"{what}: {len(self._poses)} pose{'s' if len(self._poses) != 1 else ''}"

    def _per_pocket(self) -> List[Tuple[int, float, str]]:
        """(pocket, lowest Vina score of the selected poses there, selected modes) per pocket."""
        out = []
        for k in dict.fromkeys(p.pocket for p in self._poses):
            ps = [p for p in self._poses if p.pocket == k]
            modes = [p.mode for p in ps]
            text = (f"{modes[0]}-{modes[-1]}" if len(modes) > 2 and modes == list(range(modes[0], modes[-1] + 1))
                    else ", ".join(map(str, modes)))
            out.append((k, min(p.vina for p in ps), text))
        return out

    def __repr__(self):
        lines = [f"Selection {pose_name(self.protein, self.ligand)}: protein {self.protein}, ligand {self.ligand}, "
                 f"{self._what()}", f"  SMILES: {self.smiles}", f"  pocket  {_SCORE_HEADER}  modes"]
        for k, vina, modes in self._per_pocket():
            star = " *" if k == self.best_pocket else "  "
            lines.append(f"  {k:>4}{star}{vina:>{len(_SCORE_HEADER) + 2}.1f}  {modes}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        e = html.escape
        rows = "".join(
            f"<tr><td style='text-align:right'>{k}{' &#9733;' if k == self.best_pocket else ''}</td>"
            f"<td style='text-align:right'>{vina:.1f}</td><td>{e(modes)}</td></tr>"
            for k, vina, modes in self._per_pocket())
        return (f"<div><b>{e(pose_name(self.protein, self.ligand))}</b>: protein <b>{e(self.protein)}</b>, ligand "
                f"<b>{e(self.ligand)}</b>, {e(self._what())}<br><span style='font-family:monospace'>"
                f"{e(self.smiles)}</span><table><thead><tr><th>pocket</th><th>{_SCORE_HEADER}</th>"
                f"<th>modes</th></tr></thead><tbody>{rows}</tbody></table>"
                f"<small>&#9733; the pocket of the pair's highest-affinity pose ({self.vina:.1f} kcal/mol). "
                f"pair.show(), pair.bust(), pair.export()</small></div>")

    # --- 3D --------------------------------------------------------------------------------------------------------

    def show(self, path=None, *, molstar: str = "inline", theme: str = "auto", height: int = 760,
             max_ligands=_DEFAULT):
        """The PocketVina Atlas website's protein page for this pair's protein, with this pair selected: the 3D view
        (pocket chips, modes, legend, contacts, Save image) beside the tabs Top hits, Selected pair and Pockets.

        In Jupyter, Colab or VSCode it is shown below the cell; elsewhere an HTML file is written (open it in a
        browser) and its path returned. Everything is in the page: it works offline and in a saved notebook.
        path: write the page to this HTML file instead, e.g. "pair.html"; returns the path.
        molstar: "inline" (default): the 3D viewer's code is in the page (about 1 MB); "url": loaded from the
            PocketVina Atlas website when the page opens (needs internet, a much smaller page).
        theme: "auto" (default: as the browser), "light" or "dark".
        height: height of the view in a notebook, in pixels, e.g. 900.
        max_ligands: ligands listed in Top hits, highest-affinity first, e.g. 50 (default 100; 25 for a pair of the
            PocketVina Atlas, whose poses are fetched).

        The 3D view is drawn with Mol*, described in the `"Mol* Viewer: modern web app for 3D visualization and analysis
        of large biomolecular structures" <https://doi.org/10.1093/nar/gkab314>`_ paper.
        """
        options = {} if max_ligands is _DEFAULT else {"max_ligands": max_ligands}
        return self._source.show(self, path=path, molstar=molstar, theme=theme, height=height, **options)

    # --- PoseBusters -----------------------------------------------------------------------------------------------

    def bust(self, workers: Optional[int] = None) -> pd.DataFrame:
        """PoseBusters (config "dock") on the selected poses against the protein; nothing is written to disk and the
        stored results do not change. Returns one row per pose: pocket, mode, vina, pb_valid (True: every check
        passed), pb_status ("valid", "invalid", or why it was not checked: "convert_failed", "timeout", "error"),
        failed_checks, then one column per PoseBusters check (True: passed, False: failed, empty: not applicable).

        workers: processes that run the checks at the same time, e.g. 4; None (default): one per pose, at most the
            CPUs this process may use.

        PoseBusters is described in the `"PoseBusters: AI-based docking methods fail to generate physically valid poses
        or generalise to novel sequences" <https://doi.org/10.1039/D3SC04185A>`_ paper.
        """
        from . import bust as pb

        if workers is not None and (isinstance(workers, bool) or not isinstance(workers, numbers.Integral)
                                    or workers < 1):
            raise ValueError(f"workers must be a number of processes (1, 2, ...) or None, got {workers!r}")
        _, cfg = pb._config("dock")
        names = pb._check_names(cfg)
        pb.require_posebusters()
        protein_text = self._source.protein_pdb(self.protein)
        payloads = []
        for p in self._poses:
            if self._top_sdf and (p.pocket, p.mode) == (self.best_pocket, self.best_mode):
                # the stored SDF of the best pose, as bust_results(select="best") checks it
                payloads.append(("sdf", self._top_sdf.encode("utf-8", "surrogateescape")))
            else:
                props = {"_Name": self.ligand, "protein_id": self.protein, "ligand_idx": self.ligand_idx,
                         "pocket": p.pocket, "mode": p.mode, "affinity": p.vina}
                payloads.append(("pdbqt", p.model, self.smiles_docked, props))
        n_workers = int(workers) if workers is not None else min(pb.default_workers(), len(payloads))
        # temp files only for PoseBusters versions without in-memory loading (bust._load_from_file)
        with tempfile.TemporaryDirectory(prefix="pocketvina_pb_") as tmp:
            pool = pb._Pool(cfg, n_workers, pb.DEFAULT_TIMEOUTS, tmp)
            try:
                results = pool.check(protein_text, [lambda x=x: x for x in payloads], label=self.name)
            finally:
                pool.close()
        rows = []
        for p, result in zip(self._poses, results):
            row = {"pocket": p.pocket, "mode": p.mode, "vina": p.vina, "pb_valid": None, "pb_status": None,
                   "failed_checks": None, **{n: None for n in names}}
            if "values" in result:
                failed = [n for n, v in zip(names, result["values"]) if v is False]
                row.update(pb_valid=not failed, pb_status="invalid" if failed else "valid",
                           failed_checks=", ".join(failed) or None, **dict(zip(names, result["values"])))
            else:
                row["pb_status"] = result.get("status", "error")
                logger.warning("%s pocket %d mode %d: not checked (%s): %s", self.name, p.pocket, p.mode,
                               row["pb_status"], result.get("error"))
            rows.append(row)
        df = pd.DataFrame(rows, columns=["pocket", "mode", "vina", "pb_valid", "pb_status", "failed_checks", *names])
        for col in ["pb_valid", *names]:
            df[col] = df[col].astype("boolean")
        return df

    # --- files -----------------------------------------------------------------------------------------------------

    def _path(self, path, fmt: str, protein: bool) -> str:
        """The poses' file (``_target``); a file named by ``path`` must end in .<fmt>, checked before anything is
        written (with protein=True first: not the protein's file name)."""
        target = _target(path, f"{self.name}.{fmt}")
        if protein and os.path.basename(target) == f"PV_{name_part(self.protein)}.pdb":
            raise ValueError(f"{target} is the protein's file: give the poses another name, e.g. {self.name}.{fmt}")
        ext = os.path.splitext(target)[1].lower().lstrip(".")
        if ext in EXPORT_FORMATS and ext != fmt:
            raise ValueError(f"{target} ends in .{ext} but fmt={fmt!r}: pass fmt={ext!r} for a {ext.upper()} file")
        if ext != fmt:
            other = "pdbqt" if fmt == "sdf" else "sdf"
            raise ValueError(f"{target}: poses are written as {fmt.upper()}; use a .{fmt} name, or fmt={other!r} "
                             f"for .{other}, or a / at the end for a folder")
        return target

    def _sdf(self) -> bytes:
        from . import convert

        records = []
        for p in self._poses:
            props = {"_Name": self.ligand, "protein_id": self.protein, "ligand_id": self.ligand,
                     "ligand_idx": self.ligand_idx, "pocket": p.pocket, "mode": p.mode, "affinity": p.vina,
                     "rmsd_lb": p.rmsd_lb, "rmsd_ub": p.rmsd_ub}
            sdf, status = convert.pose_to_sdf(p.model, self.smiles_docked, props)
            if sdf:
                records.append(sdf)
            else:
                logger.warning("%s: pocket %d mode %d does not convert to SDF (%s), left out (fmt='pdbqt' has it)",
                               self.name, p.pocket, p.mode, status)
        if not records:
            raise ValueError(f"{self.name}: no selected pose converts to SDF; use fmt='pdbqt'")
        return b"".join(records)

    def _pdbqt(self) -> bytes:
        """The selected models, numbered 1, 2, ... in this file, each with a REMARK naming its pocket and mode."""
        out = []
        for n, p in enumerate(self._poses, 1):
            first, _, rest = p.model.partition(b"\n")
            remark = (f"REMARK POCKETVINA protein {self.protein} ligand {self.ligand} pocket {p.pocket} "
                      f"mode {p.mode}\n").encode("utf-8", "surrogateescape")
            model = f"MODEL {n:>8}\n".encode() + remark + rest
            out.append(model if model.endswith(b"\n") else model + b"\n")
        return b"".join(out)

    def export(self, path=None, fmt: str = "sdf", *, protein: bool = False,
               ligand: bool = True) -> Union[str, List[str]]:
        """Write the selected poses into one file; returns its path. With protein=True also the protein.

        path: the file, e.g. "1fbl_molecule1.sdf" (.sdf, or .pdbqt with fmt="pdbqt"), or a folder, e.g. "poses",
            made if needed: a path without an extension is a folder ("my.poses" has one, "results_v2" and ".poses"
            none). None (default): ``<name>.<fmt>`` in the working folder, e.g. PV_1fbl_molecule1_top.sdf. With
            ligand=False the protein's file, e.g. "1fbl.pdb", or a folder.
        fmt: "sdf" (default): one record per pose (converted from the docked model with the ligand's SMILES), with
            the protein, ligand, pocket, mode and Vina score (affinity, kcal/mol) as properties; "pdbqt": the docked
            models as the engine wrote them, one MODEL per pose. Only for the poses: the protein is always PDB.
        protein: True: also write the protein the poses were docked into as a PDB file, ``PV_<protein>.pdb`` next to
            the poses, e.g. PV_1fbl.pdb; for a pair of the PocketVina Atlas the AlphaFold model the drugs were docked
            into. False (default): only the poses.
        ligand: True (default): write the poses. False, with protein=True: only the protein, the same as
            ``export_protein``.

        Returns the path of the file written, e.g. ".../PV_1fbl_molecule1_top.sdf"; with protein=True (and
        ligand=True) a list of the two paths, poses first, e.g. [".../PV_1fbl_molecule1_top.sdf", ".../PV_1fbl.pdb"].
        """
        for arg, value in (("protein", protein), ("ligand", ligand)):
            if not isinstance(value, bool):
                raise ValueError(f"{arg} must be True or False, got {value!r}")
        if not (ligand or protein):
            raise ValueError("ligand=False and protein=False: nothing to write (pair.export() writes the poses, "
                             "pair.export(protein=True) the protein too, pair.export(ligand=False, protein=True) "
                             "only the protein)")
        if fmt not in EXPORT_FORMATS:
            raise ValueError(f"fmt must be one of {EXPORT_FORMATS}, got {fmt!r} (fmt is the format of the poses; "
                             f"the protein is always PDB)")
        if not ligand:
            return self.export_protein(path)
        target = self._path(path, fmt, protein)
        data = self._sdf() if fmt == "sdf" else self._pdbqt()
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = f"{target}.tmp-{os.getpid()}"
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        logger.info("%s: %d pose(s) written to %s", self.name, len(self._poses), target)
        if protein:
            return [target, self.export_protein(os.path.dirname(target))]
        return target

    def export_protein(self, path=None) -> str:
        """Write the protein the poses were docked into as a PDB file; returns its path. The same as
        ``pair.export(ligand=False, protein=True)``.

        path: the file, e.g. "1fbl.pdb" (.pdb), or a folder, e.g. "proteins", made if needed (a path without an
            extension, as for export); None (default): ``PV_<protein>.pdb`` in the working folder, e.g. PV_1fbl.pdb.
            The input structure as PDB (a PDBQT or mmCIF input converted), without crystal waters; for a pair of the
            PocketVina Atlas the AlphaFold model the drugs were docked into.
        """
        target = _target(path, f"PV_{name_part(self.protein)}.pdb")
        if path is None and os.path.isdir(target):  # a folder PV_<protein>.pdb here: write into it, as 0.3.0 did
            target = os.path.join(target, os.path.basename(target))
        if os.path.splitext(target)[1].lower() != ".pdb":
            raise ValueError(f"{target}: the protein is written as PDB; use a .pdb name, or a / at the end for a "
                             "folder")
        text = self._source.protein_pdb(self.protein)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = f"{target}.tmp-{os.getpid()}"
        try:
            with open(tmp, "w") as f:
                f.write(text)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        logger.info("%s: protein written to %s", self.protein, target)
        return target


__all__ = ["select", "Selection", "Pose", "pose_name", "name_part"]
