"""Read a results directory in the "atlas" layout (docs/dev/interfaces.md): tables, poses, structures, exports.

Only committed proteins (those with ``manifest/<protein_id>.json``) are read. Parquet parts of a protein that is
being written, or was left half committed by a crash, are ignored by the DataFrame readers and the DuckDB views.
"""
import csv
import gzip
import json
import logging
import operator
import os
import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import pdbqt
from . import posefile, tables
from .writer import FORMAT, ResultsDir

logger = logging.getLogger(__name__)

TABLES = ("ligands", "proteins", "pockets", "pocket_runs", "modes", "pairs")
# combined_results.csv of the legacy "files" output
COMBINED_COLUMNS = ("protein", "molecule_name", "affinity (kcal/mol)", "dist from rmsd l.b.", "dist from rmsd u.b.",
                    "mode", "pocket rank")
BEST_COLUMNS = ("protein_id", "ligand_idx", "ligand_id", "vina", "best_pocket", "best_mode", "rmsd_lb", "rmsd_ub",
                "sdf_status", "pb_status")
# best_scores(): the same columns whether PoseBusters ran or not (it runs only sometimes; see posebusters())
BEST_SCORES_COLUMNS = ("protein_id", "ligand_id", "vina", "best_pocket", "best_mode", "ligand_efficiency")
EXPORT_WHAT = ("best", "all")
EXPORT_FORMATS = ("sdf", "pdbqt", "csv")
MAX_OPEN_POSE_FILES = 16
# ligand part of export file names: "<protein_id>-pocket<k>-<stem>_best.pdbqt" must stay under NAME_MAX (255 bytes)
MAX_STEM = 100
_STRUCTURE_EXTS = (".pdb.gz", ".cif.gz", ".pdbqt.gz", ".pdb", ".cif", ".pdbqt")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")
# DuckDB's read_parquet options; without hive_partitioning = false, a key=value folder anywhere in the results path
# ("/scratch/run=1/results") becomes an extra column of every view (pq.read_table: partitioning=None, the same)
_READ_PARQUET = "union_by_name = true, hive_partitioning = false"
# PoseBusters results (pocketvina.bust): columns pairs() adds, join keys, schema metadata of a part
PAIRS_DERIVED = ("pb_valid", "ligand_efficiency")
PB_KEY = ("protein_id", "ligand_idx", "pocket", "mode")
PB_JOIN = ("protein_id", "ligand_idx", "best_pocket", "best_mode")
PB_CHECKS_META = b"pocketvina.pb_checks"
PB_POSES_META = b"pocketvina.poses_sha256"
# columns of the DuckDB pairs view (and of pairs() with PoseBusters results)
PAIRS_VIEW = tables.PAIRS.append(pa.field("pb_valid", pa.bool_())).append(pa.field("ligand_efficiency", pa.float64()))

ProteinArg = Optional[Union[str, Iterable[str]]]


def _text(data: bytes) -> str:
    # surrogateescape: text.encode("utf-8", "surrogateescape") gives the stored bytes back, even if not UTF-8
    return data.decode("utf-8", "surrogateescape")


def short_floats(values) -> np.ndarray:
    """float32 scores as the float64 of their shortest decimal (``-5.2``, not ``-5.199999809``), i.e. the value the
    engine printed, so CSV exports match the legacy ``combined_results.csv`` text."""
    arr = pa.array(np.asarray(values, dtype=np.float32))
    return arr.cast(pa.string()).cast(pa.float64()).to_numpy(zero_copy_only=False)


def _to_csv(df: pd.DataFrame, path_or_file, **kw) -> None:
    """``df.to_csv`` with "\\n" line ends on every OS and pandas version."""
    try:
        df.to_csv(path_or_file, lineterminator="\n", **kw)
    except TypeError:  # pandas < 1.5 calls it line_terminator
        df.to_csv(path_or_file, line_terminator="\n", **kw)


def _sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _sql_list(strings: Iterable[str]) -> str:
    return "[" + ", ".join(_sql_str(s) for s in strings) + "]"


def _concat(parts: List[pa.Table]) -> pa.Table:
    try:
        return pa.concat_tables(parts, promote_options="permissive")
    except TypeError:  # pyarrow < 14
        return pa.concat_tables(parts, promote=True)


def _read_parts(files: List[str], columns: Optional[List[str]] = None) -> pd.DataFrame:
    parts = [pq.read_table(f, columns=columns, partitioning=None).replace_schema_metadata(None) for f in files]
    return (parts[0] if len(parts) == 1 else _concat(parts)).to_pandas()


class Results:
    """A results directory written by ``pipeline.run_atlas`` or ``io.importer.import_legacy``.

    ``protein`` arguments are protein ids; ``ligand`` arguments a ligand_id (str) or a ligand_idx (int); ``pocket``
    a P2Rank rank (1-based); ``mode`` 1-based. Unknown proteins, ligands and missing poses raise ``KeyError``.
    Pose files are kept open (at most ``MAX_OPEN_POSE_FILES``); use as a context manager or call ``close()``.
    """

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        self._dir = ResultsDir(self.path)
        manifest = self._dir.run_manifest()
        if manifest is None:
            raise FileNotFoundError(f"not a PocketVina results directory (no MANIFEST.json): {self.path}")
        if manifest.get("format") != FORMAT:
            raise ValueError(f"{self._dir.manifest_file}: format {manifest.get('format')!r}, expected {FORMAT!r}")
        self._manifest = manifest
        self._readers = OrderedDict()
        self._ligand_ids = None  # ligand_idx -> ligand_id
        self._ligand_index = None  # ligand_id -> ligand_idx
        self._n_heavy_by_idx = None  # ligand_idx -> heavy atoms of the docked fragment

    def __repr__(self):
        return f"Results({self.path!r})"

    def __fspath__(self):
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Close the open pose files (they are reopened when needed)."""
        while self._readers:
            self._readers.popitem()[1].close()

    # --- proteins and tables ---------------------------------------------------------------------------------------

    def manifest(self, protein: Optional[str] = None) -> dict:
        """``MANIFEST.json`` of the run, or ``manifest/<protein>.json`` of one protein."""
        if protein is None:
            return dict(self._manifest)
        return self._dir.protein_manifest(self._protein(protein))

    def proteins(self) -> List[str]:
        """Ids of the committed proteins, sorted."""
        return sorted(self._dir.done_proteins())

    def _protein(self, protein) -> str:
        if not isinstance(protein, str):
            raise TypeError(f"protein must be a protein_id string, not {type(protein).__name__}")
        try:
            done = self._dir.is_done(protein)
        except ValueError:  # not a valid protein_id
            done = False
        if not done:
            raise KeyError(f"protein {protein!r} is not in {self.path} (not docked, or not finished)")
        return protein

    def _protein_ids(self, protein: ProteinArg) -> List[str]:
        if protein is None:
            return self.proteins()
        if isinstance(protein, str):
            return [self._protein(protein)]
        return [self._protein(p) for p in protein]

    def _part_files(self, name: str, protein: ProteinArg = None) -> List[str]:
        if name == "ligands":
            return [self._dir.ligands_file] if os.path.isfile(self._dir.ligands_file) else []
        folder = os.path.join(self.path, "tables", name)
        files = (os.path.join(folder, f"{pid}.parquet") for pid in self._protein_ids(protein))
        return [f for f in files if os.path.isfile(f)]

    def table(self, name: str, protein: ProteinArg = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Table ``name`` (one of ``TABLES``, or another folder under ``tables/``) as one DataFrame, for one protein
        id, a list of them or (None) all committed proteins, in protein order."""
        if not isinstance(name, str) or not name or "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"invalid table name {name!r}")
        schema = tables.SCHEMAS.get(name)
        if name != "ligands" and schema is None and not os.path.isdir(os.path.join(self.path, "tables", name)):
            raise KeyError(f"no table {name!r} in {self.path}")
        if name == "ligands" and protein is not None:
            raise ValueError("the ligands table is not split by protein")
        columns = list(columns) if columns is not None else None
        files = self._part_files(name, protein)
        if not files:
            empty = schema.empty_table() if schema is not None else pa.table({})
            return (empty.select(columns) if columns is not None else empty).to_pandas()
        parts = [pq.read_table(f, columns=columns, partitioning=None) for f in files]
        return (parts[0] if len(parts) == 1 else _concat(parts)).to_pandas()

    def ligands(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        return self.table("ligands", columns=columns)

    def pockets(self, protein: ProteinArg = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        return self.table("pockets", protein, columns)

    def modes(self, protein: ProteinArg = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        return self.table("modes", protein, columns)

    def pairs(self, protein: ProteinArg = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Best pose per protein x ligand: the ``pairs`` table, then ``pb_valid`` (only when there are PoseBusters
        results, see ``posebusters``) and ``ligand_efficiency`` (``-vina / n_heavy``, heavy atoms of the docked
        fragment; NaN without a pose). ``pb_status`` / ``pb_valid`` are those of the PoseBusters row of the best pose
        (same pocket and mode), if any. ``columns`` may name these derived columns too."""
        cols = list(columns) if columns is not None else None
        if protein is not None and not isinstance(protein, str):
            protein = list(protein)  # read twice (pairs parts, then posebusters parts): a generator would be used up
        if cols is None:
            df = self.table("pairs", protein)
        else:
            read = [c for c in cols if c not in PAIRS_DERIVED]
            if "ligand_efficiency" in cols:
                read += ["ligand_idx", "vina"]
            if {"pb_status", "pb_valid"} & set(cols):
                read += [*PB_JOIN, "pb_status"]
            df = self.table("pairs", protein, list(dict.fromkeys(read)))
        if cols is None or {"pb_status", "pb_valid"} & set(cols):
            df = self._with_posebusters(df, protein, need_valid=cols is not None and "pb_valid" in cols)
        if cols is None or "ligand_efficiency" in cols:
            df["ligand_efficiency"] = self._ligand_efficiency(df)
        return df if cols is None else df[cols]

    def best_scores(self, protein: ProteinArg = None, top: Optional[int] = None) -> pd.DataFrame:
        """Best (lowest) Vina score of each protein x ligand, best first: one row per docked pair with protein_id,
        ligand_id, vina (kcal/mol), best_pocket, best_mode, ligand_efficiency (-vina / heavy atoms); no PoseBusters
        columns, also when PoseBusters ran (see ``posebusters()``). Ligands without a pose are left out.

        protein: a protein id (the file name without extension), e.g. "1fbl", or a list, e.g. ["1fbl", "2W83"];
        None (default): all proteins.
        top: only the first N rows, e.g. 10; None (default): all rows.
        """
        from .. import viz

        return viz.table(self, top=top, protein=protein)[list(BEST_SCORES_COLUMNS)]

    def select(self, protein: str, ligand, pocket=None, mode=1):
        """One protein x ligand pair, to show, check or save (a ``pocketvina.selection.Selection``):
        ``pair = results.select("1fbl", "molecule1")``, then ``pair.show()``, ``pair.bust()``, ``pair.export()``.

        protein: a protein id as in best_scores(), e.g. "1fbl".
        ligand: a ligand id (its name in the SMILES file) as in best_scores(), e.g. "molecule1".
        pocket: a pocket number, e.g. 2; None (default): the pocket of the pair's best pose; "all": every pocket the
            ligand was docked into.
        mode: 1 (default): the best mode of the pocket; 2, 3, ...: that mode; "all": every mode of the pocket(s).
        """
        from ..selection import select

        return select(self, protein, ligand, pocket=pocket, mode=mode)

    def show(self, protein=None, *, path=None, molstar: str = "inline", theme: str = "auto", height: int = 760,
             max_ligands: Optional[int] = 100):
        """The results as on the PocketVina Atlas website's protein page: the 3D view (pocket chips, modes, legend,
        contacts, Save image) beside the tabs Top hits (pick a ligand there), Selected pair and Pockets, with a protein
        menu when there are several proteins. In Jupyter, Colab or VSCode it is shown below the cell; elsewhere an
        HTML file is written (open it in a browser) and its path returned.

        protein: a protein id, e.g. "1fbl", or a list of them; None (default): all proteins.
        path: write the page to this HTML file instead, e.g. "results.html"; returns the path.
        molstar: "inline" (default): the 3D viewer's code is in the page (works offline); "url": loaded from the
            PocketVina Atlas website when the page opens (a much smaller page; needs internet).
        theme: "auto" (default: as the browser), "light" or "dark".
        height: height of the view in a notebook, in pixels, e.g. 900.
        max_ligands: ligands per protein in the page (best first), e.g. 50; None: all (a large page for a large run).
        """
        from .. import workspace

        return workspace.show(self, protein, path=path, molstar=molstar, theme=theme, height=height,
                              max_ligands=max_ligands)

    def _best_scores_head(self, protein: ProteinArg, n: int) -> Tuple[pd.DataFrame, int]:
        """``(best_scores(protein, top=n), number of rows of best_scores(protein))`` without reading every pair into
        memory (DuckDB: a count and ORDER BY ... LIMIT over the pairs parts), for runs with millions of pairs."""
        import duckdb

        files = self._part_files("pairs", protein)
        if not files:
            return self.best_scores(protein, top=n), 0
        # the rows and the order of viz.table: a pose (best_pocket >= 1, vina not NaN), by vina, protein, ligand_idx
        docked = (f"FROM read_parquet({_sql_list(files)}, {_READ_PARQUET}) "
                  "WHERE best_pocket >= 1 AND NOT isnan(vina)")
        con = duckdb.connect()
        try:
            total = con.execute(f"SELECT count(*) {docked}").fetchall()[0][0]
            df = con.execute(f"SELECT protein_id, ligand_idx, ligand_id, vina, best_pocket, best_mode {docked} "
                             f"ORDER BY vina, protein_id, ligand_idx LIMIT {operator.index(n)}").df()
        finally:
            con.close()
        df["ligand_efficiency"] = self._ligand_efficiency(df)
        return df[list(BEST_SCORES_COLUMNS)], total

    def _n_heavy(self) -> np.ndarray:
        """Heavy atoms of the docked fragment by ligand_idx (float, NaN if unknown or 0)."""
        if self._n_heavy_by_idx is None:
            lig = self.ligands(columns=["ligand_idx", "n_heavy"])
            n = np.full(len(lig), np.nan)
            n[lig["ligand_idx"].to_numpy()] = pd.to_numeric(lig["n_heavy"], errors="coerce").to_numpy(dtype=float)
            n[n <= 0] = np.nan
            self._n_heavy_by_idx = n
        return self._n_heavy_by_idx

    def _ligand_efficiency(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.zeros(0)
        n_heavy = self._n_heavy()
        idx = df["ligand_idx"].to_numpy().astype(np.int64)
        heavy = np.full(len(idx), np.nan)
        known = (idx >= 0) & (idx < len(n_heavy))
        heavy[known] = n_heavy[idx[known]]
        with np.errstate(invalid="ignore", divide="ignore"):
            return -short_floats(df["vina"]) / heavy  # the score the engine printed, e.g. -6.3, not -6.3000002

    def _with_posebusters(self, df: pd.DataFrame, protein: ProteinArg, need_valid: bool) -> pd.DataFrame:
        files = self._posebusters_files(protein)
        if not files:
            if need_valid:
                df["pb_valid"] = pd.array([pd.NA] * len(df), dtype="boolean")
            return df
        pb = _read_parts(files, [*PB_KEY, "pb_status", "pb_valid"])
        pb = pb.drop_duplicates(list(PB_KEY)).rename(columns={"pocket": "best_pocket", "mode": "best_mode",
                                                                "pb_status": "_pb_status"})
        merged = df.merge(pb, how="left", on=list(PB_JOIN), sort=False, validate="many_to_one")
        merged.index = df.index
        found = merged["_pb_status"].notna()
        df["pb_status"] = merged["_pb_status"].where(found, df["pb_status"])
        df["pb_valid"] = merged["pb_valid"].astype("boolean")
        return df

    # --- PoseBusters -----------------------------------------------------------------------------------------------

    def _posebusters_current(self, path: str, pid: str) -> bool:
        """True if the posebusters part ``path`` was made from the pose file the protein has now (a protein docked
        again keeps its old part until it is checked again)."""
        try:
            meta = pq.read_schema(path).metadata or {}
        except (OSError, pa.ArrowException) as e:
            logger.warning("%s: unreadable PoseBusters table %s: %s", pid, path, e)
            return False
        made_from = meta.get(PB_POSES_META)
        if not made_from:  # not written by pocketvina.bust
            return True
        files = (self._dir.protein_manifest(pid) or {}).get("files", {})
        return made_from.decode() == (files.get(f"poses/{pid}.poses") or {}).get("sha256")

    def _posebusters_files(self, protein: ProteinArg = None) -> List[str]:
        folder = os.path.join(self.path, "tables", "posebusters")
        if not os.path.isdir(folder):
            return []
        out = []
        for pid in self._protein_ids(protein):
            path = os.path.join(folder, f"{pid}.parquet")
            if os.path.isfile(path) and self._posebusters_current(path, pid):
                out.append(path)
        return out

    def posebusters(self, protein: ProteinArg = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """PoseBusters results (``tables.POSEBUSTERS``, written by ``pocketvina.bust.bust_results``) of one protein, a
        list or all committed proteins; empty if there are none. Results of a protein docked again since it was
        checked are left out. Bit ``b`` of ``pb_mask`` / ``pb_na_mask`` is check ``posebusters_checks(protein)[b]``."""
        columns = list(columns) if columns is not None else None
        files = self._posebusters_files(protein)
        if not files:
            empty = tables.POSEBUSTERS.empty_table()
            return (empty.select(columns) if columns is not None else empty).to_pandas()
        return _read_parts(files, columns)

    def posebusters_checks(self, protein: str) -> List[str]:
        """Names of the PoseBusters checks in the mask bits of a protein's results (``bust.check_names``)."""
        pid = self._protein(protein)
        files = self._posebusters_files(pid)
        if not files:
            raise KeyError(f"{pid}: no PoseBusters results")
        meta = pq.read_schema(files[0]).metadata or {}
        return json.loads(meta.get(PB_CHECKS_META, b"[]"))

    def pocket_runs(self, protein: ProteinArg = None, columns: Optional[List[str]] = None) -> pd.DataFrame:
        return self.table("pocket_runs", protein, columns)

    def duckdb(self):
        """In-memory DuckDB connection with the views ``ligands, proteins, pockets, pocket_runs, modes, pairs`` over
        the Parquet files of the committed proteins (as they are now: reconnect to see proteins committed later), and
        ``posebusters`` when there are PoseBusters results (see ``posebusters``).

        The ``pairs`` view has the columns of ``pairs()``: the ``pairs`` table with ``pb_status`` / ``pb_valid`` of
        the PoseBusters row of the best pose (same pocket and mode; ``pb_valid`` is NULL without one, also when there
        are no PoseBusters results) and ``ligand_efficiency`` (``-vina / n_heavy``, NULL if ``n_heavy`` is unknown).
        """
        import duckdb

        con = duckdb.connect()
        for name in TABLES:
            if name == "pairs":
                continue  # below: it joins the ligands and the PoseBusters results
            files = self._part_files(name)
            if not files:
                con.register(name, tables.SCHEMAS[name].empty_table())
                continue
            # an explicit file list, not a glob: a view is re-read at every query, and a glob would then also match
            # the parts of a protein that another process is committing (no manifest yet)
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet({_sql_list(files)}, {_READ_PARQUET})")
        pb_files = self._posebusters_files()
        if pb_files:
            con.execute(f"CREATE VIEW posebusters AS SELECT * FROM read_parquet({_sql_list(pb_files)}, "
                        f"{_READ_PARQUET})")
        self._pairs_view(con, pb_files)
        return con

    def _pairs_view(self, con, pb_files: List[str]) -> None:
        """The DuckDB ``pairs`` view (see ``duckdb``): the same rows and values as ``pairs()``."""
        files = self._part_files("pairs")
        if not files:
            con.register("pairs", PAIRS_VIEW.empty_table())
            return
        if pb_files:
            # pairs(): the first row of each pose (in protein order, then file order) when a pose has several; the
            # file positions come from a join (list_position over the file list per row costs rows x files)
            key = ", ".join(f"r.{k}" for k in PB_KEY)
            file_pos = ", ".join(f"({_sql_str(f)}, {n})" for n, f in enumerate(pb_files))
            pb_join = f"""
            LEFT JOIN (
                SELECT {key}, r.pb_status, r.pb_valid
                FROM read_parquet({_sql_list(pb_files)}, {_READ_PARQUET}, filename = true,
                                  file_row_number = true) AS r
                LEFT JOIN (VALUES {file_pos}) AS f(filename, file_pos) ON f.filename = r.filename
                QUALIFY row_number() OVER (PARTITION BY {key} ORDER BY f.file_pos, r.file_row_number) = 1
            ) AS b ON {" AND ".join(f"b.{k} = p.{j}" for k, j in zip(PB_KEY, PB_JOIN))}"""
            pb_status, pb_valid = "coalesce(b.pb_status, p.pb_status)", "b.pb_valid"
        else:
            pb_join, pb_status, pb_valid = "", "p.pb_status", "CAST(NULL AS BOOLEAN)"
        # the score as the engine printed it (short_floats: -6.3, not -6.300000190734863)
        con.execute(f"""
            CREATE VIEW pairs AS
            SELECT p.* REPLACE ({pb_status} AS pb_status), {pb_valid} AS pb_valid,
                   CASE WHEN l.n_heavy > 0 THEN -CAST(CAST(p.vina AS VARCHAR) AS DOUBLE) / l.n_heavy END
                       AS ligand_efficiency
            FROM read_parquet({_sql_list(files)}, {_READ_PARQUET}) AS p
            LEFT JOIN (SELECT ligand_idx, n_heavy FROM ligands) AS l ON l.ligand_idx = p.ligand_idx{pb_join}""")

    # --- ligands ---------------------------------------------------------------------------------------------------

    def _load_ligand_index(self):
        if self._ligand_ids is None:
            df = self.ligands(columns=["ligand_idx", "ligand_id"])
            ids = np.empty(len(df), dtype=object)
            ids[df["ligand_idx"].to_numpy()] = df["ligand_id"].to_numpy()
            self._ligand_ids = ids
            self._ligand_index = {lig: i for i, lig in enumerate(ids)}
        return self._ligand_ids

    def ligand_idx(self, ligand) -> int:
        """``ligand_idx`` of a ligand_id (str) or a ligand_idx (int, checked)."""
        ids = self._load_ligand_index()
        if isinstance(ligand, str):
            try:
                return self._ligand_index[ligand]
            except KeyError:
                raise KeyError(f"no ligand {ligand!r} in {self.path}") from None
        if isinstance(ligand, (bool, np.bool_)):
            raise TypeError("ligand must be a ligand_id (str) or a ligand_idx (int), not bool")
        try:
            i = operator.index(ligand)
        except TypeError:
            raise TypeError(f"ligand must be a ligand_id (str) or a ligand_idx (int), not {type(ligand).__name__}") \
                from None
        if not 0 <= i < len(ids):
            raise KeyError(f"ligand_idx {i} out of range 0..{len(ids) - 1}")
        return i

    def ligand_id(self, ligand) -> str:
        return self._load_ligand_index()[self.ligand_idx(ligand)]

    # --- poses and structures --------------------------------------------------------------------------------------

    def _reader(self, pid: str) -> posefile.Reader:
        reader = self._readers.pop(pid, None)
        if reader is None:
            reader = posefile.Reader(os.path.join(self.path, "poses", f"{pid}.poses"))
            while len(self._readers) >= MAX_OPEN_POSE_FILES:
                self._readers.popitem(last=False)[1].close()
        self._readers[pid] = reader
        return reader

    def _block(self, protein, ligand, pocket) -> bytes:
        pid, i = self._protein(protein), self.ligand_idx(ligand)
        reader = self._reader(pid)
        k = operator.index(pocket)
        if not 1 <= k <= reader.K or i >= reader.N:
            raise KeyError(f"{pid}: no pocket {k} for ligand_idx {i} (pockets 1..{reader.K}, {reader.N} ligands)")
        data = reader.modes(i, k)
        if data is None:
            raise KeyError(f"{pid}: no engine output for ligand {ligand!r} in pocket {k}")
        return data

    def pose_block(self, protein: str, ligand, pocket: int) -> str:
        """The engine output (all MODELs) of a ligand in a pocket, byte for byte as the engine wrote it."""
        return _text(self._block(protein, ligand, pocket))

    def pose_model(self, protein: str, ligand, pocket: int, mode: int) -> str:
        """One MODEL (``MODEL ... ENDMDL``) of the engine output; ``IndexError`` if there is no such mode."""
        return _text(pdbqt.model(self._block(protein, ligand, pocket), operator.index(mode)))

    def top_sdf(self, protein: str, ligand) -> Optional[str]:
        """SDF of the ligand's best pose in the protein (section A), or None if there is none."""
        pid, i = self._protein(protein), self.ligand_idx(ligand)
        reader = self._reader(pid)
        if i >= reader.N:
            return None
        data = reader.top(i)
        return None if data is None else _text(data)

    def structure_path(self, protein: str) -> str:
        """Path of the stored input structure (``structures/<protein_id>.pdb.gz``, ``.pdbqt.gz``, ...)."""
        pid = self._protein(protein)
        files = (self._dir.protein_manifest(pid) or {}).get("files", {})
        listed = sorted(rel for rel in files if rel.startswith("structures/"))
        candidates = [os.path.join(self.path, *rel.split("/")) for rel in listed]
        candidates += [os.path.join(self.path, "structures", pid + ext) for ext in _STRUCTURE_EXTS]
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"{pid}: no structure file in {os.path.join(self.path, 'structures')}")

    def structure(self, protein: str) -> bytes:
        """The input structure (PDB, mmCIF or PDBQT text), decompressed."""
        path = self.structure_path(protein)
        with open(path, "rb") as f:
            data = f.read()
        return gzip.decompress(data) if path.endswith(".gz") else data

    # --- exports ---------------------------------------------------------------------------------------------------

    def to_combined_csv(self, path, protein: ProteinArg = None) -> str:
        """Write every mode as a row of the legacy ``combined_results.csv`` (``COMBINED_COLUMNS``; ``protein`` is
        ``<protein_id>-pocket<k>``, ``molecule_name`` the ligand_id); returns the path. Rows are in protein, pocket,
        ligand_idx, mode order; scores are written as the engine printed them."""
        ids = self._load_ligand_index()
        pids = self._protein_ids(protein)
        path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp-{os.getpid()}"
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                csv.writer(f, lineterminator="\n").writerow(COMBINED_COLUMNS)
                for pid in pids:
                    m = self.modes(pid, columns=["ligand_idx", "pocket", "mode", "affinity", "rmsd_lb", "rmsd_ub"])
                    if m.empty:
                        continue
                    m = m.sort_values(["pocket", "ligand_idx", "mode"], kind="stable")
                    pocket = m["pocket"].to_numpy().astype(np.int64)
                    df = pd.DataFrame({
                        "protein": [f"{pid}-pocket{k}" for k in pocket],
                        "molecule_name": ids[m["ligand_idx"].to_numpy()],
                        "affinity (kcal/mol)": short_floats(m["affinity"]),
                        "dist from rmsd l.b.": short_floats(m["rmsd_lb"]),
                        "dist from rmsd u.b.": short_floats(m["rmsd_ub"]),
                        "mode": m["mode"].to_numpy().astype(np.int64),
                        "pocket rank": pocket,
                    })
                    _to_csv(df, f, header=False, index=False)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return path

    def _best_csv(self, path, pids) -> str:
        frames = []
        for pid in pids:
            p = self.pairs(pid, columns=list(BEST_COLUMNS))
            for col in ("vina", "rmsd_lb", "rmsd_ub"):
                p[col] = short_floats(p[col])
            for col in ("best_pocket", "best_mode"):  # -1 = no pose
                p[col] = p[col].astype("Int64").mask(p[col] < 1)
            frames.append(p)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(BEST_COLUMNS))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _to_csv(df, path, index=False)
        return path

    def _file_stems(self) -> np.ndarray:
        """A file-name-safe, case-insensitively unique stem of at most ``MAX_STEM`` characters (+ a ``_<i>`` suffix
        when needed) per ligand_idx."""
        ids = self._load_ligand_index()
        stems, seen = np.empty(len(ids), dtype=object), set()
        for i, lig in enumerate(ids):
            stem = _UNSAFE_CHARS.sub("_", str(lig))[:MAX_STEM]
            if not stem.strip("."):
                stem = f"ligand{i}"
            if stem.lower() in seen:
                base, n = f"{stem}_{i}", 1
                stem = base
                while stem.lower() in seen:  # "<stem>_<i>" may be another ligand's id
                    n += 1
                    stem = f"{base}_{n}"
            seen.add(stem.lower())
            stems[i] = stem
        return stems

    def export(self, out_dir, what: str = "best", fmt: str = "sdf", protein: ProteinArg = None) -> List[str]:
        """Write poses or tables as plain files under ``out_dir``; returns the paths written.

        - ``fmt="sdf"``: ``what="best"`` the stored best pose per protein x ligand
          (``<protein_id>/<protein_id>-pocket<k>-<ligand>_best.sdf``); ``what="all"`` every mode of every engine
          output converted with ``pocketvina.convert`` (needs the docked SMILES;
          ``<protein_id>/<protein_id>-pocket<k>-<ligand>_out.sdf``, modes that do not convert are left out).
        - ``fmt="pdbqt"``: the best MODEL (``..._best.pdbqt``), or every engine output as written (``..._out.pdbqt``).
        - ``fmt="csv"``: ``best_poses.csv`` (the pairs table), or ``combined_results.csv`` (``to_combined_csv``).
        """
        if what not in EXPORT_WHAT:
            raise ValueError(f"what must be one of {EXPORT_WHAT}, got {what!r}")
        if fmt not in EXPORT_FORMATS:
            raise ValueError(f"fmt must be one of {EXPORT_FORMATS}, got {fmt!r}")
        out = os.path.abspath(os.path.expanduser(os.fspath(out_dir)))
        pids = self._protein_ids(protein)
        if fmt == "csv":
            if what == "all":
                return [self.to_combined_csv(os.path.join(out, "combined_results.csv"), pids)]
            return [self._best_csv(os.path.join(out, "best_poses.csv"), pids)]
        os.makedirs(out, exist_ok=True)
        stems = self._file_stems()
        written: List[str] = []
        failed: Dict[str, int] = {}
        for pid in pids:
            if what == "best":
                self._export_best(out, pid, fmt, stems, written)
            elif fmt == "pdbqt":
                self._export_blocks(out, pid, stems, written)
            else:
                self._export_all_sdf(out, pid, stems, written, failed)
        if failed:
            logger.warning("export: %d pose(s) not converted to SDF (%s)", sum(failed.values()),
                           ", ".join(f"{k}: {v}" for k, v in sorted(failed.items())))
        logger.info("export: %d file(s) written to %s", len(written), out)
        return written

    @staticmethod
    def _write(path: str, data: bytes, written: List[str]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        written.append(path)

    def _export_best(self, out, pid, fmt, stems, written):
        pairs = self.pairs(pid, columns=["ligand_idx", "best_pocket", "best_mode", "top_length"])
        reader = self._reader(pid)
        for i, k, mode, top_length in pairs.itertuples(index=False):
            if k < 1 or (fmt == "sdf" and not top_length):
                continue
            path = os.path.join(out, pid, f"{pid}-pocket{k}-{stems[i]}_best.{fmt}")
            if fmt == "sdf":
                self._write(path, reader.top(int(i)), written)
            else:
                self._write(path, pdbqt.model(reader.modes(int(i), int(k)), int(mode)), written)

    def _export_blocks(self, out, pid, stems, written):
        runs = self.pocket_runs(pid, columns=["ligand_idx", "pocket", "length"])
        reader = self._reader(pid)
        for i, k, length in runs.itertuples(index=False):
            if length:
                self._write(os.path.join(out, pid, f"{pid}-pocket{k}-{stems[i]}_out.pdbqt"),
                            reader.modes(int(i), int(k)), written)

    def _export_all_sdf(self, out, pid, stems, written, failed):
        from .. import convert

        smiles = self.ligands(columns=["ligand_idx", "smiles_docked"]).set_index("ligand_idx")["smiles_docked"]
        ids = self._load_ligand_index()
        runs = self.pocket_runs(pid, columns=["ligand_idx", "pocket", "status", "length"])
        reader = self._reader(pid)
        for i, k, status, length in runs.itertuples(index=False):
            if status != "ok" or not length:
                continue
            smi = smiles.get(i)
            block = reader.modes(int(i), int(k))
            records = []
            for m in pdbqt.parse_modes(block):
                if not isinstance(smi, str) or not smi:
                    failed["no SMILES"] = failed.get("no SMILES", 0) + 1
                    continue
                # the fields of the best-pose SDFs (pipeline.run_atlas, importer)
                props = {"_Name": ids[i], "protein_id": pid, "ligand_idx": int(i), "pocket": int(k), "mode": m.mode,
                         "affinity": m.affinity}
                sdf, sdf_status = convert.pose_to_sdf(block[m.raw_offset:m.raw_offset + m.raw_length], smi, props)
                if sdf:
                    records.append(sdf)
                else:
                    failed[sdf_status] = failed.get(sdf_status, 0) + 1
            if records:
                self._write(os.path.join(out, pid, f"{pid}-pocket{k}-{stems[i]}_out.sdf"), b"".join(records),
                            written)
