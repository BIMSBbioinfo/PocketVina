"""Parquet tables of a results directory: schemas, atomic writes, reads (docs/output-format.md, "Tables").

Column names mirror the PocketVina Atlas tables (``af_id`` -> ``protein_id``, ``cidx`` -> ``ligand_idx``,
``catalog_id`` -> ``ligand_id``). Columns not marked nullable must not hold nulls. NaN means "missing" (null) only in
nullable columns; in a required float column it is stored as NaN.
"""
import glob
import os
import uuid
from collections.abc import Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_OPTIONS = dict(compression="zstd", compression_level=9)  # same as the Atlas importer

RUN_STATUS = ("ok", "no_output", "parse_error", "engine_error", "skipped")
SDF_STATUS = ("ok", "convert_failed", "no_pose")


def _req(name, typ):
    return pa.field(name, typ, nullable=False)


def _opt(name, typ):
    return pa.field(name, typ, nullable=True)


_SHA = pa.binary(32)

LIGANDS = pa.schema([
    _req("ligand_idx", pa.int32()),
    _req("ligand_id", pa.string()),
    _req("smiles_input", pa.string()),
    _opt("smiles_docked", pa.string()),
    _opt("inchikey", pa.string()),
    _opt("n_heavy", pa.int16()),
    _req("prep_method", pa.string()),
    _opt("seed", pa.int64()),
    _req("status", pa.string()),
    _opt("error", pa.string()),
    _opt("pdbqt", pa.string()),
])

PROTEINS = pa.schema([
    _req("protein_id", pa.string()),
    _opt("uniprot", pa.string()),
    _req("source_path", pa.string()),
    _req("structure_file", pa.string()),
    _req("receptor_sha256", _SHA),
    _req("n_pockets", pa.int16()),
    _req("n_pockets_docked", pa.int16()),
    _req("status", pa.string()),
    _opt("error", pa.string()),
])

POCKETS = pa.schema([
    _req("protein_id", pa.string()),
    _req("pocket", pa.int16()),
    _req("name", pa.string()),
    _req("score", pa.float32()),
    _opt("probability", pa.float32()),
    _req("center_x", pa.float32()),
    _req("center_y", pa.float32()),
    _req("center_z", pa.float32()),
    _req("size_x", pa.float32()),
    _req("size_y", pa.float32()),
    _req("size_z", pa.float32()),
    _opt("residue_ids", pa.string()),
])

POCKET_RUNS = pa.schema([
    _req("protein_id", pa.string()),
    _req("ligand_idx", pa.int32()),
    _req("ligand_id", pa.string()),
    _req("pocket", pa.int8()),
    _req("status", pa.string()),
    _req("n_modes", pa.int8()),
    _opt("best_affinity", pa.float32()),
    _req("offset", pa.uint64()),
    _req("length", pa.uint32()),
    _req("raw_length", pa.uint32()),
    _opt("sha256", _SHA),
    _opt("seed", pa.int64()),
    _opt("runtime_s", pa.float32()),
    _opt("error", pa.string()),
])

MODES = pa.schema([
    _req("protein_id", pa.string()),
    _opt("uniprot", pa.string()),
    _req("ligand_idx", pa.int32()),
    _req("pocket", pa.int8()),
    _req("mode", pa.int8()),
    _req("affinity", pa.float32()),
    _req("rmsd_lb", pa.float32()),
    _req("rmsd_ub", pa.float32()),
    _req("block_offset", pa.uint64()),
    _req("block_length", pa.uint32()),
    _req("raw_offset", pa.uint32()),
    _req("raw_length", pa.uint32()),
])

PAIRS = pa.schema([
    _req("protein_id", pa.string()),
    _opt("uniprot", pa.string()),
    _req("ligand_idx", pa.int32()),
    _req("ligand_id", pa.string()),
    _req("vina", pa.float32()),
    _req("best_pocket", pa.int8()),
    _req("best_mode", pa.int8()),
    _req("rmsd_lb", pa.float32()),
    _req("rmsd_ub", pa.float32()),
    _opt("pb_status", pa.string()),
    _req("top_offset", pa.uint64()),
    _req("top_length", pa.uint32()),
    _req("top_raw_length", pa.uint32()),
    _opt("top_sha256", _SHA),
    _req("sdf_status", pa.string()),
])

# PoseBusters checks of docked poses (pocketvina.bust; docs/dev/interfaces_phase2.md). Bit b of pb_mask: check b
# passed; of pb_na_mask: check b gave no result. The check names, in bit order, are a JSON list in the schema
# metadata of each part ("pocketvina.pb_checks"; bust.check_names). pocket = mode = -1: the ligand has no pose.
POSEBUSTERS = pa.schema([
    _req("protein_id", pa.string()),
    _opt("uniprot", pa.string()),
    _req("ligand_idx", pa.int32()),
    _req("ligand_id", pa.string()),
    _req("pocket", pa.int8()),
    _req("mode", pa.int8()),
    _req("pb_config", pa.string()),
    _req("pb_version", pa.string()),
    _req("pb_status", pa.string()),
    _opt("pb_valid", pa.bool_()),
    _req("pb_mask", pa.int32()),
    _req("pb_na_mask", pa.int32()),
    _opt("failed_checks", pa.string()),
    _opt("n_heavy_docked", pa.int16()),
    _opt("runtime_s", pa.float32()),
    _opt("error", pa.string()),
])

# table name (folder under tables/, or ligands.parquet) -> schema
SCHEMAS = dict(ligands=LIGANDS, proteins=PROTEINS, pockets=POCKETS, pocket_runs=POCKET_RUNS, modes=MODES,
               pairs=PAIRS)


def to_arrow(rows, schema: pa.Schema) -> pa.Table:
    """``rows`` (list of dicts, ``pandas.DataFrame``, dict of columns or ``pyarrow.Table``) as a table with exactly
    ``schema``. Missing columns are null; unknown columns, nulls in required columns and values that do not fit the
    column type raise ``ValueError``."""
    if isinstance(rows, pa.Table):
        rows = {name: rows.column(name) for name in rows.column_names}
    if isinstance(rows, pd.DataFrame):
        names, n = list(rows.columns), len(rows)

        def column(name):
            return rows[name]
    elif isinstance(rows, Mapping):
        names = list(rows)
        lengths = {len(v) for v in rows.values()}
        if len(lengths) > 1:
            raise ValueError(f"columns have different lengths: {sorted(lengths)}")
        n = lengths.pop() if lengths else 0

        def column(name):
            return rows[name]
    else:
        rows = list(rows)
        names, n = list(dict.fromkeys(k for r in rows for k in r)), len(rows)

        def column(name):
            return [r.get(name) for r in rows]

    unknown = [c for c in names if c not in schema.names]
    if unknown:
        raise ValueError(f"columns not in the schema: {unknown}")
    arrays = []
    for field in schema:
        if field.name not in names:
            arr = pa.nulls(n, field.type)
        else:
            values = column(field.name)
            if isinstance(values, pa.ChunkedArray):
                values = values.combine_chunks()
            try:
                if isinstance(values, pa.Array):
                    arr = values.cast(field.type)
                elif pa.types.is_integer(field.type):
                    # infer, then a safe cast: pa.array(..., type=int) silently truncates 1.5 to 1
                    arr = pa.array(values, from_pandas=field.nullable).cast(field.type)
                else:
                    arr = pa.array(values, type=field.type, from_pandas=field.nullable)
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, TypeError, ValueError,
                    OverflowError) as e:
                raise ValueError(f"column {field.name!r}: {e}") from e
        if not field.nullable and arr.null_count:
            raise ValueError(f"column {field.name!r} is required but has {arr.null_count} null value(s)")
        arrays.append(arr)
    return pa.Table.from_arrays(arrays, schema=schema)


def write_table(rows, schema: pa.Schema, path) -> int:
    """Write ``rows`` (see ``to_arrow``) as a zstd-compressed Parquet file at ``path``, atomically: a temporary file
    next to it is renamed over ``path``. Creates the parent folders. Returns the number of rows."""
    table = to_arrow(rows, schema)
    # absolute: pyarrow reads a relative path to a new file such as "run:1/x.parquet" as a URI with scheme "run"
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        pq.write_table(table, tmp, **PARQUET_OPTIONS)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return table.num_rows


def _files(p: str) -> list:
    if os.path.isdir(p):
        return sorted(glob.glob(os.path.join(glob.escape(p), "*.parquet")))
    if os.path.exists(p):
        return [p]
    if any(c in p for c in "*?["):
        return sorted(f for f in glob.glob(p) if os.path.isfile(f))
    return []


def read_table(path_or_glob, columns=None, schema: pa.Schema = None) -> pd.DataFrame:
    """One DataFrame from a Parquet file, a glob (``tables/modes/*.parquet``) or a folder of ``*.parquet`` files,
    concatenated in file-name order. ``columns`` selects columns. If there is no such file, returns an empty
    DataFrame with the columns of ``schema`` when given, else raises ``FileNotFoundError``."""
    columns = list(columns) if columns is not None else None
    path = os.fspath(path_or_glob)
    files = _files(path)
    if not files:
        if schema is None:
            raise FileNotFoundError(f"no Parquet file at {path}")
        empty = schema.empty_table()
        return (empty.select(columns) if columns is not None else empty).to_pandas()
    # absolute: see write_table; partitioning=None: older pyarrow (21) adds a column per key=value folder of the path
    parts = [pq.read_table(os.path.abspath(f), columns=columns, partitioning=None) for f in files]
    if len(parts) == 1:
        return parts[0].to_pandas()
    try:
        table = pa.concat_tables(parts, promote_options="permissive")
    except TypeError:  # pyarrow < 14
        table = pa.concat_tables(parts, promote=True)
    return table.to_pandas()
