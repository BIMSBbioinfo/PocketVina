"""The ``pocketvina`` command line (``python -m pocketvina``): docking runs, PoseBusters checks, views, exports and
queries of a results directory, the import of legacy runs, a GPU benchmark and the P2Rank install.

    pocketvina setup p2rank                               # once: P2Rank into ~/.cache/pocketvina (a run does it too)
    pocketvina dock --proteins proteins.txt --ligands ligands.csv --out results --top-pockets 5
    pocketvina dock --config run.toml --seed 7            # flags given here override the TOML file
    pocketvina bust --results results
    pocketvina info --results results
    pocketvina query --results results "SELECT ligand_id, vina FROM pairs ORDER BY vina LIMIT 10"
    pocketvina query best --results results --top 20
    pocketvina show --results results --protein 1fbl --ligand molecule1 --out pose.html
    pocketvina export --results results --what best --format sdf --out poses
    pocketvina import-legacy output --results results --smiles ligands.csv
    pocketvina bench gpu --protein 1fbl.pdb --ligands ligands.csv --procs 1,2,3 --out bench
    pocketvina atlas search imatinib                      # the PocketVina Atlas (pocketvina.atlas.Atlas)
    pocketvina atlas show ABL1 --ligand imatinib --out ABL1_imatinib.html
    pocketvina atlas export ABL1 imatinib --fmt sdf --protein --out poses
    pocketvina atlas download ABL1 EGFR --out my_atlas

Exit codes: 0 ok, 1 usage or configuration error, 2 partial failure (some proteins or engine calls failed), 3 nothing
docked. The modules that need pyarrow, DuckDB, PoseBusters or a viewer are imported by the commands that use them, so
``dock --format files`` works without pyarrow.
"""
import ast
import contextlib
import dataclasses
import json
import logging
import math
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

import click
from click.core import ParameterSource

from . import __version__, engine, gpu, p2rank
from .main import (
    CITATION,
    DEFAULT_LIGANDS_PER_CALL,
    MAX_TOP_POCKETS,
    STRUCTURE_KINDS,
    P2RankConfig,
    PocketConfig,
    PocketVinaConfig,
    PocketVinaGPU,
    read_database_list,
    read_protein_list,
)

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_USAGE = 1    # usage or configuration error
EXIT_PARTIAL = 2  # some proteins (or engine calls) failed
EXIT_NOTHING = 3  # nothing docked
EXIT_ABORTED = 130  # Ctrl-C
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # as PocketVinaGPU's
RANDOM_SEED = "random"
# dock options that are paths: relative ones in a --config file are relative to the file's folder
PATH_OPTIONS = ("proteins", "database_path", "protein_list", "ligands", "out", "tmp_dir")
# --config keys that name an option by its PocketVinaConfig field
CONFIG_ALIASES = {"protein_paths": "proteins", "smiles_file": "ligands", "output_dir": "out",
                  "output_format": "format", "vram_per_proc_mib": "vram_per_proc"}
# dock options passed to PocketVinaConfig as they are (option name -> field)
CONFIG_FIELDS = {"database_path": "database_path", "protein_list": "protein_list", "top_pockets": "top_pockets",
                 "pockets": "pockets", "pockets_dir": "pockets_dir", "num_modes": "num_modes",
                 "energy_range": "energy_range", "ligand_prep": "ligand_prep",
                 "ligands_per_call": "ligands_per_call", "batch_size": "batch_size", "gpus": "gpus",
                 "procs_per_gpu": "procs_per_gpu", "vram_per_proc": "vram_per_proc_mib",
                 "engine_timeout": "engine_timeout", "posebusters": "posebusters",
                 "posebusters_workers": "posebusters_workers", "tmp_dir": "tmp_dir", "gzip_level": "gzip_level"}
BUST_SELECT = ("best", "top_k", "per_pocket")  # bust.SELECT (bust imports pyarrow)
EXPORT_WHAT = ("best", "all")
EXPORT_FORMATS = ("sdf", "pdbqt", "csv")
VIEW_BACKENDS = ("auto", "molstar", "py3dmol")
ATLAS_SEARCH_TOP = 20  # atlas.SEARCH_TOP (pocketvina.atlas imports pandas: only the atlas commands import it)
ATLAS_EXPORT_FORMATS = ("sdf", "pdbqt")  # selection.EXPORT_FORMATS
BENCH_CSV = "bench_gpu.csv"
BENCH_JSON = "bench_gpu.json"
BENCH_COLUMNS = ("procs", "wall_s", "ligands_docked", "ligands_per_hour", "speedup", "proc_runtime_s", "s_per_ligand",
                 "slowdown", "peak_vram_mib", "gpu_util_mean", "gpu_util_max", "gpu_mem_used_max_mib", "failed_calls")
NVIDIA_SMI_GPU = ("nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits")
NVIDIA_SMI_APPS = ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits")
NVIDIA_SMI_TIMEOUT_S = 10
P2RANK_SOURCES = {"P2RANK_HOME": "$P2RANK_HOME", "PATH": "prank on PATH",
                  "package": "the package folder of an older source install", "cache": "the user cache"}
_POCKET_SUFFIX = re.compile(r"-pocket\d+$")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


# --- errors, logging, helpers --------------------------------------------------------------------------------------

class CliError(click.ClickException):
    """An error message (no traceback) and the exit code."""

    def __init__(self, message: str, exit_code: int = EXIT_USAGE):
        super().__init__(message)
        self.exit_code = exit_code


class _Group(click.Group):
    """``click.Group`` whose usage errors exit with code 1 (click's own code 2 means a partial failure here)."""

    def main(self, args=None, prog_name=None, complete_var=None, standalone_mode=True, **extra):
        if not standalone_mode:
            return super().main(args, prog_name, complete_var, standalone_mode, **extra)
        try:
            code = super().main(args, prog_name, complete_var, False, **extra)
        except click.ClickException as e:
            e.show()
            sys.exit(EXIT_USAGE if isinstance(e, click.UsageError) else e.exit_code)
        except click.Abort:
            click.echo("Aborted!", err=True)
            sys.exit(EXIT_ABORTED)
        sys.exit(code if isinstance(code, int) else EXIT_OK)


def _message(e: BaseException) -> str:
    if isinstance(e, KeyError) and e.args:
        return str(e.args[0])
    return str(e) or type(e).__name__


@contextlib.contextmanager
def _user_errors(also: tuple = ()):
    """Errors of the inputs or settings (a bad value, a missing file, protein or ligand, a missing optional package)
    as a message and exit code 1 instead of a traceback; ``-v`` prints the traceback too. ``also``: more exception
    types to report so (the atlas commands: OSError, e.g. no connection to the Atlas)."""
    try:
        yield
    except click.ClickException:
        raise
    except (ValueError, KeyError, ImportError, FileNotFoundError, NotADirectoryError, IsADirectoryError,
            PermissionError, *also) as e:
        ctx = click.get_current_context(silent=True)
        if ctx is not None and (ctx.find_object(dict) or {}).get("log_level") == logging.DEBUG:
            click.echo(traceback.format_exc(), err=True, nl=False)
        message = _message(e)
        if ctx is not None and any(p.name == "force" for p in ctx.command.params):
            message = message.replace("force=True", "--force")  # ResultsDir.init_run names the Python argument
        raise CliError(message) from e


def _setup_logging(ctx: click.Context) -> None:
    """Log to stderr at the level of ``-v``/``-q`` (handlers already installed, e.g. by ``PocketVinaGPU``, stay)."""
    level = (ctx.find_object(dict) or {}).get("log_level", logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(format=LOG_FORMAT)
    root.setLevel(level)


def _echo_table(df, max_colwidth: int = 60) -> None:
    if df.empty:
        click.echo("  ".join(str(c) for c in df.columns))
        click.echo("(0 rows)")
    else:
        click.echo(df.to_string(index=False, max_colwidth=max_colwidth))


def _write_csv(df, path: str) -> None:
    if path == "-":
        click.echo(df.to_csv(index=False), nl=False)
        return
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    click.echo(f"{len(df)} rows written to {path}")


def _counts(counter) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _plain_number(value):
    """20.0 -> 20 (so a box given on the command line is stored as the Python API default is)."""
    return int(value) if isinstance(value, float) and value.is_integer() else value


class _SeedType(click.ParamType):
    name = "seed"

    def convert(self, value, param, ctx):
        if isinstance(value, str) and value.strip().lower() in (RANDOM_SEED, "none"):
            return RANDOM_SEED
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            self.fail(f"{value!r} is not an integer or 'random'", param, ctx)
        try:
            return int(value)
        except (TypeError, ValueError):
            self.fail(f"{value!r} is not an integer or 'random'", param, ctx)


class _Finite:
    """Mixin of a float type: nan and inf are errors (they pass ``click.FloatRange``'s range checks)."""

    def convert(self, value, param, ctx):
        rv = super().convert(value, param, ctx)
        if not math.isfinite(rv):
            self.fail(f"{value!r} is not a finite number", param, ctx)
        return rv


class _FiniteFloat(_Finite, click.types.FloatParamType):
    pass


class _FiniteFloatRange(_Finite, click.FloatRange):
    pass


class _BoxSizeType(_FiniteFloatRange):
    """A box edge (A): a finite number > 0 (the engine takes no other, and an atlas run would have registered the
    run's settings in --out before it found out)."""

    def __init__(self):
        super().__init__(0, min_open=True)


THREAD_TYPE = click.IntRange(engine.MIN_THREAD, 2**31 - 1)  # engine.write_config's limits


class _ProcsType(click.ParamType):
    name = "N|auto"

    def convert(self, value, param, ctx):
        try:
            return gpu.check_procs_per_gpu(int(value) if isinstance(value, str) and value.strip().isdigit()
                                           else value)
        except ValueError as e:
            self.fail(str(e), param, ctx)


def _parse_numbers(value, kind, name: str) -> list:
    items = [v.strip() for v in str(value).split(",") if v.strip()]
    try:
        return [kind(v) for v in items]
    except ValueError:
        raise click.BadParameter(f"{value!r}: {name}") from None


def _procs_option(ctx, param, value) -> List[int]:
    procs = _parse_numbers(value, int, "comma-separated process counts, e.g. 1,2,3")
    if not procs or any(n < 1 for n in procs):
        raise click.BadParameter(f"{value!r}: process counts must be integers >= 1, e.g. 1,2,3")
    return list(dict.fromkeys(procs))


def _timeouts_option(ctx, param, value):
    if value is None or str(value).strip().lower() == "none":
        return None
    timeouts = _parse_numbers(value, float, "comma-separated seconds, e.g. 30,300, or 'none'")
    if not timeouts or not all(0 < t < float("inf") for t in timeouts):
        raise click.BadParameter(f"{value!r}: timeouts must be positive seconds, e.g. 30,300")
    return tuple(timeouts)


def _is_structure_file(path) -> bool:
    name = os.path.basename(os.fspath(path)).lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return name.endswith(tuple(STRUCTURE_KINDS))


def structure_paths(path) -> List[str]:
    """Absolute structure paths that ``--proteins`` names: a protein list file (one path per line, relative to the
    list's folder; ``main.read_protein_list``), a folder (its structure files, sorted by name) or one structure."""
    path = os.path.abspath(os.path.expanduser(os.fspath(path)))
    if os.path.isdir(path):
        found = sorted(os.path.join(path, n) for n in os.listdir(path)
                       if not n.startswith(".") and _is_structure_file(n) and os.path.isfile(os.path.join(path, n)))
        if not found:
            kinds = ", ".join(f"*{k}" for k in STRUCTURE_KINDS)
            raise ValueError(f"{path}: no structure files ({kinds}, optionally gzipped)")
        return found
    if _is_structure_file(path):
        return [path]
    return read_protein_list(path)


def _listed_protein_ids(path) -> List[str]:
    from .io.writer import protein_id_from_path

    return [protein_id_from_path(p) for p in structure_paths(path)]


def _results_option(f):
    return click.option("--results", required=True, type=click.Path(exists=True, file_okay=False),
                        help="Results directory (atlas layout).")(f)


# --- the group -----------------------------------------------------------------------------------------------------

@click.group(cls=_Group, context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 110})
@click.version_option(__version__, "--version", prog_name="pocketvina")
@click.option("-v", "--verbose", is_flag=True, help="Debug messages, and tracebacks of errors.")
@click.option("-q", "--quiet", is_flag=True, help="Only warnings and errors in the log.")
@click.pass_context
def cli(ctx, verbose, quiet):
    """PocketVina: GPU docking into the pockets P2Rank predicts, and the results it writes.

    Exit codes: 0 ok, 1 usage or configuration error, 2 partial failure (some proteins or engine calls failed),
    3 nothing docked.
    """
    ctx.obj = dict(log_level=logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO)


def main(args: Optional[Sequence[str]] = None) -> None:
    """Entry point of the ``pocketvina`` command (and ``python -m pocketvina``)."""
    cli.main(args=list(args) if args is not None else None)


# --- dock ----------------------------------------------------------------------------------------------------------

def _read_toml(path: str) -> dict:
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        try:
            import tomli as tomllib
        except ImportError:
            raise CliError(f"--config {path}: reading TOML needs Python 3.11 or later, or the tomli package "
                           "(pip install tomli)") from None
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:  # TOML is UTF-8
        raise CliError(f"--config {path}: not valid TOML: {e}") from None


def _cast_silently(param: click.Parameter, value) -> bool:
    """A TOML value that click would turn into another one without an error, or fail on with a traceback: a boolean
    for an option that is not a flag (``true`` -> 1), anything but a boolean for a flag (``1``), or a fraction for an
    integer option (``2.5`` -> 2)."""
    for v in value if isinstance(value, list) else [value]:
        if isinstance(v, bool) != bool(getattr(param, "is_flag", False)):
            return True
        if isinstance(v, float) and not v.is_integer() and isinstance(param.type, click.types.IntParamType):
            return True
    return False


def _config_file_options(ctx: click.Context, path: str) -> dict:
    """The dock options set in a TOML file (option names in snake_case, or ``PocketVinaConfig`` field names), checked
    and converted like the command-line flags; relative paths are relative to the file's folder."""
    data = _read_toml(path)
    base = os.path.dirname(os.path.abspath(path))
    params = {p.name: p for p in ctx.command.params if p.name != "config"}
    out, keys = {}, {}
    for key, value in data.items():
        name = CONFIG_ALIASES.get(key, key)
        if name not in params:
            raise CliError(f"--config {path}: unknown key {key!r} (keys: {', '.join(sorted(params))})")
        if name in keys:
            raise CliError(f"--config {path}: {keys[name]!r} and {key!r} set the same option")
        keys[name] = key
        if name in PATH_OPTIONS and isinstance(value, str):
            value = os.path.join(base, os.path.expanduser(value))
        elif name == "gpus" and isinstance(value, list):
            value = ",".join(str(v) for v in value)
        elif name == "box" and isinstance(value, (int, float)) and not isinstance(value, bool):
            value = [value] * 3
        if (not isinstance(value, (str, int, float, list)) or (isinstance(value, list) and name != "box")
                or (name in PATH_OPTIONS and not isinstance(value, str))  # click.Path would os.stat() a number
                or _cast_silently(params[name], value)):
            raise CliError(f"--config {path}: {key} = {value!r} is not a valid value")  # a table, a date, ...
        try:
            out[name] = params[name].type_cast_value(ctx, value)
        except click.BadParameter as e:
            raise CliError(f"--config {path}: {key}: {e.message}") from None
    return out


def _dock_options(ctx: click.Context, opts: dict) -> dict:
    """The dock options: flags given on the command line, else the ``--config`` file's value, else the default."""
    opts = dict(opts)
    config = opts.pop("config")
    from_file = _config_file_options(ctx, config) if config else {}
    given = {name for name in opts
             if ctx.get_parameter_source(name) not in (None, ParameterSource.DEFAULT, ParameterSource.DEFAULT_MAP)}
    # the proteins are one setting of two kinds: one given here replaces the file's, whichever kind that is
    if "proteins" in given:
        from_file.pop("database_path", None)
        from_file.pop("protein_list", None)
    if given & {"database_path", "protein_list"}:
        from_file.pop("proteins", None)
    for name, value in from_file.items():
        if name not in given:
            opts[name] = value
    if opts["proteins"] is not None and (opts["database_path"] is not None or opts["protein_list"] is not None):
        raise click.UsageError("--proteins and --database-path/--protein-list exclude each other")
    if opts["proteins"] is None and (opts["database_path"] is None or opts["protein_list"] is None):
        raise click.UsageError("give --proteins (or --database-path with --protein-list), here or in --config")
    for name in ("ligands", "out"):
        if opts[name] is None:
            raise click.UsageError(f"missing --{name} (here or in --config)")
    return opts


def _dock_config(opts: dict, protein_list_file: Optional[str]) -> PocketVinaConfig:
    kw = {field: opts[name] for name, field in CONFIG_FIELDS.items() if opts[name] is not None}
    if opts["seed"] is not None:
        kw["seed"] = None if opts["seed"] == RANDOM_SEED else opts["seed"]
    pocket = {}
    if opts["box"] is not None:
        pocket.update(zip(("size_x", "size_y", "size_z"), (_plain_number(v) for v in opts["box"])))
    if opts["thread"] is not None:
        pocket["thread"] = opts["thread"]
    if pocket:
        kw["pocket_config"] = PocketConfig(**pocket)
    p2rank = {k: opts[f"p2rank_{k}"] for k in ("model", "threads") if opts[f"p2rank_{k}"] is not None}
    if p2rank:
        kw["p2rank_config"] = P2RankConfig(**p2rank)
    return PocketVinaConfig(protein_paths=protein_list_file, smiles_file=opts["ligands"], output_dir=opts["out"],
                            output_format=opts["format"] or "atlas", **kw)


def _manifest_stamp(rd, protein_id: str) -> Optional[tuple]:
    """Identity of a protein's manifest file (None: not done); a commit replaces the file, so the stamp changes."""
    try:
        st = os.stat(rd.protein_manifest_file(protein_id))
    except OSError:
        return None
    return st.st_ino, st.st_mtime_ns, st.st_size


def _dock_atlas(pv: PocketVinaGPU, force: bool) -> int:
    with _user_errors():  # pyarrow missing: a message (the "files" format works without it)
        from . import pipeline
        from .io.writer import ResultsDir

        rd = ResultsDir(pv.config.output_dir)
        listed = [p.protein_id for p in pipeline.protein_inputs(pv)]
    # by manifest file, not by name: --force with another ligand list makes done proteins not done (init_run), and
    # the run docks them again
    before = {pid: _manifest_stamp(rd, pid) for pid in rd.done_proteins()}
    error = None
    click.echo(CITATION, err=True)  # on stderr: stdout is for the run's summary
    try:
        with _user_errors():
            pipeline.run_atlas(pv, force=force)
    except RuntimeError as e:  # also RunAborted: the engine or P2Rank cannot run; or no ligand could be prepared
        error = _message(e)
        logger.debug("run stopped", exc_info=True)
    failed = dict(getattr(pv, "failed_proteins", None) or {})
    done = [pid for pid in listed if rd.is_done(pid)]
    new = [pid for pid in done if before.get(pid) is None or _manifest_stamp(rd, pid) != before[pid]]
    summaries = {pid: (rd.protein_manifest(pid) or {}).get("summary") or {} for pid in done}
    n_pose = sum(_int(s.get("n_with_pose")) for s in summaries.values())
    calls_failed = sum(_int(summaries[pid].get("engine_runs_failed")) for pid in new)
    click.echo(f"results: {rd.path}")
    click.echo(f"proteins: {len(new)} docked, {len(done) - len(new)} done before, {len(failed)} failed "
               f"(of {len(listed)})")
    click.echo(f"best poses: {n_pose}")
    for pid, reason in failed.items():
        click.echo(f"failed: {pid}: {reason}")
    if calls_failed:
        click.echo(f"engine calls failed: {calls_failed} (their ligands are engine_error runs; see the log)")
    if error:
        click.echo(f"error: {error}", err=True)
    if error or failed or calls_failed:
        if failed or error:
            click.echo("rerun the same command to retry what failed (proteins done are skipped)", err=True)
        return EXIT_PARTIAL if n_pose else EXIT_NOTHING
    return EXIT_OK if n_pose else EXIT_NOTHING


def _dock_files(pv: PocketVinaGPU) -> int:
    import pandas as pd

    cfg = pv.config
    with _user_errors():
        pv.execute_docking()
    path = os.path.join(cfg.output_dir, "combined_results.csv")
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = pd.DataFrame(columns=["protein"])
    docked = {_POCKET_SUFFIX.sub("", str(p)) for p in df.get("protein", pd.Series(dtype=object)).dropna()}
    n_listed = len(read_protein_list(cfg.protein_paths) if cfg.protein_paths else read_database_list(cfg.protein_list))
    click.echo(f"output: {cfg.output_dir}")
    click.echo(f"combined results: {path} ({len(df)} rows)")
    click.echo(f"proteins with poses: {len(docked)} of {n_listed}")
    if df.empty:
        return EXIT_NOTHING
    return EXIT_PARTIAL if len(docked) < n_listed else EXIT_OK


DOCK_EPILOG = """\b
--config run.toml: the options above as keys, in snake_case (or PocketVinaConfig field names); paths are relative to
the file's folder; flags given on the command line override it. For example:
  proteins = "proteins.txt"
  ligands = "ligands.csv"
  out = "results"
  top_pockets = 5
  box = [20, 20, 20]
  seed = 42
  procs_per_gpu = "auto"
\b
Exit codes: 0 ok; 1 usage or configuration error; 2 some proteins or engine calls failed (rerun to retry them:
proteins done are skipped); 3 nothing docked.
"""


@cli.command(epilog=DOCK_EPILOG)
@click.option("--proteins", type=click.Path(exists=True),
              help="Protein list file (one structure per line, relative to the list's folder), a folder of structures "
                   "(*.pdb, *.ent, *.cif, *.mmcif, *.pdbqt, optionally .gz) or one structure file.")
@click.option("--database-path", type=click.Path(exists=True, file_okay=False),
              help="Database workflow (precomputed pockets), with --protein-list, instead of --proteins.")
@click.option("--protein-list", type=click.Path(exists=True, dir_okay=False),
              help="Database workflow: CSV of (database, id) rows with a header.")
@click.option("--ligands", type=click.Path(exists=True, dir_okay=False),
              help="Ligand CSV: a header row, then molecule_name,smiles (.txt with the same content works too).")
@click.option("--out", type=click.Path(file_okay=False), help="Results directory (atlas) or output folder (files).")
@click.option("--format", type=click.Choice(["atlas", "files"]),
              help="Output layout: atlas (pose files and Parquet tables, resumable) or files (one PDBQT per ligand "
                   "and pocket, combined_results.csv).  [default: atlas]")
@click.option("--top-pockets", type=click.IntRange(1, MAX_TOP_POCKETS),
              help="Dock the N best-ranked pockets.  [default: all]")
@click.option("--pockets-dir", type=click.Path(exists=True, file_okay=False),
              help="atlas: folder of P2Rank predictions <protein>.csv (e.g. from P2RankProcessor): those proteins "
                   "are docked into them without running P2Rank.")
@click.option("--pockets", help="Dock only these pocket ranks, e.g. 10, 10-12 or 1,3,10-12 (not with --top-pockets).  "
                                "[default: all]")
@click.option("--box", nargs=3, type=_BoxSizeType(), metavar="X Y Z",
              help="Docking box size (A).  [default: 20 20 20]")
@click.option("--thread", type=THREAD_TYPE, help="Engine threads.  [default: 8000]")
@click.option("--num-modes", type=click.IntRange(1), help="Poses per ligand and pocket.  [default: 20]")
@click.option("--energy-range", type=click.FloatRange(0), help="kcal/mol above the best pose.  [default: 20]")
@click.option("--seed", type=_SeedType(), help="Seed of the ligand embedding and the engine, or 'random'.  "
                                               "[default: 42]")
@click.option("--ligand-prep", type=click.Choice(["standard", "legacy"]),
              help="standard (largest fragment, seeded) or legacy (pre-v0.2, to regenerate old results).  "
                   "[default: standard]")
@click.option("--ligands-per-call", type=click.IntRange(0),
              help="atlas: ligands per engine call, 0: all ligands of a pocket in one call.  "
                   f"[default: --batch-size if set, else {DEFAULT_LIGANDS_PER_CALL}]")
@click.option("--batch-size", type=click.IntRange(1),
              help="files: ligands per batch folder and engine call (atlas: as --ligands-per-call if that is not "
                   "set).  [default: 10]")
@click.option("--gpus", help="atlas: the GPU to dock on, an nvidia-smi index or UUID; 'all' or a comma-separated list "
                             "for several.  [default: the first visible GPU]")
@click.option("--procs-per-gpu", type=_ProcsType(),
              help="atlas: pockets docked at the same time on the GPU, or 'auto' (as many as its free memory holds).  "
                   "[default: auto]")
@click.option("--vram-per-proc", type=click.IntRange(1), metavar="MIB",
              help=f"GPU memory one engine needs.  [default: {gpu.DEFAULT_VRAM_PER_PROC_MIB}]")
@click.option("--engine-timeout", type=click.FloatRange(0, min_open=True), metavar="S",
              help="Seconds one engine call may take.  [default: no limit]")
@click.option("--posebusters/--no-posebusters", default=None,
              help="atlas: check the best poses with PoseBusters while docking.  [default: no]")
@click.option("--posebusters-workers", type=click.IntRange(1), help="PoseBusters worker processes.")
@click.option("--tmp-dir", type=click.Path(file_okay=False),
              help="Folder for temp files.  [default: $POCKETVINA_TMPDIR, $SLURM_TMPDIR, $TMPDIR, /tmp]")
@click.option("--p2rank-model", help="P2Rank model.  [default: alphafold_conservation_hmm]")
@click.option("--p2rank-threads", type=click.IntRange(1), help="P2Rank threads.  [default: 4]")
@click.option("--gzip-level", type=click.IntRange(-1, 9), help="atlas: pose file compression.  [default: 6]")
@click.option("--config", type=click.Path(exists=True, dir_okay=False), metavar="RUN.TOML",
              help="TOML file with these options (see below).")
@click.option("--force", is_flag=True,
              help="atlas: replace the run definition of a results directory that holds a run with other settings "
                   "or ligands.")
@click.pass_context
def dock(ctx, **cli_opts):
    """Dock ligands into the P2Rank pockets of proteins.

    The options mirror PocketVinaConfig. The atlas format skips the proteins already in --out, so a stopped run
    resumes where it was; the effective settings are stored in its MANIFEST.json.
    """
    opts = _dock_options(ctx, cli_opts)
    list_dir = None
    # SIGTERM unwinds this block too (the run's own handler would kill the process before the finally clause)
    with engine.exit_on_sigterm():
        try:
            protein_list_file = opts["proteins"]
            if protein_list_file is not None and (os.path.isdir(protein_list_file)
                                                  or _is_structure_file(protein_list_file)):
                with _user_errors():
                    paths = structure_paths(protein_list_file)
                    list_dir = engine.resolve_tmp_dir(opts["tmp_dir"])
                protein_list_file = os.path.join(list_dir, "proteins.txt")
                with open(protein_list_file, "w") as f:
                    f.writelines(f"{p}\n" for p in paths)
            with _user_errors():
                config = _dock_config(opts, protein_list_file)
                pv = PocketVinaGPU(config)
            _setup_logging(ctx)  # after PocketVinaGPU: it adds the "files" format's pocketvina_summary.log
            logger.debug("dock settings: %s", {k: v for k, v in opts.items() if v is not None})
            if config.output_format == "atlas":
                code = _dock_atlas(pv, bool(opts["force"]))
            else:
                if opts["force"]:
                    logger.warning("--force applies to the atlas format only")
                code = _dock_files(pv)
        finally:
            if list_dir is not None:
                shutil.rmtree(list_dir, ignore_errors=True)
    ctx.exit(code)


# --- bust, show, export --------------------------------------------------------------------------------------------

@cli.command()
@_results_option
@click.option("--select", type=click.Choice(BUST_SELECT), default="best", show_default=True,
              help="best: the stored best pose of each ligand; top_k: its K best modes over all pockets; "
                   "per_pocket: its K best modes in each pocket.")
@click.option("--top-k", type=click.IntRange(1), default=1, show_default=True, help="K of top_k and per_pocket.")
@click.option("--config", "pb_config", default="dock", show_default=True,
              help="PoseBusters config (dock, dock_fast, ...).")
@click.option("--workers", type=click.IntRange(1), help="Worker processes.  [default: the CPUs available, <= 16]")
@click.option("--timeouts", default="30,300", show_default=True, callback=_timeouts_option,
              help="Seconds per pose for each attempt, comma-separated ('none': no limit); a pose still running "
                   "after the last one is a 'timeout' row.")
@click.option("--overwrite", is_flag=True, help="Check proteins again that have PoseBusters results.")
@click.option("--protein", "proteins", multiple=True, help="Only this protein (repeatable).  [default: all]")
@click.option("--tmp-dir", type=click.Path(file_okay=False), help="Folder for temp files.")
@click.pass_context
def bust(ctx, results, select, top_k, pb_config, workers, timeouts, overwrite, proteins, tmp_dir):
    """Check docked poses with PoseBusters (pip install posebusters).

    Writes tables/posebusters/<protein_id>.parquet; proteins that have it already are skipped unless --overwrite.
    """
    _setup_logging(ctx)
    with _user_errors():
        from . import bust as pb
        from .io.results import Results

        pb.require_posebusters()
        done = pb.bust_results(results, select=select, top_k=top_k, config=pb_config, workers=workers,
                               timeouts=timeouts, proteins=list(proteins) or None, overwrite=overwrite,
                               tmp_dir=tmp_dir)
        with Results(results) as res:
            pids = list(proteins) or res.proteins()
            status = Counter(res.posebusters(pids, columns=["pb_status"])["pb_status"])
    click.echo(f"PoseBusters ({pb_config}, {select}): {len(done)} proteins checked, {len(pids) - len(done)} had "
               "results already")
    click.echo(f"pb_status: {_counts(status) or 'no rows'}")


def _ligand_arg(res, ligand: str):
    """A ligand_id, or a ligand_idx when no ligand has that id."""
    try:
        res.ligand_idx(ligand)
        return ligand
    except KeyError:
        if ligand.strip().isdigit():
            return int(ligand)
        raise


@cli.command()
@_results_option
@click.option("--protein", required=True, help="protein_id.")
@click.option("--ligand", required=True, help="ligand_id, or ligand_idx (row of tables/ligands.parquet).")
@click.option("--pocket", type=click.IntRange(1), help="Pocket (P2Rank rank).  [default: the best pose]")
@click.option("--mode", type=click.IntRange(1), help="Mode of --pocket.  [default: 1]")
@click.option("--out", type=click.Path(dir_okay=False),
              help="HTML file to write.  [default: <protein>-<ligand>.html here]")
@click.option("--backend", type=click.Choice(VIEW_BACKENDS), default="auto", show_default=True,
              help="Viewer: molstar (MolViewSpec), py3dmol, or auto (molstar if installed).")
@click.pass_context
def show(ctx, results, protein, ligand, pocket, mode, out, backend):
    """Write a standalone HTML page with a 3D view of a pose and print its path.

    The page embeds the structures; it loads the viewer's JavaScript from a CDN when opened.
    """
    _setup_logging(ctx)
    if mode is not None and pocket is None:
        raise click.UsageError("--mode needs --pocket (without --pocket the best pose is shown)")
    with _user_errors():
        from . import viz
        from .io.results import Results

        with Results(results) as res:
            lig = _ligand_arg(res, ligand)
            lid = res.ligand_id(lig)
            if out is None:
                where = "" if pocket is None else f"-pocket{pocket}-mode{mode or 1}"
                out = _UNSAFE_CHARS.sub("_", f"{protein}-{lid}{where}")[:200] + ".html"
            viz.to_html(res, protein, lig, pocket, mode or 1, path=out, backend=backend)
    click.echo(os.path.abspath(os.path.expanduser(out)))  # as viz.to_html wrote it


@cli.command()
@_results_option
@click.option("--what", type=click.Choice(EXPORT_WHAT), default="best", show_default=True,
              help="best: the best pose per protein and ligand; all: every engine output.")
@click.option("--format", "fmt", type=click.Choice(EXPORT_FORMATS), default="sdf", show_default=True,
              help="sdf, pdbqt, or csv (best: best_poses.csv; all: the legacy combined_results.csv).")
@click.option("--out", required=True, type=click.Path(file_okay=False), help="Folder to write into.")
@click.option("--protein", "proteins", multiple=True, help="Only this protein (repeatable).  [default: all]")
@click.pass_context
def export(ctx, results, what, fmt, out, proteins):
    """Export poses or tables as plain files (one folder per protein)."""
    _setup_logging(ctx)
    with _user_errors():
        from .io.results import Results

        with Results(results) as res:
            files = res.export(out, what=what, fmt=fmt, protein=list(proteins) or None)
    click.echo(f"{len(files)} files written to {os.path.abspath(os.path.expanduser(out))}")  # as Results.export


# --- info, query ---------------------------------------------------------------------------------------------------

def _staged_ids(path: str) -> set:
    """Protein ids with a pose file or a proteins/pairs table, committed or not."""
    ids = set()
    for folder, suffix in (("poses", ".poses"), (os.path.join("tables", "proteins"), ".parquet"),
                           (os.path.join("tables", "pairs"), ".parquet")):
        try:
            names = os.listdir(os.path.join(path, folder))
        except OSError:
            continue
        ids.update(n[:-len(suffix)] for n in names if n.endswith(suffix))
    return ids


def results_info(path, proteins=None) -> dict:
    """What ``info`` prints: run settings, counts, and the proteins with problems. ``proteins``: a protein list,
    folder or structure file whose proteins that are not done are listed (they failed, or are not docked yet)."""
    from .io.results import Results
    from .io.writer import ResultsDir

    with Results(path) as res:
        manifest = res.manifest()
        pids = res.proteins()
        rd = ResultsDir(res.path)
        summaries = {pid: (rd.protein_manifest(pid) or {}).get("summary") or {} for pid in pids}
        ligands = Counter(res.ligands(columns=["status"])["status"])
        pb = Counter(res.posebusters(columns=["pb_status"])["pb_status"])
        runs, problems = Counter(), {}
        for pid, s in summaries.items():
            runs.update(s.get("runs_by_status") or {})
            notes = []
            if s.get("status") not in (None, "ok"):
                notes.append(f"status {s['status']}")
            if _int(s.get("engine_runs_failed")):
                notes.append(f"{s['engine_runs_failed']} of {s.get('engine_runs')} engine runs failed")
            if "n_with_pose" in s and not _int(s["n_with_pose"]):
                notes.append("no pose")
            if notes:
                problems[pid] = "; ".join(notes)
        info = dict(results=res.path, format=manifest.get("format"), format_version=manifest.get("format_version"),
                    pocketvina_version=manifest.get("pocketvina_version"), created=manifest.get("created"),
                    params=manifest.get("params") or {},
                    ligands=dict(total=sum(ligands.values()), by_status=dict(ligands)),
                    proteins=len(pids), pocket_runs=dict(total=sum(runs.values()), by_status=dict(runs)),
                    pairs=dict(total=sum(_int(s.get("n_pairs")) for s in summaries.values()),
                               with_pose=sum(_int(s.get("n_with_pose")) for s in summaries.values())),
                    posebusters=dict(pb) or None, problems=problems,
                    incomplete=sorted(_staged_ids(res.path) - set(pids)))
    if proteins is not None:
        done = set(pids)
        info["not_done"] = [pid for pid in _listed_protein_ids(proteins) if pid not in done]
    return info


def _echo_info(info: dict) -> None:
    click.echo(f"results: {info['results']}")
    click.echo(f"format: {info['format']} {info['format_version']} (pocketvina {info['pocketvina_version']}, "
               f"created {info['created']})")
    click.echo("params:")
    for key, value in sorted(info["params"].items()):
        click.echo(f"  {key}: {json.dumps(value)}")
    lig = info["ligands"]
    click.echo(f"ligands: {lig['total']} ({_counts(lig['by_status'])})")
    click.echo(f"proteins: {info['proteins']} done")
    runs = info["pocket_runs"]
    click.echo(f"pocket runs: {runs['total']}" + (f" ({_counts(runs['by_status'])})" if runs["total"] else ""))
    click.echo(f"best poses: {info['pairs']['with_pose']} of {info['pairs']['total']} protein x ligand pairs")
    if info["posebusters"]:
        click.echo(f"posebusters: {_counts(info['posebusters'])}")
    if info["problems"]:
        click.echo(f"problems ({len(info['problems'])} proteins):")
        for pid, note in info["problems"].items():
            click.echo(f"  {pid}: {note}")
    if info["incomplete"]:
        click.echo(f"incomplete ({len(info['incomplete'])}: files without a manifest, stopped while committing or "
                   "being committed now):")
        for pid in info["incomplete"]:
            click.echo(f"  {pid}")
    if "not_done" in info:
        click.echo(f"not done ({len(info['not_done'])} of the --proteins list: failed, or not docked yet):")
        for pid in info["not_done"]:
            click.echo(f"  {pid}")


@cli.command()
@_results_option
@click.option("--proteins", type=click.Path(exists=True),
              help="Protein list, folder or structure: also list its proteins that are not done (failed, or not "
                   "docked yet).")
@click.option("--json", "as_json", is_flag=True, help="Print JSON.")
@click.pass_context
def info(ctx, results, proteins, as_json):
    """Summarize a results directory: settings, counts and proteins with problems.

    Proteins that failed are not committed, so a results directory alone does not know them: --proteins lists the
    proteins of the run's list that are not done.
    """
    _setup_logging(ctx)
    with _user_errors():
        data = results_info(results, proteins)
    if as_json:
        click.echo(json.dumps(data, indent=2, sort_keys=True))
    else:
        _echo_info(data)


QUERY_EPILOG = """\b
Views: ligands, proteins, pockets, pocket_runs, modes, pairs, and posebusters when there are PoseBusters results.
  pocketvina query --results R "SELECT ligand_id, vina FROM pairs ORDER BY vina LIMIT 10"
  pocketvina query --results R --csv modes.csv "SELECT * FROM modes WHERE affinity < -9"
  pocketvina query best --results R --top 20
"""


@cli.command(epilog=QUERY_EPILOG)
@click.argument("sql")
@_results_option
@click.option("--top", type=click.IntRange(0), default=20, show_default=True, help="query best: number of pairs.")
@click.option("--protein", "proteins", multiple=True, help="query best: only this protein (repeatable).")
@click.option("--csv", "csv_path", type=click.Path(dir_okay=False, allow_dash=True),
              help="Write the rows as CSV to this file ('-': standard output) instead of printing a table.")
@click.pass_context
def query(ctx, sql, results, top, proteins, csv_path):
    """Run DuckDB SQL over the results tables, or "query best": the best protein x ligand pairs.

    query best: best pose per protein and ligand, best Vina score first, with ligand efficiency (and PoseBusters
    status when checked).
    """
    _setup_logging(ctx)
    best = sql.strip().lower() == "best"
    if not best:
        for name in ("top", "proteins"):
            if ctx.get_parameter_source(name) == ParameterSource.COMMANDLINE:
                raise click.UsageError(f"--{name.rstrip('s')} applies to 'query best' only")
    with _user_errors():
        from .io.results import Results

        with Results(results) as res:
            if best:
                from . import viz

                df = viz.table(res, top=top, protein=list(proteins) or None)
            else:
                import duckdb

                con = res.duckdb()
                try:
                    df = con.execute(sql).df()
                except duckdb.Error as e:
                    raise CliError(f"query failed: {e}") from None
                finally:
                    con.close()
    if csv_path:
        _write_csv(df, csv_path)
    else:
        _echo_table(df)


# --- import-legacy -------------------------------------------------------------------------------------------------

@cli.command("import-legacy")
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--results", required=True, type=click.Path(file_okay=False), help="Results directory to write.")
@click.option("--smiles", type=click.Path(exists=True, dir_okay=False),
              help="The run's ligand CSV (molecule_name,smiles): ligand order and best-pose SDFs.")
@click.option("--tmp-dir", type=click.Path(file_okay=False), help="Folder for temp files.")
@click.option("--gzip-level", type=click.IntRange(-1, 9), default=6, show_default=True,
              help="Pose file compression.")
@click.option("--force", is_flag=True, help="Replace another run definition in --results and import again.")
@click.pass_context
def import_legacy(ctx, output_dir, results, smiles, tmp_dir, gzip_level, force):
    """Convert a legacy PocketVina output folder (the files format) into an atlas results directory.

    Importing into a results directory that holds this import already resumes it.
    """
    _setup_logging(ctx)
    with _user_errors(), engine.exit_on_sigterm():  # SIGTERM: the importer's temp folder is removed first
        from .io.importer import import_legacy as run_import
        from .io.results import Results

        path = run_import(output_dir, results, smiles, tmp_dir=tmp_dir, gzip_level=gzip_level, force=force)
        with Results(path) as res:
            n = len(res.proteins())
    click.echo(f"{path}: {n} proteins")


# --- the PocketVina Atlas ------------------------------------------------------------------------------------------

class _NumberOrAll(click.ParamType):
    """A pocket or mode number (1, 2, ...) or 'all'."""

    name = "N|all"

    def convert(self, value, param, ctx):
        text = str(value).strip().lower()
        if text == "all":
            return "all"
        if text.isdigit() and int(text) >= 1:
            return int(text)
        self.fail(f"{value!r} is not a number (1, 2, ...) or 'all'", param, ctx)


def _atlas_option(f):
    return click.option("--atlas", "location", metavar="PATH_OR_URL",
                        help="The Atlas: a release folder (e.g. one written by `pocketvina atlas download`) or its "
                             "URL.  [default: $POCKETVINA_ATLAS, else the public release]")(f)


def _atlas(location):
    from .atlas import Atlas

    return Atlas(location)


# the Python hints in the Atlas's error messages, as the atlas commands say them
_ATLAS_SEARCH_HINT = re.compile(r"atlas\.search\(('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\)")
_ATLAS_HINTS = ((re.compile(r"Atlas\('([^']*)'\)"), r"--atlas \1"),
                (re.compile(r"Atlas\(\) for the public release"), "no --atlas for the public release"),
                (re.compile(r"Atlas\(\) reads every protein"), "The public release (no --atlas) has every protein"),
                (re.compile(r"\bdownload\(\)"), "pocketvina atlas download"),
                (re.compile(r"\bfmt='(\w+)'"), r"--fmt \1"))


@contextlib.contextmanager
def _atlas_errors(location):
    """``_user_errors`` for the atlas commands: also OSError (e.g. no connection to the Atlas), and the hints of the
    messages name the command line ("pocketvina atlas search ABL", "--atlas PATH", "--fmt pdbqt"), not Python."""
    try:
        with _user_errors(also=(OSError,)):
            yield
    except CliError as e:
        where = f" --atlas {shlex.quote(location)}" if location else ""
        e.message = _ATLAS_SEARCH_HINT.sub(
            lambda m: f"pocketvina atlas search {shlex.quote(ast.literal_eval(m.group(1)))}{where}", e.message)
        for python, command_line in _ATLAS_HINTS:
            e.message = python.sub(command_line, e.message)
        raise


def _pair_options(f):
    f = click.option("--mode", type=_NumberOrAll(), metavar="N|all",
                     help="A mode, or 'all'.  [default: 1, the best mode of the pocket]")(f)
    return click.option("--pocket", type=_NumberOrAll(), metavar="N|all",
                        help="A pocket (P2Rank rank), or 'all': each pocket.  [default: the pocket of the best "
                             "pose]")(f)


@cli.group("atlas")
def atlas_group():
    """The PocketVina Atlas: docking results of drugs against the human proteome, computed in advance (in Python:
    pocketvina.Atlas).

    PROTEIN is any identifier of a protein (gene, UniProt accession, entry name, Entrez, HGNC or Ensembl ID), LIGAND
    any identifier of a drug (name, synonym, PVL ID, ChEMBL ID, catalogue number, InChIKey or SMILES).
    """


@atlas_group.command("search")
@click.argument("query")
@click.option("--top", type=click.IntRange(0), default=ATLAS_SEARCH_TOP, show_default=True, help="Number of rows.")
@_atlas_option
@click.pass_context
def atlas_search(ctx, query, top, location):
    """Print the proteins and drugs whose identifiers match QUERY (also the start or part of a name), best match
    first: kind, id (UniProt accession or PVL ID), name, the identifier that matched and its text."""
    _setup_logging(ctx)
    with _atlas_errors(location):
        df = _atlas(location).search(query, top=top)
    _echo_table(df)


@atlas_group.command("show")
@click.argument("protein")
@click.option("--ligand", help="A drug: the page with this pair selected.")
@_pair_options
@click.option("--out", type=click.Path(dir_okay=False),
              help="HTML file to write.  [default: PV_<UniProt>.html, or the pair's file name, here]")
@_atlas_option
@click.pass_context
def atlas_show(ctx, protein, ligand, pocket, mode, out, location):
    """Write the protein's page as an HTML file and print its path: the 3D view beside its best drugs (Top hits),
    Selected pair, Pockets and Download, with the protein's information. --ligand selects a pair (--pocket and --mode
    its poses)."""
    _setup_logging(ctx)
    if ligand is None and (pocket is not None or mode is not None):
        raise click.UsageError("--pocket and --mode need --ligand (they pick poses of a pair)")
    with _atlas_errors(location):
        from .selection import name_part

        atlas = _atlas(location)
        if ligand is None:
            uniprot = atlas._uniprot(protein)
            path = atlas.show(uniprot, path=out or f"PV_{name_part(uniprot)}.html")
        else:
            pair = atlas.select(protein, ligand, pocket=pocket, mode=1 if mode is None else mode)
            path = pair.show(path=out or f"{pair.name}.html")
    click.echo(path)


@atlas_group.command("export")
@click.argument("protein")
@click.argument("ligand")
@_pair_options
@click.option("--fmt", type=click.Choice(ATLAS_EXPORT_FORMATS), default="sdf", show_default=True,
              help="sdf, or pdbqt (the docked models as the engine wrote them).")
@click.option("--protein", "with_protein", is_flag=True,
              help="Also write the protein (the AlphaFold model the drugs were docked into) as a PDB file.")
@click.option("--out", type=click.Path(),
              help="File (.sdf or .pdbqt) or folder to write into.  [default: the pair's file name, here]")
@_atlas_option
@click.pass_context
def atlas_export(ctx, protein, ligand, pocket, mode, fmt, with_protein, out, location):
    """Write poses of a protein x drug pair (by default its best pose) into one SDF or PDBQT file, and print its
    path; --protein also writes the protein."""
    _setup_logging(ctx)
    with _atlas_errors(location):
        pair = _atlas(location).select(protein, ligand, pocket=pocket, mode=1 if mode is None else mode)
        if out is not None and not os.path.splitext(out)[1]:
            os.makedirs(out, exist_ok=True)  # a folder
        paths = [pair.export(out, fmt=fmt)]
        if with_protein:
            paths.append(pair.export_protein(os.path.dirname(paths[0])))
    for path in paths:
        click.echo(path)


@atlas_group.command("download")
@click.argument("proteins", nargs=-1, required=True)
@click.option("--out", required=True, type=click.Path(file_okay=False),
              help="Folder to write: a new one, or one written by an earlier download (it is completed).")
@_atlas_option
@click.pass_context
def atlas_download(ctx, proteins, out, location):
    """Download what the Atlas has on PROTEINS into a folder that --atlas FOLDER (and Python's Atlas(FOLDER)) reads
    offline, and print its path. The size is printed before it starts; the files are checked against the release's
    sha256."""
    _setup_logging(ctx)
    with _atlas_errors(location):
        path = _atlas(location).download(list(proteins), out)
    click.echo(path)


# --- setup ---------------------------------------------------------------------------------------------------------

@cli.group("setup")
def setup_group():
    """Install what runs need besides the Python package (once per machine or user)."""


@setup_group.command("p2rank")
@click.option("--dir", "root", type=click.Path(file_okay=False), metavar="DIR",
              help=f"Install into DIR/{p2rank.CACHE_NAME}; runs use it once P2RANK_HOME is set to that folder.  "
                   "[default: the user cache, ~/.cache/pocketvina]")
@click.option("--from", "archive", type=click.Path(exists=True, dir_okay=False), metavar="TAR.GZ",
              help=f"Unpack this p2rank_{p2rank.P2RANK_VERSION}.tar.gz (downloaded elsewhere) instead of downloading "
                   "it.")
@click.option("--force", is_flag=True, help="Install again, even if a P2Rank is found.")
@click.option("--check", is_flag=True, help="Only show the P2Rank and Java that runs use; install nothing.")
@click.pass_context
def setup_p2rank(ctx, root, archive, force, check):
    """Download P2Rank (the pocket predictor) once, and check the Java 17+ it needs.

    Nothing is installed if a P2Rank is found: $P2RANK_HOME, P2Rank's `prank` launcher on PATH, the
    p2rank_2.5 folder of an older source install, or the user cache ($XDG_CACHE_HOME/pocketvina or
    ~/.cache/pocketvina). Otherwise P2Rank 2.5.1 (263 MiB) is downloaded from its GitHub release, checked against
    its pinned sha256 and unpacked so that the folder appears complete or not at all. A run without P2Rank downloads
    it too; this does it ahead of a run, e.g. on a login node that has internet access. Without internet access,
    download the release file elsewhere and pass it with --from.

    Exit code 1 if P2Rank is missing (--check) or cannot be installed, or if there is no Java 17 or later.
    """
    _setup_logging(ctx)
    if check and (root or archive or force):
        raise CliError("--check installs nothing: use it without --dir, --from and --force")
    problems = []
    found = p2rank.find_p2rank()
    if check or (found is not None and not (root or archive or force)):
        if found is not None:
            click.echo(f"P2Rank: {found.home} (found: {P2RANK_SOURCES[found.source]})")
        else:
            problems.append(p2rank.missing_message())
    else:
        target = os.path.join(os.path.abspath(os.path.expanduser(root)) if root else p2rank.cache_root(),
                              p2rank.CACHE_NAME)
        there = p2rank.is_p2rank_home(target) and not force
        try:
            home = os.path.dirname(p2rank.ensure_p2rank(root=root, archive=archive, force=force,
                                                        progress=sys.stderr.isatty()))
        except p2rank.P2RankError as e:
            raise CliError(str(e)) from e
        click.echo(f"P2Rank: {home} ({'installed already' if there else f'P2Rank {p2rank.P2RANK_VERSION} installed'})")
        used = p2rank.p2rank_home()
        if used != home:
            click.echo(f"Set P2RANK_HOME={home} for runs to use it" + (f" (now they use {used})." if used else "."))
    try:
        click.echo(f"Java: {p2rank.check_java()} ({p2rank.java_command()})")
    except p2rank.JavaError as e:
        problems.append(str(e))
    for problem in problems:
        click.echo(f"Error: {problem}", err=True)
    if problems:
        ctx.exit(EXIT_USAGE)


# --- bench gpu -----------------------------------------------------------------------------------------------------

@cli.group()
def bench():
    """Benchmarks."""


def _nvidia_smi(args: Sequence[str]) -> Optional[str]:
    try:
        return subprocess.run(list(args), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              timeout=NVIDIA_SMI_TIMEOUT_S, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _number(text: str) -> Optional[float]:
    try:
        value = float(text)
    except ValueError:  # e.g. "[N/A]"
        return None
    return value if math.isfinite(value) else None


class _GpuSampler:
    """Samples one GPU with nvidia-smi every ``interval`` seconds, in a thread, while a configuration runs: its
    utilisation and memory use, and the memory of each engine of the benchmark (a process whose environment holds
    ``pipeline.SLOT_ENV=<marker>...``; Linux ``/proc``)."""

    def __init__(self, g: gpu.GPU, needle: bytes, interval: float):
        self.uuid, self.select, self.needle, self.interval = g.uuid, gpu.device_id(g), needle, interval
        self.util: List[float] = []
        self.mem_used: List[float] = []
        self.proc_mem: Dict[int, float] = {}  # pid -> peak MiB
        self.max_procs = 0
        self._ours: Dict[int, bool] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pocketvina-bench-sampler", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def _run(self):
        while not self._stop.wait(self.interval):
            self.sample()

    def _is_ours(self, pid: int) -> bool:
        if pid not in self._ours:
            try:
                with open(f"/proc/{pid}/environ", "rb") as f:
                    self._ours[pid] = any(v.startswith(self.needle) for v in f.read().split(b"\0"))
            except OSError:  # gone, or not ours to read
                self._ours[pid] = False
        return self._ours[pid]

    def sample(self) -> None:
        for line in (_nvidia_smi(NVIDIA_SMI_GPU + ("-i", self.select)) or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[0] == self.uuid:
                util, used = _number(parts[1]), _number(parts[2])
                if util is not None:
                    self.util.append(util)
                if used is not None:
                    self.mem_used.append(used)
        n = 0
        for line in (_nvidia_smi(NVIDIA_SMI_APPS) or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3 or parts[0] != self.uuid or not parts[1].isdigit() or not self._is_ours(int(parts[1])):
                continue
            n += 1
            used = _number(parts[2])
            if used is not None:
                pid = int(parts[1])
                self.proc_mem[pid] = max(used, self.proc_mem.get(pid, 0.0))
        self.max_procs = max(self.max_procs, n)

    def stats(self) -> dict:
        def stat(values, f):
            return round(f(values), 1) if values else None

        return dict(peak_vram_mib=stat(list(self.proc_mem.values()), max), procs_seen=self.max_procs,
                    gpu_util_mean=stat(self.util, lambda v: sum(v) / len(v)), gpu_util_max=stat(self.util, max),
                    gpu_mem_used_max_mib=stat(self.mem_used, max))


@dataclasses.dataclass
class _Bench:
    """Engine calls of a GPU benchmark: one protein pocket, one GPU, fixed settings."""

    work: str
    receptor: str
    pocket: int
    center: tuple
    box: tuple
    thread: int
    seed: int
    device: Optional[str]       # CUDA_VISIBLE_DEVICES of the engines (None: the environment's)
    kernel_cache: str
    marker: str                 # pipeline.SLOT_ENV prefix of this benchmark's engines
    timeout: Optional[float]

    def call(self, tag: str, lig_dir: str, stems: Dict[str, int]) -> dict:
        """Run one engine call; its run time and how many of its ligands got poses (the outputs are removed).
        ``rejected``: the engine failed because it could parse none of the call's ligands ("No valid ligands")."""
        from .pipeline import _LOG_PARSE_ERROR, SLOT_ENV

        out_dir = os.path.join(self.work, "out", tag)
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        config = engine.write_config(os.path.join(self.work, "configs", f"{tag}.txt"), self.receptor, lig_dir,
                                     out_dir, self.center, self.box, self.thread, self.pocket, seed=self.seed,
                                     opencl_binary_path=self.kernel_cache)
        env = {SLOT_ENV: f"{self.marker}{tag}"}
        if self.device is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.device
        try:
            result = engine.run_engine(config, cwd=os.path.join(self.work, "cwd", tag), env_extra=env,
                                       log_path=os.path.join(self.work, "logs", f"{tag}.log"), timeout=self.timeout)
            # killed: an output it was writing may be cut short, so none counts (as in pipeline._run_call)
            outputs = ({} if result.timed_out else
                       engine.collect_outputs(out_dir, self.receptor, self.pocket, stems, remove=True))
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
        lines = [line.strip() for line in result.log_tail.splitlines() if line.strip()]
        docked = sum(1 for v in outputs.values() if v)
        return dict(tag=tag, ligands=len(stems), docked=docked,
                    runtime_s=round(result.runtime_s, 3), returncode=result.returncode, timed_out=result.timed_out,
                    ok=result.ok, out_of_memory=gpu.is_out_of_memory(result.log_tail),
                    rejected=(not result.ok and not result.timed_out and not docked
                              and _LOG_PARSE_ERROR in result.log_tail),
                    error=None if result.ok else (lines[-1][:500] if lines else f"exit code {result.returncode}"))

    def configuration(self, n: int, dirs: List[tuple], sampler: Optional[_GpuSampler]) -> tuple:
        """Run ``n`` engine calls at the same time (``dirs``: their (ligand dir, stems)); (row, calls)."""
        from .pipeline import _kill_engines

        sampling = sampler if sampler is not None else contextlib.nullcontext()
        with sampling, ThreadPoolExecutor(max_workers=n, thread_name_prefix="pocketvina-bench") as pool:
            t0 = time.monotonic()
            futures = []
            try:
                for p, (lig_dir, stems) in enumerate(dirs):
                    futures.append(pool.submit(self.call, f"procs{n}_{p + 1}", lig_dir, stems))
                calls = [f.result() for f in futures]
            except BaseException:  # Ctrl-C, SIGTERM, a call that raised: the engines run in the pool's threads
                for f in futures:
                    f.cancel()
                # until every thread is back: a thread may start its engine after a kill (the pool waits for it)
                while not all(f.done() for f in futures):
                    _kill_engines(self.marker)
                    wait(futures, timeout=0.25)
                raise
            wall = time.monotonic() - t0
        ok = [c for c in calls if c["ok"]]
        docked = sum(c["docked"] for c in calls)
        row = dict(procs=n, wall_s=round(wall, 3), ligands_docked=docked,
                   ligands_per_hour=round(docked * 3600 / wall, 1) if wall > 0 else None,
                   proc_runtime_s=round(sum(c["runtime_s"] for c in ok) / len(ok), 3) if ok else None,
                   s_per_ligand=(round(sum(c["runtime_s"] / c["ligands"] for c in ok) / len(ok), 4)
                                 if ok else None),
                   failed_calls=len(calls) - len(ok), out_of_memory_calls=sum(c["out_of_memory"] for c in calls))
        row.update(sampler.stats() if sampler is not None else
                   dict(peak_vram_mib=None, procs_seen=None, gpu_util_mean=None, gpu_util_max=None,
                        gpu_mem_used_max_mib=None))
        return row, calls


def _bench_device(spec: Optional[str]) -> Optional[gpu.GPU]:
    gpus = gpu.list_gpus()
    if spec is None:
        return gpus[0] if gpus else None
    if not gpus:
        raise CliError(f"--gpu {spec}: nvidia-smi lists no GPU here")
    chosen = gpu.select_gpus(gpus, spec)
    if len(chosen) != 1:
        raise click.UsageError("--gpu takes one GPU (the benchmark runs on one)")
    return chosen[0]


def _bench_ligands(path: str, n: int, method: str, seed: int) -> list:
    """``n`` prepared ligands: the first ``n`` of the file, the ones that were prepared repeated as needed."""
    from .ligands import prepare_ligands, read_smiles_file

    rows = read_smiles_file(path)
    if not rows:
        raise CliError(f"no ligands in {path}")
    records = prepare_ligands(rows[:n], method=method, seed=seed)
    ok = [r for r in records if r.status == "ok"]
    if not ok:
        raise CliError(f"none of the {len(records)} ligands could be prepared "
                       f"({dict(Counter(r.status for r in records))})", EXIT_NOTHING)
    if len(ok) < n:
        logger.info("%d ligands prepared, repeated to %d", len(ok), n)
    return [dataclasses.replace(ok[j % len(ok)], ligand_idx=j) for j in range(n)]


def _bench_warmup(bench: _Bench, records: list, lig_root: str) -> dict:
    """The warm-up call, before any configuration: it builds the OpenCL kernels if the cache has none, so that no
    configuration times that (or several engines build them at once). One ligand; the engine builds the kernels only
    after it has parsed a ligand, so a ligand it rejects is followed by the next distinct one
    (``pipeline.KERNEL_BUILD_TRIES`` at most). ``CliError`` (exit code 3) if no warm-up call docks its ligand."""
    from .ligands import write_ligand_dir
    from .pipeline import KERNEL_BUILD_TRIES

    tries = []
    for rec in records:  # records repeats the ligands that were prepared
        if len(tries) == KERNEL_BUILD_TRIES:
            break
        if all(rec.pdbqt != t.pdbqt for t in tries):
            tries.append(rec)
    for n, rec in enumerate(tries, 1):
        folder = os.path.join(lig_root, f"warmup{n}")
        try:
            warmup = bench.call("warmup", folder, write_ligand_dir([rec], folder))
        except OSError as e:  # no engine
            raise CliError(str(e), EXIT_NOTHING) from None
        if warmup["docked"] or not warmup["rejected"]:
            break
        logger.info("warm-up: the engine rejected ligand %s (%s), trying the next one", rec.ligand_id,
                    warmup["error"])
    if not warmup["ok"] and not warmup["rejected"]:
        raise CliError(f"the warm-up engine call failed: {warmup['error']}", EXIT_NOTHING)
    if not warmup["docked"]:
        why = (f"the engine rejected the first {len(tries)} prepared ligands ({warmup['error']})" if warmup["rejected"]
               else "the engine exited with code 0 but wrote no pose")
        raise CliError(f"the warm-up engine call docked nothing: {why}", EXIT_NOTHING)
    return warmup


def _bench_pocket(protein: str, rank: int, center, work: str, p2rank_model: str) -> tuple:
    if center is not None:
        return tuple(center)
    from . import pipeline

    try:
        csv_path = pipeline.predict_pockets(protein, os.path.join(work, "p2rank"), P2RankConfig(model=p2rank_model))
    except RuntimeError as e:  # also RunAborted
        raise CliError(f"cannot predict the pockets of {protein}: {e} (or give the box centre with --center)") \
            from None
    pockets = pipeline.read_pockets_csv(csv_path)
    found = [p for p in pockets if p.rank == rank]
    if not found:
        raise CliError(f"{protein}: P2Rank predicted {len(pockets)} pockets, there is no pocket {rank}")
    return tuple(found[0].center)


def _check_out_folder(path: str) -> None:
    """``CliError`` unless the folder ``path`` exists or can be created, and can be written into: checked before a
    benchmark that may run for an hour writes its results there (``path`` itself is not created yet)."""
    existing = path
    while not os.path.lexists(existing) and os.path.dirname(existing) != existing:
        existing = os.path.dirname(existing)
    if not os.path.isdir(existing):
        raise CliError(f"--out {path}: {existing} is not a folder")
    if not os.access(existing, os.W_OK | os.X_OK):
        raise CliError(f"--out {path}: cannot write into {existing}")


def _echo_bench(rows: List[dict]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows, columns=list(BENCH_COLUMNS))
    for col in df.columns:  # na_rep covers float columns only: a column of None (no nvidia-smi) is printed "None"
        if df[col].isna().all():
            df[col] = "-"
    click.echo(df.to_string(index=False, na_rep="-"))


def run_bench_gpu(protein: str, ligands: str, out: str, *, n_ligands: int = 100, procs: Sequence[int] = (1, 2, 3),
                  pocket: int = 1, gpu_spec: Optional[str] = None, center=None, box=(20, 20, 20), thread: int = 8000,
                  seed: int = 42, ligand_prep: str = "standard", split: bool = False,
                  engine_timeout: Optional[float] = None, sample_interval: float = 0.5,
                  p2rank_model: str = P2RankConfig.model, tmp_dir: Optional[str] = None) -> dict:
    """The ``bench gpu`` workload; writes ``bench_gpu.csv`` and ``bench_gpu.json`` into ``out``, returns the JSON."""
    from . import pipeline
    from .io.writer import protein_id_from_path
    from .ligands import write_ligand_dir

    procs = list(procs)
    if split and max(procs) > n_ligands:
        raise click.UsageError(f"--split: {max(procs)} processes need at least as many ligands (--n-ligands "
                               f"{n_ligands})")
    out = os.path.abspath(os.path.expanduser(out))
    _check_out_folder(out)  # not after the benchmark: its results would be lost
    g = _bench_device(gpu_spec)
    device = gpu.device_id(g) if g is not None else None
    if g is None:
        logger.warning("nvidia-smi lists no GPU: the engines use the default GPU; GPU memory and utilisation are not "
                       "sampled")
    else:
        fit = gpu.fitting_procs(g, gpu.DEFAULT_VRAM_PER_PROC_MIB)
        logger.info("benchmark on GPU %d (%s, %s of %s MiB free)", g.index, g.name, g.memory_free_mib,
                    g.memory_total_mib)
        if fit is not None and max(procs) > fit:
            logger.warning("GPU %d: its free memory holds about %d engines of %d MiB; more may run out of memory",
                           g.index, fit, gpu.DEFAULT_VRAM_PER_PROC_MIB)
    protein = os.path.abspath(os.path.expanduser(protein))
    tmp = engine.resolve_tmp_dir(tmp_dir)
    marker = f"bench{os.getpid()}-{secrets.token_hex(4)}-"
    try:
        records = _bench_ligands(ligands, n_ligands, ligand_prep, seed)
        pid = protein_id_from_path(protein)
        receptor = os.path.join(tmp, "receptor", f"{pid}.pdbqt")
        os.makedirs(os.path.dirname(receptor))
        with open(receptor, "w") as f:
            f.write(pipeline.receptor_pdbqt(protein))
        box_center = _bench_pocket(protein, pocket, center, tmp, p2rank_model)
        bench = _Bench(os.path.join(tmp, "runs"), receptor, pocket, box_center, tuple(box), thread, seed, device,
                       pipeline._kernel_cache_for(device), marker, engine_timeout)
        lig_root = os.path.join(tmp, "ligands")
        warmup = _bench_warmup(bench, records, lig_root)
        logger.info("warm-up call: %.1f s", warmup["runtime_s"])
        shared = None
        rows, calls = [], []
        for n in procs:
            if split:
                dirs = [(d, write_ligand_dir(records[p * n_ligands // n:(p + 1) * n_ligands // n], d))
                        for p, d in enumerate(os.path.join(lig_root, f"split{n}_{p + 1}") for p in range(n))]
            else:
                if shared is None:
                    shared = (os.path.join(lig_root, "all"), write_ligand_dir(records, os.path.join(lig_root, "all")))
                dirs = [shared] * n
            logger.info("%d engine process%s at the same time ...", n, "es" * (n > 1))
            sampler = (_GpuSampler(g, f"{pipeline.SLOT_ENV}={marker}".encode(), sample_interval)
                       if g is not None else None)
            row, done = bench.configuration(n, dirs, sampler)
            for c in done:
                c["procs"] = n
            rows.append(row)
            calls += done
            logger.info("%d processes: %.1f s, %s ligands/hour, %d failed calls", n, row["wall_s"],
                        row["ligands_per_hour"], row["failed_calls"])
    except BaseException:
        from .pipeline import _kill_engines

        _kill_engines(marker)
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    base = next((r for r in sorted(rows, key=lambda r: r["procs"]) if r["s_per_ligand"]), None)
    for row in rows:
        row["speedup"] = (round(row["ligands_per_hour"] / base["ligands_per_hour"], 3)
                          if base and base["ligands_per_hour"] and row["ligands_per_hour"] is not None else None)
        row["slowdown"] = (round(row["s_per_ligand"] / base["s_per_ligand"], 3)
                           if base and row["s_per_ligand"] is not None else None)
    report = dict(created=datetime.now(timezone.utc).isoformat(timespec="seconds"), pocketvina_version=__version__,
                  protein=protein, protein_id=pid, ligands=os.path.abspath(ligands), n_ligands=n_ligands,
                  split=split, pocket=pocket, center=list(box_center), box=list(box), thread=thread, seed=seed,
                  ligand_prep=ligand_prep, engine=engine.engine_path(), kernel_cache=bench.kernel_cache,
                  gpu=g._asdict() if g is not None else None, sample_interval_s=sample_interval,
                  baseline_procs=base["procs"] if base else None, warmup=warmup,
                  configurations=[{c: r.get(c) for c in BENCH_COLUMNS + ("procs_seen", "out_of_memory_calls")}
                                  for r in rows],
                  calls=calls)
    os.makedirs(out, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows, columns=list(BENCH_COLUMNS)).to_csv(os.path.join(out, BENCH_CSV), index=False)
    tmp_json = os.path.join(out, f"{BENCH_JSON}.tmp-{os.getpid()}")
    with open(tmp_json, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_json, os.path.join(out, BENCH_JSON))
    return report


BENCH_EPILOG = """\b
Columns: wall_s (all processes of a configuration), ligands_per_hour (ligands docked, all processes), speedup
(ligands/hour relative to the fewest processes measured, normally 1), proc_runtime_s (mean run time of one engine
call), s_per_ligand (its time per ligand), slowdown (s_per_ligand relative to the fewest processes: how much slower
each process runs when it shares the GPU), peak_vram_mib (most GPU memory one engine used), gpu_util_mean/max (%),
gpu_mem_used_max_mib (the whole GPU). GPU columns are sampled with nvidia-smi ("-" without it).
\b
Exit codes: 0 ok; 1 usage or input error; 2 some engine calls failed (e.g. out of GPU memory); 3 none docked.
"""


@bench.command("gpu", epilog=BENCH_EPILOG)
@click.option("--protein", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Structure (PDB, mmCIF or PDBQT, optionally gzipped).")
@click.option("--ligands", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Ligand CSV (molecule_name,smiles); its ligands are repeated if it has fewer than --n-ligands.")
@click.option("--n-ligands", type=click.IntRange(1), default=100, show_default=True,
              help="Ligands per engine call.")
@click.option("--procs", default="1,2,3", show_default=True, callback=_procs_option,
              help="Engine processes at the same time on the GPU, one configuration each, comma-separated.")
@click.option("--pocket", type=click.IntRange(1), default=1, show_default=True, help="Pocket (P2Rank rank).")
@click.option("--out", required=True, type=click.Path(file_okay=False),
              help=f"Folder for {BENCH_CSV} and {BENCH_JSON}.")
@click.option("--gpu", "gpu_spec", help="GPU (nvidia-smi index or UUID).  [default: the first visible]")
@click.option("--center", nargs=3, type=_FiniteFloat(), metavar="X Y Z",
              help="Box centre instead of P2Rank's --pocket.")
@click.option("--box", nargs=3, type=_BoxSizeType(), default=(20.0, 20.0, 20.0), show_default=True,
              metavar="X Y Z", help="Box size (A).")
@click.option("--thread", type=THREAD_TYPE, default=8000, show_default=True, help="Engine threads.")
@click.option("--seed", type=click.IntRange(0, 2**31 - 1), default=42, show_default=True,
              help="Seed of the ligand embedding and the engine.")
@click.option("--ligand-prep", type=click.Choice(["standard", "legacy"]), default="standard", show_default=True)
@click.option("--split", is_flag=True,
              help="The processes share the --n-ligands ligands (n/N each) instead of each docking all of them.")
# nan or inf: a busy loop or a dead sampler thread, and an engine timeout run_engine rejects after the P2Rank step
@click.option("--engine-timeout", type=_FiniteFloatRange(0, min_open=True), metavar="S",
              help="Seconds one engine call may take.  [default: no limit]")
@click.option("--sample-interval", type=_FiniteFloatRange(0.05), default=0.5, show_default=True,
              help="Seconds between nvidia-smi samples.")
@click.option("--p2rank-model", default=P2RankConfig.model, show_default=True)
@click.option("--tmp-dir", type=click.Path(file_okay=False), help="Folder for temp files.")
@click.pass_context
def bench_gpu(ctx, protein, ligands, n_ligands, procs, pocket, out, gpu_spec, center, box, thread, seed, ligand_prep,
              split, engine_timeout, sample_interval, p2rank_model, tmp_dir):
    """Measure docking throughput with 1, 2, 3, ... engine processes sharing one GPU (procs_per_gpu).

    For each --procs value N, N engine processes start at the same time on the GPU, each docking the same
    --n-ligands ligands into the same pocket with the same settings (one engine call of a run with
    procs_per_gpu=N; --split shares the ligands out instead). A warm-up call first builds the OpenCL kernels if
    needed. The table is printed and written to --out.
    """
    _setup_logging(ctx)
    with _user_errors(), engine.exit_on_sigterm():
        report = run_bench_gpu(protein, ligands, out, n_ligands=n_ligands, procs=procs, pocket=pocket,
                               gpu_spec=gpu_spec, center=center, box=box, thread=thread, seed=seed,
                               ligand_prep=ligand_prep, split=split, engine_timeout=engine_timeout,
                               sample_interval=sample_interval, p2rank_model=p2rank_model, tmp_dir=tmp_dir)
    g = report["gpu"]
    click.echo(f"GPU: {g['index']} {g['name']} ({g['uuid']})" if g else "GPU: default (no nvidia-smi)")
    click.echo(f"{report['protein_id']} pocket {report['pocket']}, {report['n_ligands']} ligands per "
               f"{'configuration (split)' if report['split'] else 'process'}, thread={report['thread']}, "
               f"warm-up {report['warmup']['runtime_s']:.1f} s")
    _echo_bench(report["configurations"])
    out = os.path.abspath(os.path.expanduser(out))  # as run_bench_gpu wrote it
    click.echo(f"written: {os.path.join(out, BENCH_CSV)}, {os.path.join(out, BENCH_JSON)}")
    calls = report["calls"]
    failed = [c for c in calls if not c["ok"]]
    for c in failed:
        click.echo(f"failed: {c['tag']}: {c['error']}", err=True)
    if not any(c["docked"] for c in calls):  # also when every call exited with code 0
        click.echo("error: no engine call docked a ligand", err=True)
        ctx.exit(EXIT_NOTHING)
    if failed:
        ctx.exit(EXIT_PARTIAL)
