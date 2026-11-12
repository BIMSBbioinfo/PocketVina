#!/usr/bin/env python3
"""Stand-in for the PocketVina-GPU engine binary, so the pipeline can be tested without a GPU.

Same command line (``--config <file>`` and the other ``--<option> <value>`` flags) and config keys as
``pocketvina/main/main.cpp``, with the same option checks. For every regular ``*.pdbqt`` file in ``ligand_directory``
(sorted by name, as main.cpp does; folders and other files are skipped) it writes
``<out_dir>/<receptor_base>-pocket<k>-<stem>_out.pdbqt`` and the matching ``.json``,
named and formatted like the real engine's output (main.cpp:405-451, 827-846). Every MODEL block holds the input
ligand's lines unchanged under a ``REMARK VINA RESULT:`` line. Affinities (kcal/mol, one decimal, best first) and
RMSDs are a deterministic function of (receptor base name, pocket, ligand stem, mode, seed); without ``seed`` the
fake engine uses a fixed value, so runs stay reproducible.

Failures:
- a ligand file containing ``FAIL``: no output for that ligand and exit code 1; the other ligands are written.
- a ligand the engine cannot parse (an atom type such as ``Si``, no ``TORSDOF``, a second ``ROOT``; see
  ``check_ligand``): parse error on stderr and no output for it, exit code unchanged, as the real engine does.
- no valid ligand, a missing or invalid option, an unreadable receptor or ligand directory: exit code 1.
- an output path the engine would put in the filesystem root (see ``dock``): exit code 1 instead of writing there.

Test hooks (environment):
  POCKETVINA_FAKE_ENGINE_LOG    append one JSON line per call (argv, cwd, options, some env vars, outputs, exit code,
                                start and end time as ``time.time()``)
  POCKETVINA_FAKE_ENGINE_SLEEP  seconds to sleep before docking (for timeout tests)
  POCKETVINA_FAKE_ENGINE_WAIT_FOR  a folder: before docking, the call writes ``started-<pid>`` into it and waits until
                                a file ``go`` is there, so a test decides when calls end, without timing margins; after
                                ``WAIT_FOR_S`` seconds without ``go`` it docks anyway, so a test fails instead of
                                hanging. Its log line then has ``"go": true`` (``false``: it gave up waiting)
  POCKETVINA_FAKE_ENGINE_SHARED_RNG  "1": as in the real engine, whose ligands of one call share one random number
                                generator, a ligand's scores also depend on its position among the call's valid
                                ligands (``score_modes(..., position=)``), so ligand order and chunking matter

Standard library only, so it runs under any Python 3 interpreter.
"""
import hashlib
import json
import os
import re
import sys
import time

STR_OPTS = ("receptor", "flex", "ligand", "ligand_directory", "output_directory", "opencl_binary_path", "out", "log")
INT_OPTS = ("pocket_rank", "thread", "search_depth", "rilc_bfgs", "seed", "num_modes")
FLOAT_OPTS = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z", "energy_range", "weight_gauss1",
              "weight_gauss2", "weight_repulsion", "weight_hydrophobic", "weight_hydrogen", "weight_rot")
SWITCHES = ("randomize_only",)
CLI_SWITCHES = ("help", "help_advanced", "version")
BOX = ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z")
DEFAULTS = {"pocket_rank": 1, "num_modes": 20, "energy_range": 20.0}
# lib/atom_constants.h: atom_kind_data + atom_equivalence_data ("Se") + non_ad_metal_names
ACCEPTED_TYPES = frozenset("C A N O P S H F I NA OA SA HD Mg Mn Zn Ca Fe Cl Br Se Cu Na K Hg Co U Cd Ni".split())
FAIL_MARKER = "FAIL"
LOGGED_ENV = ("POCKETVINA_KERNEL_SRC", "POCKETVINA_KERNEL_CACHE", "OPENCL_ROOT", "LD_LIBRARY_PATH",
              "CUDA_VISIBLE_DEVICES", "POCKETVINA_DIE_WITH_PARENT", "POCKETVINA_ENGINE_SLOT")
USAGE = "Usage: PocketVina-GPU --config <file> [--<option> <value> ...]"
WAIT_FOR_S = 30  # POCKETVINA_FAKE_ENGINE_WAIT_FOR: how long a call waits for "go"


class UsageError(Exception):
    pass


def _convert(key, value):
    try:
        if key in INT_OPTS:
            return int(value)
        if key in FLOAT_OPTS:
            return float(value)
    except ValueError:
        raise UsageError(f"the argument ('{value}') for option '{key}' is invalid") from None
    if key in SWITCHES:
        if value.lower() in ("", "1", "true", "yes", "on"):
            return True
        if value.lower() in ("0", "false", "no", "off"):
            return False
        raise UsageError(f"the argument ('{value}') for option '{key}' is invalid")
    return value


def parse_config(text):
    """``[(key, value)]`` of a boost::program_options config file (``key = value``, ``#`` starts a comment)."""
    pairs = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise UsageError(f"the options configuration file contains an invalid line '{line}'")
        key, value = line.split("=", 1)
        pairs.append((key.strip(), value.strip()))
    return pairs


def parse_argv(argv):
    """Command-line options as a dict (``config`` is the config file path, not yet read)."""
    opts = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("--") or len(arg) == 2:
            raise UsageError(f"unexpected command-line argument '{arg}'")
        name, eq, value = arg[2:].partition("=")
        if name in CLI_SWITCHES or name in SWITCHES:
            opts[name] = _convert(name, value) if (eq and name in SWITCHES) else True
            i += 1
            continue
        if name != "config" and name not in STR_OPTS + INT_OPTS + FLOAT_OPTS:
            raise UsageError(f"unrecognised option '--{name}'")
        if not eq:
            if i + 1 >= len(argv):
                raise UsageError(f"the required argument for option '--{name}' is missing")
            value = argv[i + 1]
            i += 1
        i += 1
        if name in opts:
            raise UsageError(f"option '{name}' cannot be specified more than once")
        opts[name] = value if name == "config" else _convert(name, value)
    return opts


def load_options(argv):
    """Command line + config file; as with boost, a command-line value wins over the config file."""
    opts = parse_argv(argv)
    if "config" in opts:
        try:
            with open(opts["config"]) as f:
                text = f.read()
        except OSError:
            raise UsageError(f"could not open \"{opts['config']}\" for reading") from None
        seen = set()
        for key, value in parse_config(text):
            if key not in STR_OPTS + INT_OPTS + FLOAT_OPTS + SWITCHES:
                raise UsageError(f"unrecognised option '{key}'")
            if key in seen:
                raise UsageError(f"option '{key}' cannot be specified more than once")
            seen.add(key)
            opts.setdefault(key, _convert(key, value))
    for key, value in DEFAULTS.items():
        opts.setdefault(key, value)
    return opts


def validate(opts):
    """The checks main.cpp makes before docking, in the same order."""
    if "receptor" not in opts:
        raise UsageError("Missing receptor.")
    if "thread" not in opts:
        raise UsageError("Missing thread, the recommend thread value is 8000")
    if opts["thread"] < 1000:
        raise UsageError("thread must be at least 1000")
    if "ligand" in opts and "ligand_directory" in opts:
        raise UsageError("Can not define ligand and ligand_directory at the same time")
    if opts["num_modes"] < 1:
        raise UsageError("num_modes must be 1 or greater")
    missing = [k for k in BOX if k not in opts]
    if missing:
        raise UsageError(f"the option '{missing[0]}' is required but missing")
    if min(opts["size_x"], opts["size_y"], opts["size_z"]) <= 0:
        raise UsageError("Search space dimensions should be positive")
    if "ligand" not in opts and "ligand_directory" not in opts:
        raise UsageError("Missing ligand")
    if opts.get("search_depth", 1) < 1:
        raise UsageError("search_depth must be 1 or greater")


def _stem(path):
    """boost::filesystem::basename: file name without its last extension."""
    return os.path.splitext(os.path.basename(path))[0]


def ligand_stem(path):
    """Ligand file name up to its first ``.pdbqt``: the ``<stem>`` in ``<stem>_out.pdbqt``."""
    name = os.path.basename(path)
    cut = name.find(".pdbqt")
    return name[:cut] if cut >= 0 else name


def output_paths(out_name, receptor, pocket):
    """(pdbqt, json, json key) for one ligand, from its engine ``out_name`` (main.cpp:410-416, 447)."""
    prefix = f"{_stem(receptor)}-pocket{pocket}-{_stem(out_name)}"
    pdbqt = os.path.dirname(out_name) + "/" + prefix + ".pdbqt"
    return pdbqt, pdbqt[:pdbqt.rfind(".")] + ".json", prefix + "-results"


def directory_out_name(ligand_directory, out_dir, ligand_path):
    """The engine's ``out_names`` entry for a ligand found in ``ligand_directory`` (main.cpp:835-843)."""
    cut = ligand_path.find(".pdbqt")
    tmp = (ligand_path[:cut] if cut >= 0 else ligand_path) + "_out.pdbqt"
    return out_dir + tmp[len(ligand_directory):]


def _unit(*parts):
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


def score_modes(receptor, pocket, stem, seed, num_modes=20, energy_range=20.0, position=None):
    """``[(affinity, rmsd_lb, rmsd_ub)]``, best first, as the engine reports them for this ligand × pocket.

    ``receptor`` is the receptor base name; ``seed`` None means "no seed given"; ``position`` (0-based, among the
    valid ligands of the call) is used with ``POCKETVINA_FAKE_ENGINE_SHARED_RNG=1`` only. Affinities are kept in
    tenths so that the energy_range cut and the one-decimal output are exact.
    """
    seed = "auto" if seed is None else int(seed)
    if position is not None:
        seed = f"{seed}@{int(position)}"
    best = -(30 + int(_unit(receptor, pocket, stem, 1, seed) * 80))  # -3.0 .. -10.9 kcal/mol
    limit = best + round(energy_range * 10)
    tenths, modes = best, []
    for mode in range(1, num_modes + 1):
        if mode == 1:
            lb = ub = 0.0
        else:
            tenths += int(_unit(receptor, pocket, stem, mode, seed) * 5)  # 0.0 .. 0.4 above the previous mode
            if tenths > limit:
                break
            lb = round(1.0 + 7.0 * _unit(receptor, pocket, stem, mode, seed, "lb"), 3)
            ub = round(lb + 0.5 + 6.0 * _unit(receptor, pocket, stem, mode, seed, "ub"), 3)
        modes.append((tenths / 10, lb, ub))
    return modes


def check_ligand(text):
    """None if the engine would accept this ligand PDBQT, else ``(line_number, reason)``.

    The structure checks of lib/parse_pdbqt.cpp (parse_pdbqt_ligand): ``ROOT`` .. ``ENDROOT``, then ``BRANCH`` ..
    ``ENDBRANCH`` blocks and exactly one ``TORSDOF`` outside them; empty, ``REMARK`` and ``WARNING`` lines are
    skipped, any other tag (``END``, ``TER``, a second ``ROOT`` as in an untruncated multi-fragment file) is an error.
    """
    unknown = "Unknown or inappropriate tag"
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    state, depth, n_atoms, torsdof = "start", 0, 0, False
    for n, line in enumerate(lines, 1):
        if not line or line.startswith(("WARNING", "REMARK")):
            continue
        if line.startswith("MODEL"):
            return n, 'Unexpected multi-MODEL input. Use "vina_split" first?'
        if line.startswith(("ATOM  ", "HETATM")):
            if state != "root" and depth == 0:
                return n, unknown
            ad_type = line[77:].strip()  # omit_whitespace(str, 78, 79) reads to the end of the line
            if ad_type not in ACCEPTED_TYPES:
                return n, (f'ATOM syntax incorrect: "{ad_type}" is not a valid AutoDock type. '
                           "Note that AutoDock atom types are case-sensitive.")
            n_atoms += 1
        elif state == "start":
            if not line.startswith("ROOT"):
                return n, unknown
            state = "root"
        elif state == "root":
            if not line.startswith("ENDROOT"):
                return n, unknown
            state = "body"
        elif line.startswith("BRANCH"):
            depth += 1
        elif line.startswith("ENDBRANCH") and depth:
            depth -= 1
        elif line.startswith("TORSDOF") and depth == 0:
            if torsdof:
                return n, "TORSDOF can occur only once"
            if not re.match(r"\s*\+?\d", line[len("TORSDOF"):]):
                return n, "Syntax error"
            torsdof = True
        else:
            return n, unknown
    if n_atoms == 0:
        return len(lines), "No atoms in the ligand"
    if not torsdof:
        return len(lines), "Missing TORSDOF"
    return None


def render_pdbqt(ligand_text, modes):
    body = "".join(line + "\n" for line in ligand_text.splitlines() if line.strip())
    return "".join(f"MODEL {i}\nREMARK VINA RESULT: {e:9.1f}  {lb:9.3f}  {ub:9.3f}\n{body}ENDMDL\n"
                   for i, (e, lb, ub) in enumerate(modes, 1))


def render_json(key, modes, pocket):
    """Same text as the engine's nlohmann ``dump(4)``: sorted keys, 4-space indent, raw UTF-8, no final newline."""
    rows = [{"mode": i, "affinity (kcal/mol)": float(f"{e:.1f}"), "dist from rmsd l.b.": float(f"{lb:.3f}"),
             "dist from rmsd u.b.": float(f"{ub:.3f}"), "pocket rank": pocket}
            for i, (e, lb, ub) in enumerate(modes, 1)]
    return json.dumps({key: rows}, indent=4, sort_keys=True, ensure_ascii=False)


def wait_for_go(folder):
    """POCKETVINA_FAKE_ENGINE_WAIT_FOR: write ``<folder>/started-<pid>``, then wait for ``<folder>/go``.

    True when ``go`` came, False when it did not within ``WAIT_FOR_S`` seconds.
    """
    with open(os.path.join(folder, f"started-{os.getpid()}"), "w"):
        pass
    go = os.path.join(folder, "go")
    deadline = time.monotonic() + WAIT_FOR_S
    while not os.path.exists(go):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def dock(opts, say, warn):
    """Write the outputs; returns (exit code, written pdbqt paths, failed ligand paths)."""
    receptor, pocket = opts["receptor"], opts["pocket_rank"]
    if not os.path.isfile(receptor):
        raise UsageError(f'could not open "{receptor}" for reading')
    if "ligand_directory" in opts:
        lig_dir = opts["ligand_directory"]
        out_dir = opts.get("output_directory", lig_dir + "_out")
        if not os.path.isdir(lig_dir):
            raise UsageError(f"File system error: directory_iterator::directory_iterator: {lig_dir}")
        if not os.path.isdir(out_dir):
            try:
                os.mkdir(out_dir)
            except OSError as e:
                raise UsageError(f"File system error: create_directory: {e.strerror}: \"{out_dir}\"") from None
        jobs = [(p, directory_out_name(lig_dir, out_dir, p))
                for p in (os.path.join(lig_dir, n) for n in sorted(os.listdir(lig_dir)) if n.endswith(".pdbqt"))
                if os.path.isfile(p)]
        say("Using virtual sreening mode\n")
        say(f"Output will be in the directory {out_dir}")
    else:
        lig = opts["ligand"]
        out = opts.get("out") or (lig[:-6] if lig.endswith(".pdbqt") else lig) + "_out.pdbqt"
        jobs = [(lig, out)]
        say("Using single ligand docking mode\n")

    valid, failed = [], []
    for lig_path, out_name in jobs:
        try:
            with open(lig_path, encoding="utf-8", errors="surrogateescape") as f:
                text = f.read()
        except OSError:
            raise UsageError(f'could not open "{lig_path}" for reading') from None
        if FAIL_MARKER in text:
            warn(f'fake engine: "{lig_path}" contains {FAIL_MARKER}, no output written')
            failed.append(lig_path)
            continue
        err = check_ligand(text)
        if err:
            warn(f'\nParse error on line {err[0]} in file "{lig_path}": {err[1]}')
            continue
        valid.append((lig_path, out_name, text))
    if not valid:
        warn("No valid ligands in the input directory")
        return 1, [], failed

    seed = opts.get("seed")
    say(f"Using random seed: {seed if seed is not None else 'auto (fake engine: fixed)'}")
    written = []
    shared_rng = os.environ.get("POCKETVINA_FAKE_ENGINE_SHARED_RNG") == "1"
    for position, (lig_path, out_name, text) in enumerate(valid):
        modes = score_modes(_stem(receptor), pocket, ligand_stem(lig_path), seed, opts["num_modes"],
                            opts["energy_range"], position=position if shared_rng else None)
        pdbqt, json_path, key = output_paths(out_name, receptor, pocket)
        if not os.path.dirname(out_name):
            # the engine would write "/<prefix>.pdbqt" here (a ligand path without a directory, or a
            # ligand_directory ending in "/" with a relative output_directory); never write into the root
            raise UsageError(f'could not open "{pdbqt}" for writing (fake engine: refusing to write into "/")')
        say(f"Writing ligand {pdbqt} output...")
        try:
            with open(pdbqt, "w", encoding="utf-8", errors="surrogateescape") as f:
                f.write(render_pdbqt(text, modes))
            with open(json_path, "w", encoding="utf-8", errors="surrogateescape") as f:
                f.write(render_json(key, modes, pocket))
        except OSError:
            raise UsageError(f'could not open "{pdbqt}" for writing') from None
        say(f"JSON file successfully saved to: {json_path}")
        written.append(pdbqt)
    return (1 if failed else 0), written, failed


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    lines, opts, written, failed, code, go = [], {}, [], [], 1, None
    t0 = time.time()

    def say(msg):
        lines.append(msg)
        print(msg, flush=True)

    def warn(msg):
        print(msg, file=sys.stderr, flush=True)

    try:
        opts = load_options(argv)
        say(f"Pocket rank: {opts['pocket_rank']}")
        if opts.get("help") or opts.get("help_advanced"):
            say(USAGE)
            code = 0
            return code
        if opts.get("version"):
            say("PocketVina-GPU fake engine (tests/fake_engine.py)")
            code = 0
            return code
        validate(opts)
        delay = float(os.environ.get("POCKETVINA_FAKE_ENGINE_SLEEP") or 0)
        if delay > 0:
            time.sleep(delay)
        if os.environ.get("POCKETVINA_FAKE_ENGINE_WAIT_FOR"):
            go = wait_for_go(os.environ["POCKETVINA_FAKE_ENGINE_WAIT_FOR"])
        code, written, failed = dock(opts, say, warn)
        say(f"QuickVina 2-GPU3 total runtime = {time.time() - t0:.3f} s")
        if opts.get("log"):
            with open(opts["log"], "w") as f:
                f.write("\n".join(lines) + "\n")
        return code
    except UsageError as e:
        warn(f"\n\nError: {e}\n\n{USAGE}")
        code = 1
        return code
    finally:
        log = os.environ.get("POCKETVINA_FAKE_ENGINE_LOG")
        if log:
            entry = {"argv": argv, "cwd": os.getcwd(), "options": opts, "outputs": written, "failed": failed,
                     "returncode": code, "env": {k: os.environ.get(k) for k in LOGGED_ENV}, "start": t0,
                     "end": time.time()}
            if go is not None:
                entry["go"] = go
            # one write() on an O_APPEND fd, so lines from concurrent engine calls never interleave
            fd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, (json.dumps(entry, sort_keys=True) + "\n").encode())
            finally:
                os.close(fd)


if __name__ == "__main__":
    sys.exit(main())
