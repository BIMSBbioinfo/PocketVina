"""The documentation in docs/: every ``pocketvina`` command it shows exists with the options it uses, its Python and
TOML examples parse, its pages are all in the table of contents (docs/dev/ is not), and, where the Sphinx packages of
docs/requirements.txt are installed too (``pip install ".[dev,docs]"``), it builds without warnings
(``sphinx-build -W``; without them the build is checked by running sphinx-build itself, as the docs workflow does).
The Python run of the quickstart page and the examples of the visualisation and PoseBusters pages that build on it run
as written, with the fake engine and the fixture P2Rank pockets (``run_section``, also used for the README).
"""
import ast
import contextlib
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import click
import pytest

from pocketvina import cli

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.glob("*.md"))
FENCE = re.compile(r"^(?P<indent> *)```(?P<lang>[^\n`]*)\n(?P<body>.*?)^(?P=indent)```", re.S | re.M)
INLINE = re.compile(r"(?<!`)`(pocketvina [^`]+)`(?!`)")
DOC_ROLE = re.compile(r"\{doc\}`(?:[^`<]*<)?(?P<target>[^`>]+)>?`")
TOCTREE = re.compile(r"^```\{toctree\}\n(?P<body>.*?)^```", re.S | re.M)
# the tests of an sdist (MANIFEST.in ships tests/, not docs/): nothing to check
pytestmark = pytest.mark.skipif(not DOCS.is_dir() and (DOCS.parent / "PKG-INFO").is_file(),
                                reason="an sdist has no docs/")


def _blocks(lang_prefixes):
    for page in PAGES:
        for m in FENCE.finditer(page.read_text(encoding="utf-8")):
            lang = m.group("lang").strip()
            if lang.startswith(lang_prefixes):
                body = "\n".join(line[len(m.group("indent")):] for line in m.group("body").splitlines())
                yield page.name, body


def code_blocks(text: str, lang: str) -> list:
    """The bodies of the ``lang`` code blocks of a Markdown text, unindented."""
    return ["\n".join(line[len(m.group("indent")):] for line in m.group("body").splitlines())
            for m in FENCE.finditer(text) if m.group("lang").strip() == lang]


def section(text: str, title: str) -> str:
    """The ``## title`` section of a Markdown text, up to the next ``## `` heading."""
    m = re.search(rf"^## {re.escape(title)}\n(?P<body>.*?)(?=^## |\Z)", text, re.S | re.M)
    assert m, f"no section {title!r}"
    return m.group("body")


def _shell_commands(text):
    """Lines of a shell block, continuation lines joined."""
    joined = re.sub(r"\\\n\s*", " ", text)
    return [line.strip() for line in joined.splitlines() if line.strip()]


def documented_commands():
    """(page, argv after "pocketvina") of every pocketvina command line in a bash block or an inline code span."""
    found = []
    for page, body in _blocks(("bash",)):
        for line in _shell_commands(body):
            if line.startswith("pocketvina "):
                found.append((page, line))
    for page in PAGES:
        text = FENCE.sub("", page.read_text(encoding="utf-8"))
        found += [(page.name, m.group(1)) for m in INLINE.finditer(text)]
    commands = []
    for page, line in found:
        argv = [a for a in shlex.split(line, comments=True)[1:] if a != "..."]  # "...": options left out
        commands.append((page, argv))
    return commands


def parse(argv):
    """Parse ``pocketvina <argv>`` with click's parser, without running it or converting values (files need not
    exist): unknown commands and options raise ``click.UsageError``; returns the leaf command and the arguments left
    over (none for a valid command line)."""
    cmd, name, parent, args = cli.cli, "pocketvina", None, list(argv)
    while True:
        # with the command's context settings, as when it runs: click keeps the help option a command makes first
        ctx = click.Context(cmd, info_name=name, parent=parent, **cmd.context_settings)
        _, rest, _ = cmd.make_parser(ctx).parse_args(args=args)
        if not isinstance(cmd, click.Group) or not rest:
            return cmd, rest
        name, args = rest[0], rest[1:]
        sub = cmd.get_command(ctx, name)
        if sub is None:
            raise click.UsageError(f"no command {name!r} in {ctx.command_path}")
        cmd, parent = sub, ctx


COMMANDS = documented_commands()


def test_docs_show_commands():
    pages = {page for page, _ in COMMANDS}
    assert {"quickstart.md", "installation.md", "hpc.md", "output-format.md", "posebusters.md"} <= pages
    used = {argv[0] for _, argv in COMMANDS if argv and not argv[0].startswith("-")}
    # every command of the CLI is shown somewhere besides the generated reference
    assert set(cli.cli.list_commands(click.Context(cli.cli))) <= used


@pytest.mark.parametrize("page,argv", COMMANDS, ids=[f"{p}:{' '.join(a)[:60]}" for p, a in COMMANDS])
def test_documented_command_parses(page, argv):
    cmd, leftover = parse(argv)
    assert not leftover, f"{page}: unexpected arguments {leftover} for {cmd.name}"


def test_parse_rejects_unknown_options_and_commands():
    with pytest.raises(click.UsageError):
        parse(["dock", "--no-such-option", "1"])
    with pytest.raises(click.UsageError):
        parse(["no-such-command"])
    assert parse(["query", "--results", "r", "SELECT 1", "extra"])[1] == ["extra"]


def test_python_examples_compile():
    blocks = list(_blocks(("python",)))
    assert len(blocks) >= 8
    for page, body in blocks:
        try:
            ast.parse(body)
        except SyntaxError as e:
            pytest.fail(f"{page}: {e}\n{body}")


def test_toml_examples_are_dock_options():
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    names = {p.name for p in cli.dock.params}
    blocks = list(_blocks(("toml",)))
    assert blocks
    for page, body in blocks:
        keys = set(tomllib.loads(body))
        assert keys <= names | set(cli.CONFIG_ALIASES), f"{page}: unknown keys {keys - names}"


def test_every_page_is_in_the_toctree_and_dev_is_not():
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    listed = [line.strip() for m in TOCTREE.finditer(index) for line in m.group("body").splitlines()
              if line.strip() and not line.startswith(":")]
    assert sorted(listed) == sorted(p.stem for p in PAGES if p.name != "index.md")
    assert not any(entry.startswith("dev") for entry in listed)
    conf = (DOCS / "conf.py").read_text(encoding="utf-8")
    assert re.search(r"exclude_patterns = \[[^\]]*\"dev\"", conf)


def test_doc_references_exist():
    for page in PAGES:
        for m in DOC_ROLE.finditer(page.read_text(encoding="utf-8")):
            assert (DOCS / f"{m.group('target')}.md").is_file(), f"{page.name}: {{doc}} {m.group('target')}"


def test_sphinx_build_without_warnings(tmp_path):
    for module in ("sphinx", "myst_parser", "furo", "sphinx_click"):
        pytest.importorskip(module)
    result = subprocess.run([sys.executable, "-m", "sphinx", "-W", "-q", "-b", "html", "-d", str(tmp_path / "doctrees"),
                             str(DOCS), str(tmp_path / "html")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-5000:]
    assert (tmp_path / "html" / "cli.html").is_file()


def test_click_help_markup():
    """docs/conf.py turns click help into reStructuredText: "*.pdb" is no emphasis and a Markdown-style `name` is
    code, not reST's default role (italics); double backticks stay as they are."""
    import runpy

    conf = runpy.run_path(str(DOCS / "conf.py"))
    text = conf["_click_text"]
    assert text("P2Rank's `prank` launcher on PATH") == "P2Rank's ``prank`` launcher on PATH"
    assert text("structures (*.pdb, *.cif)") == r"structures (\*.pdb, \*.cif)"
    assert text("the ``pocketvina`` command, ``*.pdb``") == "the ``pocketvina`` command, ``*.pdb``"
    assert text("a ` b") == "a ` b"


# --- examples, run ---------------------------------------------------------------------------------------------------

# the example ligands the examples dock in the tests (quick to embed), with the two example proteins
EXAMPLE_LIGANDS = ("molecule1", "molecule2", "molecule3", "molecule5")


def example_inputs(workdir: Path, ligands=EXAMPLE_LIGANDS) -> None:
    """``workdir/example``: the repository's example proteins and the given ligands of its ligands.txt."""
    example = workdir / "example"
    shutil.copytree(ROOT / "example" / "proteins", example / "proteins")
    rows = (ROOT / "example" / "ligands.txt").read_text().splitlines()
    kept = [rows[0]] + [r for r in rows[1:] if r.split(",")[0] in ligands]
    (example / "ligands.txt").write_text("\n".join(kept) + "\n")


@contextlib.contextmanager
def _chdir(path):  # contextlib.chdir: Python 3.11+
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def run_section(text: str, workdir: Path, ns=None) -> dict:
    """Run the code of a Markdown section in ``workdir`` as a user pastes it: its shell commands (but `conda ...` and
    the `pocketvina` command lines, which ``parse`` checks), then its Python blocks one after the other in one
    namespace (``ns``, e.g. that of an earlier section); returns the namespace, with what the Python code printed
    added as ``ns["_stdout"]``."""
    for line in (line for body in code_blocks(text, "bash") for line in _shell_commands(body)):
        if not line.startswith(("conda ", "pocketvina ")):
            subprocess.run(["bash", "-c", line], cwd=workdir, check=True)
    ns = {"__name__": "__main__"} if ns is None else ns
    out = io.StringIO()
    with _chdir(workdir), contextlib.redirect_stdout(out):
        for body in code_blocks(text, "python"):
            exec(compile(body, "<docs example>", "exec"), ns)
    ns["_stdout"] = ns.get("_stdout", "") + out.getvalue()
    return ns


@pytest.fixture
def stub_p2rank(monkeypatch):
    """P2Rank's pockets from the fixture predictions (3 pockets), receptors cropped (as in test_pipeline); without
    PoseBusters, ``Selection.bust`` returns None."""
    from pocketvina import main as pv_main
    from pocketvina import pipeline, selection
    from test_pipeline import cropped_receptor, stub_predict_pockets

    pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    monkeypatch.setattr(pipeline, "predict_pockets", stub_predict_pockets(n_pockets=3))
    monkeypatch.setattr(pipeline, "receptor_pdbqt", cropped_receptor())
    monkeypatch.setattr(pv_main, "receptor_pdbqt", cropped_receptor())
    try:
        import posebusters  # noqa: F401
    except ImportError:
        monkeypatch.setattr(selection.Selection, "bust", lambda self, workers=None: None)


def test_quickstart_python_runs(tmp_path, fake_engine_env, stub_p2rank):
    """quickstart.md, "Quick Start", then the examples of visualisation.md and posebusters.md that use its
    ``pocketvina`` and ``pair``."""
    from pocketvina.io.results import BEST_SCORES_COLUMNS

    example_inputs(tmp_path)
    page = (DOCS / "quickstart.md").read_text(encoding="utf-8")
    ns = run_section(section(page, "Quick Start"), tmp_path)
    assert "Best scores:" in ns["_stdout"] and ns["pocketvina"].failed_proteins == {}
    best = ns["pocketvina"].best_scores()
    assert list(best.columns) == list(BEST_SCORES_COLUMNS) and len(best) == 2 * len(EXAMPLE_LIGANDS)
    pair = ns["pair"]  # the last selection: all modes of the pocket of the highest-affinity pose
    assert pair.name == f"PV_1fbl_molecule1_pk{pair.best_pocket}_modes"
    for name in ("PV_results.html", "results.html", f"{pair.name}.html", f"{pair.name}.sdf", "PV_1fbl.pdb"):
        assert (tmp_path / name).is_file(), name

    visualisation = (DOCS / "visualisation.md").read_text(encoding="utf-8")
    run_section(section(visualisation, "The 3D page"), tmp_path, ns)
    assert (tmp_path / "results.html").is_file() and (tmp_path / "PV_1fbl.html").is_file()
    assert (tmp_path / "PV_1fbl_molecule1_pk2_modes.html").is_file()
    run_section(section(visualisation, "Desktop viewers"), tmp_path, ns)
    assert (tmp_path / "PV_1fbl_molecule1_pk2_modes.sdf").is_file() and (tmp_path / "1fbl_molecule1.pdbqt").is_file()
    run_section(section((DOCS / "posebusters.md").read_text(encoding="utf-8"), "One pair"), tmp_path, ns)
    if ns["checks"] is not None:
        assert list(ns["checks"].columns[:6]) == ["pocket", "mode", "vina", "pb_valid", "pb_status", "failed_checks"]
        assert len(ns["checks"]) == len(ns["pair"].poses) > 1


def test_selection_table_names_the_files_export_writes(tmp_path, fake_engine_env, stub_p2rank):
    """Each row of the select() table: the selection it describes, exported, has the file name in the table."""
    from pocketvina import PocketVinaConfig, PocketVinaGPU

    text = section((DOCS / "visualisation.md").read_text(encoding="utf-8"), "Selecting poses")
    rows = re.findall(r"^\| (`[^|]*`|\(nothing more\)) \| [^|]+ \| `(PV_[^`]+)`", text, re.M)
    assert len(rows) >= 5
    example_inputs(tmp_path)
    (tmp_path / "proteins.txt").write_text("example/proteins/1fbl.pdb\nexample/proteins/2W83.pdb\n")
    config = PocketVinaConfig(protein_paths=str(tmp_path / "proteins.txt"),
                              smiles_file=str(tmp_path / "example" / "ligands.txt"), output_dir=str(tmp_path / "r"),
                              output_format="atlas", top_pockets=2)
    pv = PocketVinaGPU(config)
    with contextlib.redirect_stdout(io.StringIO()):
        pv.execute_docking()
    (tmp_path / "out").mkdir()
    for args, name in rows:
        kwargs = {} if args.startswith("(") else eval(f"dict({args.strip('`')})")
        pair = pv.select(protein="1fbl", ligand="molecule1", **kwargs)
        if "if pocket 1 has the highest-affinity pose" in text and kwargs == {"mode": "all"} and pair.best_pocket != 1:
            continue
        assert os.path.basename(pair.export(tmp_path / "out")) == name, (args, name)
