"""The quickstart notebook (``notebooks/quickstart_colab.ipynb``) and its setup script (``notebooks/colab_setup.py``).

The notebook is executed headless with nbclient as a user runs it outside Colab (its install cell installs nothing),
in a copy of the repository's layout (``notebooks/`` as the working folder, the example proteins and a few example
ligands in ``../example``). What makes it run without a GPU stays out of the notebook: this test prepends one cell
that stubs P2Rank (the fixture pockets) and the receptor conversion (cropped receptors, as ``test_pipeline`` does),
and the fake engine docks (``fake_engine_env``). The kernel runs this interpreter, with ``pocketvina`` imported from
this checkout. Skipped without nbformat, nbclient or ipykernel; the structure checks need only nbformat.

The Atlas notebook (``notebooks/atlas_colab.ipynb``, no GPU) runs the same way, on the fixture release of the PocketVina
Atlas (``$POCKETVINA_ATLAS``), without internet.
"""
import ast
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "quickstart_colab.ipynb"
SETUP_SCRIPT = REPO_ROOT / "notebooks" / "colab_setup.py"
KERNEL_NAME = "pocketvina-notebook-test"
CELL_TIMEOUT_S = 300
# tags of the code cells, in notebook order
CODE_TAGS = ["install", "dock", "load", "show", "select", "pair-show", "select-all", "bust", "export"]
SHOW_TAGS = ("show", "pair-show", "select-all")
MAX_CODE_LINES = 30
ATLAS_NOTEBOOK = REPO_ROOT / "notebooks" / "atlas_colab.ipynb"
ATLAS_CODE_TAGS = ["install", "atlas", "search", "protein", "best-scores", "select", "bust", "export", "download"]
# the cell this test puts in front of the notebook: P2Rank and the receptor conversion stubbed, no GPU listed
CI_CELL = f"""import sys
sys.path.insert(0, {str(TESTS_DIR)!r})
import test_pipeline
test_pipeline.install_stubs(n_pockets=3)"""


def _tag(cell) -> str:
    tags = cell.get("metadata", {}).get("tags", [])
    return tags[0] if tags else ""


def _code_cells(nb) -> list:
    return [c for c in nb.cells if c.cell_type == "code"]


def _cell(nb, tag: str):
    return next(c for c in _code_cells(nb) if _tag(c) == tag)


def _outputs(cell, mime: str) -> list:
    return [o["data"][mime] for o in cell.get("outputs", []) if mime in o.get("data", {})]


def _text(cell) -> str:
    """Everything a cell printed or displayed, as one string."""
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append(out.get("text", ""))
        for value in out.get("data", {}).values():
            parts.append(value if isinstance(value, str) else json.dumps(value))
    return "\n".join(parts)


def _setup_module():
    spec = importlib.util.spec_from_file_location("colab_setup_under_test", SETUP_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(str(NOTEBOOK), as_version=4)


# --- the notebook file ---------------------------------------------------------------------------------------------

def test_notebook_is_valid_short_and_clean(notebook):
    nbformat.validate(notebook)
    assert [_tag(c) for c in _code_cells(notebook)] == CODE_TAGS
    for cell in _code_cells(notebook):
        assert cell.outputs == [] and cell.execution_count is None, f"stored outputs in the {_tag(cell)} cell"
        # Python 3.9 syntax (the oldest Python PocketVina supports), no IPython magics or shell escapes
        ast.parse(cell.source, feature_version=(3, 9))
        assert len(cell.source.splitlines()) <= MAX_CODE_LINES, _tag(cell)
    assert notebook.metadata["accelerator"] == "GPU"  # Colab opens it on a GPU runtime
    assert notebook.metadata["kernelspec"]["name"] == "python3"
    # a title with the Colab badge, and a markdown cell above every code cell
    kinds = [c.cell_type for c in notebook.cells]
    assert kinds[0] == "markdown" and notebook.cells[0].source.startswith("# ")
    assert "colab.research.google.com/github/BIMSBbioinfo/PocketVina/blob/main/notebooks/quickstart_colab.ipynb" \
        in notebook.cells[0].source
    assert "Run all" in notebook.cells[0].source and "token" not in notebook.cells[0].source  # a public repository
    assert all(kinds[i - 1] == "markdown" for i, kind in enumerate(kinds) if kind == "code")


def test_notebook_uses_the_simple_api(notebook):
    sources = "\n".join(c.source for c in _code_cells(notebook))
    dock = _cell(notebook, "dock").source
    for setting in ('output_format="atlas"', "htvs_mode=True", "PocketVinaGPU(config", ".execute_docking()"):
        assert setting in dock
    for call in ("pocketvina.show()", 'pocketvina.select(protein="1fbl", ligand="molecule1")', "pair.show()",
                 'pocket="all"', "pair.bust()", "pair.export()"):
        assert call in sources, call
    # no test plumbing, no lower-level modules
    for unwanted in ("POCKETVINA_", "fake_engine", "CI", "viz.", "bust.", "/tests"):
        assert unwanted not in sources, unwanted
    install = _cell(notebook, "install").source
    assert '"google.colab" in sys.modules' in install and "colab_setup.setup(REPO_URL, REPO_REF)" in install
    assert len(install.splitlines()) <= 11


# --- the setup script ----------------------------------------------------------------------------------------------

def test_colab_clone_follows_repo_ref(tmp_path, monkeypatch):
    """The clone step (Colab only): a ref older than the notebook stops with a clear message, and setting another
    REPO_REF afterwards checks it out in the existing clone."""
    if shutil.which("git") is None:
        pytest.skip("needs git")
    # git must not reach another repository: run from a git hook, GIT_DIR / GIT_INDEX_FILE / GIT_WORK_TREE would make
    # the commands below (and the script's fetch + checkout --force) act on the caller's repository instead
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no user or system git config (signing, hooks, default branch)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    src = tmp_path / "src"
    src.mkdir()

    def git(*args):
        subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args], cwd=src,
                       check=True, capture_output=True)

    git("init", "-q")
    git("symbolic-ref", "HEAD", "refs/heads/main")
    (src / "pocketvina").mkdir()
    (src / "pocketvina" / "__init__.py").write_text("")
    git("add", "-A")
    git("commit", "-q", "-m", "v0.1")
    git("checkout", "-q", "-b", "dev")
    (src / "pocketvina" / "selection.py").write_text("")
    git("add", "-A")
    git("commit", "-q", "-m", "v0.2")

    setup = _setup_module()
    monkeypatch.setattr(setup, "sys", type("FakeSys", (), {"modules": {}, "executable": sys.executable}))
    log = tmp_path / "install.log"
    clone = tmp_path / "clone"
    with pytest.raises(RuntimeError, match="set REPO_REF"):
        setup.clone_repo(src.as_uri(), "main", clone, log=log)
    assert (clone / ".git").is_dir() and not (clone / "pocketvina" / "selection.py").exists()
    setup.clone_repo(src.as_uri(), "dev", clone, log=log)
    assert (clone / "pocketvina" / "selection.py").is_file()
    (clone / "pocketvina" / "boost_1_77_0").mkdir()  # not in git: kept
    setup.clone_repo(src.as_uri(), "dev", clone, log=log)  # already there
    # pocketvina imported before the ref changed: the new checkout needs a fresh session
    setup.sys.modules["pocketvina"] = object()
    with pytest.raises(RuntimeError, match="Restart session"):
        setup.clone_repo(src.as_uri(), "main", clone, log=log)
    assert not (clone / "pocketvina" / "selection.py").exists() and (clone / "pocketvina" / "boost_1_77_0").is_dir()


def test_colab_apt_install_survives_a_failing_update(tmp_path, monkeypatch):
    """The apt step (Colab only) with a stand-in apt-get: a failing `apt-get update` (one broken third-party apt
    source on the runtime) does not stop the install, a failing `apt-get install` does."""
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "apt_calls.txt"
    apt = bin_dir / "apt-get"
    apt.write_text(f'#!/bin/sh\necho "$*" >> {shlex.quote(str(calls))}\n'
                   'if [ "$1" = update ] || [ "$FAKE_APT_INSTALL_FAILS" = 1 ]; then exit 100; fi\n')
    apt.chmod(0o755)
    if not os.access(apt, os.X_OK):
        pytest.skip(f"cannot execute {apt} (noexec temp dir?)")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.delenv("FAKE_APT_INSTALL_FAILS", raising=False)
    # the script's `bash -c` must not read the caller's startup files, which may reset PATH (and so run the real
    # apt-get): Debian/Ubuntu bash sources ~/.bashrc in a shell started over SSH ($SSH_CLIENT, $SHLVL < 2)
    for name in ("BASH_ENV", "ENV", "SSH_CLIENT", "SSH2_CLIENT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    setup = _setup_module()
    log = tmp_path / "install.log"
    setup.apt_install(["pkg-a", "pkg-b"], log=log)
    assert calls.read_text().splitlines() == ["update -qq", "install -y -qq pkg-a pkg-b"]
    monkeypatch.setenv("FAKE_APT_INSTALL_FAILS", "1")
    with pytest.raises(RuntimeError, match="exit code 100: DEBIAN_FRONTEND=noninteractive apt-get install"):
        setup.apt_install(["pkg-a"], log=log)
    assert "apt-get update failed" in log.read_text()


def test_setup_outside_colab_installs_nothing(tmp_path):
    """Outside Colab, ``setup`` only makes ``pocketvina`` importable: an installed one, else this checkout's (as with
    a development environment that runs the tests from the checkout)."""
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import colab_setup; colab_setup.setup('url', 'ref'); "
            "import pocketvina; print('IMPORTED', pocketvina.__file__)")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    run = subprocess.run([sys.executable, "-c", code, str(SETUP_SCRIPT.parent)], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-3000:]
    imported = Path(run.stdout.split("IMPORTED ")[1].strip())
    if imported.parent.parent == REPO_ROOT:
        assert f"working folder {tmp_path}" in run.stdout
    assert sorted(os.listdir(tmp_path)) == []  # nothing written


# --- executed --------------------------------------------------------------------------------------------------------

def _repo_listing() -> set:
    return {p.relative_to(REPO_ROOT).as_posix() for folder in (REPO_ROOT, NOTEBOOK.parent) for p in folder.iterdir()}


def _kernel_spec(root: Path) -> None:
    """A kernel that runs this interpreter (the "python3" kernel spec may belong to another environment)."""
    folder = root / "kernels" / KERNEL_NAME
    folder.mkdir(parents=True)
    spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "PocketVina notebook test", "language": "python"}
    (folder / "kernel.json").write_text(json.dumps(spec))


def _execute(notebook, workdir: Path, tmp_path: Path, monkeypatch, first_cell: str = ""):
    """A copy of the notebook (the module fixture stays unexecuted), executed in ``workdir`` by a kernel that runs this
    interpreter, with ``pocketvina`` and the setup script imported from this checkout; ``first_cell`` runs before the
    notebook and is not in the result."""
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    jupyter = tmp_path / "jupyter"
    _kernel_spec(jupyter)
    path = [str(REPO_ROOT), str(SETUP_SCRIPT.parent), os.environ.get("PYTHONPATH")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, path)))
    monkeypatch.setenv("JUPYTER_PATH", str(jupyter))
    monkeypatch.setenv("JUPYTER_RUNTIME_DIR", str(tmp_path / "jupyter_runtime"))
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ipython"))
    nb = nbformat.from_dict(json.loads(json.dumps(notebook)))
    if first_cell:
        nb.cells.insert(0, nbformat.v4.new_code_cell(first_cell))
    client = nbclient.NotebookClient(nb, kernel_name=KERNEL_NAME, timeout=CELL_TIMEOUT_S, allow_errors=False,
                                     resources={"metadata": {"path": str(workdir)}})
    client.execute()
    if first_cell:
        nb.cells.pop(0)
    return nb


@pytest.fixture
def executed(notebook, fake_engine_env, tmp_path, monkeypatch):
    """(executed notebook without the CI cell, its working folder, files in the repository's top folder and
    notebooks/ before)."""
    pytest.importorskip("duckdb")
    from test_docs import example_inputs

    workdir = tmp_path / "work" / "notebooks"
    workdir.mkdir(parents=True)
    example_inputs(workdir.parent)  # ../example
    monkeypatch.setenv("POCKETVINA_TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    before = _repo_listing()
    nb = _execute(notebook, workdir, tmp_path, monkeypatch, first_cell=CI_CELL)
    return nb, workdir, before


def test_notebook_runs(executed, fake_engine_env):
    from pocketvina.io.results import Results
    from pocketvina.workspace import NOT_SHOWN

    nb, workdir, repo_before = executed
    for cell in _code_cells(nb):
        errors = [o for o in cell.outputs if o.output_type == "error"]
        assert not errors, f"{_tag(cell)} cell: {errors}"
    assert fake_engine_env.calls(), "the notebook did not dock with the fake engine"
    import pocketvina

    assert f"PocketVina {pocketvina.__version__}" in _text(_cell(nb, "install"))

    # the dock cell's settings reached the run (htvs_mode: the 5 best pockets), and its best scores table appeared
    results = workdir / "results"
    params = json.loads((results / "MANIFEST.json").read_text())["params"]
    assert (params["output_format"], params["top_pockets"]) == ("atlas", 5)
    with Results(str(results)) as res:
        assert res.proteins() == ["1fbl", "2W83"]
        n_pairs = len(res.best_scores())
    table = "\n".join(_outputs(_cell(nb, "dock"), "text/html"))
    assert "ligand_efficiency" in table and "molecule1" in table and table.count("<tr") == n_pairs + 1
    # the results folder opened again without docking: the same table
    assert "\n".join(_outputs(_cell(nb, "load"), "text/html")).count("<tr") == n_pairs + 1

    # the 3D pages: an iframe holding the whole page, with a line for where it is not drawn
    for tag in SHOW_TAGS:
        html = _outputs(_cell(nb, tag), "text/html")
        assert len(html) == 1 and html[0].startswith("<iframe srcdoc=") and len(html[0]) > 100_000, tag
        assert _outputs(_cell(nb, tag), "text/plain") == [NOT_SHOWN], tag
    assert "PV_1fbl_molecule1" in "\n".join(_outputs(_cell(nb, "select"), "text/html"))

    # PoseBusters and the export: the best pose in each docked pocket of 1fbl
    with Results(str(results)) as res:
        n_pockets = len(res.select("1fbl", "molecule1", pocket="all").poses)
    bust = "\n".join(_outputs(_cell(nb, "bust"), "text/html"))
    assert "pb_valid" in bust and "failed_checks" in bust and bust.count("<tr") == n_pockets + 1
    exported = workdir / "PV_1fbl_molecule1_pocketall_top.sdf"
    assert repr(str(exported)) in _text(_cell(nb, "export"))
    assert exported.read_text().count("$$$$") == n_pockets
    assert _repo_listing() == repo_before


# --- the Atlas notebook ----------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def atlas_notebook():
    return nbformat.read(str(ATLAS_NOTEBOOK), as_version=4)


def test_atlas_notebook_is_valid_short_and_clean(atlas_notebook):
    nb = atlas_notebook
    nbformat.validate(nb)
    assert [_tag(c) for c in _code_cells(nb)] == ATLAS_CODE_TAGS
    for cell in _code_cells(nb):
        assert cell.outputs == [] and cell.execution_count is None, f"stored outputs in the {_tag(cell)} cell"
        ast.parse(cell.source, feature_version=(3, 9))
        assert len(cell.source.splitlines()) <= MAX_CODE_LINES, _tag(cell)
    assert "accelerator" not in nb.metadata  # no GPU: Colab's default runtime
    assert nb.metadata["kernelspec"]["name"] == "python3"
    kinds = [c.cell_type for c in nb.cells]
    assert kinds[0] == "markdown" and nb.cells[0].source.startswith("# ")
    assert "colab.research.google.com/github/BIMSBbioinfo/PocketVina/blob/main/notebooks/atlas_colab.ipynb" \
        in nb.cells[0].source
    assert "Run all" in nb.cells[0].source
    assert all(kinds[i - 1] == "markdown" for i, kind in enumerate(kinds) if kind == "code")


def test_atlas_notebook_uses_the_simple_api(atlas_notebook):
    sources = "\n".join(c.source for c in _code_cells(atlas_notebook))
    for call in ("atlas = Atlas()", 'atlas.search("imatinib")', 'atlas.protein("ABL1")', 'best_scores(protein="ABL1"',
                 'best_scores(ligand="imatinib"', 'atlas.select(protein="ABL1", ligand="imatinib", pocket="all")',
                 "pair.show()", "pair.bust()", "pair.export()", "pair.export_protein()",
                 'atlas.download(proteins=["ABL1"], out="my_atlas")', 'Atlas("my_atlas")'):
        assert call in sources, call
    # no test plumbing, and none of the quickstart's GPU setup
    for unwanted in ("POCKETVINA_", "/tests", "colab_setup", "apt", "java", "OpenCL", "git clone"):
        assert unwanted not in sources, unwanted
    install = _cell(atlas_notebook, "install").source
    assert '"google.colab" in sys.modules' in install and '"pip", "install"' in install


@pytest.fixture
def atlas_executed(atlas_notebook, atlas_dir, tmp_path, monkeypatch):
    """(executed Atlas notebook, its working folder): it reads the fixture release, not the internet."""
    monkeypatch.setenv("POCKETVINA_ATLAS", str(atlas_dir))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    workdir = tmp_path / "work"
    workdir.mkdir()
    before = _repo_listing()
    try:
        nb = _execute(atlas_notebook, workdir, tmp_path, monkeypatch)
    except Exception as e:  # only the error's type: nbclient's message holds the traceback, which may name drugs
        raise AssertionError(f"the Atlas notebook failed: {getattr(e, 'ename', type(e).__name__)}") from None
    assert _repo_listing() == before
    return nb, workdir


def test_atlas_notebook_runs(atlas_executed):
    import pocketvina
    from pocketvina.workspace import NOT_SHOWN

    nb, workdir = atlas_executed
    for cell in _code_cells(nb):
        errors = [o.ename for o in cell.outputs if o.output_type == "error"]
        assert not errors, f"{_tag(cell)} cell: {errors}"
    assert f"PocketVina {pocketvina.__version__}" in _text(_cell(nb, "install"))
    assert "PocketVina Atlas" in "\n".join(_outputs(_cell(nb, "atlas"), "text/html"))
    assert "<tr" in "\n".join(_outputs(_cell(nb, "search"), "text/html"))
    assert "P00519" in "\n".join(_outputs(_cell(nb, "protein"), "text/html"))  # ABL1
    tables = _outputs(_cell(nb, "best-scores"), "text/html")  # of the protein, then of the drug
    assert len(tables) == 2 and all("vina" in t and "ligand_efficiency" in t for t in tables)
    # the 3D page: an iframe holding the whole page, with a line for where it is not drawn
    html = _outputs(_cell(nb, "select"), "text/html")
    assert len(html) == 1 and html[0].startswith("<iframe srcdoc=") and len(html[0]) > 100_000
    assert _outputs(_cell(nb, "select"), "text/plain") == [NOT_SHOWN]
    assert "pb_valid" in "\n".join(_outputs(_cell(nb, "bust"), "text/html"))
    for name in ("PV_P00519_PVL00129_pocketall_top.sdf", "PV_P00519.pdb"):
        assert (workdir / name).is_file() and name in _text(_cell(nb, "export")), name
    assert (workdir / "my_atlas" / "meta.json").is_file()
    assert "P00519" in "\n".join(_outputs(_cell(nb, "download"), "text/html"))
