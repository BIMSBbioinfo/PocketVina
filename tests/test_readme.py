"""README.md: the one-line install, and its relative links and #anchors point to files and headings that exist; what
the install instructions build inside the checkout is git-ignored. The examples the README points to are in
docs/quickstart.md: its Quick Start runs as written (its shell command and Python blocks, one after the other in one
namespace) with the fake engine and the fixture P2Rank pockets, and once for real on a GPU (``POCKETVINA_GPU_TESTS=1``,
P2Rank and Java needed); so does "P2Rank first, docking later"."""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from test_docs import (
    EXAMPLE_LIGANDS,
    FENCE,
    _shell_commands,
    example_inputs,
    run_section,
    section,
    stub_p2rank,  # noqa: F401  (a fixture)
)

ROOT = Path(__file__).resolve().parent.parent
TEXT = (ROOT / "README.md").read_text(encoding="utf-8")
SDIST = (ROOT / "PKG-INFO").is_file()  # the tests of an sdist, not of a checkout
QUICKSTART = ROOT / "docs" / "quickstart.md"  # the README's examples (not in an sdist: MANIFEST.in)
INSTALLATION = ROOT / "docs" / "installation.md"
LINK = re.compile(r"\]\((?P<md>[^)\s]+)\)|\b(?:src|href)=\"(?P<html>[^\"]+)\"")
HEADING = re.compile(r"^#{1,6} +(?P<title>.+?) *$", re.M)
COMMENT = re.compile(r"<!--.*?-->", re.S)
CODE_SPAN = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")


def _blocks(lang, text=TEXT):
    return [m.group("body") for m in FENCE.finditer(text) if m.group("lang").strip() == lang]


def _section(title: str) -> str:
    """The text of the README's ``## title`` section, up to the next ``## `` heading."""
    return section(TEXT, title)


def _doc(title: str, page: Path = QUICKSTART) -> str:
    """The ``## title`` section of a docs page (skipped where docs/ is not there, e.g. in an sdist)."""
    if not page.is_file():
        pytest.skip(f"no {page.relative_to(ROOT)} (an sdist)")
    return section(page.read_text(encoding="utf-8"), title)


def _anchor(title: str) -> str:
    # GitHub's (and MyST's) heading anchors: lower case, punctuation dropped, spaces as hyphens
    return re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")


def _anchors(text: str) -> set:
    return {_anchor(m.group("title")) for m in HEADING.finditer(COMMENT.sub("", FENCE.sub("", text)))}


def test_readme_install_is_one_line():
    """Installation: one pip command, no OpenCL exports, extras or setup step (P2Rank is downloaded at the first run);
    the README shows no other command or code."""
    install = _section("Installation")
    commands = [line for body in _blocks("bash", install) for line in _shell_commands(body)]
    assert commands == ["pip install pocketvina"]
    for unwanted in ("OCL_ICD_VENDORS", "LD_LIBRARY_PATH", "[bust", "setup p2rank"):
        assert unwanted not in install
    assert {m.group("lang").strip() for m in FENCE.finditer(TEXT)} == {"bash", "bibtex"}  # no examples


def test_install_is_short():
    """docs/installation.md: a conda environment with Java and `pip install pocketvina`; from source: clone, the same
    environment, `pip install .`."""
    if not INSTALLATION.is_file():
        pytest.skip("no docs/installation.md (an sdist)")
    pypi, source = _blocks("bash", INSTALLATION.read_text(encoding="utf-8"))[:2]
    env = ['conda create -n pocketvina -c conda-forge python=3.11 "openjdk>=17" -y', "conda activate pocketvina"]
    assert _shell_commands(pypi) == env + ["pip install pocketvina"]
    assert _shell_commands(source) == ["git clone https://github.com/BIMSBbioinfo/PocketVina", "cd PocketVina", *env,
                                       "pip install ."]


def test_quick_start_uses_the_v01_config():
    """The full v0.1 example (README.md of 0.1.0) with output_format="atlas", then the new methods."""
    quick_start = _doc("Quick Start")
    code = "\n".join(_blocks("python", quick_start))
    for name in ("protein_paths=", "smiles_file=", "output_dir=", "batch_size=", 'output_format="atlas"',
                 "P2RankConfig(", "threads=", "visualizations=", "vis_copy_proteins=", "model=",
                 "PocketConfig(", "size_x=", "size_y=", "size_z=", "thread=", "PocketVinaGPU(config, htvs_mode=True)",
                 ".execute_docking()", ".best_scores(", ".show()", ".select(", 'pocket="all"', 'mode="all"',
                 "pair.show()", "pair.bust()", "pair.export()"):
        assert name in code, name
    # PoseBusters is a step of its own (pair.bust()), not a column of the tables
    assert "pb_" not in quick_start


def test_readme_links_exist():
    text = COMMENT.sub("", FENCE.sub("", TEXT))
    anchors = _anchors(TEXT)
    targets = [m.group("md") or m.group("html") for m in LINK.finditer(text)]
    assert len(targets) >= 10
    for target in targets:
        if target.startswith("#"):
            assert target[1:] in anchors, f"no heading for {target}"
        elif not re.match(r"[a-z]+:", target):
            if SDIST and not (ROOT / target.split("/")[0]).exists():
                continue  # an sdist (MANIFEST.in) has no docs/, recipe/, app/ or doc/
            path, _, anchor = target.partition("#")
            assert (ROOT / path).exists(), f"no file {target}"
            if anchor:
                assert anchor in _anchors((ROOT / path).read_text(encoding="utf-8")), f"no heading for {target}"


@pytest.mark.parametrize("path", [".venv/pyvenv.cfg", "conda-bld/noarch/repodata.json", "docs/_build/html/index.html"])
def test_build_folders_in_the_checkout_are_ignored(path):
    """`python -m venv .venv` (docs/installation.md), `rattler-build ... --output-dir conda-bld` (recipe/README.md)
    and `sphinx-build -b html docs docs/_build/html` leave `git status` clean."""
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("needs git and a git checkout")
    run = subprocess.run([git, "-C", str(ROOT), "check-ignore", "-q", "--no-index", path])
    assert run.returncode == 0, f"{path} is not git-ignored"


# --- the Quick Start, run -------------------------------------------------------------------------------------------

def check_quick_start(ns: dict, workdir: Path) -> None:
    """What the Quick Start says it does, in ``workdir`` after it ran there (``run_section``)."""
    from pocketvina import PocketVinaGPU, Results
    from pocketvina.io.results import BEST_SCORES_COLUMNS

    results = workdir / "results"
    proteins = {"1fbl", "2W83"}
    # the results folder (docs/output-format.md)
    for name in ("MANIFEST.json", "manifest/<protein>.json", "poses/<protein>.poses", "structures"):
        for protein in proteins:
            assert (results / name.replace("<protein>", protein)).exists(), name
    tables = {p.stem for p in (results / "tables").iterdir()}  # ligands.parquet, pairs/, ...
    assert tables >= {"ligands", "proteins", "pockets", "pocket_runs", "modes", "pairs"}, tables

    # execute_docking showed the best scores (printed: not in a notebook)
    assert "Best scores:" in ns["_stdout"] and "molecule1" in ns["_stdout"]
    pv = ns["pocketvina"]
    best = pv.best_scores()
    assert list(best.columns) == list(BEST_SCORES_COLUMNS)
    assert set(best["protein_id"]) == proteins and set(best["ligand_id"]) == set(EXAMPLE_LIGANDS)
    assert best["vina"].is_monotonic_increasing
    assert not any(c.startswith("pb_") for c in best.columns)
    one = pv.best_scores(protein="1fbl", top=5)
    assert len(one) == min(5, len(EXAMPLE_LIGANDS)) and set(one["protein_id"]) == {"1fbl"}
    # a results folder of an earlier session
    assert PocketVinaGPU(ns["config"]).best_scores().equals(best)
    assert Results(str(results)).best_scores().equals(best)

    # the 3D pages: written as HTML files outside a notebook
    assert (workdir / "PV_results.html").stat().st_size > 100_000
    assert (workdir / "results.html").stat().st_size > 100_000
    # the last selection: every mode of the best pocket, shown, checked and exported
    pair = ns["pair"]
    assert (pair.protein, pair.ligand, pair.pocket, pair.mode) == ("1fbl", "molecule1", None, "all")
    assert pair.name == f"PV_1fbl_molecule1_pk{pair.best_pocket}_modes"
    assert (workdir / f"{pair.name}.html").is_file()
    sdf = workdir / f"{pair.name}.sdf"
    assert sdf.is_file() and sdf.read_text().count("$$$$") == len(pair.poses)
    assert re.search(r"PV_1fbl_molecule1_pk\d+_modes\.sdf", _doc("Quick Start"))


@pytest.mark.usefixtures("stub_p2rank")
def test_quick_start_runs(tmp_path, fake_engine_env):
    """The Quick Start with the fake engine, P2Rank's fixture pockets and cropped receptors (``stub_p2rank``), the
    example proteins and a few example ligands."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    example_inputs(workdir)
    ns = run_section(_doc("Quick Start"), workdir)
    check_quick_start(ns, workdir)
    assert fake_engine_env.calls()
    # 2 proteins x their 5 best pockets (htvs_mode; 6 pockets with a pose in the fixtures), batch_size=10: one engine
    # call per docked pocket
    assert len(fake_engine_env.calls()) == 6


def test_p2rank_first(tmp_path, monkeypatch, fake_engine_env):
    """The section "P2Rank first, docking later" as written (P2Rank stubbed: the fixture pockets): P2RankProcessor
    writes the pockets, and the docking reads them without running P2Rank again."""
    from pocketvina import main as pv_main
    from pocketvina import pipeline
    from pocketvina.io.writer import ResultsDir
    from test_pipeline import cropped_receptor, write_pockets

    def run_p2rank(self, pdb_file):
        base = os.path.splitext(os.path.basename(pdb_file))[0]
        write_pockets(os.path.join(self.pockets_dir, f"{base}.csv"), 3)
        return True

    def no_p2rank(*args, **kwargs):
        raise AssertionError("P2Rank ran again")

    monkeypatch.setattr(pv_main.P2RankProcessor, "run_p2rank", run_p2rank)
    monkeypatch.setattr(pv_main, "receptor_pdbqt", cropped_receptor())
    monkeypatch.setattr(pipeline, "receptor_pdbqt", cropped_receptor())
    monkeypatch.setattr(pipeline, "predict_pockets", no_p2rank)
    example_inputs(tmp_path)
    (tmp_path / "proteins.txt").write_text("example/proteins/1fbl.pdb\nexample/proteins/2W83.pdb\n")
    run_section(_doc("P2Rank first, docking later"), tmp_path)
    pockets = tmp_path / "pockets" / "proteins" / "pockets"
    assert sorted(p.name for p in pockets.iterdir()) == ["1fbl.csv", "2W83.csv"]
    assert ResultsDir(tmp_path / "results_library1").done_proteins() == {"1fbl", "2W83"}


@pytest.mark.gpu
def test_quick_start_on_the_gpu(tmp_path):
    """The Quick Start for real: P2Rank, the engine on the GPU, PoseBusters (a few example ligands)."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    example_inputs(workdir)
    ns = run_section(_doc("Quick Start"), workdir)
    check_quick_start(ns, workdir)
    print(ns["_stdout"])
