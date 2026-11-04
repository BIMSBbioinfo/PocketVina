"""Packaging: the sdist and wheel built from a copy of the checkout (setuptools' PEP 517 hooks, as ``pip install .``
and ``python -m build`` call them) and the engine Makefile's settings and ``make install``.

Building must neither download nor compile anything: compilers, make, wget, curl, conda, apt-get and sudo are
replaced by tripwire scripts on PATH that record any call. The copy holds the Boost/P2Rank trees and stale kernel
binaries of an older source install and the SOURCES.txt of an older build (which every build reads again), listing
files that are not part of the package; all of them must stay out of both archives.
"""
import ast
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
import tarfile
import zipfile
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
PACKAGE_DIR = REPO_ROOT / "pocketvina"
ENGINE = PACKAGE_DIR / "PocketVina-GPU"
TRIPWIRES = ("gcc", "g++", "cc", "c++", "make", "cmake", "wget", "curl", "conda", "mamba", "apt-get", "sudo", "git")
BUILD_TIMEOUT_S = 300
VERSION = "0.3.0"
KERNELS = sorted(p.name for p in (PACKAGE_DIR / "OpenCL" / "src" / "kernels").glob("*.cl"))
MAKE_VARS = ("BOOST_ROOT", "BOOST_LIB_PATH", "OPENCL_ROOT", "OPENCL_LIB_PATH", "PREFIX", "DESTDIR", "CXX", "CXXFLAGS",
             "LDFLAGS", "MAKEFLAGS", "MFLAGS")

ELF_PLATFORMS = {b"\x3e\0": "linux_x86_64", b"\xb7\0": "linux_aarch64"}  # e_machine (little-endian) -> tag
VIEWER_FILES = {f"pocketvina/viewer/{n}" for n in (
    "viewer.js", "viewer.css", "bridge.js", "workspace.js", "workspace.css", "README.md",
    "molstar-pv-5.13.1/molstar-pv.js", "molstar-pv-5.13.1/molstar-pv.css", "molstar-pv-5.13.1/LICENSE",
    "molstar-pv-5.13.1/THIRD-PARTY-LICENSES.txt", "molstar-pv-5.13.1/SOURCES.txt")}
NOT_PACKAGED = ("benchmark/astex/main.py", "doc/assets/pocketvina_figure.png", "app/app.py", "scripts_test/run.sh",
                "Dockerfile", ".dockerignore", ".gitignore")
needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="needs GNU make")


def _needs_setuptools_77():
    """The build backend of pyproject.toml (setuptools >= 77) in this Python, else skip."""
    setuptools = pytest.importorskip("setuptools")
    if int(setuptools.__version__.split(".")[0]) < 77:
        pytest.skip(f"setuptools {setuptools.__version__} < 77, the build backend pyproject.toml requires")


def _copy_checkout(dst: Path, engine: bytes = None) -> Path:
    """The files a build reads (as in a git checkout) plus an older install's and an older build's leftovers;
    ``engine`` replaces the engine binary (b"" removes it)."""
    skip = shutil.ignore_patterns("__pycache__", "*.pyc", "boost_1_77_0", "p2rank_2.5", ".pytest_cache",
                                  ".ipynb_checkpoints")
    dst.mkdir(parents=True)
    for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "LICENSE"):
        shutil.copy2(REPO_ROOT / name, dst / name)
    for name in ("pocketvina", "tests", "example", "notebooks"):
        shutil.copytree(REPO_ROOT / name, dst / name, ignore=skip, symlinks=True)
    pkg = dst / "pocketvina"
    (pkg / "p2rank_2.5" / "bin").mkdir(parents=True)
    (pkg / "p2rank_2.5" / "prank").write_text("#!/bin/sh\n")
    (pkg / "boost_1_77_0" / "lib").mkdir(parents=True)
    (pkg / "boost_1_77_0" / "lib" / "libboost_system.a").write_bytes(b"!<arch>\n")
    for stale in ("Kernel1_Opt.bin", "Kernel2_Opt.bin"):  # git-tracked leftovers of v0.1
        (pkg / stale).write_bytes(b"stale")
    # git-tracked files that are not part of the package, listed in the SOURCES.txt of a v0.1 build
    for name in NOT_PACKAGED:
        (dst / name).parent.mkdir(parents=True, exist_ok=True)
        (dst / name).write_text("not packaged\n")
    (dst / "pocketvina.egg-info").mkdir()
    (dst / "pocketvina.egg-info" / "SOURCES.txt").write_text("\n".join(
        NOT_PACKAGED + ("pocketvina/p2rank_2.5/prank", "pocketvina/boost_1_77_0/lib/libboost_system.a",
                        "pocketvina/Kernel1_Opt.bin", "pocketvina/__init__.py")) + "\n")
    if engine is not None:
        (pkg / "PocketVina-GPU").unlink()
        if engine:
            (pkg / "PocketVina-GPU").write_bytes(engine)
            (pkg / "PocketVina-GPU").chmod(0o755)
    return dst


def _tripwire_env(tmp_path: Path) -> dict:
    """The environment with tripwire scripts first on PATH; each call is appended to ``tripwire_calls.log``."""
    bin_dir = tmp_path / "tripwires"
    if not bin_dir.is_dir():
        bin_dir.mkdir()
        for tool in TRIPWIRES:
            exe = bin_dir / tool
            exe.write_text(f'#!/bin/sh\necho "{tool} $*" >> "{tmp_path / "tripwire_calls.log"}"\nexit 1\n')
            exe.chmod(0o755)
    return dict(os.environ, PATH=f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}", PYTHONDONTWRITEBYTECODE="1")


def _build(tmp_path: Path, src: Path, hook: str) -> Path:
    """Run setuptools' PEP 517 ``hook`` (build_sdist / build_wheel) in ``src``; the archive built."""
    out = tmp_path / "dist"
    out.mkdir(exist_ok=True)
    code = f"import sys\nfrom setuptools import build_meta\nprint(build_meta.{hook}(sys.argv[1]))\n"
    run = subprocess.run([sys.executable, "-c", code, str(out)], cwd=src, env=_tripwire_env(tmp_path),
                         capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
    assert run.returncode == 0, run.stdout + run.stderr
    log = tmp_path / "tripwire_calls.log"
    assert not log.exists(), f"the build ran: {log.read_text()}"
    return out / run.stdout.strip().splitlines()[-1]


def z_bytes(whl: Path, name: str) -> bytes:
    with zipfile.ZipFile(whl) as z:
        return z.read(name)


def _files(root: Path) -> set:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def _elf_header(machine: int) -> bytes:
    """The first bytes of a 64-bit little-endian ELF executable for ``machine``."""
    return b"\x7fELF\x02\x01\x01" + b"\0" * 9 + (2).to_bytes(2, "little") + machine.to_bytes(2, "little") + b"\0" * 44


# --- wheel and sdist -----------------------------------------------------------------------------------------------

def test_setup_py_has_no_install_side_effects():
    """setup.py only tags the wheel: no subprocess, network or file system work (pre-v0.2 it downloaded P2Rank and
    Boost, compiled the engine and ran conda/apt-get from egg_info, develop and install)."""
    tree = ast.parse((REPO_ROOT / "setup.py").read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported == {"os", "sysconfig", "setuptools", "setuptools.command.bdist_wheel"}
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert {c for c in calls if c.startswith("os.")} <= {"os.path.join", "os.path.dirname", "os.path.abspath",
                                                         "os.path.isfile"}
    commands = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & {"install", "develop", "egg_info"}
    assert not commands


def test_wheel_contents(tmp_path):
    """One wheel for every Python 3, tagged for the engine's platform; package data = engine + kernel sources."""
    _needs_setuptools_77()
    src = _copy_checkout(tmp_path / "src")
    before = _files(src)
    whl = _build(tmp_path, src, "build_wheel")
    plat = ELF_PLATFORMS.get(ENGINE.read_bytes()[18:20], sysconfig.get_platform().replace("-", "_").replace(".", "_"))
    assert whl.name == f"pocketvina-{VERSION}-py3-none-{plat}.whl"  # x86_64 for the binary in git
    with zipfile.ZipFile(whl) as z:
        names = set(z.namelist())
        info = {i.filename: i for i in z.infolist()}
        meta = z.read(f"pocketvina-{VERSION}.dist-info/METADATA").decode()
        wheel = z.read(f"pocketvina-{VERSION}.dist-info/WHEEL").decode()
        entry_points = z.read(f"pocketvina-{VERSION}.dist-info/entry_points.txt").decode()
        engine = z.read("pocketvina/PocketVina-GPU")
    assert engine == ENGINE.read_bytes()
    assert (info["pocketvina/PocketVina-GPU"].external_attr >> 16) & stat.S_IXUSR  # pip keeps the mode
    assert "Root-Is-Purelib: true" in wheel  # the files sit at the root, only the tag is not "any"
    py = {n for n in names if n.endswith(".py")}
    assert {"pocketvina/__init__.py", "pocketvina/cli.py", "pocketvina/p2rank.py", "pocketvina/io/results.py"} <= py
    assert all(n.startswith(("pocketvina/", f"pocketvina-{VERSION}.dist-info/")) for n in names)
    assert {f"pocketvina/OpenCL/src/kernels/{k}" for k in KERNELS} <= names
    assert {"pocketvina/OpenCL/inc/kernel2.h", "pocketvina/OpenCL/inc/commonMacros.h"} <= names
    # the notebook viewer (pocketvina.workspace): every file it reads, the Mol* bundle's licences; not the site's
    # original viewer.js (pocketvina/viewer/upstream/, kept in the sdist to compare)
    assert VIEWER_FILES <= names and not any("viewer/upstream" in n for n in names)
    assert z_bytes(whl, "pocketvina/viewer/molstar-pv-5.13.1/molstar-pv.js") == (
        PACKAGE_DIR / "viewer" / "molstar-pv-5.13.1" / "molstar-pv.js").read_bytes()
    for unwanted in (".bin", "boost_1_77_0", "p2rank_2.5", "QuickVina2", "pocketvina/lib/", "pocketvina/main/",
                     "Makefile", "wrapcl.cpp", "tests/", "example/"):
        assert not any(unwanted in n for n in names), unwanted
    assert "[console_scripts]\npocketvina = pocketvina.cli:main" in entry_points
    assert f"Version: {VERSION}" in meta and "Requires-Python: >=3.9" in meta
    requires = {line.split(": ", 1)[1] for line in meta.splitlines() if line.startswith("Requires-Dist: ")}
    assert {"numpy", "pandas<3", "rdkit", "openbabel-wheel", "pyarrow", "duckdb", "click", "tqdm"} <= requires
    assert 'tomli; python_version < "3.11"' in requires
    # PoseBusters and the viewers are dependencies since 0.2; the bust and viz extras are empty (old commands work)
    assert {"posebusters", "py3Dmol", 'molviewspec; python_version >= "3.10"'} <= requires
    assert not any(r.endswith(('extra == "bust"', 'extra == "viz"')) for r in requires)
    # Meeko's PyPI package declares no dependencies: the extra brings what importing it needs, and a Meeko that
    # imports on Python 3.9 (0.8 does not)
    assert {f'{dist}; extra == "meeko"' for dist in ("scipy", "gemmi")} <= requires
    assert {'meeko; python_version >= "3.10" and extra == "meeko"',
            'meeko<0.8; python_version < "3.10" and extra == "meeko"'} <= requires
    extras = {line.split(": ", 1)[1] for line in meta.splitlines() if line.startswith("Provides-Extra: ")}
    assert extras == {"bust", "viz", "meeko", "app", "dev", "docs"}
    # the source tree gets only build artefacts (in-tree builds write build/ and *.egg-info, as pip does)
    added = {f for f in _files(src) - before if not f.startswith(("build/", "pocketvina.egg-info/"))}
    assert added == set() and _files(src) >= before


@pytest.mark.parametrize("engine, plat", [
    (b"", "any"),  # no binary: a pure wheel (engine from conda or $POCKETVINA_ENGINE)
    (_elf_header(183), "linux_aarch64"),
    (b"#!/bin/sh\necho not ELF\n", sysconfig.get_platform().replace("-", "_").replace(".", "_")),
], ids=["no-engine", "aarch64", "not-elf"])
def test_wheel_tag_follows_the_engine(tmp_path, engine, plat):
    _needs_setuptools_77()
    whl = _build(tmp_path, _copy_checkout(tmp_path / "src", engine=engine), "build_wheel")
    assert whl.name == f"pocketvina-{VERSION}-py3-none-{plat}.whl"
    with zipfile.ZipFile(whl) as z:
        assert ("pocketvina/PocketVina-GPU" in z.namelist()) == bool(engine)


def test_sdist_contents(tmp_path):
    """The sdist rebuilds the wheel and the engine (make -C pocketvina, with Boost from elsewhere) and has the tests."""
    _needs_setuptools_77()
    sdist = _build(tmp_path, _copy_checkout(tmp_path / "src"), "build_sdist")
    assert sdist.name == f"pocketvina-{VERSION}.tar.gz"
    with tarfile.open(sdist) as tar:
        members = {m.name.split("/", 1)[1]: m for m in tar.getmembers() if "/" in m.name}
    names = set(members)
    for needed in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "LICENSE", "pocketvina/Makefile",
                   "pocketvina/PocketVina-GPU", "pocketvina/main/main.cpp", "pocketvina/lib/main_procedure_cl.cpp",
                   "pocketvina/OpenCL/src/wrapcl.cpp", "pocketvina/OpenCL/inc/wrapcl.h", "pocketvina/p2rank.py",
                   "tests/conftest.py", "tests/fake_engine.py", "tests/data/smiles.csv", "example/ligands.txt",
                   "notebooks/quickstart_colab.ipynb", "pocketvina/viewer/upstream/viewer.js"):
        assert needed in names, needed
    assert VIEWER_FILES <= names
    assert {f"pocketvina/OpenCL/src/kernels/{k}" for k in KERNELS} <= names
    assert members["pocketvina/PocketVina-GPU"].mode & stat.S_IXUSR
    for unwanted in ("boost_1_77_0", "p2rank_2.5", "QuickVina2", "Kernel1_Opt.bin", "__pycache__", ".pyc"):
        assert not any(unwanted in n for n in names), unwanted
    assert not names & set(NOT_PACKAGED)  # listed in the older build's SOURCES.txt


def test_sdist_tests_skip_what_only_the_repository_has(tmp_path):
    """The tests of the docs, README links and conda recipe run in an unpacked sdist, which has no docs/, recipe/,
    app/ or doc/: they skip what is missing there instead of failing."""
    _needs_setuptools_77()
    pytest.importorskip("yaml")  # tests/test_recipe.py
    sdist = _build(tmp_path, _copy_checkout(tmp_path / "src"), "build_sdist")
    with tarfile.open(sdist) as tar:
        tar.extractall(tmp_path / "unpacked", **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    root = tmp_path / "unpacked" / f"pocketvina-{VERSION}"
    assert (root / "PKG-INFO").is_file() and not (root / "docs").exists() and not (root / "recipe").exists()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "RATTLER_BUILD")}
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_docs.py",
                          "tests/test_readme.py", "tests/test_recipe.py"], cwd=root, capture_output=True, text=True,
                         timeout=BUILD_TIMEOUT_S, env=dict(env, PYTHONDONTWRITEBYTECODE="1"))
    assert run.returncode == 0, run.stdout[-5000:] + run.stderr[-2000:]
    # the skip reasons (-ra in pyproject.toml's addopts)
    assert "an sdist has no docs/" in run.stdout and "an sdist has no recipe/" in run.stdout, run.stdout[-5000:]


# --- the engine Makefile -------------------------------------------------------------------------------------------

def _make(*args, cwd=PACKAGE_DIR, env_extra=None, check=True) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in MAKE_VARS}
    env.update(env_extra or {})
    return subprocess.run(["make", "--no-print-directory", "-C", str(cwd), *args], env=env, capture_output=True,
                          text=True, timeout=60, check=check)


def _boost_source_tree(root: Path) -> Path:
    for name in ("thread.cpp", "once.cpp"):
        path = root / "libs" / "thread" / "src" / "pthread" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("// stand-in\n")
    (root / "lib").mkdir()
    (root / "lib" / "libboost_system.a").write_bytes(b"!<arch>\n")
    return root


def _installed_boost(root: Path, system_stub: bool) -> Path:
    (root / "include" / "boost").mkdir(parents=True)
    (root / "lib").mkdir()
    for lib in ("program_options", "filesystem", "thread") + (("system",) if system_stub else ()):
        (root / "lib" / f"libboost_{lib}.so").write_bytes(b"")
    return root


@needs_make
def test_makefile_defaults_build_as_before(tmp_path):
    """Default settings: gcc, the Boost source tree (Boost.Thread compiled in, static libraries), CUDA's OpenCL."""
    boost = _boost_source_tree(tmp_path / "boost_1_77_0")
    cmd = _make("-n", "out", env_extra={"BOOST_ROOT": str(boost)}).stdout
    assert cmd.startswith("gcc -o PocketVina-GPU ")
    assert f"-I{boost} -I{boost}/boost " in cmd and "-I/usr/local/cuda/include" in cmd
    assert f"{boost}/libs/thread/src/pthread/thread.cpp {boost}/libs/thread/src/pthread/once.cpp" in cmd
    assert "-lboost_program_options -lboost_system -lboost_filesystem" in cmd and "-lboost_thread" not in cmd
    assert "-L/usr/local/cuda/lib64" in cmd and "rpath" not in cmd
    assert "-O3" in cmd and cmd.rstrip().endswith("-DNDEBUG") and "BUILD_KERNEL_FROM_SOURCE" not in cmd
    assert "BUILD_KERNEL_FROM_SOURCE" in _make("-n", "source", env_extra={"BOOST_ROOT": str(boost)}).stdout
    printed = _make("-s", "print").stdout
    assert f"BOOST_ROOT = {PACKAGE_DIR}/boost_1_77_0" in printed and "CXX = gcc" in printed
    assert "OPENCL_ROOT = /usr/local/cuda" in printed and "PREFIX = /usr/local" in printed
    # the names before v0.2, on the command line as the README had them
    old = _make("-n", "out", f"BOOST_LIB_PATH={boost}", "OPENCL_LIB_PATH=/opt/cuda").stdout
    assert f"{boost}/libs/thread/src/pthread/once.cpp" in old and "-I/opt/cuda/include" in old


@needs_make
@pytest.mark.parametrize("system_stub", [True, False], ids=["boost-1.88", "boost-1.89"])
def test_makefile_settings_from_the_environment(tmp_path, system_stub):
    """An installed (e.g. conda) Boost is linked dynamically; compiler and flags come from the environment."""
    boost = _installed_boost(tmp_path / "conda", system_stub)
    env = {"BOOST_ROOT": str(boost), "OPENCL_ROOT": "/opt/ocl", "CXX": "x86_64-conda-linux-gnu-c++",
           "CXXFLAGS": "-O2 -fPIC", "LDFLAGS": "-Wl,--as-needed -L/opt/extra"}
    cmd = _make("-n", "out", env_extra=env).stdout
    assert cmd.startswith("x86_64-conda-linux-gnu-c++ -o PocketVina-GPU -O2 -fPIC ")
    assert cmd.index("-O2") < cmd.index("-O3")  # the engine's -O3 wins
    assert f"-I{boost}/include" in cmd and "-I/opt/ocl/include" in cmd and "-L/opt/ocl/lib64 -L/opt/ocl/lib" in cmd
    assert "libs/thread" not in cmd and "-lboost_thread" in cmd and f"-Wl,-rpath,{boost}/lib" in cmd
    assert ("-lboost_system" in cmd) == system_stub
    assert cmd.index("-Wl,--as-needed") < cmd.index("-lboost_program_options")  # before the libraries
    assert "CXX = x86_64-conda-linux-gnu-c++" in _make("-s", "print", env_extra=env).stdout


@needs_make
def test_makefile_install(tmp_path):
    src = tmp_path / "pocketvina"
    src.mkdir()
    shutil.copy2(PACKAGE_DIR / "Makefile", src / "Makefile")
    shutil.copytree(PACKAGE_DIR / "OpenCL", src / "OpenCL")
    missing = _make("install", f"PREFIX={tmp_path / 'never'}", cwd=src, check=False)
    assert missing.returncode != 0 and "run make first" in missing.stderr and not (tmp_path / "never").exists()
    (src / "PocketVina-GPU").write_text("#!/bin/sh\necho engine\n")
    (src / "PocketVina-GPU").chmod(0o755)
    _make("install", cwd=src, env_extra={"PREFIX": str(tmp_path / "prefix")})
    exe = tmp_path / "prefix" / "bin" / "pocketvina-engine"
    share = tmp_path / "prefix" / "share" / "pocketvina" / "OpenCL"
    assert exe.read_text() == (src / "PocketVina-GPU").read_text() and stat.S_IMODE(exe.stat().st_mode) == 0o755
    assert sorted(p.name for p in (share / "src" / "kernels").iterdir()) == KERNELS
    headers = sorted(p.name for p in (src / "OpenCL" / "inc").iterdir())
    assert sorted(p.name for p in (share / "inc").iterdir()) == headers
    staged = _make("-n", "install", "PREFIX=/opt/pv", "DESTDIR=/stage", cwd=src).stdout
    assert "install -m 755 PocketVina-GPU /stage/opt/pv/bin/pocketvina-engine" in staged
    assert "/stage/opt/pv/share/pocketvina/OpenCL/src/kernels" in staged

    # what pocketvina runs with that prefix on PATH and no package binary (e.g. the conda packages)
    code = ("import sys; from pocketvina import engine; engine.PACKAGE_DIR = sys.argv[1]; "
            "print(engine.engine_path()); print(engine.kernel_source_dir())")
    run = subprocess.run([sys.executable, "-c", code, str(tmp_path / "no_binary")], capture_output=True, text=True,
                         env=dict(os.environ, PATH=f"{exe.parent}{os.pathsep}{os.environ.get('PATH', '')}",
                                  PYTHONPATH=str(REPO_ROOT)), timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == [str(exe), str(tmp_path / "prefix" / "share" / "pocketvina")]
