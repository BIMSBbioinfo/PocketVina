"""The conda recipe ``recipe/recipe.yaml`` (rattler-build, recipe format v1) and its engine build script.

Neither conda nor rattler-build is needed: the recipe is read as YAML and checked against pyproject.toml and the
package, and ``recipe/build_engine.sh`` runs on a copy of the engine sources with a stand-in compiler (nothing is
compiled). With ``rattler-build`` on PATH the recipe is also rendered. Building and installing the packages for real
is described in recipe/README.md.
"""
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

import pocketvina
from pocketvina import engine, p2rank

yaml = pytest.importorskip("yaml")

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
RECIPE_DIR = REPO_ROOT / "recipe"
PACKAGE_DIR = REPO_ROOT / "pocketvina"
BUILD_SCRIPT = RECIPE_DIR / "build_engine.sh"
# pyproject.toml dependency -> conda-forge package
CONDA_NAMES = {"openbabel-wheel": "openbabel", "duckdb": "python-duckdb"}
needs_make = pytest.mark.skipif(shutil.which("make") is None or shutil.which("bash") is None,
                                reason="needs GNU make and bash")
# the tests of an sdist (MANIFEST.in ships tests/, not recipe/)
NO_RECIPE = not RECIPE_DIR.is_dir() and (REPO_ROOT / "PKG-INFO").is_file()
needs_recipe = pytest.mark.skipif(NO_RECIPE, reason="an sdist has no recipe/")


def _pyproject() -> dict:
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        tomllib = pytest.importorskip("tomli")
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recipe() -> dict:
    if NO_RECIPE:
        pytest.skip("an sdist has no recipe/")
    return yaml.safe_load((RECIPE_DIR / "recipe.yaml").read_text(encoding="utf-8"))


def _output(recipe: dict, name: str) -> dict:
    (out,) = [o for o in recipe["outputs"] if o["package"]["name"] == name]
    return out


def _render(text: str, context: dict) -> str:
    """``${{ name }}`` of the recipe's context replaced by its value."""
    return re.sub(r"\$\{\{\s*(\w+)\s*\}\}", lambda m: str(context[m.group(1)]), text)


def _spec(requirement: str) -> tuple:
    """``(name, version constraint without spaces)`` of a conda match spec or a PEP 508 requirement (no marker)."""
    requirement = requirement.split(";")[0].strip()
    m = re.match(r"([A-Za-z0-9_.\-]+)\s*(.*)", requirement)
    return m.group(1).lower(), m.group(2).replace(" ", "")


# --- recipe.yaml ---------------------------------------------------------------------------------------------------

def test_recipe_version_is_the_package_version(recipe):
    assert recipe["schema_version"] == 1
    assert recipe["context"]["version"] == pocketvina.__version__
    assert recipe["recipe"]["version"] == "${{ version }}"
    assert [o["package"]["name"] for o in recipe["outputs"]] == ["pocketvina-engine", "pocketvina"]


def test_engine_output(recipe):
    out = _output(recipe, "pocketvina-engine")
    assert out["build"]["script"] == BUILD_SCRIPT.name
    assert out["build"]["skip"] == ['target_platform != "linux-64"']
    reqs = out["requirements"]
    assert "${{ compiler('cxx') }}" in reqs["build"] and "make" in reqs["build"]
    assert {_spec(r)[0] for r in reqs["host"]} == {"libboost-devel", "ocl-icd", "opencl-headers"}
    assert reqs["run"] == ["ocl-icd", "ocl-icd-system"]  # the loader, and the system's ICDs (NVIDIA driver) for it
    scripts = [line for test in out["tests"] for line in test.get("script", [])]
    assert "pocketvina-engine --help" in scripts
    contents = [test["package_contents"] for test in out["tests"] if "package_contents" in test][0]
    assert contents["bin"] == [engine.ENGINE_COMMAND]
    installed = {f for f in contents["files"] if f.startswith(engine.INSTALLED_KERNEL_DIR + "/")}
    assert {f"{engine.INSTALLED_KERNEL_DIR}/OpenCL/src/kernels/{k}" for k in ("kernel1.cl", "kernel2.cl")} <= installed


def test_python_output_dependencies_follow_pyproject(recipe):
    """Run requirements = pyproject.toml's dependencies (conda names), Java for P2Rank, the engine of this version."""
    project = _pyproject()["project"]
    out = _output(recipe, "pocketvina")
    assert out["build"]["noarch"] == "python"
    run = [_render(r, recipe["context"]) for r in out["requirements"]["run"]]
    specs = dict(_spec(r) for r in run)
    expected = {}
    for dep in project["dependencies"]:
        name, constraint = _spec(dep)
        expected[CONDA_NAMES.get(name, name)] = constraint
    assert {k: v for k, v in specs.items() if k in expected} == expected
    assert set(specs) - set(expected) == {"python", "openjdk", "pocketvina-engine"}
    assert specs["openjdk"].split(",")[0] == f">={p2rank.JAVA_MIN}"
    assert specs["pocketvina-engine"] == f"=={pocketvina.__version__}"

    python_min = recipe["context"]["python_min"]
    assert specs["python"] == f">={python_min}"
    assert f"python {python_min}.*" in [_render(r, recipe["context"]) for r in out["requirements"]["host"]]
    # never older than pyproject.toml's requires-python
    lowest = tuple(int(x) for x in project["requires-python"].lstrip(">=").split("."))
    assert tuple(int(x) for x in python_min.split(".")) >= lowest
    tests = [t["python"] for t in out["tests"] if "python" in t]
    assert tests and tests[0]["python_version"] == "${{ python_min }}.*" and "pocketvina" in tests[0]["imports"]


def test_python_output_entry_points_follow_pyproject(recipe):
    scripts = _pyproject()["project"]["scripts"]
    out = _output(recipe, "pocketvina")
    assert out["build"]["python"]["entry_points"] == [f"{name} = {target}" for name, target in scripts.items()]
    lines = out["build"]["script"]
    assert lines[0] == "rm -f pocketvina/PocketVina-GPU"  # noarch: never the prebuilt engine of the source tree
    assert "--no-deps" in lines[-1] and "--no-build-isolation" in lines[-1]
    tests = [line for t in out["tests"] for line in t.get("script", [])]
    assert "pocketvina --help" in tests and 'python -c "import pocketvina"' in tests


def _included(path: str, patterns) -> bool:
    """Whether a source ``filter`` include glob matches ``path`` (``*`` within a folder, ``**`` across folders)."""
    for pattern in patterns:
        regex = re.escape(pattern).replace(r"\*\*", "\0").replace(r"\*", "[^/]*").replace("\0", ".*")
        if re.fullmatch(regex, path):
            return True
    return False


def test_source_filter_has_everything_the_packages_are_built_from(recipe):
    include = recipe["source"]["filter"]["include"]
    tracked = ["pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "LICENSE", "pocketvina/Makefile"]
    for sub in ("main", "lib", "OpenCL", "io", "viewer"):
        tracked += [p.relative_to(REPO_ROOT).as_posix() for p in (PACKAGE_DIR / sub).rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts]
    tracked += [p.relative_to(REPO_ROOT).as_posix() for p in PACKAGE_DIR.glob("*.py")]
    missing = [p for p in tracked if not _included(p, include)]
    assert not missing
    # what neither package is built from: the prebuilt engine, kernel binaries, Boost/P2Rank trees of older installs,
    # and rattler-build output folders inside the checkout (recipe/README.md's conda-bld/, the default ./output)
    for path in ("pocketvina/PocketVina-GPU", "pocketvina/Kernel1_Opt.bin", "pocketvina/boost_1_77_0/boost/config.hpp",
                 "pocketvina/p2rank_2.5/prank", "pocketvina/QuickVina2-GPU-2-1/Kernel1_Opt.bin", "tests/conftest.py",
                 "benchmark/utils/x.py", "example/ligands.txt", "test/x.py",
                 "conda-bld/noarch/pocketvina-0.3.2-pyh4616a5c_0.conda",
                 "output/bld/rattler-build_pocketvina_h4616a5c_1/work/pocketvina/cli.py"):
        assert not _included(path, include), path


@needs_recipe
def test_recipe_renders(tmp_path):
    """rattler-build accepts the recipe (render only: no solve, no download, no build)."""
    exe = os.environ.get("RATTLER_BUILD") or shutil.which("rattler-build")
    if not exe:
        pytest.skip("rattler-build is not on PATH (or $RATTLER_BUILD)")
    run = subprocess.run([exe, "build", "--recipe", str(RECIPE_DIR), "--render-only", "--output-dir",
                          str(tmp_path / "out"), "--log-style", "plain", "--color", "never"],
                         capture_output=True, text=True, timeout=300, cwd=tmp_path, stdin=subprocess.DEVNULL)
    assert run.returncode == 0, run.stdout + run.stderr
    start = run.stdout.index("[")
    rendered, _ = json.JSONDecoder().raw_decode(run.stdout[start:])
    by_name = {r["recipe"]["package"]["name"]: r for r in rendered}
    assert set(by_name) == {"pocketvina-engine", "pocketvina"}
    assert by_name["pocketvina"]["build_configuration"]["target_platform"] == "noarch"
    assert by_name["pocketvina-engine"]["recipe"]["package"]["version"] == pocketvina.__version__
    assert f"pocketvina-engine =={pocketvina.__version__}" in by_name["pocketvina"]["recipe"]["requirements"]["run"]


# --- build_engine.sh -----------------------------------------------------------------------------------------------

def _engine_sources(root: Path) -> Path:
    """A source tree as rattler-build prepares it ($SRC_DIR), with a prebuilt engine that must not be packaged."""
    pkg = root / "pocketvina"
    pkg.mkdir(parents=True)
    shutil.copy2(PACKAGE_DIR / "Makefile", pkg / "Makefile")
    for sub in ("main", "OpenCL"):
        shutil.copytree(PACKAGE_DIR / sub, pkg / sub)
    (pkg / "lib").mkdir()
    (pkg / "lib" / "cache.cpp").write_text("// stand-in\n")
    (pkg / "PocketVina-GPU").write_text("#!/bin/sh\necho prebuilt\n")
    (pkg / "PocketVina-GPU").chmod(0o755)
    return root


def _host_prefix(root: Path, boost_has_convenience: bool) -> Path:
    """$PREFIX with an installed Boost (libboost-devel) and OpenCL loader + headers (ocl-icd, opencl-headers)."""
    fs = root / "include" / "boost" / "filesystem"
    fs.mkdir(parents=True)
    if boost_has_convenience:
        (fs / "convenience.hpp").write_text("// Boost <= 1.84\n")
    (root / "include" / "CL").mkdir()
    (root / "lib").mkdir()
    for lib in ("boost_program_options", "boost_filesystem", "boost_thread", "OpenCL"):
        (root / "lib" / f"lib{lib}.so").write_bytes(b"")
    return root


def _fake_compiler(path: Path, log: Path) -> Path:
    """A compiler that records its arguments and writes a shell script as the ``-o`` output."""
    path.write_text("#!/bin/sh\n"
                    f'printf "%s\\n" "$*" >> "{log}"\n'
                    'out=""; prev=""\n'
                    'for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done\n'
                    'printf "#!/bin/sh\\necho built\\n" > "$out"\n'
                    'chmod 755 "$out"\n')
    path.chmod(0o755)
    return path


@needs_make
@needs_recipe
@pytest.mark.parametrize("boost_has_convenience", [False, True], ids=["boost-1.85+", "boost-1.84"])
def test_build_engine_script(tmp_path, boost_has_convenience):
    src = _engine_sources(tmp_path / "work")
    prefix = _host_prefix(tmp_path / "host", boost_has_convenience)
    log = tmp_path / "compiler.log"
    env = {k: v for k, v in os.environ.items() if k not in ("BOOST_ROOT", "BOOST_LIB_PATH", "OPENCL_ROOT",
                                                           "OPENCL_LIB_PATH", "DESTDIR", "MAKEFLAGS", "MFLAGS")}
    env.update(PREFIX=str(prefix), SRC_DIR=str(src), CXX=str(_fake_compiler(tmp_path / "c++", log)),
               CXXFLAGS="-O2 -isystem " + str(prefix / "include"), LDFLAGS="-L" + str(prefix / "lib"))
    run = subprocess.run(["bash", str(BUILD_SCRIPT)], cwd=src, env=env, capture_output=True, text=True, timeout=120,
                         stdin=subprocess.DEVNULL)
    assert run.returncode == 0, run.stdout + run.stderr

    exe = prefix / "bin" / engine.ENGINE_COMMAND
    assert exe.read_text() == "#!/bin/sh\necho built\n" and stat.S_IMODE(exe.stat().st_mode) == 0o755
    share = prefix / engine.INSTALLED_KERNEL_DIR / "OpenCL"
    kernels = sorted(p.name for p in (PACKAGE_DIR / "OpenCL" / "src" / "kernels").glob("*.cl"))
    assert sorted(p.name for p in (share / "src" / "kernels").iterdir()) == kernels
    assert sorted(p.name for p in (share / "inc").iterdir()) == sorted(
        p.name for p in (PACKAGE_DIR / "OpenCL" / "inc").iterdir())

    (cmd,) = log.read_text().splitlines()
    assert f"-I{prefix}/include" in cmd and f"-L{prefix}/lib" in cmd and "-lboost_thread" in cmd
    assert "-lboost_system" not in cmd and f"-Wl,-rpath,{prefix}/lib" in cmd and "libs/thread" not in cmd
    shim = src / "boost_compat" / "boost" / "filesystem" / "convenience.hpp"
    if boost_has_convenience:
        assert not shim.exists() and "boost_compat" not in cmd
    else:
        assert cmd.startswith(f"-o PocketVina-GPU -I{src}/boost_compat -O2 ")
        text = shim.read_text()
        assert "inline std::string basename(const path& p) { return p.stem().string(); }" in text
    # nothing written next to the recipe or into the checkout
    assert not (PACKAGE_DIR / "boost_compat").exists() and not (RECIPE_DIR / "boost_compat").exists()


@needs_recipe
def test_build_engine_script_syntax():
    if shutil.which("bash") is None:
        pytest.skip("needs bash")
    assert os.access(BUILD_SCRIPT, os.X_OK)
    run = subprocess.run(["bash", "-n", str(BUILD_SCRIPT)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert BUILD_SCRIPT.read_text().startswith("#!/usr/bin/env bash\n")


def test_noarch_package_finds_the_conda_engine(tmp_path):
    """What the pocketvina package's own test asserts in a conda environment, with a fake prefix: no engine in the
    package directory -> <prefix>/bin/pocketvina-engine and the kernel sources in <prefix>/share/pocketvina."""
    prefix = tmp_path / "env"
    (prefix / "bin").mkdir(parents=True)
    exe = prefix / "bin" / engine.ENGINE_COMMAND
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    shutil.copytree(PACKAGE_DIR / "OpenCL", prefix / engine.INSTALLED_KERNEL_DIR / "OpenCL")
    code = ("import os, sys; from pocketvina import engine as e; e.PACKAGE_DIR = sys.argv[1]; "
            "print(e.engine_path()); print(e.kernel_source_dir())")
    env = {k: v for k, v in os.environ.items() if k not in ("POCKETVINA_ENGINE", "POCKETVINA_KERNEL_SRC")}
    env.update(PATH=f"{prefix / 'bin'}{os.pathsep}{env.get('PATH', '')}", PYTHONPATH=str(REPO_ROOT))
    run = subprocess.run([sys.executable, "-c", code, str(tmp_path / "site-packages" / "pocketvina")],
                         capture_output=True, text=True, env=env, timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == [str(exe), str(prefix / engine.INSTALLED_KERNEL_DIR)]
