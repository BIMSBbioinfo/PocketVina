"""Setup of the quickstart notebook (``quickstart_colab.ipynb``): its install cell calls ``setup(REPO_URL, REPO_REF)``.

On Google Colab (after the cell cloned the repository into ``/content/PocketVina``) it installs what
docs/installation.md does with conda, in Colab's own Python:

1. apt: Java 17 (for P2Rank), the OpenCL ICD loader and ``clinfo``;
2. ``/etc/OpenCL/vendors/nvidia.icd``: Colab has the NVIDIA driver and its OpenCL library, but no ICD file that points
   to it, so without it OpenCL sees no GPU;
3. the clone at ``REPO_REF``, for the example files (a ``REPO_REF`` changed later is checked out in the same clone);
4. ``pip install pocketvina`` from PyPI (with PoseBusters and the viewers). If its prebuilt engine does not run here, or
   with ``build=True``, the engine is compiled from the clone (Boost 1.77 and ``make``; takes a while) and the clone
   is installed instead;
5. P2Rank (``pocketvina setup p2rank``) and a check that OpenCL lists the GPU.

Every step is skipped when it is done already, so the cell can run again (e.g. after a runtime restart); command output
goes to ``/content/pocketvina_install.log``. Then the notebook works in the clone's ``notebooks/`` folder, as a local
copy does (the example inputs are in ``../example``).

Elsewhere nothing is installed; if ``pocketvina`` cannot be imported, the checkout this file is in is used.
"""
import importlib
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

COLAB_REPO_DIR = Path("/content/PocketVina")
INSTALL_LOG = Path("/content/pocketvina_install.log")
APT_PACKAGES = ["openjdk-17-jre-headless", "ocl-icd-libopencl1", "clinfo"]
NVIDIA_LIB_DIRS = ["/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu", "/usr/local/nvidia/lib64"]
BOOST_URL = "https://archives.boost.io/release/1.77.0/source/boost_1_77_0.tar.gz"
# modules a Colab session may have imported before the install: a version pip changes needs a restart of the session
RESTART_IF_CHANGED = ("numpy", "pandas", "pyarrow")


def in_colab() -> bool:
    return "google.colab" in sys.modules


def sh(cmd, cwd=None, log=None):
    """Run a bash command with its output appended to the log; raise with the end of the log if it fails."""
    log = Path(log or INSTALL_LOG)
    with open(log, "a") as f:
        f.write(f"\n$ {cmd}\n")
        f.flush()
        code = subprocess.run(["bash", "-c", f"set -eo pipefail; {cmd}"], cwd=cwd, stdout=f,
                              stderr=subprocess.STDOUT).returncode
    if code:
        tail = log.read_text(errors="replace")[-3000:]
        raise RuntimeError(f"exit code {code}: {cmd}\n--- end of {log} ---\n{tail}")


def apt_install(packages, log=None):
    """apt-get install, after an update that may fail: a broken third-party apt source on the runtime makes
    `apt-get update` exit non-zero, but the Ubuntu packages can still be installed."""
    sh("apt-get update -qq || echo 'WARNING: apt-get update failed for some sources, installing anyway'", log=log)
    sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq " + " ".join(packages), log=log)


def clone_repo(repo_url, repo_ref, repo_dir=COLAB_REPO_DIR, log=None):
    """``repo_dir`` at ``repo_ref`` of ``repo_url``: cloned if missing, checked out again when the URL or ref changed
    since (files not in git, such as a compiled Boost, stay)."""
    repo_dir = Path(repo_dir)
    ref_file = repo_dir / ".git" / "pocketvina_notebook_ref"  # what the notebook checked out
    ref = f"{repo_url} {repo_ref}"
    if not (repo_dir / ".git").is_dir():
        sh(f"git clone -q --depth 1 --branch {repo_ref} {repo_url} {repo_dir}", log=log)
    elif ref_file.is_file() and ref_file.read_text() != ref:
        sh(f"git fetch -q --depth 1 {repo_url} {repo_ref} && git checkout -q --force FETCH_HEAD", cwd=repo_dir, log=log)
        if "pocketvina" in sys.modules:  # the version imported before is still loaded
            ref_file.write_text(ref)
            raise RuntimeError(f"{repo_dir} is now at {repo_ref}: Runtime > Restart session, then Run all")
    ref_file.write_text(ref)  # also for the clone made by the notebook's cell, at this ref
    if not (repo_dir / "pocketvina" / "selection.py").is_file():
        raise RuntimeError(f"{repo_url} at {repo_ref} is older than this notebook (no pocketvina/selection.py): set "
                           "REPO_REF to a branch or tag that has it")


def engine_runs(path) -> bool:
    try:
        return subprocess.run([str(path), "--version"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def build_engine(pkg: Path, log=None):
    """Boost 1.77 as static libraries in pocketvina/boost_1_77_0, then ``make`` (the Makefile's defaults: that Boost
    and the CUDA toolkit's OpenCL in /usr/local/cuda, which Colab has)."""
    boost = pkg / "boost_1_77_0"
    if not (boost / "lib" / "libboost_program_options.a").is_file():
        print("   Boost 1.77 ...")
        if not (boost / "bootstrap.sh").is_file():
            sh(f"wget -q {BOOST_URL} -O boost.tar.gz && tar -xzf boost.tar.gz && rm boost.tar.gz", cwd=pkg, log=log)
        sh(f"./bootstrap.sh --with-libraries=program_options,system,filesystem,thread && ./b2 --prefix={boost} "
           f"link=static runtime-link=static threading=multi variant=release -j{os.cpu_count() or 2} install",
           cwd=boost, log=log)
    print("   make ...")
    sh("make clean && make", cwd=pkg, log=log)
    if not engine_runs(pkg / "PocketVina-GPU"):
        raise RuntimeError(f"the engine compiled in {pkg} does not run, see {log or INSTALL_LOG}")


def _installed_engine() -> Path:
    """The engine binary of the installed pocketvina package (found without importing it)."""
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("pocketvina")
    return Path(list(spec.submodule_search_locations)[0]) / "PocketVina-GPU"


def install_colab(repo_url, repo_ref, repo_dir=COLAB_REPO_DIR, build=False):
    """Steps 1-5 of the module docstring."""
    t0 = time.time()
    repo_dir = Path(repo_dir)
    print("1/5 Java 17, OpenCL loader (apt)")
    if subprocess.run(["dpkg", "-s", *APT_PACKAGES], capture_output=True).returncode != 0:
        apt_install(APT_PACKAGES)
    java_homes = sorted(Path("/usr/lib/jvm").glob("java-17-openjdk-*"))
    if not java_homes:
        raise RuntimeError(f"no Java 17 in /usr/lib/jvm after installing {APT_PACKAGES[0]}")
    os.environ["JAVA_HOME"] = str(java_homes[0])  # P2Rank runs $JAVA_HOME/bin/java
    if not os.environ["PATH"].startswith(f"{java_homes[0]}/bin:"):
        os.environ["PATH"] = f"{java_homes[0]}/bin:{os.environ['PATH']}"

    print("2/5 OpenCL ICD file of the NVIDIA driver")
    icd = Path("/etc/OpenCL/vendors/nvidia.icd")
    if not icd.is_file():
        libs = [p for name in ("libnvidia-opencl.so.1", "libnvidia-opencl.so.*")  # the versioned file if no .so.1
                for d in NVIDIA_LIB_DIRS for p in sorted(Path(d).glob(name))]
        icd.parent.mkdir(parents=True, exist_ok=True)
        icd.write_text(f"{libs[0] if libs else 'libnvidia-opencl.so.1'}\n")

    print(f"3/5 PocketVina repository ({repo_ref}), for the example files")
    clone_repo(repo_url, repo_ref, repo_dir)

    print("4/5 pip install pocketvina (with PoseBusters and the viewers)")
    if build or importlib.util.find_spec("pocketvina") is None:
        loaded = {m: getattr(sys.modules[m], "__version__", None) for m in RESTART_IF_CHANGED if m in sys.modules}
        if not build:
            sh(f"{sys.executable} -m pip install -q pocketvina")
        if build or not engine_runs(_installed_engine()):
            print("   compiling the engine from the repository")
            build_engine(repo_dir / "pocketvina")
            sh(f"{sys.executable} -m pip install -q {repo_dir}")
        importlib.invalidate_caches()
        from importlib.metadata import version

        changed = [m for m, v in loaded.items() if v is not None and version(m) != v]
        if changed or "pocketvina" in sys.modules:
            raise RuntimeError(f"pip changed {', '.join(changed) or 'pocketvina'}, which this session has loaded "
                               "already: Runtime > Restart session, then Run all")

    print("5/5 P2Rank, OpenCL devices")
    sh(f"{sys.executable} -m pocketvina setup p2rank")
    devices = subprocess.run(["clinfo", "-l"], capture_output=True, text=True).stdout.strip()
    print("   OpenCL:", devices.replace("\n", "\n           ") if devices else "no device (not a GPU runtime?)")
    print(f"done in {time.time() - t0:.0f} s (log: {INSTALL_LOG})")


def setup(repo_url, repo_ref, build=False):
    """On Colab: install (``install_colab``; ``build=True`` compiles the engine even if the prebuilt one runs), then
    work in the clone's notebooks/ folder. Elsewhere: nothing is installed; the checkout this file is in is used if
    ``pocketvina`` cannot be imported."""
    if in_colab():
        install_colab(repo_url, repo_ref, build=build)
        os.chdir(COLAB_REPO_DIR / "notebooks")
    elif importlib.util.find_spec("pocketvina") is None:
        checkout = Path(__file__).resolve().parent.parent
        if (checkout / "pocketvina" / "__init__.py").is_file():
            sys.path.insert(0, str(checkout))
    if importlib.util.find_spec("pocketvina") is None:
        raise RuntimeError("PocketVina is not installed: pip install pocketvina")
    import pocketvina

    print(f"PocketVina {pocketvina.__version__} ({Path(pocketvina.__file__).parent}); working folder {Path.cwd()}")
