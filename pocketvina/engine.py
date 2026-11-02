"""The PocketVina-GPU docking engine: config files, environment, OpenCL kernel cache, output collection and the
temp folders runs work in (``resolve_tmp_dir``, also ``pocketvina.io.writer.resolve_tmp_dir``).

The engine is ``$POCKETVINA_ENGINE``, else the binary in the package (``pocketvina/PocketVina-GPU``: pip installs and
source checkouts, rebuilt with ``make -C pocketvina``), else ``pocketvina-engine`` on ``PATH`` or in this Python's
``bin/`` (``make install``, the conda package); see ``engine_path``. It compiles the kernel sources next to it:
``<prefix>/share/pocketvina`` for an installed ``<prefix>/bin/pocketvina-engine``, else the package directory
(``kernel_source_dir``).

The engine compiles its two OpenCL kernels the first time it runs on a GPU and saves them in the config's
``opencl_binary_path`` (see ``pocketvina/lib/main_procedure_cl.cpp``). ``write_config`` points that at
``kernel_cache_dir()``, a per-user folder keyed by engine, kernel sources and GPU, so the package directory is never
written to, and engines running at the same time share one cache.

``run_engine`` starts the engine in a process group of its own, so that a timeout or an interrupt kills the engine and
anything it started; the engine is told to die with the Python process that started it (``POCKETVINA_DIE_WITH_PARENT``,
``pocketvina/main/main.cpp``) for the cases where that process is killed outright.

``$OCL_ICD_VENDORS`` (the v0.1 README had users set it to ``/etc/OpenCL/vendors``) makes some OpenCL loaders, e.g.
the CUDA toolkit's ``libOpenCL.so.1``, find no platform at all: the engine then fails at once with
``CL_PLATFORM_NOT_FOUND_KHR``. ``run_engine`` runs such a call once more without the variable, and when that works,
leaves it out of every later engine call of this process (one warning says so).

Standard library only: the "files" output format docks without pyarrow.

The engine is based on QuickVina2-GPU 2.1 (`"Vina-GPU 2.1: towards further optimizing docking speed and precision of
AutoDock Vina and its derivatives" <https://doi.org/10.1101/2023.11.04.565429>`_), QuickVina 2 (`"Fast, accurate, and
reliable molecular docking with QuickVina 2" <https://doi.org/10.1093/bioinformatics/btv082>`_) and AutoDock Vina
(`"AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and
multithreading" <https://doi.org/10.1002/jcc.21334>`_).
"""
import contextlib
import functools
import hashlib
import logging
import math
import numbers
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .pdbqt import engine_output_name

logger = logging.getLogger(__name__)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_NAME = "PocketVina-GPU"  # the binary in the package directory
ENGINE_COMMAND = "pocketvina-engine"  # an installed engine: `make install` (<prefix>/bin), the conda package
INSTALLED_KERNEL_DIR = os.path.join("share", "pocketvina")  # its kernel sources, relative to <prefix>
KERNEL_FILES = ("Kernel1_Opt.bin", "Kernel2_Opt.bin")
KERNEL_SOURCE_DIRS = ("OpenCL/src/kernels", "OpenCL/inc")  # relative to $POCKETVINA_KERNEL_SRC
MIN_THREAD = 1000  # the engine rejects fewer
LOG_TAIL_CHARS = 4000
TMP_ENV_VARS = ("POCKETVINA_TMPDIR", "SLURM_TMPDIR", "TMPDIR")
DIE_WITH_PARENT_ENV = "POCKETVINA_DIE_WITH_PARENT"  # main.cpp: <pid> -> the engine is killed when that process dies
OCL_VENDORS_ENV = "OCL_ICD_VENDORS"
NO_PLATFORM_ERROR = "CL_PLATFORM_NOT_FOUND_KHR"  # OpenCL/src/wrapcl.cpp prints "Err-1001:CL_PLATFORM_NOT_FOUND_KHR"
_INT32 = (-2 ** 31, 2 ** 31 - 1)
_PROCESS_GROUPS = hasattr(os, "killpg") and hasattr(os, "setsid")  # POSIX
_RUNNING: set = set()  # pids of the engines running now (run_engine), for the memory sampler of an atlas run
_RUNNING_LOCK = threading.Lock()
_THIS_PROCESS = object()  # kernel_cache_dir/kernel_cache_key default: this process's $CUDA_VISIBLE_DEVICES
# True once an engine call that found no OpenCL platform with $OCL_ICD_VENDORS worked without it (see run_engine)
_drop_ocl_vendors = False
_drop_ocl_vendors_lock = threading.Lock()


@dataclass(frozen=True)
class EngineResult:
    """One engine call: exit code, wall time and the end of its combined stdout/stderr."""

    returncode: int
    runtime_s: float
    log_tail: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class _Terminated(BaseException):
    """SIGTERM received inside ``exit_on_sigterm``."""


def _raise_terminated(signum, frame):
    raise _Terminated()


@contextlib.contextmanager
def exit_on_sigterm():
    """Run the block (or decorated function) so that SIGTERM (``kill <pid>``, a batch scheduler ending the job)
    unwinds it like Ctrl-C: ``finally`` clauses kill the engine and remove temp folders, then the process dies of
    SIGTERM as it would have. Only in the main thread, and only while SIGTERM has its default action (a handler set
    by the application, or by an outer ``exit_on_sigterm``, is left alone)."""
    if threading.current_thread() is not threading.main_thread() or \
            signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
        yield
        return
    signal.signal(signal.SIGTERM, _raise_terminated)
    try:
        yield
    except _Terminated:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)


def _usable_dir(path) -> bool:
    return os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)


def resolve_tmp_dir(explicit: Optional[str] = None) -> str:
    """A fresh private temp folder (``mkdtemp(prefix="pocketvina_")``) inside the first of ``explicit``,
    ``$POCKETVINA_TMPDIR``, ``$SLURM_TMPDIR``, ``$TMPDIR``, ``tempfile.gettempdir()``.

    ``explicit`` is created if needed; an environment variable naming a missing or read-only folder is skipped.
    The path returned is absolute. The caller removes the folder when done.
    """
    if explicit:
        base = os.path.expanduser(os.fspath(explicit))
        os.makedirs(base, exist_ok=True)
    else:
        base = None
        for var in TMP_ENV_VARS:
            value = os.environ.get(var)
            if not value:
                continue
            if _usable_dir(value):
                base = value
                break
            logger.warning("$%s=%s is not a writable folder, ignored", var, value)
        base = base or tempfile.gettempdir()
    # mkdtemp returns a relative path for a relative dir before Python 3.12
    return tempfile.mkdtemp(prefix="pocketvina_", dir=os.path.abspath(base))


def _executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def engine_path() -> str:
    """The engine executable: ``$POCKETVINA_ENGINE`` (a path, or a command on ``PATH``), else the package binary
    ``pocketvina/PocketVina-GPU``, else ``pocketvina-engine`` on ``PATH`` or in ``<sys.prefix>/bin`` (a conda env
    used without activating it, e.g. as a Jupyter kernel). If there is none, the package binary's path (``run_engine``
    reports it missing)."""
    path = os.environ.get("POCKETVINA_ENGINE")
    if path:
        if os.sep not in path:
            return shutil.which(path) or path
        return os.path.abspath(os.path.expanduser(path))
    packaged = os.path.join(PACKAGE_DIR, ENGINE_NAME)
    if _executable(packaged):
        return packaged
    installed = shutil.which(ENGINE_COMMAND)
    if installed:
        return os.path.abspath(installed)
    in_prefix = os.path.join(sys.prefix, "bin", ENGINE_COMMAND)
    return in_prefix if _executable(in_prefix) else packaged


def installed_kernel_dir(exe: str) -> Optional[str]:
    """``<prefix>/share/pocketvina`` of an engine installed as ``<prefix>/bin/<engine>`` (the path given or the file
    a symlink points to) if it holds the kernel sources, else None."""
    for path in dict.fromkeys((os.path.abspath(exe), os.path.realpath(exe))):
        share = os.path.join(os.path.dirname(os.path.dirname(path)), INSTALLED_KERNEL_DIR)
        if os.path.isdir(os.path.join(share, KERNEL_SOURCE_DIRS[0])):
            return share
    return None


def kernel_source_dir() -> str:
    """Root of the kernel sources the engine compiles (``OpenCL/src/kernels`` and ``OpenCL/inc`` under it):
    ``$POCKETVINA_KERNEL_SRC``, else ``installed_kernel_dir`` of an engine outside the package directory, else the
    package directory."""
    explicit = os.environ.get("POCKETVINA_KERNEL_SRC")
    if explicit:
        return os.path.abspath(explicit)
    exe = engine_path()
    if not _within(exe, PACKAGE_DIR):
        share = installed_kernel_dir(exe)
        if share:
            return share
    return PACKAGE_DIR


def kernel_cache_dir(visible_devices=_THIS_PROCESS) -> str:
    """Folder for the compiled OpenCL kernels (the config's ``opencl_binary_path``); the engine creates it.

    ``$POCKETVINA_KERNEL_CACHE`` if set, else ``~/.cache/pocketvina/kernels/<key>`` (``$XDG_CACHE_HOME`` instead of
    ``~/.cache`` when set; the system temp dir if there is no usable home), where ``<key>`` is
    ``kernel_cache_key(visible_devices)``: the folder for an engine run with ``CUDA_VISIBLE_DEVICES=visible_devices``
    (None: unset), by default with this process's.
    """
    explicit = os.environ.get("POCKETVINA_KERNEL_CACHE")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    base = os.environ.get("XDG_CACHE_HOME", "")
    if not os.path.isabs(base):  # unset, or relative (invalid in the XDG spec)
        base = os.path.join(os.path.expanduser("~"), ".cache")
    if not os.path.isabs(base):  # no home directory: never a path relative to the working directory
        base = os.path.join(tempfile.gettempdir(), f"pocketvina-{_user_tag()}")
    return os.path.join(base, "pocketvina", "kernels", kernel_cache_key(visible_devices))


def kernel_cache_key(visible_devices=_THIS_PROCESS) -> str:
    """16 hex digits of sha256 over the engine binary, the kernel sources and the GPU name/driver (``nvidia-smi``) of
    the GPU an engine with ``CUDA_VISIBLE_DEVICES=visible_devices`` uses (None: unset; default: this process's).

    Kernels built for another GPU model, driver or kernel source are then never offered to the engine (it would also
    reject them and rebuild, but caches for different GPUs sharing a home directory would keep replacing each other).
    """
    if visible_devices is _THIS_PROCESS:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    h = hashlib.sha256()
    for part in (_file_sha256(engine_path()), _tree_sha256(kernel_source_dir(), KERNEL_SOURCE_DIRS),
                 *_gpu_identity(visible_devices)):
        h.update(part.encode("utf-8", "surrogateescape") + b"\0")
    return h.hexdigest()[:16]


def _user_tag() -> str:
    return str(os.getuid()) if hasattr(os, "getuid") else "user"


def _file_sha256(path: str) -> str:
    try:
        st = os.stat(path)
    except OSError:
        return ""  # a missing engine is reported by run_engine
    try:
        return _file_sha256_cached(path, st.st_size, st.st_mtime_ns)
    except OSError:  # e.g. an execute-only engine: still changes when it is rebuilt
        return f"stat:{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}"


@functools.lru_cache(maxsize=16)
def _file_sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_sha256(root: str, subdirs: Iterable[str]) -> str:
    h = hashlib.sha256()
    for sub in subdirs:
        top = os.path.join(root, sub)
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames.sort()
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                h.update(os.path.relpath(path, root).encode("utf-8", "surrogateescape") + b"\0")
                h.update(_file_sha256(path).encode() + b"\0")
    return h.hexdigest()


@functools.lru_cache(maxsize=8)
def _gpu_identity(visible_devices: Optional[str]) -> tuple:
    """Sorted distinct ``"<name>|<driver>"`` of the GPUs the engine may use; empty without ``nvidia-smi``.

    The engine takes the first OpenCL GPU, i.e. the first entry of ``CUDA_VISIBLE_DEVICES`` when that names an
    ``nvidia-smi`` index or UUID; otherwise all GPUs count.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader"],
                             stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    gpus = []
    for line in out.splitlines():
        head = line.split(",", 2)
        tail = head[-1].rsplit(",", 1)
        if len(head) == 3 and len(tail) == 2:  # a GPU name may hold a comma
            gpus.append(tuple(p.strip() for p in (head[0], head[1], tail[0], tail[1])))
    first = (visible_devices or "").split(",")[0].strip()
    if first:
        chosen = [g for g in gpus if first == g[0] or (first.startswith("GPU-") and g[1].startswith(first))]
        if chosen:
            gpus = chosen[:1]
    return tuple(sorted({f"{name}|{driver}" for _, _, name, driver in gpus}))


def _check_path(name: str, value, *, forbid_pdbqt: bool = False) -> str:
    path = os.path.abspath(os.fspath(value))
    # boost::program_options config files: '#' starts a comment anywhere, values are stripped, one per line
    if any(c in path for c in "#\n\r") or path != path.strip():
        raise ValueError(f"{name} {path!r}: the engine config cannot hold '#', line breaks or edge whitespace")
    if forbid_pdbqt and ".pdbqt" in path:
        raise ValueError(f"{name} {path!r}: the engine cuts ligand paths at the first '.pdbqt', rename the folder")
    return path


def _check_int(name: str, value, minimum: int, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        bound = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ValueError(f"{name} must be {bound}, got {value}")
    return value


def _check_number(name: str, value, *, positive: bool = False, non_negative: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if (positive and value <= 0) or (non_negative and value < 0):
        raise ValueError(f"{name} must be {'> 0' if positive else '>= 0'}, got {value!r}")
    return str(int(value)) if isinstance(value, numbers.Integral) else repr(float(value))


def _check_xyz(name: str, values: Sequence, positive: bool = False) -> list:
    values = list(values)
    if len(values) != 3:
        raise ValueError(f"{name} must have 3 values (x, y, z), got {len(values)}")
    return [_check_number(f"{name}_{axis}", v, positive=positive) for axis, v in zip("xyz", values)]


def write_config(path, receptor, ligand_dir, out_dir, center, size, thread, pocket, seed=None, num_modes=None,
                 energy_range=None, opencl_binary_path=None) -> str:
    """Write an engine config file (``key = value`` lines) and return its absolute path.

    Paths are written absolute, since the engine runs in a temp dir. ``out_dir`` None leaves the engine default
    ``<ligand_dir>_out``; the engine creates ``out_dir`` but not its parent. ``opencl_binary_path`` None means
    ``kernel_cache_dir()``. ``seed``, ``num_modes`` and ``energy_range`` are written only when given (engine
    defaults: a random seed, 20 modes, 20 kcal/mol). Values the engine would reject raise ``ValueError``.
    """
    centers = _check_xyz("center", center)
    sizes = _check_xyz("size", size, positive=True)
    lines = [("receptor", _check_path("receptor", receptor)),
             ("ligand_directory", _check_path("ligand_dir", ligand_dir, forbid_pdbqt=True))]
    if out_dir is not None:
        lines.append(("output_directory", _check_path("out_dir", out_dir)))
    lines.append(("opencl_binary_path",
                  _check_path("opencl_binary_path", kernel_cache_dir() if opencl_binary_path is None
                              else opencl_binary_path)))
    lines += [(f"center_{axis}", v) for axis, v in zip("xyz", centers)]
    lines += [(f"size_{axis}", v) for axis, v in zip("xyz", sizes)]
    lines.append(("thread", str(_check_int("thread", thread, MIN_THREAD, _INT32[1]))))
    lines.append(("pocket_rank", str(_check_int("pocket", pocket, 1, _INT32[1]))))
    if seed is not None:
        lines.append(("seed", str(_check_int("seed", seed, *_INT32))))
    if num_modes is not None:
        lines.append(("num_modes", str(_check_int("num_modes", num_modes, 1, _INT32[1]))))
    if energy_range is not None:
        lines.append(("energy_range", _check_number("energy_range", energy_range, non_negative=True)))

    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write("".join(f"{key} = {value}\n" for key, value in lines))
    return path


def engine_env(env_extra: Optional[Mapping[str, Optional[str]]] = None) -> Dict[str, str]:
    """``os.environ`` plus what the engine needs; ``env_extra`` is applied last (a None value unsets a variable).

    ``$POCKETVINA_DIE_WITH_PARENT`` is this process's pid: the engine (Linux) is killed when this process dies.
    ``$OCL_ICD_VENDORS`` is left out once ``run_engine`` found that the engine works only without it.
    """
    env = dict(os.environ)
    if _drop_ocl_vendors:
        env.pop(OCL_VENDORS_ENV, None)
    lib_dirs = [d for d in (os.path.join(PACKAGE_DIR, "boost_1_77_0", "stage", "lib"),
                            os.path.join(PACKAGE_DIR, "boost_1_77_0", "lib")) if os.path.isdir(d)]
    # empty entries would mean "the working directory"
    lib_path = os.pathsep.join(lib_dirs + [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p])
    if lib_path:
        env["LD_LIBRARY_PATH"] = lib_path
    else:
        env.pop("LD_LIBRARY_PATH", None)
    env["OPENCL_ROOT"] = os.path.join(PACKAGE_DIR, "OpenCL")
    env["POCKETVINA_KERNEL_SRC"] = kernel_source_dir()
    env[DIE_WITH_PARENT_ENV] = str(os.getpid())
    for key, value in (env_extra or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = str(value)
    return env


def _within(path: str, root: str) -> bool:
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _tail(f, limit: int = LOG_TAIL_CHARS) -> str:
    """Last ``limit`` characters of a log file, with ``\\r`` progress redraws reduced to their final state."""
    f.flush()
    f.seek(0, os.SEEK_END)
    start = max(0, f.tell() - 8 * limit)
    f.seek(start)
    text = f.read().decode("utf-8", "replace")
    lines = []
    for line in text.split("\n"):
        parts = [p for p in line.split("\r") if p]
        lines.append(parts[-1] if parts else "")
    text = "\n".join(lines)
    if start > 0 or len(text) > limit:
        text = text[-limit:]
        cut = text.find("\n")
        text = text[cut + 1:] if 0 <= cut < len(text) - 1 else text
    return text


def check_timeout(name: str, value) -> Optional[float]:
    """``value`` as float seconds (None: no limit); ``ValueError`` unless it is a finite number > 0."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite number of seconds > 0 or None, got {value!r}")
    return float(value)


def _kill_group(proc: subprocess.Popen) -> int:
    """SIGKILL the engine's process group (the engine and anything it started), then reap the engine.

    Only called before ``proc`` is reaped, so its pid (== the group id) cannot have been reused.
    """
    if _PROCESS_GROUPS:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:  # e.g. the group is gone already
            proc.kill()
    else:
        proc.kill()
    return proc.wait()


def ocl_vendors_dropped() -> bool:
    """True once this process leaves ``$OCL_ICD_VENDORS`` out of the engine's environment (see ``run_engine``)."""
    return _drop_ocl_vendors


def _no_platform(result: EngineResult) -> bool:
    return not result.ok and not result.timed_out and NO_PLATFORM_ERROR in result.log_tail


def _drop_ocl_vendors_from_now_on(value: str) -> None:
    global _drop_ocl_vendors
    with _drop_ocl_vendors_lock:
        if _drop_ocl_vendors:
            return
        _drop_ocl_vendors = True
    logger.warning("the engine finds no OpenCL platform with OCL_ICD_VENDORS=%s set, but works without it: it is "
                   "left out of the engine's environment from now on (to skip this retry, unset it, e.g. remove "
                   "`export OCL_ICD_VENDORS=...` from ~/.bashrc)", value)


def running_engines() -> List[int]:
    """pids of the engines that ``run_engine`` calls of this process are running now."""
    with _RUNNING_LOCK:
        return list(_RUNNING)


def run_engine(config_path, *, cwd, env_extra=None, log_path=None, timeout=None) -> EngineResult:
    """Run ``<engine> --config <config_path>`` in ``cwd`` (created if missing; never the package directory).

    stdout and stderr go to ``log_path`` (overwritten) or to an anonymous temp file. The engine runs in a process
    group of its own (POSIX): after ``timeout`` seconds (None: no limit) the whole group is killed and ``timed_out``
    is set; an exception while waiting (Ctrl-C, SIGTERM under ``exit_on_sigterm``) kills it too. A missing or
    non-executable engine raises ``OSError``; engine failures are reported through ``returncode``.

    A call that fails with ``CL_PLATFORM_NOT_FOUND_KHR`` while ``$OCL_ICD_VENDORS`` is set runs once more without
    it (that result is returned); if it then finds a platform, every later call of this process goes without it.
    """
    timeout = check_timeout("timeout", timeout)
    cwd = os.path.abspath(os.fspath(cwd))
    if _within(cwd, PACKAGE_DIR):
        raise ValueError(f"the engine must not run inside the package directory {PACKAGE_DIR}: use a temp dir")
    os.makedirs(cwd, exist_ok=True)
    cmd = [engine_path(), "--config", os.path.abspath(os.fspath(config_path))]
    if log_path is not None:
        log_path = os.path.abspath(os.fspath(log_path))
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    env = engine_env(env_extra)
    result = _run(cmd, cwd, env, log_path, timeout)
    vendors = env.get(OCL_VENDORS_ENV)
    if vendors and _no_platform(result):
        logger.debug("engine found no OpenCL platform with OCL_ICD_VENDORS=%s, trying again without it", vendors)
        env.pop(OCL_VENDORS_ENV)
        result = _run(cmd, cwd, env, log_path, timeout)
        if not _no_platform(result):
            _drop_ocl_vendors_from_now_on(vendors)
    return result


def _run(cmd: list, cwd: str, env: Dict[str, str], log_path: Optional[str], timeout: Optional[float]) \
        -> EngineResult:
    """One engine process (see ``run_engine``)."""
    exe = cmd[0]
    log = open(log_path, "w+b") if log_path is not None else tempfile.TemporaryFile()
    with log:
        logger.debug("running %s in %s", " ".join(cmd), cwd)
        t0 = time.monotonic()
        try:
            # a session of its own: a terminal's or notebook's signals reach only Python, which kills the group
            proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=_PROCESS_GROUPS)
        except OSError as e:
            raise type(e)(e.errno, f"{e.strerror}: cannot run the PocketVina engine {exe!r} (build it with "
                                   "`make -C pocketvina` in a source checkout, install the conda package "
                                   "pocketvina-engine, or set POCKETVINA_ENGINE to an engine)") from e
        timed_out = False
        with _RUNNING_LOCK:
            _RUNNING.add(proc.pid)
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = _kill_group(proc)
        except BaseException:  # e.g. KeyboardInterrupt: do not leave the engine running on the GPU
            _kill_group(proc)
            raise
        finally:
            with _RUNNING_LOCK:
                _RUNNING.discard(proc.pid)
        runtime_s = time.monotonic() - t0
        log_tail = _tail(log)
    if timed_out:
        log_tail += f"\n[pocketvina] engine killed (with its process group) after the {timeout:g} s timeout"
        logger.warning("engine timed out after %g s, killed (config %s)", timeout, cmd[-1])
    elif returncode != 0:
        logger.debug("engine exited with code %d (config %s)", returncode, cmd[-1])
    return EngineResult(returncode=returncode, runtime_s=runtime_s, log_tail=log_tail, timed_out=timed_out)


def collect_outputs(out_dir, receptor_pdbqt, pocket, ligand_stems, *, remove=False) -> Dict[str, Optional[bytes]]:
    """``{stem: output PDBQT bytes, or None if the engine wrote none}`` for the ligands ``<stem>.pdbqt`` docked into
    ``pocket``; with ``remove=True`` each output and its ``.json`` are deleted once read."""
    out_dir = os.fspath(out_dir)
    outputs: Dict[str, Optional[bytes]] = {}
    for stem in ligand_stems:
        path = os.path.join(out_dir, engine_output_name(os.fspath(receptor_pdbqt), pocket, f"{stem}.pdbqt"))
        try:
            with open(path, "rb") as f:
                outputs[stem] = f.read()
        except FileNotFoundError:
            outputs[stem] = None
        if remove:
            for p in (path, path[:-len(".pdbqt")] + ".json"):
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
    return outputs
