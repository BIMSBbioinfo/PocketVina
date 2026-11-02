"""P2Rank, the pocket predictor PocketVina docks into: where it is installed, a one-time download of the release
PocketVina is tested with, and the Java it needs.

``p2rank_home()`` is the first P2Rank folder (one with a ``prank`` launcher) of

1. ``$P2RANK_HOME``,
2. the folder of the ``prank`` on ``PATH`` (P2Rank's launcher, or a link or wrapper script to it; symlinks resolved;
   not a compiled ``prank`` without P2Rank's ``bin/p2rank.jar`` next to it, such as the PRANK sequence aligner that
   bioconda and Debian install under that name),
3. ``pocketvina/p2rank_2.5/`` in the package directory (source installs before v0.2 unpacked it there),
4. the user cache ``~/.cache/pocketvina/p2rank-2.5.1/`` (``$XDG_CACHE_HOME`` instead of ``~/.cache`` when set).

``ensure_p2rank()`` returns the ``prank`` launcher of that folder and, if there is none, downloads the official
P2Rank 2.5.1 release once (checked against its pinned sha256) and unpacks it into the user cache; the folder appears
complete or not at all, and processes starting at the same time (e.g. SLURM array jobs) download it only once.
``pocketvina setup p2rank`` does the same ahead of a run, e.g. on a login node that has internet access. Nothing is
ever written into the package directory.

P2Rank needs Java 17 or later: its launcher runs ``$JAVA_HOME/bin/java`` when ``JAVA_HOME`` is set, else ``java`` on
``PATH``. ``check_java()`` finds that Java the same way and explains how to install one.

Standard library only (tqdm draws the download progress bar when asked to).

P2Rank is described in the `"P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding
sites from protein structure" <https://doi.org/10.1186/s13321-018-0285-8>`_ paper.
"""
import contextlib
import hashlib
import http.client
import logging
import os
import posixpath
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
import zlib
from typing import Iterator, NamedTuple, Optional

logger = logging.getLogger(__name__)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
P2RANK_VERSION = "2.5.1"
P2RANK_URL = f"https://github.com/rdk/p2rank/releases/download/{P2RANK_VERSION}/p2rank_{P2RANK_VERSION}.tar.gz"
# the release asset's sha256 (as GitHub also lists it for p2rank_2.5.1.tar.gz) and size in bytes
P2RANK_SHA256 = "d243f2d9036ac053fefb9407b5fe1c85f4fe077c519fd975ac585e995feab274"
P2RANK_SIZE = 275_625_956
ARCHIVE_TOP = f"p2rank_{P2RANK_VERSION}"  # the archive's only top-level folder
CACHE_NAME = f"p2rank-{P2RANK_VERSION}"  # the folder ensure_p2rank creates in the cache (or in ``root``)
LEGACY_NAME = "p2rank_2.5"  # in the package directory: source installs before v0.2
LAUNCHER = "prank"
JAVA_MIN = 17  # P2Rank 2.5 is compiled for Java 17
DOWNLOAD_TIMEOUT_S = 60  # per connection attempt and read
_CHUNK = 1 << 20
_JAVA_VERSION_RE = re.compile(r'version "([^"]+)"')
_JAVA_HELP = ('install one, e.g. `conda install -c conda-forge "openjdk>=17"`, `sudo apt-get install '
              'openjdk-17-jre-headless` (Debian/Ubuntu) or a JDK from https://adoptium.net, and put its `java` first '
              'on PATH or set JAVA_HOME to it')


class P2RankError(RuntimeError):
    """P2Rank is not installed and cannot be installed here (no network, a bad archive, a full disk...)."""


class JavaError(RuntimeError):
    """No Java, or one older than P2Rank needs."""


class P2RankInstall(NamedTuple):
    """A P2Rank found by ``find_p2rank``: its folder and where it was found."""

    home: str
    source: str  # "P2RANK_HOME", "PATH", "package" or "cache"

    @property
    def launcher(self) -> str:
        return os.path.join(self.home, LAUNCHER)


# --- where P2Rank is -----------------------------------------------------------------------------------------------

def cache_root() -> str:
    """The user cache folder of PocketVina: ``~/.cache/pocketvina`` (``$XDG_CACHE_HOME/pocketvina`` when that is set;
    under the system temp dir if there is no home directory), as for the engine's kernel cache."""
    base = os.environ.get("XDG_CACHE_HOME", "")
    if not os.path.isabs(base):  # unset, or relative (invalid in the XDG spec)
        base = os.path.join(os.path.expanduser("~"), ".cache")
    if not os.path.isabs(base):  # no home directory: never a path relative to the working directory
        tag = str(os.getuid()) if hasattr(os, "getuid") else "user"
        base = os.path.join(tempfile.gettempdir(), f"pocketvina-{tag}")
    return os.path.join(base, "pocketvina")


def cache_home() -> str:
    """Where ``ensure_p2rank`` puts P2Rank by default: ``<cache_root()>/p2rank-2.5.1``."""
    return os.path.join(cache_root(), CACHE_NAME)


def is_p2rank_home(path) -> bool:
    """Whether ``path`` is a folder with a ``prank`` launcher."""
    return bool(path) and os.path.isfile(os.path.join(os.fspath(path), LAUNCHER))


def _is_p2rank_launcher(path: str) -> bool:
    """Whether the ``prank`` at ``path`` can be P2Rank's launcher: P2Rank's ``bin/p2rank.jar`` is next to it, or it is
    a script (e.g. a wrapper that runs a P2Rank elsewhere). Not a compiled program, such as the PRANK sequence
    aligner (conda package ``bioconda::prank``, Debian package ``prank``)."""
    if os.path.isfile(os.path.join(os.path.dirname(path), "bin", "p2rank.jar")):
        return True
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"#!"
    except OSError:
        return False


def find_p2rank() -> Optional[P2RankInstall]:
    """The first P2Rank of ``$P2RANK_HOME``, ``prank`` on ``PATH``, the package folder ``p2rank_2.5/`` and the user
    cache (see the module docstring); None if there is none. A ``$P2RANK_HOME`` without a ``prank`` is skipped with a
    warning, a ``prank`` on ``PATH`` that is not P2Rank (e.g. the PRANK aligner) silently."""
    env = os.environ.get("P2RANK_HOME")
    if env:
        home = os.path.abspath(os.path.expanduser(env))
        if is_p2rank_home(home):
            return P2RankInstall(home, "P2RANK_HOME")
        logger.warning("$P2RANK_HOME=%s has no %s launcher, ignored", env, LAUNCHER)
    on_path = shutil.which(LAUNCHER)
    if on_path:
        # the launcher finds its jars next to the path it is run by: run the file a symlink points to
        real = os.path.realpath(on_path)
        home = os.path.dirname(real if os.path.basename(real) == LAUNCHER else os.path.abspath(on_path))
        if is_p2rank_home(home) and _is_p2rank_launcher(os.path.join(home, LAUNCHER)):
            return P2RankInstall(home, "PATH")
        logger.debug("%s on PATH is not P2Rank's launcher (no bin/p2rank.jar next to it), ignored", on_path)
    for home, source in ((os.path.join(PACKAGE_DIR, LEGACY_NAME), "package"), (cache_home(), "cache")):
        if is_p2rank_home(home):
            return P2RankInstall(home, source)
    return None


def p2rank_home() -> Optional[str]:
    """The folder of the P2Rank PocketVina runs (see ``find_p2rank``), or None if there is none."""
    found = find_p2rank()
    return found.home if found else None


# --- installing it -------------------------------------------------------------------------------------------------

def missing_message(home: Optional[str] = None) -> str:
    """Why P2Rank is not found and what to do (``home``: the folder it was expected in, default ``cache_home()``)."""
    home = home or cache_home()
    return (f"P2Rank not found ($P2RANK_HOME, `prank` on PATH, {os.path.join(PACKAGE_DIR, LEGACY_NAME)}, {home}): "
            f"run `pocketvina setup p2rank` (downloads P2Rank {P2RANK_VERSION}, {P2RANK_SIZE / 2**20:.0f} MiB, "
            "once), or set P2RANK_HOME to an unpacked P2Rank 2.5 folder")


def _offline_message() -> str:
    return (f"Without internet access here: download {P2RANK_URL} on a machine that has it, then run "
            f"`pocketvina setup p2rank --from p2rank_{P2RANK_VERSION}.tar.gz`, or set P2RANK_HOME to an unpacked "
            "P2Rank 2.5 folder")


def ensure_p2rank(download: bool = True, *, root=None, archive=None, force: bool = False,
                  progress: bool = False) -> str:
    """The ``prank`` launcher of the P2Rank to run, installing P2Rank 2.5.1 first if there is none.

    Without ``root``, ``archive`` and ``force`` the P2Rank of ``find_p2rank()`` is used if there is one. Otherwise
    P2Rank goes to ``<root>/p2rank-2.5.1`` (``root`` default: ``cache_root()``), unless it is there already (and not
    ``force``): unpacked from ``archive`` (a ``p2rank_2.5.1.tar.gz`` downloaded elsewhere) or, with ``download``,
    downloaded from ``P2RANK_URL``; either must match ``P2RANK_SHA256``. ``progress`` shows a download progress bar
    (tqdm, on stderr). Raises ``P2RankError`` with what to do instead if P2Rank is missing and cannot be installed.
    A P2Rank installed into another ``root`` is found by later runs only through ``$P2RANK_HOME``.
    """
    if root is None and archive is None and not force:
        found = find_p2rank()
        if found:
            return found.launcher
    root = os.path.abspath(os.path.expanduser(os.fspath(root))) if root is not None else cache_root()
    home = os.path.join(root, CACHE_NAME)
    if is_p2rank_home(home) and not force:
        return os.path.join(home, LAUNCHER)
    if archive is None and not download:
        raise P2RankError(missing_message(home))
    try:
        os.makedirs(root, exist_ok=True)
    except OSError as e:
        raise P2RankError(f"cannot create {root} for P2Rank: {e}") from e
    with _install_lock(root) as waited:
        if is_p2rank_home(home) and (waited or not force):
            return os.path.join(home, LAUNCHER)  # installed meanwhile by another process
        if waited and archive is None:
            # that process could not install it either: no point in every waiting job downloading in turn
            raise P2RankError(f"another process could not install P2Rank into {home} (see its log). "
                              f"{_offline_message()}")
        _install(home, archive, progress)
    return os.path.join(home, LAUNCHER)


@contextlib.contextmanager
def _install_lock(root: str) -> Iterator[bool]:
    """An exclusive lock on ``<root>/.p2rank-2.5.1.lock`` (POSIX ``flock``); yields whether another process held it
    first. Without file locks (Windows, some network filesystems) nothing is locked; the install is still atomic."""
    try:
        import fcntl
    except ImportError:
        yield False
        return
    path = os.path.join(root, f".{CACHE_NAME}.lock")
    try:
        f = open(path, "a")
    except OSError:
        yield False
        return
    with f:
        waited = False
        try:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                waited = True
                logger.warning("waiting for another process installing P2Rank into %s", root)
                fcntl.flock(f, fcntl.LOCK_EX)
            locked = True
        except OSError:  # a filesystem without flock
            locked = False
        try:
            yield waited
        finally:
            if locked:
                fcntl.flock(f, fcntl.LOCK_UN)


def _install(home: str, archive, progress: bool) -> None:
    """Unpack ``archive`` (or the downloaded release) into a temp folder next to ``home``, then rename it to
    ``home`` (replacing an older one)."""
    parent = os.path.dirname(home)
    try:
        work = tempfile.mkdtemp(prefix=f".{CACHE_NAME}.tmp-", dir=parent)
    except OSError as e:
        raise P2RankError(f"cannot write into {parent}: {e}") from e
    try:
        if archive is None:
            archive = os.path.join(work, f"p2rank_{P2RANK_VERSION}.tar.gz")
            logger.warning("downloading P2Rank %s (%.0f MiB) into %s, once", P2RANK_VERSION, P2RANK_SIZE / 2**20,
                           home)
            _download(P2RANK_URL, archive, progress)
        else:
            archive = os.path.abspath(os.path.expanduser(os.fspath(archive)))
            _check_archive(archive)
        unpacked = os.path.join(work, "unpacked")
        _unpack(archive, unpacked)
        src = os.path.join(unpacked, ARCHIVE_TOP)
        launcher = os.path.join(src, LAUNCHER)
        os.chmod(launcher, os.stat(launcher).st_mode | 0o111)
        old = None
        if os.path.lexists(home):  # force, or a folder without a launcher (e.g. made by hand): replace it
            old = os.path.join(work, "old")
            os.rename(home, old)
        try:
            os.rename(src, home)
        except OSError:
            if is_p2rank_home(home):  # another process was faster (a filesystem without flock)
                return
            if old is not None and not os.path.lexists(home):
                os.rename(old, home)  # keep what was there
            raise
        logger.info("P2Rank %s installed in %s", P2RANK_VERSION, home)
    except OSError as e:
        raise P2RankError(f"cannot install P2Rank into {home}: {e}") from e
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _download(url: str, path: str, progress: bool) -> None:
    """``url`` to ``path`` (streamed), checked against ``P2RANK_SHA256``."""
    h = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "pocketvina"})
    with open(path, "wb") as f:
        try:
            response = urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_S)
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as e:
            reason = getattr(e, "reason", None) or e
            raise P2RankError(f"cannot download P2Rank {P2RANK_VERSION} from {url}: {reason}. "
                              f"{_offline_message()}") from e
        bar = None
        with response:
            if progress:
                from tqdm import tqdm

                total = int(response.headers.get("Content-Length") or 0) or P2RANK_SIZE
                bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=f"P2Rank {P2RANK_VERSION}")
            try:
                while True:
                    try:
                        chunk = response.read(_CHUNK)
                    except (http.client.HTTPException, OSError) as e:
                        raise P2RankError(f"the P2Rank download from {url} broke off after {size} bytes: {e}. "
                                          f"Try again. {_offline_message()}") from e
                    if not chunk:
                        break
                    f.write(chunk)  # OSError (e.g. a full disk): _install reports it
                    h.update(chunk)
                    size += len(chunk)
                    if bar is not None:
                        bar.update(len(chunk))
            finally:
                if bar is not None:
                    bar.close()
    if h.hexdigest() != P2RANK_SHA256:
        raise P2RankError(f"the P2Rank download from {url} is damaged or not P2Rank {P2RANK_VERSION} ({size} bytes, "
                          f"sha256 {h.hexdigest()}; expected {P2RANK_SIZE} bytes, sha256 {P2RANK_SHA256}): try again")


def _check_archive(path: str) -> None:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                h.update(chunk)
    except OSError as e:
        raise P2RankError(f"cannot read {path}: {e}") from e
    if h.hexdigest() != P2RANK_SHA256:
        raise P2RankError(f"{path} is not the P2Rank {P2RANK_VERSION} release file p2rank_{P2RANK_VERSION}.tar.gz "
                          f"(sha256 {h.hexdigest()}, expected {P2RANK_SHA256}); download it from {P2RANK_URL}")


def _member_name(m: tarfile.TarInfo) -> str:
    """The path of archive entry ``m``; ``P2RankError`` unless it is a plain file or folder inside ``ARCHIVE_TOP/``
    (the release holds nothing else)."""
    name = m.name[2:] if m.name.startswith("./") else m.name
    parts = name.split("/")
    if posixpath.isabs(name) or ".." in parts or parts[0] != ARCHIVE_TOP or not (m.isfile() or m.isdir()):
        raise P2RankError(f"unexpected entry {m.name!r} in the P2Rank archive")
    return name


def _unpack(archive: str, dest: str) -> None:
    """Unpack ``archive`` into ``dest`` in one pass, entry by entry after checking it (``_member_name``)."""
    data_filter = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}  # 3.12, 3.9-3.11 security releases
    launcher = f"{ARCHIVE_TOP}/{LAUNCHER}"
    found = False
    try:
        with tarfile.open(archive, "r|*") as tar:
            for m in tar:
                name = _member_name(m)
                found = found or (name == launcher and m.isfile())
                tar.extract(m, dest, **data_filter)
    except (tarfile.TarError, EOFError, zlib.error) as e:
        raise P2RankError(f"cannot unpack the P2Rank archive {archive}: {e}") from e
    if not found:
        raise P2RankError(f"the P2Rank archive {archive} has no {launcher}")


# --- Java ----------------------------------------------------------------------------------------------------------

def java_command() -> Optional[str]:
    """The ``java`` P2Rank's launcher runs: ``$JAVA_HOME/bin/java`` when ``JAVA_HOME`` is set, else ``java`` on
    ``PATH``; None if that does not exist."""
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        exe = os.path.join(java_home, "bin", "java")
        return exe if os.path.isfile(exe) and os.access(exe, os.X_OK) else None
    return shutil.which("java")


def parse_java_version(text: str) -> Optional[int]:
    """The major version in ``java -version`` output (``1.8.0_292`` -> 8, ``17.0.8`` -> 17, ``21`` -> 21)."""
    m = _JAVA_VERSION_RE.search(text)
    if not m:
        return None
    parts = re.split(r"[._+-]", m.group(1))
    try:
        major = int(parts[0])
        return int(parts[1]) if major == 1 and len(parts) > 1 else major
    except ValueError:
        return None


def java_version(java: Optional[str] = None) -> Optional[int]:
    """Major version of ``java`` (default ``java_command()``); None if it is missing or does not say."""
    java = java or java_command()
    if not java:
        return None
    try:
        out = subprocess.run([java, "-version"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace", timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_java_version(out)


def check_java(minimum: int = JAVA_MIN) -> int:
    """The major version of the Java P2Rank would run; ``JavaError`` saying how to get one if it is missing or older
    than ``minimum``."""
    java = java_command()
    if java is None:
        java_home = os.environ.get("JAVA_HOME")
        where = (f"$JAVA_HOME is {java_home}, but {os.path.join(java_home, 'bin', 'java')} does not exist"
                 if java_home else "`java` is not on PATH")
        raise JavaError(f"P2Rank needs Java {minimum} or later, but there is no Java ({where}): {_JAVA_HELP}")
    version = java_version(java)
    if version is None:
        raise JavaError(f"P2Rank needs Java {minimum} or later: cannot tell the version of {java} (`{java} -version` "
                        f"failed). If it is not a working Java, {_JAVA_HELP}")
    if version < minimum:
        raise JavaError(f"P2Rank needs Java {minimum} or later, but {java} is Java {version}: {_JAVA_HELP}")
    return version


def java_problem(minimum: int = JAVA_MIN) -> Optional[str]:
    """What is wrong with the Java P2Rank would run (``check_java``'s message), or None if it is fine."""
    try:
        check_java(minimum)
    except JavaError as e:
        return str(e)
    return None
