"""pocketvina.p2rank: where P2Rank is found, its one-time download into the user cache, the Java check, and the
``pocketvina setup p2rank`` command.

Nothing here downloads the real release: a small stand-in archive with the same layout (``p2rank_2.5.1/prank``, ...)
is served from a ``file://`` URL, with ``P2RANK_SHA256`` patched to its digest. Java is a fake ``java`` script.
"""
import hashlib
import io
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import urllib.error
from pathlib import Path

import pytest
from click.testing import CliRunner

from pocketvina import cli, p2rank, pipeline
from pocketvina.main import P2RankConfig, P2RankProcessor

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
TOP = p2rank.ARCHIVE_TOP
TOOLS = ("sh", "cp", "basename", "dirname", "cat", "sleep")  # what the fake prank scripts run


def _exe(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    if not os.access(path, os.X_OK):
        pytest.skip(f"cannot execute {path} (noexec temp dir?)")
    return path


def make_home(path: Path) -> Path:
    """A folder that looks like an unpacked P2Rank."""
    _exe(path / "prank", "echo fake prank\n")
    (path / "bin").mkdir(exist_ok=True)
    (path / "bin" / "p2rank.jar").write_bytes(b"jar")
    return path


def make_archive(path: Path, entries=None, top=TOP) -> Path:
    """A gzipped tar with the release's layout; ``entries`` = {name: bytes | None (folder) | TarInfo} to use instead."""
    if entries is None:
        entries = {f"{top}": None, f"{top}/prank": b"#!/bin/sh\necho prank $@\n", f"{top}/bin": None,
                   f"{top}/bin/p2rank.jar": b"jar" * 100, f"{top}/models/default/model.zst": b"model"}
    with tarfile.open(path, "w:gz") as tar:
        for name, data in entries.items():
            if isinstance(data, tarfile.TarInfo):
                tar.addfile(data)
                continue
            info = tarfile.TarInfo(name)
            if data is None:
                info.type, info.mode = tarfile.DIRTYPE, 0o775
                tar.addfile(info)
            else:
                info.size, info.mode = len(data), 0o664  # prank not executable: ensure_p2rank makes it so
                tar.addfile(info, io.BytesIO(data))
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """No P2Rank and no Java anywhere: no $P2RANK_HOME or $JAVA_HOME, PATH = ``bin`` (for fake prank/java scripts)
    + a folder with only the shell tools the fakes use, an empty package dir, a fresh user cache."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tools = tmp_path / "tools"
    tools.mkdir()
    for tool in TOOLS:
        found = shutil.which(tool)
        if found is None:
            pytest.skip(f"needs {tool}")
        (tools / tool).symlink_to(found)
    monkeypatch.delenv("P2RANK_HOME", raising=False)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{tools}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(p2rank, "PACKAGE_DIR", str(tmp_path / "pkg"))
    return tmp_path


@pytest.fixture
def release(env, monkeypatch):
    """A stand-in release archive served from a file:// URL, pinned in the module; returns its path."""
    (env / "release").mkdir()
    archive = make_archive(env / "release" / f"p2rank_{p2rank.P2RANK_VERSION}.tar.gz")
    monkeypatch.setattr(p2rank, "P2RANK_URL", archive.as_uri())
    monkeypatch.setattr(p2rank, "P2RANK_SHA256", sha256(archive))
    monkeypatch.setattr(p2rank, "P2RANK_SIZE", archive.stat().st_size)
    return archive


@pytest.fixture
def downloads(monkeypatch):
    """Counts the downloads (``_download`` calls)."""
    calls = []
    real = p2rank._download

    def counting(url, path, progress):
        calls.append(url)
        return real(url, path, progress)

    monkeypatch.setattr(p2rank, "_download", counting)
    return calls


def fake_java(bin_dir: Path, version_line: str) -> Path:
    """A ``java`` that prints ``version_line`` to stderr for ``-version``, as Java does."""
    return _exe(bin_dir / "java", f"echo '{version_line}' >&2\n")


def leftovers(root: Path) -> list:
    """Temp folders or downloads an install left behind in ``root``."""
    return sorted(p.name for p in root.iterdir() if p.name.startswith(f".{p2rank.CACHE_NAME}.tmp-")) \
        if root.is_dir() else []


# --- where P2Rank is found -----------------------------------------------------------------------------------------

def test_find_order(env, monkeypatch, caplog):
    cache = Path(p2rank.cache_home())
    assert cache == env / "cache" / "pocketvina" / f"p2rank-{p2rank.P2RANK_VERSION}"
    assert p2rank.find_p2rank() is None and p2rank.p2rank_home() is None
    make_home(cache)
    assert p2rank.find_p2rank() == (str(cache), "cache")
    legacy = make_home(env / "pkg" / "p2rank_2.5")
    assert p2rank.find_p2rank() == (str(legacy), "package")
    on_path = make_home(env / "conda" / "share" / "p2rank")
    (env / "bin" / "prank").symlink_to(on_path / "prank")  # as a conda package links it: the real folder is used
    found = p2rank.find_p2rank()
    assert found == (str(on_path), "PATH") and found.launcher == str(on_path / "prank")
    explicit = make_home(env / "explicit")
    monkeypatch.setenv("P2RANK_HOME", str(explicit))
    assert p2rank.find_p2rank() == (str(explicit), "P2RANK_HOME") and p2rank.p2rank_home() == str(explicit)
    monkeypatch.setenv("P2RANK_HOME", str(env / "nothing_here"))
    with caplog.at_level(logging.WARNING, logger="pocketvina.p2rank"):
        assert p2rank.find_p2rank().source == "PATH"
    assert "P2RANK_HOME" in caplog.text and "ignored" in caplog.text


def test_prank_wrapper_on_path_is_run_as_it_is(env):
    """A ``prank`` on PATH that links to a file of another name (a wrapper script): its own folder is the home."""
    wrapper = _exe(env / "opt" / "p2rank_wrapper.sh", "echo wrapped\n")
    (env / "bin" / "prank").symlink_to(wrapper)
    assert p2rank.find_p2rank() == (str(env / "bin"), "PATH")


def test_prank_aligner_on_path_is_not_p2rank(env):
    """The PRANK sequence aligner (bioconda::prank, Debian prank) is a compiled ``prank`` without P2Rank's jar next
    to it: skipped, so the next P2Rank is used, or downloaded."""
    aligner = env / "bin" / "prank"
    aligner.write_bytes(b"\x7fELF\x02\x01\x01" + b"\0" * 57)
    aligner.chmod(0o755)
    assert p2rank.find_p2rank() is None
    legacy = make_home(env / "pkg" / "p2rank_2.5")
    assert p2rank.find_p2rank() == (str(legacy), "package")
    (env / "bin" / "bin").mkdir()  # P2Rank's jar next to a launcher: P2Rank, whatever the launcher is
    (env / "bin" / "bin" / "p2rank.jar").write_bytes(b"jar")
    assert p2rank.find_p2rank() == (str(env / "bin"), "PATH")


def test_cache_root_without_xdg_or_home(env, monkeypatch):
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(env / "home"))
    assert p2rank.cache_root() == str(env / "home" / ".cache" / "pocketvina")
    monkeypatch.setenv("XDG_CACHE_HOME", "relative")  # invalid in the XDG spec: ignored
    assert p2rank.cache_root() == str(env / "home" / ".cache" / "pocketvina")
    monkeypatch.setattr(os.path, "expanduser", lambda p: p)  # no home directory
    assert p2rank.cache_root().startswith(os.path.abspath(os.sep))  # never relative to the working directory


def test_importing_pocketvina_changes_no_environment_variable(tmp_path):
    """Before v0.2 the import set P2RANK_HOME, PATH, LD_LIBRARY_PATH and OPENCL_ROOT."""
    code = ("import os, sys; keys = ('P2RANK_HOME', 'PATH', 'LD_LIBRARY_PATH', 'OPENCL_ROOT'); "
            "before = [os.environ.get(k) for k in keys]; import pocketvina; "
            "sys.exit(0 if [os.environ.get(k) for k in keys] == before else 1)")
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    run = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True,
                         timeout=120)
    assert run.returncode == 0, run.stderr
    import pocketvina

    assert pocketvina.p2rank_home == os.path.join(pocketvina.package_dir, "p2rank_2.5")  # pre-v0.2 attribute


# --- ensure_p2rank -------------------------------------------------------------------------------------------------

def test_ensure_downloads_once_into_the_cache(env, release, downloads):
    launcher = Path(p2rank.ensure_p2rank())
    assert launcher == Path(p2rank.cache_home()) / "prank" and os.access(launcher, os.X_OK)
    assert (launcher.parent / "bin" / "p2rank.jar").read_bytes() == b"jar" * 100
    assert downloads == [release.as_uri()]
    assert p2rank.ensure_p2rank() == str(launcher) and downloads == [release.as_uri()]  # found: not again
    assert p2rank.find_p2rank().source == "cache"
    assert leftovers(launcher.parent.parent) == []


def test_ensure_uses_a_p2rank_that_is_found(env, release, downloads):
    home = make_home(env / "pkg" / "p2rank_2.5")
    assert p2rank.ensure_p2rank() == str(home / "prank") and downloads == []
    assert p2rank.ensure_p2rank(download=False) == str(home / "prank")


def test_ensure_without_download(env, release):
    with pytest.raises(p2rank.P2RankError, match="pocketvina setup p2rank") as e:
        p2rank.ensure_p2rank(download=False)
    assert "P2RANK_HOME" in str(e.value) and not os.path.exists(p2rank.cache_root())


def test_ensure_offline(env, release, monkeypatch):
    def no_network(*args, **kwargs):
        raise urllib.error.URLError(OSError(101, "Network is unreachable"))

    monkeypatch.setattr(p2rank.urllib.request, "urlopen", no_network)
    with pytest.raises(p2rank.P2RankError, match="Network is unreachable") as e:
        p2rank.ensure_p2rank()
    assert "--from" in str(e.value) and p2rank.P2RANK_URL in str(e.value)
    assert not os.path.exists(p2rank.cache_home()) and leftovers(Path(p2rank.cache_root())) == []


def test_ensure_rejects_a_damaged_download(env, release, monkeypatch):
    monkeypatch.setattr(p2rank, "P2RANK_SHA256", "0" * 64)
    with pytest.raises(p2rank.P2RankError, match="damaged"):
        p2rank.ensure_p2rank()
    assert not os.path.exists(p2rank.cache_home()) and leftovers(Path(p2rank.cache_root())) == []


def test_ensure_download_broken_off(env, release, monkeypatch):
    class Response(io.BytesIO):
        headers = {}

        def read(self, n=-1):
            if self.tell():
                raise ConnectionResetError(104, "Connection reset by peer")
            return super().read(10)

    monkeypatch.setattr(p2rank.urllib.request, "urlopen", lambda *a, **k: Response(release.read_bytes()))
    with pytest.raises(p2rank.P2RankError, match="broke off after 10 bytes"):
        p2rank.ensure_p2rank()
    assert not os.path.exists(p2rank.cache_home())


def test_ensure_from_archive_and_root(env, release, downloads):
    root = env / "shared"
    launcher = p2rank.ensure_p2rank(root=root, archive=release)
    assert launcher == str(root / p2rank.CACHE_NAME / "prank") and downloads == []
    assert p2rank.find_p2rank() is None  # another root: only through $P2RANK_HOME
    other = make_archive(env / "other.tar.gz", top=TOP)
    other.write_bytes(other.read_bytes() + b"\0" * 512)  # another file: not the pinned release
    with pytest.raises(p2rank.P2RankError, match="is not the P2Rank"):
        p2rank.ensure_p2rank(root=env / "elsewhere", archive=other)
    assert not (env / "elsewhere" / p2rank.CACHE_NAME).exists()


def test_ensure_force_replaces_an_install(env, release, downloads):
    home = Path(p2rank.cache_home())
    p2rank.ensure_p2rank()
    (home / "stale").write_text("old")
    p2rank.ensure_p2rank(force=True)
    assert len(downloads) == 2 and not (home / "stale").exists() and (home / "prank").is_file()


def test_ensure_replaces_a_folder_without_launcher(env, release):
    home = Path(p2rank.cache_home())
    home.mkdir(parents=True)
    (home / "partial").write_text("made by hand")
    p2rank.ensure_p2rank()
    assert (home / "prank").is_file() and not (home / "partial").exists()


@pytest.mark.parametrize("entry", ["../evil", "/abs/evil", f"{TOP}/../../evil", "other_top/prank", "symlink"])
def test_unsafe_archives_are_refused(env, monkeypatch, entry):
    if entry == "symlink":
        link = tarfile.TarInfo(f"{TOP}/bin/link")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        entries = {f"{TOP}": None, f"{TOP}/prank": b"x", f"{TOP}/bin/link": link}
    else:
        entries = {f"{TOP}": None, f"{TOP}/prank": b"x", entry: b"evil"}
    archive = make_archive(env / "bad.tar.gz", entries)
    monkeypatch.setattr(p2rank, "P2RANK_SHA256", sha256(archive))
    with pytest.raises(p2rank.P2RankError, match="unexpected entry"):
        p2rank.ensure_p2rank(archive=archive)
    assert not os.path.exists(p2rank.cache_home()) and not (env / "evil").exists()


def test_archive_without_launcher_is_refused(env, monkeypatch):
    archive = make_archive(env / "nolauncher.tar.gz", {f"{TOP}": None, f"{TOP}/README.md": b"readme"})
    monkeypatch.setattr(p2rank, "P2RANK_SHA256", sha256(archive))
    with pytest.raises(p2rank.P2RankError, match="has no"):
        p2rank.ensure_p2rank(archive=archive)
    assert not os.path.exists(p2rank.cache_home())


class _WaitSeen(logging.Handler):
    """Sets ``event`` when ensure_p2rank logs that it waits for another process's install."""

    def __init__(self):
        super().__init__()
        self.event = threading.Event()

    def emit(self, record):
        if "waiting for another process" in record.getMessage():
            self.event.set()


@pytest.mark.parametrize("fail", [False, True], ids=["ok", "failed"])
def test_concurrent_installs_download_once(env, release, monkeypatch, fail):
    """Two processes (here threads: flock locks are per open file) installing at once: one downloads, the other
    waits and then uses its P2Rank; if that download failed, the waiting one does not download again."""
    calls, started, seen = [], threading.Event(), _WaitSeen()
    real = p2rank._download

    def download(url, path, progress):  # holds the lock until the other thread waits for it
        calls.append(url)
        started.set()
        assert seen.event.wait(20)
        if fail:
            raise p2rank.P2RankError("cannot download (test)")
        return real(url, path, progress)

    monkeypatch.setattr(p2rank, "_download", download)
    p2rank.logger.addHandler(seen)
    results = {}

    def install(name):
        try:
            results[name] = p2rank.ensure_p2rank()
        except p2rank.P2RankError as e:
            results[name] = e

    try:
        first = threading.Thread(target=install, args=("first",))
        first.start()
        assert started.wait(20)
        second = threading.Thread(target=install, args=("second",))
        second.start()
        first.join(30)
        second.join(30)
    finally:
        p2rank.logger.removeHandler(seen)
    assert len(calls) == 1
    if fail:
        assert "cannot download (test)" in str(results["first"])
        assert "another process could not install" in str(results["second"])
    else:
        assert results["first"] == results["second"] == os.path.join(p2rank.cache_home(), "prank")
    assert leftovers(Path(p2rank.cache_root())) == []


def test_nothing_is_written_into_the_package_dir(env, release, monkeypatch):
    pkg = env / "pkg"
    pkg.mkdir()
    p2rank.ensure_p2rank()
    assert list(pkg.iterdir()) == []


# --- Java ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, major", [
    ('openjdk version "17.0.8" 2023-07-18', 17),
    ('java version "1.8.0_292"', 8),
    ('openjdk version "21" 2023-09-19', 21),
    ('openjdk version "17.0.14-internal" 2025-01-21', 17),
    ('Picked up JAVA_TOOL_OPTIONS: -Xmx1g\nopenjdk version "11.0.2" 2019-01-15', 11),
    ("no version here", None),
    ('openjdk version "x.y"', None),
])
def test_parse_java_version(text, major):
    assert p2rank.parse_java_version(text) == major


def test_check_java(env, monkeypatch):
    with pytest.raises(p2rank.JavaError, match="no Java .*`java` is not on PATH") as e:
        p2rank.check_java()
    assert "openjdk" in str(e.value) and p2rank.java_problem() == str(e.value)
    fake_java(env / "bin", 'java version "1.8.0_292"')
    with pytest.raises(p2rank.JavaError, match="is Java 8"):
        p2rank.check_java()
    fake_java(env / "bin", 'openjdk version "17.0.8" 2023-07-18')
    assert p2rank.check_java() == 17 and p2rank.java_problem() is None
    assert p2rank.java_command() == str(env / "bin" / "java")
    # P2Rank's launcher prefers $JAVA_HOME/bin/java: so does the check
    monkeypatch.setenv("JAVA_HOME", str(env / "jdk11"))
    with pytest.raises(p2rank.JavaError, match="JAVA_HOME"):
        p2rank.check_java()
    fake_java(env / "jdk11" / "bin", 'openjdk version "11.0.2" 2019-01-15')
    with pytest.raises(p2rank.JavaError, match="is Java 11"):
        p2rank.check_java()
    _exe(env / "jdk11" / "bin" / "java", "exit 1\n")
    with pytest.raises(p2rank.JavaError, match="cannot tell the version"):
        p2rank.check_java()


# --- the pipeline and the files format use it ----------------------------------------------------------------------

def test_predict_pockets_without_p2rank_stops_the_run(env, release, monkeypatch, example_dir):
    monkeypatch.setattr(p2rank, "P2RANK_SHA256", "0" * 64)  # the download fails
    with pytest.raises(pipeline.RunAborted, match="damaged"):
        pipeline.predict_pockets(example_dir / "proteins" / "1fbl.pdb", env / "work")


def test_predict_pockets_downloads_p2rank_and_runs_it(env, release, monkeypatch, example_dir, downloads):
    """No P2Rank: the stand-in release is downloaded into the cache, and its prank is what runs."""
    pockets_csv = TESTS_DIR / "data" / "legacy_run" / "proteins" / "pockets" / "AF-Q96LA5-F1-model_v4.csv"
    body = ('while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift;; -f) f="$2"; shift;; esac; shift; done\n'
            f'cp "{pockets_csv}" "$out/$(basename "$f")_predictions.csv"\n')
    archive = make_archive(env / "release" / "p2rank-run.tar.gz", {f"{TOP}": None,
                                                                    f"{TOP}/prank": ("#!/bin/sh\n" + body).encode()})
    monkeypatch.setattr(p2rank, "P2RANK_URL", archive.as_uri())
    monkeypatch.setattr(p2rank, "P2RANK_SHA256", sha256(archive))
    csv_path = pipeline.predict_pockets(example_dir / "proteins" / "1fbl.pdb", env / "work")
    assert len(pipeline.read_pockets_csv(csv_path)) == 22 and len(downloads) == 1


def test_predict_pockets_explains_a_missing_java(env, example_dir, monkeypatch):
    """The real launcher without a Java exits with 127: the error says how to install Java."""
    make_home(env / "p2rank")
    _exe(env / "p2rank" / "prank", "echo 'prank: line 36: java: command not found' >&2\nexit 127\n")
    monkeypatch.setenv("P2RANK_HOME", str(env / "p2rank"))
    with pytest.raises(pipeline.RunAborted, match="exit code 127; P2Rank needs Java 17 or later") as e:
        pipeline.predict_pockets(example_dir / "proteins" / "1fbl.pdb", env / "work")
    assert "openjdk" in str(e.value) and "command not found" in str(e.value)


def test_p2rank_processor_runs_the_p2rank_found(env, tmp_path, example_dir, monkeypatch):
    home = make_home(env / "p2rank")
    pockets_csv = TESTS_DIR / "data" / "legacy_run" / "proteins" / "pockets" / "AF-Q96LA5-F1-model_v4.csv"
    _exe(home / "prank", 'while [ $# -gt 0 ]; do case "$1" in -o) out="$2"; shift;; -f) f="$2"; shift;; esac; '
                         f'shift; done\ncp "{pockets_csv}" "$out/$(basename "$f")_predictions.csv"\n')
    monkeypatch.setenv("P2RANK_HOME", str(home))
    proc = P2RankProcessor(P2RankConfig(output_dir=str(tmp_path / "out")))
    assert proc.run_p2rank(str(example_dir / "proteins" / "1fbl.pdb"))
    assert (tmp_path / "out" / "proteins" / "pockets" / "1fbl.csv").is_file()


def test_p2rank_processor_tries_to_install_p2rank_once(env, release, tmp_path, example_dir, monkeypatch, downloads,
                                                       caplog):
    """No P2Rank and its download fails (no network, a full disk...): the proteins of a list fail without one more
    download attempt each (with a network that drops packets each would wait for the timeout)."""
    monkeypatch.setattr(p2rank, "P2RANK_SHA256", "0" * 64)  # the download fails
    proteins = [str(example_dir / "proteins" / name) for name in ("1fbl.pdb", "2W83.pdb")]
    proc = P2RankProcessor(P2RankConfig(output_dir=str(tmp_path / "out")))
    with caplog.at_level(logging.ERROR, logger="P2RankProcessor"):
        results = proc.process_file_list(proteins)
    assert results == dict.fromkeys(proteins, False) and len(downloads) == 1
    assert caplog.text.count("damaged") == 2  # both proteins say why


def test_p2rank_processor_logs_why_p2rank_failed(env, tmp_path, example_dir, monkeypatch, caplog):
    home = make_home(env / "p2rank")
    _exe(home / "prank", "echo 'Exception in thread main java.lang.UnsupportedClassVersionError' >&2\nexit 1\n")
    fake_java(env / "bin", 'openjdk version "11.0.2" 2019-01-15')
    monkeypatch.setenv("P2RANK_HOME", str(home))
    proc = P2RankProcessor(P2RankConfig(output_dir=str(tmp_path / "out")))
    with caplog.at_level(logging.ERROR, logger="P2RankProcessor"):
        assert not proc.run_p2rank(str(example_dir / "proteins" / "1fbl.pdb"))
    assert "UnsupportedClassVersionError" in caplog.text and "is Java 11" in caplog.text


# --- pocketvina setup p2rank ---------------------------------------------------------------------------------------

def invoke(*args):
    return CliRunner().invoke(cli.cli, ["setup", "p2rank", *map(str, args)], catch_exceptions=False)


def test_setup_p2rank_downloads_once(env, release, downloads):
    fake_java(env / "bin", 'openjdk version "17.0.8" 2023-07-18')
    result = invoke()
    assert result.exit_code == 0, result.output
    home = p2rank.cache_home()
    assert f"P2Rank: {home} (P2Rank {p2rank.P2RANK_VERSION} installed)" in result.output
    assert "Java: 17" in result.output and len(downloads) == 1
    result = invoke()
    assert result.exit_code == 0 and f"P2Rank: {home} (found: the user cache)" in result.output
    assert len(downloads) == 1
    result = invoke("--check")
    assert result.exit_code == 0 and "found: the user cache" in result.output and len(downloads) == 1


def test_setup_p2rank_dir_and_from(env, release, downloads):
    fake_java(env / "bin", 'openjdk version "21" 2023-09-19')
    result = invoke("--dir", env / "opt", "--from", release)
    assert result.exit_code == 0, result.output
    home = env / "opt" / p2rank.CACHE_NAME
    assert (home / "prank").is_file() and downloads == []
    assert f"Set P2RANK_HOME={home} for runs to use it." in result.output  # not found by runs otherwise
    make_home(env / "pkg" / "p2rank_2.5")
    result = invoke("--dir", env / "opt")
    assert f"(now they use {env / 'pkg' / 'p2rank_2.5'})" in result.output
    result = invoke("--dir", env / "opt")
    assert result.exit_code == 0 and "installed already" in result.output and downloads == []


def test_setup_p2rank_problems_exit_with_1(env, release, monkeypatch):
    result = invoke("--check")  # no P2Rank, no Java
    assert result.exit_code == cli.EXIT_USAGE
    assert "pocketvina setup p2rank" in result.output and "no Java" in result.output
    assert not os.path.exists(p2rank.cache_root())
    result = invoke()  # installs P2Rank, but there is no Java for it
    assert result.exit_code == cli.EXIT_USAGE and "installed" in result.output and "openjdk" in result.output
    monkeypatch.setattr(p2rank, "P2RANK_SHA256", "0" * 64)
    result = invoke("--force")
    assert result.exit_code == cli.EXIT_USAGE and "damaged" in result.output
    result = invoke("--check", "--force")
    assert result.exit_code == cli.EXIT_USAGE and "--check installs nothing" in result.output
