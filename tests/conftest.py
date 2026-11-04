"""Shared pytest fixtures and the ``gpu`` marker.

``tests/data/`` holds a subset of a real PocketVina run (copied from ``test/test_karin/``: AF-Q96LA5-F1-model_v4
docked with the example ligands; SMILES from ``test/example/ligands.txt``):

  smiles.csv                    molecule_name,smiles of the three fixture ligands (checked against the PDBQTs)
  legacy_run/                   a legacy ("files" format) PocketVina output_dir
    batch_1/<name>_cleaned.pdbqt                     engine input ligands: molecule2, molecule5, molecule10
    batch_1_out/AF-Q96LA5-F1-model_v4-pocket{1,2}-molecule{2,5}_cleaned_out.{pdbqt,json}   engine outputs, 9 modes
    proteins/pockets/AF-Q96LA5-F1-model_v4.csv       P2Rank predictions, all 22 pockets (only 1 and 2 were kept)
    proteins/structures/AF-Q96LA5-F1-model_v4.pdbqt  receptor, cut to whole residues within 22 A of pockets 1-2
    combined_results.csv, combined_results.json      the original rows/entries for these four outputs

molecule10 has a silicon atom, which the engine rejects: it has no output in any pocket, as in the original run.
molecule2 carries a carboxylic acid; molecule5 a stereocentre (unspecified in its SMILES) and an E double bond.

``tests/data/atlas/v1.1/`` is a tiny PocketVina Atlas release in the layout of the real one (three proteins with a
few drugs each; built by ``tests/data/atlas/make_fixture.py``, which says what is in it). ``atlas_dir`` is its path
and ``atlas_server`` serves it over HTTP with Range support (``http_server`` serves any folder).

Tests marked ``@pytest.mark.gpu`` are skipped unless ``POCKETVINA_GPU_TESTS=1``, tests marked
``@pytest.mark.network`` (they read the public Atlas over the internet) unless ``POCKETVINA_NETWORK_TESTS=1``.
"""
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import stat
import sys
import threading
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
DATA_DIR = TESTS_DIR / "data"
ATLAS_DIR = DATA_DIR / "atlas" / "v1.1"
PACKAGE_DIR = REPO_ROOT / "pocketvina"
FAKE_ENGINE = TESTS_DIR / "fake_engine.py"

pytest_plugins = ["pytester"]

SKIP_UNLESS = {"gpu": ("POCKETVINA_GPU_TESTS", "needs a GPU: set POCKETVINA_GPU_TESTS=1 to run"),
               "network": ("POCKETVINA_NETWORK_TESTS", "needs the internet: set POCKETVINA_NETWORK_TESTS=1 to run")}


def pytest_collection_modifyitems(config, items):
    for marker, (variable, reason) in SKIP_UNLESS.items():
        if os.environ.get(variable) == "1":
            continue
        skip = pytest.mark.skip(reason=reason)
        for item in items:
            # not `marker in item.keywords`: that also matches a parametrize id or a node of that name
            if item.get_closest_marker(marker) is not None:
                item.add_marker(skip)


def _package_entries():
    try:
        return set(os.listdir(PACKAGE_DIR))
    except OSError:
        return set()


@pytest.fixture(scope="session", autouse=True)
def _no_writes_into_package_dir():
    """Fail the session if a test leaves new non-Python files or folders in ``pocketvina/``."""
    before = _package_entries()
    yield
    new = sorted(n for n in _package_entries() - before
                 if not n.endswith((".py", ".pyc")) and n not in ("__pycache__", "io"))
    if new:
        pytest.fail(f"tests wrote into the package directory {PACKAGE_DIR}: {new}", pytrace=False)


def _data_digests():
    return {p.relative_to(DATA_DIR).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in DATA_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


@pytest.fixture(scope="session", autouse=True)
def _fixture_data_unchanged():
    """Fail the session if a test changed ``tests/data/`` (e.g. an engine run on ``real_run.ligand_dir`` without
    ``output_directory`` writes to ``<ligand_dir>_out`` == the fixture's ``batch_1_out/``, same file names)."""
    before = _data_digests()
    yield
    after = _data_digests()
    changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
    if changed:
        pytest.fail(f"tests changed the read-only fixture data in {DATA_DIR}: {changed}", pytrace=False)


@dataclass(frozen=True)
class RealRun:
    """Paths and facts of the fixture run in ``tests/data/`` (read-only: copy before writing)."""

    root: Path
    smiles_csv: Path
    protein_id: str = "AF-Q96LA5-F1-model_v4"
    ligands: tuple = ("molecule2", "molecule5", "molecule10")
    docked: tuple = ("molecule2", "molecule5")
    failed: tuple = ("molecule10",)
    pockets: tuple = (1, 2)
    n_pockets: int = 22
    n_modes: int = 9

    @property
    def receptor(self) -> Path:
        return self.root / "proteins" / "structures" / f"{self.protein_id}.pdbqt"

    @property
    def pockets_csv(self) -> Path:
        return self.root / "proteins" / "pockets" / f"{self.protein_id}.csv"

    @property
    def combined_csv(self) -> Path:
        return self.root / "combined_results.csv"

    @property
    def combined_json(self) -> Path:
        return self.root / "combined_results.json"

    @property
    def ligand_dir(self) -> Path:
        return self.root / "batch_1"

    def ligand_pdbqt(self, name: str) -> Path:
        return self.ligand_dir / f"{name}_cleaned.pdbqt"

    def output(self, name: str, pocket: int, ext: str = "pdbqt") -> Path:
        """Engine output of ligand ``name`` in ``pocket`` (``ext`` "pdbqt" or "json")."""
        return self.root / "batch_1_out" / f"{self.protein_id}-pocket{pocket}-{name}_cleaned_out.{ext}"

    def outputs(self) -> list:
        return [self.output(n, k) for n in self.docked for k in self.pockets]

    def smiles(self) -> dict:
        """``{molecule_name: smiles}`` from ``smiles.csv``."""
        with open(self.smiles_csv, newline="") as f:
            return {row["molecule_name"]: row["smiles"] for row in csv.DictReader(f)}

    def pocket_center(self, pocket: int) -> tuple:
        """(x, y, z) of a P2Rank pocket (1-based rank)."""
        with open(self.pockets_csv, newline="") as f:
            for row in csv.DictReader(f, skipinitialspace=True):
                row = {k.strip(): v.strip() for k, v in row.items()}
                if int(row["rank"]) == pocket:
                    return tuple(float(row[c]) for c in ("center_x", "center_y", "center_z"))
        raise KeyError(pocket)


class FakeEngine(str):
    """The fake engine as installed by ``fake_engine_env``: its executable path (== ``$POCKETVINA_ENGINE``).

    A ``str``, so it works wherever an engine path is expected; also has ``.path``, ``.log`` and ``.calls()``.
    """

    log: Path  # one JSON line per engine call

    def __new__(cls, path, log):
        self = super().__new__(cls, path)
        self.log = Path(log)
        return self

    def __getnewargs__(self):
        # copy/deepcopy/pickle (and dataclasses.asdict) call __new__ with these
        return str(self), str(self.log)

    @property
    def path(self) -> str:
        return str(self)

    def calls(self) -> list:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session")
def example_dir() -> Path:
    """The repository's ``example/`` folder (ligands.txt, protein_files.txt, proteins/*.pdb). Do not write into it."""
    return REPO_ROOT / "example"


@pytest.fixture(scope="session")
def real_run() -> RealRun:
    return RealRun(root=DATA_DIR / "legacy_run", smiles_csv=DATA_DIR / "smiles.csv")


@pytest.fixture
def legacy_run_dir(tmp_path) -> str:
    """A private copy of ``tests/data/legacy_run`` (a legacy output_dir), as a str path."""
    dst = tmp_path / "legacy_run"
    shutil.copytree(DATA_DIR / "legacy_run", dst)
    return str(dst)


@pytest.fixture
def tmp_results(tmp_path) -> str:
    """Path (str) of a results directory that does not exist yet."""
    return str(tmp_path / "results")


@pytest.fixture(scope="session")
def fake_engine_path() -> Path:
    return FAKE_ENGINE


@pytest.fixture
def fake_engine_env(tmp_path, monkeypatch) -> FakeEngine:
    """Point ``$POCKETVINA_ENGINE`` at the fake engine (run with this interpreter) and log its calls.

    Also sets ``$POCKETVINA_KERNEL_CACHE`` to a temp folder so nothing is written under ``~/.cache``, makes
    ``pocketvina.gpu.list_gpus`` find no GPU, so that atlas runs plan one engine slot whatever GPUs the test machine
    has (tests of several slots patch it with fake GPUs). ``$OCL_ICD_VENDORS`` is unset (a shell may set it, see
    ``pocketvina.engine.run_engine``), and the engine starts out passing it on again.
    """
    import pocketvina.engine
    import pocketvina.gpu

    monkeypatch.setattr(pocketvina.gpu, "list_gpus", lambda: [])
    monkeypatch.setattr(pocketvina.engine, "_drop_ocl_vendors", False)
    monkeypatch.delenv("OCL_ICD_VENDORS", raising=False)
    bin_dir = tmp_path / "fake_engine_bin"
    bin_dir.mkdir()
    exe = bin_dir / "PocketVina-GPU"
    exe.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(FAKE_ENGINE))} "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if not os.access(exe, os.X_OK):
        # tmp_path on a noexec mount (a hardened /tmp on some clusters): run the script through its own shebang
        if not os.access(FAKE_ENGINE, os.X_OK):
            pytest.skip(f"cannot execute {exe} (noexec temp dir?) nor {FAKE_ENGINE}; set --basetemp elsewhere")
        exe = FAKE_ENGINE
    log = tmp_path / "fake_engine_calls.jsonl"
    monkeypatch.setenv("POCKETVINA_ENGINE", str(exe))
    monkeypatch.setenv("POCKETVINA_FAKE_ENGINE_LOG", str(log))
    monkeypatch.setenv("POCKETVINA_KERNEL_CACHE", str(tmp_path / "kernel_cache"))
    monkeypatch.delenv("POCKETVINA_FAKE_ENGINE_SLEEP", raising=False)
    monkeypatch.delenv("POCKETVINA_FAKE_ENGINE_SHARED_RNG", raising=False)
    return FakeEngine(path=str(exe), log=log)


# --- a local HTTP server (the Atlas over HTTP) ---------------------------------------------------------------------

class _FileHandler(BaseHTTPRequestHandler):
    """Serves the server's folder: whole files, and one byte range when the request has a Range header and the
    server answers ranges (``server.ranges = False`` makes it send the whole file instead, status 200); a range that
    starts past the end of the file gets status 416. ``server.cut = n``: the next n answers stop halfway and the
    connection closes."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # no request log in the test output
        pass

    def do_GET(self):
        rel = urllib.parse.unquote(urllib.parse.urlparse(self.path).path).lstrip("/")
        path = Path(self.server.root, *rel.split("/"))
        self.server.requests.append((rel, self.headers.get("Range")))
        if not path.is_file() or ".." in rel.split("/"):
            self.send_error(404, "no such file")
            return
        data = path.read_bytes()
        asked = re.match(r"bytes=(\d+)-(\d+)$", self.headers.get("Range") or "")
        if asked and self.server.ranges:
            start, end = int(asked.group(1)), int(asked.group(2))
            if start >= len(data):
                self.send_error(416, "range not satisfiable")
                return
            body, status = data[start:end + 1], 206
        else:
            body, status = data, 200
        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{start + len(body) - 1}/{len(data)}")
        self.end_headers()
        if self.server.cut:
            self.server.cut -= 1
            body = body[:len(body) // 2]
            self.close_connection = True
        self.wfile.write(body)


class FixtureServer(ThreadingHTTPServer):
    """A local HTTP server for a folder: ``url`` (its base URL, ending in "/"), ``requests`` (one
    ``(path, Range header)`` per request), ``ranges`` (set it to False to answer Range requests with the whole
    file, as some servers do) and ``cut`` (set it to n to drop the connection halfway through the next n answers)."""

    daemon_threads = True

    def __init__(self, root):
        super().__init__(("127.0.0.1", 0), _FileHandler)
        self.root = str(root)
        self.requests = []
        self.ranges = True
        self.cut = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"


@pytest.fixture
def http_server():
    """``http_server(folder)`` serves that folder on 127.0.0.1 while the test runs (a ``FixtureServer``)."""
    servers = []

    def start(root) -> FixtureServer:
        server = FixtureServer(root)
        # a short poll interval: shutdown() at the end of the test waits for one
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="session")
def atlas_dir() -> Path:
    """The Atlas fixture release ``tests/data/atlas/v1.1`` (read-only: copy before writing)."""
    return ATLAS_DIR


@pytest.fixture
def atlas_server(http_server, atlas_dir) -> FixtureServer:
    """The Atlas fixture release served over HTTP (``atlas_server.url``)."""
    return http_server(atlas_dir)


@pytest.fixture
def closed_port() -> str:
    """The URL of a port on 127.0.0.1 that nothing listens on (a connection there fails at once)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/"
