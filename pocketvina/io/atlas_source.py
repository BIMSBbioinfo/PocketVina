"""Read the files of a PocketVina Atlas release: a local folder or the base URL of one (the folder with meta.json).

    src = Source("https://bimsbstatic.mdc-berlin.de/akalin/asarigun/pocketvina_atlas/data/v1.1/")  # or a folder
    src.json("proteins/P00519.json")                        # parsed JSON (the release has proteins/P00519.json.gz)
    src.range("poses/P00519.poses", 0, 128)                 # bytes of a file (over HTTP: a Range request)
    src.file("structures/AF-P00519-F1-model_v6.pdb.gz")     # a local path (over HTTP: downloaded into the cache)
    src.copy("poses/P00519.poses", "my_atlas/poses/P00519.poses", size=..., sha256=...)  # a checked copy, streamed

Whole files fetched over HTTP are kept in ``<cache_root()>/atlas/<release>/<path>`` (``pocketvina.p2rank.cache_root``)
and reused: a final release never changes. Byte ranges of a final release are kept in memory (``RANGE_CACHE`` bytes
per Source). HTTP uses http.client (standard library) with connections kept open and reused (``MAX_IDLE`` per
server; through a proxy, and to follow a redirect: urllib, one connection per request; after a permanent redirect
requests go to the new place), with a timeout and 3 retries for connection errors, server errors (5xx) and answers cut
short.

The pose files (PVATLAS1, ``pocketvina.io.posefile``) are read with ``pair_blocks`` (the mode blocks of one ligand,
every pocket), ``top_record`` (its top pose) and ``pose_records`` (both, for many ligands at once: byte ranges close
together are read in one request), the same code for a folder and a URL.
"""
import gzip
import hashlib
import http.client
import json
import logging
import operator
import os
import re
import ssl
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .. import __version__
from ..p2rank import cache_root
from . import posefile

logger = logging.getLogger(__name__)

USER_AGENT = f"pocketvina/{__version__}"
TIMEOUT = 60  # seconds an HTTP request may wait for the server (to connect, and for each read)
RETRY_DELAYS = (1, 2, 4)  # seconds before each retry of a request that could not connect or got a server error
# the release stores these files only gzip-compressed: "proteins/P00519.json" means proteins/P00519.json.gz (the
# site's GZ_DATA rule, lib/client.ts)
GZ_DATA = re.compile(r"^(search|proteins|compounds|salt_forms)/[^/]+\.json$")
_URL = re.compile(r"^https?://", re.IGNORECASE)
_CHUNK = 1 << 20  # bytes read and written at a time by Source.copy
MAX_IDLE = 16  # open connections kept per server for the next requests
IDLE_TIMEOUT = 30  # seconds a kept connection may wait for its next request (servers close idle ones)
RANGE_CACHE = 64 << 20  # bytes of byte ranges a Source keeps in memory (over HTTP, a final release)
MERGE_GAP = 64 << 10  # byte ranges of a file at most this far apart are read in one request (Source.ranges)
MERGE_MAX = 4 << 20  # ... as long as that request asks for at most this many bytes
_DRAIN = 64 << 10  # what is left of an answer is read (so its connection can be reused) if it is at most this long
_REDIRECT = (301, 302, 303, 307, 308)
_STALE = (http.client.RemoteDisconnected, ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


def _rel(path) -> str:
    """A file of the release as a relative path with "/" ("proteins/P00519.json.gz"); never one outside it."""
    rel = str(path).replace("\\", "/").lstrip("/")
    if not rel or any(part in ("", ".", "..") for part in rel.split("/")):
        raise ValueError(f"not a file of a PocketVina Atlas release: {path!r}")
    return rel


def _folder_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "release"


class Source:
    """The files of one PocketVina Atlas release: ``location`` is its folder (the one with meta.json), e.g.
    "/data/atlas/v1.1", or the http(s) URL of that folder."""

    def __init__(self, location):
        text = os.fspath(location)
        if _URL.match(text):
            self.url: Optional[str] = text.rstrip("/") + "/"
            self.path: Optional[str] = None
        else:
            self.url = None
            self.path = os.path.abspath(os.path.expanduser(text))
        self._meta: Optional[dict] = None
        self._base = self.url  # where requests go: the URL, or where a permanent redirect of it leads
        self._proxied = bool(self.url) and _proxied(self.url)  # through a proxy: urllib (it reads the settings)
        # byte ranges read over HTTP: path -> [(offset, bytes)], oldest first in _kept_order (Source.range)
        self._kept: Dict[str, List[Tuple[int, bytes]]] = {}
        self._kept_order: deque = deque()
        self._kept_bytes = 0
        self._lock = threading.Lock()

    def __getstate__(self):  # pickle, copy: without the lock and the ranges kept in memory
        state = self.__dict__.copy()
        state.update(_kept={}, _kept_order=deque(), _kept_bytes=0)
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    @property
    def location(self) -> str:
        """The folder or URL of the release."""
        return self.url or self.path

    def __repr__(self):
        return f"Source({self.location!r})"

    # --- whole files -----------------------------------------------------------------------------------------------

    @property
    def meta(self) -> dict:
        """meta.json of the release, read once (never from the cache: it names the release)."""
        if self._meta is None:
            data = self._get("meta.json")[1] if self.url else self._read_local("meta.json")
            self._meta = json.loads(data)
        return self._meta

    def json(self, path: str):
        """A JSON file of the release, parsed: "meta.json", "search/index.json", "proteins/P00519.json" (or
        "proteins/P00519.json.gz"). Gzip-compressed bytes are decompressed (the release stores search/, proteins/,
        compounds/ and salt_forms/ files as .json.gz only)."""
        rel = _rel(path)
        if GZ_DATA.match(rel):
            rel += ".gz"
        if rel == "meta.json":
            return self.meta
        data = self._read_local(rel) if self.path is not None else self._read_cached(rel)
        if data[:2] == b"\x1f\x8b":
            data = gzip.decompress(data)
        return json.loads(data)

    def file(self, path: str) -> str:
        """A whole file of the release as a local path, e.g. "structures/AF-P00519-F1-model_v6.pdb.gz": the file
        itself in a folder; for a URL its copy in the cache, downloaded on first use."""
        rel = _rel(path)
        if self.path is not None:
            return self._local(rel)
        target, reuse = self._cache_path(rel)
        if reuse and os.path.isfile(target):
            return target
        self.copy(rel, target)
        return target

    def copy(self, path: str, target, size: Optional[int] = None, sha256: Optional[str] = None,
             progress: Optional[Callable[[int], None]] = None) -> Tuple[int, str]:
        """Copy a whole file of the release to ``target`` (a local path), streamed: it is written to a temp file next
        to ``target`` and renamed when it is complete and checked, so ``target`` is the whole file or not there.
        ``size``, ``sha256``: what the file must be (the release's MANIFEST.json says it); a copy that differs is not
        kept (ValueError). Over HTTP the body must be as long as the server said (Content-Length); a body cut short is
        fetched again. ``progress(n)``: called with the number of bytes written (a negative number when a fetch starts
        again). Returns the size and sha256 (hex) of the copy."""
        rel = _rel(path)
        local = self._local(rel) if self.path is not None else None
        target = os.path.abspath(os.fspath(target))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = f"{target}.tmp-{os.getpid()}-{threading.get_ident()}"
        written = [0]  # bytes the current attempt reported to progress

        def stream(source, f) -> Tuple[int, str]:
            if written[0] and progress:
                progress(-written[0])  # a new attempt starts again from the first byte
            written[0] = 0
            f.seek(0)
            f.truncate()
            h = hashlib.sha256()
            while True:
                chunk = source.read(_CHUNK)
                if not chunk:
                    break
                try:
                    f.write(chunk)
                except OSError as e:
                    raise _WriteError(e) from e
                h.update(chunk)
                written[0] += len(chunk)
                if progress:
                    progress(len(chunk))
            return written[0], h.hexdigest()

        def fetched(response, f) -> Tuple[int, str]:
            got = stream(response, f)
            if response.length:  # the connection closed before the end of the body the server announced
                raise http.client.IncompleteRead(b"", response.length)
            return got

        try:
            with open(tmp, "wb") as f:
                if local is not None:
                    with open(local, "rb") as source:
                        n, digest = stream(source, f)
                else:
                    n, digest = self._request(rel, None, lambda response: fetched(response, f))
            if size is not None and n != size:
                raise ValueError(f"{self.location}: {rel} has {n} bytes, the release says {size}: the copy is damaged "
                                 f"or incomplete; try again")
            if sha256 is not None and digest != sha256:
                raise ValueError(f"{self.location}: {rel} is damaged (its sha256 is {digest}, the release's "
                                 f"MANIFEST.json says {sha256}); try again")
            os.replace(tmp, target)
        except _WriteError as e:
            raise e.error from None
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return n, digest

    def size(self, path: str) -> Optional[int]:
        """The size of a file of the release in bytes (over HTTP: from the answer to a request for its first byte;
        None if the server does not say). FileNotFoundError if the release has no such file."""
        rel = _rel(path)
        if self.path is not None:
            return os.path.getsize(self._local(rel))

        def read(response) -> Optional[int]:
            if response is None:  # HTTP 416: not even one byte, an empty file
                return 0
            total = re.match(r"^bytes \d+-\d+/(\d+)$", response.headers.get("Content-Range") or "")
            if response.status == 206 and total:
                return int(total.group(1))
            return response.length if response.status == 200 else None

        return self._request(rel, {"Range": "bytes=0-0"}, read)

    def _local(self, rel: str) -> str:
        full = os.path.join(self.path, *rel.split("/"))
        if not os.path.isfile(full):
            raise FileNotFoundError(f"{full} does not exist: no file {rel} in the PocketVina Atlas release {self.path}")
        return full

    def _read_local(self, rel: str) -> bytes:
        with open(self._local(rel), "rb") as f:
            return f.read()

    def _read_cached(self, rel: str) -> bytes:
        with open(self.file(rel), "rb") as f:
            return f.read()

    def _cache_path(self, rel: str) -> Tuple[str, bool]:
        """Where a file fetched over HTTP is kept, and whether a copy there may be used (only for a final release).
        A subset of a release (meta.json "subset", e.g. a download) served over HTTP gets its own folder, so that it
        never mixes with the files of the full release."""
        meta = self.meta
        name = _folder_name(str(self._release().get("label") or meta.get("version") or "unknown"))
        if meta.get("subset"):
            name += "-subset-" + hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:12]
        return os.path.join(cache_root(), "atlas", name, *rel.split("/")), self._release().get("status") == "final"

    def _release(self) -> dict:
        release = self.meta.get("release")
        return release if isinstance(release, dict) else {}

    # --- byte ranges -----------------------------------------------------------------------------------------------

    def range(self, path: str, offset: int, length: int) -> bytes:
        """``length`` bytes of a file from byte ``offset`` on (a seek and read, or an HTTP Range request; a server
        that ignores Range and sends the whole file is answered from its first offset + length bytes). Over HTTP, the
        ranges of a final release are kept in memory: a range inside one read before is not asked for again."""
        rel = _rel(path)
        offset, length = operator.index(offset), operator.index(length)
        if offset < 0 or length < 0:
            raise ValueError(f"{rel}: a byte range needs offset >= 0 and length >= 0, got {offset}, {length}")
        if length == 0:
            return b""
        if self.path is not None:
            with open(self._local(rel), "rb") as f:
                f.seek(offset)
                data = f.read(length)
        else:
            keep = self._release().get("status") == "final"
            data = self._kept_range(rel, offset, length) if keep else None
            if data is not None:
                return data
            status, data = self._get(rel, {"Range": f"bytes={offset}-{offset + length - 1}"}, limit=offset + length)
            if status != 206:
                data = data[offset:offset + length]
            if keep and len(data) == length:
                self._keep_range(rel, offset, data)
        if len(data) != length:
            raise ValueError(f"{self.location}: {rel} has no bytes {offset} to {offset + length - 1} (got {len(data)} "
                             f"of {length}); the file is shorter than the protein's JSON says")
        return data

    def ranges(self, asks: Sequence[Tuple[str, int, int]], map: Callable = map) -> List[bytes]:
        """The bytes of many ranges, ``asks``: (path, offset, length) each, in their order. Ranges of a file at most
        ``MERGE_GAP`` bytes apart are read in one request (over HTTP each request costs a round trip to the server).
        ``map``: how the requests run, e.g. ``pool.map`` of a ThreadPoolExecutor to run them at the same time
        (default: one after another)."""
        spans: Dict[str, List[List[int]]] = {}  # path -> merged [start, end) spans, by start
        for path, offset, length in sorted((_rel(p), operator.index(o), operator.index(n)) for p, o, n in asks):
            if length <= 0:
                continue
            merged = spans.setdefault(path, [])
            if merged and offset - merged[-1][1] <= MERGE_GAP and max(merged[-1][1], offset + length) - merged[-1][0] \
                    <= MERGE_MAX:
                merged[-1][1] = max(merged[-1][1], offset + length)
            else:
                merged.append([offset, offset + length])
        reads = [(path, start, end) for path, merged in spans.items() for start, end in merged]
        got = {(path, start): data for (path, start, _), data in
               zip(reads, map(lambda r: self.range(r[0], r[1], r[2] - r[1]), reads))}
        out = []
        for path, offset, length in asks:
            if length <= 0:
                out.append(self.range(path, offset, length))  # b"", or the error of a negative range
                continue
            rel = _rel(path)
            start = next(s for s, e in spans[rel] if s <= offset and offset + length <= e)
            out.append(got[rel, start][offset - start:offset - start + length])
        return out

    def _kept_range(self, rel: str, offset: int, length: int) -> Optional[bytes]:
        with self._lock:
            for start, data in self._kept.get(rel, ()):
                if start <= offset and offset + length <= start + len(data):
                    return data[offset - start:offset - start + length]
        return None

    def _keep_range(self, rel: str, offset: int, data: bytes) -> None:
        if len(data) > RANGE_CACHE // 4:
            return
        with self._lock:
            self._kept.setdefault(rel, []).append((offset, data))
            self._kept_order.append((rel, offset, data))
            self._kept_bytes += len(data)
            while self._kept_bytes > RANGE_CACHE:
                old_rel, old_offset, old = self._kept_order.popleft()
                kept = self._kept[old_rel]
                del kept[next(i for i, (s, d) in enumerate(kept) if s == old_offset and d is old)]
                self._kept_bytes -= len(old)

    # --- HTTP ------------------------------------------------------------------------------------------------------

    def _get(self, rel: str, headers: Optional[dict] = None, limit: Optional[int] = None) -> Tuple[int, bytes]:
        """(status, body) of a GET of a file of the release; at most ``limit`` bytes of the body are read. A body cut
        short by a closed connection is fetched again."""
        def read(response) -> Tuple[int, bytes]:
            if response is None:  # a byte range past the end of the file: no bytes, as a seek and read gives
                return 416, b""
            body = response.read() if limit is None else response.read(limit)
            if limit is not None and len(body) < limit and response.length:
                # the connection closed before the end of the body the server announced: try again
                raise http.client.IncompleteRead(body, response.length)
            return response.status, body

        return self._request(rel, headers, read)

    def _request(self, rel: str, headers: Optional[dict], read: Callable):
        """``read(response)`` of a GET of a file of the release (``read(None)`` for HTTP 416: a byte range past its
        end). Connection errors, server errors (5xx) and errors while ``read`` reads the body (a body cut short raises
        http.client.IncompleteRead) are tried again, 3 times; HTTP 404 raises FileNotFoundError."""
        quoted = urllib.parse.quote(rel)
        url = self._base + quoted
        headers = {"User-Agent": USER_AGENT, **(headers or {})}

        def moved(final: str):  # a permanent redirect: later requests go to the new place directly
            if final.endswith(quoted) and final != url and _URL.match(final):
                self._base = final[:-len(quoted)]
                self._proxied = _proxied(self._base)
        for attempt in range(len(RETRY_DELAYS) + 1):
            last = attempt == len(RETRY_DELAYS)
            try:
                return _urlopen(url, headers, read) if self._proxied else _fetch(url, headers, read, moved)
            except _Status as e:
                if e.code == 404:
                    raise FileNotFoundError(f"{url} does not exist (HTTP 404): no such file in the PocketVina "
                                            f"Atlas release") from None
                if e.code < 500 or last:
                    raise OSError(f"{url}: the server answered HTTP {e.code} {e.reason}") from None
                why = f"HTTP {e.code}"
            except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
                why = str(e.reason if isinstance(e, urllib.error.URLError) else e) or type(e).__name__
                if last:
                    raise ConnectionError(f"cannot reach the PocketVina Atlas at {self.url}: {why}; use "
                                          f"Atlas('/path/to/release') for a local copy") from None
            logger.info("%s: %s, trying again in %s s", url, why, RETRY_DELAYS[attempt])
            time.sleep(RETRY_DELAYS[attempt])
        raise AssertionError("unreachable")  # pragma: no cover


# --- HTTP connections ------------------------------------------------------------------------------------------------
# urllib opens a new connection for every request (it sends "Connection: close"); over HTTPS each costs several round
# trips to the server, which far from it (e.g. Colab to Berlin) makes a page of many small byte ranges slow. Requests
# go through http.client instead, on connections kept open (MAX_IDLE per server, shared by all threads); urllib is kept
# for a proxy (urllib reads the proxy settings) and to follow a redirect.

class _Status(Exception):
    """An HTTP error status (>= 400) of a request."""

    def __init__(self, code: int, reason: str):
        super().__init__(f"HTTP {code} {reason}")
        self.code, self.reason = code, reason


# (scheme, host, port, pid) -> [(open connection not in use, when it was last used)]
_idle: Dict[tuple, List[Tuple[http.client.HTTPConnection, float]]] = {}
_idle_lock = threading.Lock()
_ssl_context: Optional[ssl.SSLContext] = None


def _proxied(url: str) -> bool:
    """Whether requests to ``url`` go through a proxy (the proxy settings; urllib's rule), or the URL has a user name
    (urllib's job too)."""
    parts = urllib.parse.urlsplit(url)
    if parts.username:
        return True
    return bool(urllib.request.getproxies().get(parts.scheme.lower())) and \
        not urllib.request.proxy_bypass(parts.hostname or "")


def _fetch(url: str, headers: dict, read: Callable, moved: Optional[Callable[[str], None]] = None):
    """``read(response)`` of one GET on a kept connection (``read(None)`` for HTTP 416); an HTTP error raises
    _Status. A redirect is followed by urllib; ``moved(final URL)`` is called after a permanent one (301, 308)."""
    parts = urllib.parse.urlsplit(url)
    key = (parts.scheme.lower(), parts.hostname, parts.port, os.getpid())
    target = parts.path + (f"?{parts.query}" if parts.query else "")
    conn, reused = _connection(key)
    response, keep = None, False
    try:
        try:
            conn.request("GET", target, headers=headers)
            response = conn.getresponse()
        except ssl.SSLError:
            _forget_ssl_context()  # e.g. a certificate error: certificates set up since then are used next time
            raise
        except _STALE:
            if not reused:
                raise
            # a kept connection the server has closed in the meantime: once more, on a new one
            conn.close()
            conn, reused = _connection(key, new=True)
            conn.request("GET", target, headers=headers)
            response = conn.getresponse()
        if response.status in _REDIRECT:
            location = response.getheader("Location")
            keep = _drain(response)
            if not location:
                raise _Status(response.status, f"{response.reason} (no Location)")
            # urllib goes there (and follows any further redirect)
            return _urlopen(urllib.parse.urljoin(url, location), headers, read,
                            moved if response.status in (301, 308) else None)
        if response.status >= 400:
            keep = _drain(response)
            if response.status == 416:
                return read(None)
            raise _Status(response.status, response.reason)
        result = read(response)
        keep = _drain(response)
        return result
    finally:
        if keep and not response.will_close:
            with _idle_lock:
                idle = _idle.setdefault(key, [])
                if len(idle) < MAX_IDLE:
                    idle.append((conn, time.monotonic()))
                    conn = None
        if conn is not None:
            conn.close()


def _connection(key: tuple, new: bool = False) -> Tuple[http.client.HTTPConnection, bool]:
    """An open connection to the server of ``key`` if one is free (reused: True), else a new one."""
    global _ssl_context
    old = []
    with _idle_lock:
        idle = _idle.get(key) or []
        while idle and not new:
            conn, since = idle.pop()
            if time.monotonic() - since <= IDLE_TIMEOUT:
                break
            old.append(conn)
        else:
            conn = None
        if key[0] == "https" and _ssl_context is None:
            # urllib's context: the system's certificates, or what ssl._create_default_https_context is set to
            _ssl_context = ssl._create_default_https_context()
    for stale in old:
        stale.close()
    if conn is not None:
        return conn, True
    scheme, host, port, _ = key
    if scheme == "https":
        return http.client.HTTPSConnection(host, port, timeout=TIMEOUT, context=_ssl_context), False
    return http.client.HTTPConnection(host, port, timeout=TIMEOUT), False


def _forget_ssl_context() -> None:
    global _ssl_context
    with _idle_lock:
        _ssl_context = None


def _drain(response: http.client.HTTPResponse) -> bool:
    """Read what is left of an answer's body when it is short (an error page, a byte not read); True if the whole body
    has been read, so that its connection can serve the next request."""
    try:
        if not response.isclosed() and response.length is not None and response.length <= _DRAIN:
            response.read()
    except (http.client.HTTPException, OSError):
        return False
    return response.isclosed()


def _urlopen(url: str, headers: dict, read: Callable, moved: Optional[Callable[[str], None]] = None):
    """``_fetch`` through urllib: a new connection, with the proxy settings (as they are now) and redirects;
    ``moved(url)`` is called if ``url`` answered without redirecting further (the target of a permanent redirect)."""
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.build_opener().open(request, timeout=TIMEOUT) as response:
            if moved is not None and response.geturl() == url:
                moved(url)
            return read(response)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            return read(None)
        raise _Status(e.code, e.reason) from None


class _WriteError(Exception):
    """An error writing a copy (Source.copy): not a network error, so it is not tried again."""

    def __init__(self, error: OSError):
        super().__init__(str(error))
        self.error = error


# --- pose files ----------------------------------------------------------------------------------------------------

def pair_blocks(source: Source, ref: dict, idx: int) -> List[Optional[bytes]]:
    """The mode blocks of record ``idx`` of a pose file, pockets 1 to K: the engine's PDBQT output (every mode) of
    each pocket, or None for a pocket without one. ``ref``: the pose file as the protein's JSON describes it
    (``poses`` or ``poses_redock``: file, N, K, table_b_offset). Two byte ranges are read: the K + 1 entries of table
    B, then the K blocks, which lie next to each other in the file."""
    return pose_records(source, [(ref, idx, 0, 0)])[0][0]


def top_record(source: Source, file: str, offset: int, length: int) -> Optional[bytes]:
    """The top pose (SDF bytes) stored at [offset, offset + length) of a pose file; None when length is 0 (no top
    pose)."""
    if not length:
        return None
    return posefile.gunzip_member(source.range(file, offset, length))


def pose_records(source: Source, records: Sequence[Tuple[dict, int, int, int]],
                 map: Callable = map) -> List[Tuple[List[Optional[bytes]], Optional[bytes]]]:
    """``pair_blocks`` and ``top_record`` of many records at once: for each (ref, idx, top offset, top length), its
    mode blocks (pockets 1 to K) and its top pose (None when the length is 0). Two rounds of reads: the records' table
    B entries with their top poses, then their blocks; ranges close together are read in one request
    (``Source.ranges``). ``map``: how the requests of a round run, e.g. ``pool.map`` (default: one after another)."""
    tables, tops = [], []
    for ref, idx, top_offset, top_length in records:
        n, k, table_b = int(ref["N"]), int(ref["K"]), int(ref["table_b_offset"])
        idx = operator.index(idx)
        if not 0 <= idx < n:
            raise IndexError(f"{ref['file']}: no record {idx} (records 0 to {n - 1})")
        tables.append((ref["file"], table_b + 8 * idx * k, 8 * (k + 1)))
        tops.append((ref["file"], int(top_offset), int(top_length)))
    first = source.ranges(tables + [top for top in tops if top[2]], map)
    raw_tops = iter(first[len(tables):])
    spans, all_starts = [], []
    for (ref, idx, *_), raw in zip(records, first):
        starts = struct.unpack(f"<{int(ref['K']) + 1}Q", raw)
        if any(end < start for start, end in zip(starts, starts[1:])):
            raise ValueError(f"{ref['file']}: table B is not in order at record {idx}")
        spans.append((ref["file"], starts[0], starts[-1] - starts[0]))
        all_starts.append(starts)
    out = []
    for starts, span, top in zip(all_starts, source.ranges(spans, map), tops):
        blocks = [posefile.gunzip_member(span[start - starts[0]:end - starts[0]]) if end > start else None
                  for start, end in zip(starts, starts[1:])]
        out.append((blocks, posefile.gunzip_member(next(raw_tops)) if top[2] else None))
    return out


__all__ = ["Source", "pair_blocks", "top_record", "pose_records", "GZ_DATA"]
