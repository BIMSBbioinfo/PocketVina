"""pocketvina.io.posefile: verbatim copy of the Atlas PVATLAS1 module (byte-compatible), Reader as context manager."""
import difflib
import gzip
import importlib.util
import inspect
import json
import os
import random
import struct
import zlib
from pathlib import Path

import pytest

from pocketvina.io import posefile as pf

ATLAS_SCRIPTS = os.environ.get("POCKETVINA_ATLAS_SCRIPTS",
                               "/data/akalin/asarigun/pocketvina_atlas/_work/db_prototype/scripts")
ATLAS_POSEFILE = Path(ATLAS_SCRIPTS) / "posefile.py"


@pytest.fixture(scope="module")
def atlas_pf():
    """The original Atlas module, imported by path (skipped where the Atlas folder is not available)."""
    if not os.path.isfile(ATLAS_POSEFILE):  # also False where /data is not readable
        pytest.skip(f"PocketVina Atlas posefile.py not found: {ATLAS_POSEFILE}")
    spec = importlib.util.spec_from_file_location("_atlas_posefile", ATLAS_POSEFILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def records(N, K, seed=0, missing_top=(), missing_blocks=()):
    """Deterministic test content: SDF-like tops and PDBQT-like blocks of varying size."""
    rng = random.Random(seed)
    tops = {i: f"top {i}\n".encode() * rng.randint(1, 50) for i in range(N) if i not in missing_top}
    blocks = {j: f"MODEL 1\nblock {j}\nENDMDL\n".encode() * rng.randint(1, 80)
              for j in range(N * K) if j not in missing_blocks}
    return tops, blocks


def write(mod, path, N, K, tops, blocks, meta, order=None, level=9):
    w = mod.Writer(str(path), N, K, meta)
    for i in order if order is not None else sorted(tops):
        w.add_top(i, mod.gz_member(tops[i], level), len(tops[i]))
    w.begin_modes()
    for j in sorted(blocks):
        w.add_block(j, mod.gz_member(blocks[j], level))
    return w.close()


def test_public_names():
    for name in ("MAGIC", "VERSION", "Writer", "Reader", "gz_member", "gunzip_member", "sha256"):
        assert hasattr(pf, name), name
    assert pf.MAGIC == b"PVATLAS1" and pf.VERSION == 1 and pf.HEADER_SIZE == 128


def test_copy_is_verbatim():
    """Compared with the Atlas file, the copy only adds comment lines and Reader.__enter__/__exit__."""
    if not os.path.isfile(ATLAS_POSEFILE):  # also False where /data is not readable
        pytest.skip(f"PocketVina Atlas posefile.py not found: {ATLAS_POSEFILE}")
    atlas = ATLAS_POSEFILE.read_text().splitlines()
    ours = Path(pf.__file__).read_text().splitlines()
    added = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, atlas, ours, autojunk=False).get_opcodes():
        assert tag in ("equal", "insert"), f"changed Atlas lines {i1 + 1}-{i2}: {atlas[i1:i2]}"
        if tag == "insert":
            added += ours[j1:j2]
    code = [line for line in added if line.strip() and not line.lstrip().startswith("#")]
    assert code == ["    def __enter__(self):", "        return self",
                    "    def __exit__(self, *exc):", "        self.close()"]
    assert added[0].startswith("# Verbatim copy") and str(ATLAS_POSEFILE.name) in "\n".join(added)


def test_round_trip(tmp_path):
    N, K = 7, 3
    tops, blocks = records(N, K, missing_top={2, 5}, missing_blocks={0, 4, 5, 20})
    meta = {"protein_id": "P1", "K": K, "note": "ünïcode"}
    order = [6, 0, 3, 1, 4]  # any order, as written in affinity order
    info = write(pf, tmp_path / "p.poses", N, K, tops, blocks, meta, order=order)
    assert info["file_size"] == os.path.getsize(tmp_path / "p.poses")
    with pf.Reader(str(tmp_path / "p.poses")) as r:
        assert (r.N, r.K, r.meta) == (N, K, meta)
        for i in range(N):
            assert r.top(i) == tops.get(i)
            off, ln = r.top_range(i)
            assert (ln == 0) == (i not in tops)
            for k in range(1, K + 1):
                j = i * K + (k - 1)
                assert r.modes(i, k) == blocks.get(j)
                off, ln = r.block_range(i, k)
                assert (ln == 0) == (j not in blocks)
                assert off == r.B[j]
        # section A in the order written; each section is one valid multi-member gzip stream
        offs = [r.top_range(i)[0] for i in order]
        assert offs == sorted(offs) and offs[0] == r.sa
        with open(tmp_path / "p.poses", "rb") as f:
            data = f.read()
        assert gzip.decompress(data[r.sa:r.sb]) == b"".join(tops[i] for i in order)
        assert gzip.decompress(data[r.sb:r.size]) == b"".join(blocks[j] for j in sorted(blocks))
        assert r.B[N * K] == r.size == len(data)


def test_header_layout(tmp_path):
    tops, blocks = records(2, 2)
    write(pf, tmp_path / "p.poses", 2, 2, tops, blocks, {"a": 1})
    data = (tmp_path / "p.poses").read_bytes()
    h = pf.HEADER.unpack(data[:pf.HEADER.size])
    assert h[0] == pf.MAGIC and h[1:4] == (1, 2, 2) and h[4] == 128 and h[-1] == 12
    assert data[pf.HEADER.size:128] == b"\0" * (128 - pf.HEADER.size)
    meta_off, meta_len = h[10], h[11]
    assert meta_off == 128 and meta_len % 8 == 0
    assert json.loads(data[meta_off:meta_off + meta_len]) == {"a": 1}


def test_gz_member_deterministic():
    raw = b"MODEL 1\nREMARK VINA RESULT: -5.0 0.0 0.0\nENDMDL\n" * 10
    gz = pf.gz_member(raw)
    assert gz == pf.gz_member(raw)
    assert gz[:3] == b"\x1f\x8b\x08" and gz[4:8] == b"\0\0\0\0" and gz[9] == 255  # mtime 0, OS unknown
    assert pf.gunzip_member(gz) == raw == gzip.decompress(gz)
    assert pf.gunzip_member(pf.gz_member(b"")) == b""
    assert len(pf.gz_member(raw, 1)) >= len(pf.gz_member(raw, 9))


def test_gunzip_member_rejects():
    gz = pf.gz_member(b"abc")
    with pytest.raises(ValueError):
        pf.gunzip_member(gz + gz)
    with pytest.raises(ValueError):
        pf.gunzip_member(gz + b"x")
    with pytest.raises((ValueError, zlib.error)):
        pf.gunzip_member(gz[:-3])


def test_sha256():
    import hashlib
    assert pf.sha256(b"abc") == hashlib.sha256(b"abc").digest() and len(pf.sha256(b"")) == 32


def test_reader_context_manager(tmp_path):
    tops, blocks = records(1, 1)
    write(pf, tmp_path / "p.poses", 1, 1, tops, blocks, {})
    with pf.Reader(str(tmp_path / "p.poses")) as r:
        assert r.top(0) == tops[0]
    assert r.f.closed
    with pytest.raises(RuntimeError):
        with pf.Reader(str(tmp_path / "p.poses")) as r2:
            raise RuntimeError("boom")
    assert r2.f.closed
    assert set(inspect.signature(pf.Reader.__exit__).parameters) == {"self", "exc"}


def test_reader_rejects_other_files(tmp_path):
    bad = tmp_path / "bad.poses"
    bad.write_bytes(b"NOTATLAS" + b"\0" * 120)
    with pytest.raises(ValueError, match="not a PVATLAS1"):
        pf.Reader(str(bad))


def test_reader_prototype_entries(tmp_path):
    """Phase-1 prototype files with 16-byte table-A entries (offset, length, raw_length) are still readable."""
    N, K = 2, 2
    tops = {0: b"top zero\n", 1: b"top one\n" * 3}
    blocks = {0: b"b0\n", 3: b"b3\n" * 4}
    meta = b'{"p":1}' + b" "
    ta = 128 + len(meta)
    tb = ta + N * 16
    sa = tb + (N * K + 1) * 8
    body, A, pos = b"", [], sa
    for i in range(N):
        gz = pf.gz_member(tops[i])
        A.append((pos, len(gz), len(tops[i])))
        body, pos = body + gz, pos + len(gz)
    sb = pos
    B = []
    for j in range(N * K):
        B.append(pos)
        if j in blocks:
            gz = pf.gz_member(blocks[j])
            body, pos = body + gz, pos + len(gz)
    B.append(pos)
    hdr = pf.HEADER.pack(pf.MAGIC, 1, N, K, 128, ta, tb, sa, sb, pos, 128, len(meta), 16)
    data = (hdr + b"\0" * (128 - len(hdr)) + meta + b"".join(struct.pack("<QII", *a) for a in A)
            + struct.pack(f"<{len(B)}Q", *B) + body)
    (tmp_path / "proto.poses").write_bytes(data)
    with pf.Reader(str(tmp_path / "proto.poses")) as r:
        assert r.meta == {"p": 1}
        assert [r.top(i) for i in range(N)] == [tops[0], tops[1]]
        assert r.modes(0, 1) == blocks[0] and r.modes(1, 2) == blocks[3] and r.modes(0, 2) is None


# --- byte compatibility with the Atlas module ------------------------------------------------------------------------

def test_same_bytes_as_atlas_writer(tmp_path, atlas_pf):
    N, K = 6, 5
    tops, blocks = records(N, K, seed=3, missing_top={1}, missing_blocks={2, 7, 29})
    meta = {"format": "PVATLAS1", "protein_id": "X", "K": K}
    order = [4, 0, 5, 2, 3]
    ours = write(pf, tmp_path / "ours.poses", N, K, tops, blocks, meta, order=order)
    theirs = write(atlas_pf, tmp_path / "atlas.poses", N, K, tops, blocks, meta, order=order)
    assert ours == theirs
    assert (tmp_path / "ours.poses").read_bytes() == (tmp_path / "atlas.poses").read_bytes()
    for level in (1, 6, 9):
        assert pf.gz_member(tops[0], level) == atlas_pf.gz_member(tops[0], level)


def test_readers_interchangeable(tmp_path, atlas_pf):
    N, K = 4, 2
    tops, blocks = records(N, K, seed=5, missing_top={3}, missing_blocks={1})
    write(pf, tmp_path / "ours.poses", N, K, tops, blocks, {"by": "pocketvina"})
    write(atlas_pf, tmp_path / "atlas.poses", N, K, tops, blocks, {"by": "atlas"})
    for path in ("ours.poses", "atlas.poses"):
        a = atlas_pf.Reader(str(tmp_path / path))
        with pf.Reader(str(tmp_path / path)) as b:
            assert (a.N, a.K, a.A, a.B, a.meta) == (b.N, b.K, b.A, b.B, b.meta)
            for i in range(N):
                assert a.top(i) == b.top(i) == tops.get(i)
                for k in range(1, K + 1):
                    assert a.modes(i, k) == b.modes(i, k) == blocks.get(i * K + k - 1)
        a.close()
