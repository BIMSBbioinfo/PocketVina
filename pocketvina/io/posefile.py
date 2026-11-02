# Verbatim copy of the PocketVina Atlas module
#   /data/akalin/asarigun/pocketvina_atlas/_work/db_prototype/scripts/posefile.py
# Files written here must stay byte-identical to the Atlas ones, so do not change the format on one side only.
# Only addition: Reader.__enter__/__exit__ (context-manager support).
# In PocketVina's tables and meta, af_id is protein_id and the compound index cidx is ligand_idx.
"""PocketVina Atlas pose file, format PVATLAS1 version 1 (database-plan.md v2, section 3).

One file per protein. All integers little-endian. Offsets are absolute byte positions in the file.

  header   128 bytes   (struct HEADER below)
  meta     JSON (UTF-8), padded with spaces to a multiple of 8 bytes: af_id, uniprot, compound list hash, ...
  table A  N entries x 12 bytes: uint64 offset, uint32 length   (top pose of compound i; v2.1)
           (phase-1 prototype files used 16-byte entries with an extra uint32 raw_length; the header field
           a_entry_size says which, and Reader accepts both)
  table B  N*K + 1 entries x uint64: start of the mode block of (compound i, pocket k) at index j = i*K + (k-1);
           block j = [B[j], B[j+1]); B[N*K] = end of section B
  section A  N gzip members (fewer if poses are missing): the original <AF>-pocketK-<CAT>_cleaned_out.sdf bytes,
             written in AFFINITY order (best compound first), so the top rows of the protein page are adjacent
  section B  N*K gzip members: the original <AF>-pocketK-<CAT>_cleaned_out.pdbqt bytes (20 MODELs), in j order

Missing record: length 0 (table A) or B[j+1] == B[j] (table B).
Why table A stores (offset, length) and not N+1 starts as in the plan's sketch: section A is written in affinity
order but looked up in compound order, so "next start - start" would not be the record length.

Each record is one complete gzip member (RFC 1952, mtime 0, no file name), so
  - a byte range [offset, offset+length) can be served as a .gz download or inflated with DecompressionStream;
  - section A (or B) as a whole is a valid multi-member gzip stream: `gzip -dc` gives the records concatenated.
"""
import io, json, struct, zlib, hashlib

MAGIC = b"PVATLAS1"
VERSION = 1
HEADER_SIZE = 128
# magic, version, N, K, header_size, table_a, table_b, section_a, section_b, file_size, meta_off, meta_len, a_entry_size
HEADER = struct.Struct("<8sIIIIQQQQQQII")
A_ENTRY = struct.Struct("<QI")            # v2.1: offset, length (12 bytes)
A_ENTRY_PROTO = struct.Struct("<QII")     # phase-1 prototype: offset, length, raw_length (16 bytes)
assert HEADER.size <= HEADER_SIZE and A_ENTRY.size == 12


def gz_member(raw: bytes, level: int = 9) -> bytes:
    """One deterministic gzip member holding exactly `raw` (header mtime 0, OS byte 255 = unknown)."""
    c = zlib.compressobj(level, zlib.DEFLATED, 31)
    b = bytearray(c.compress(raw) + c.flush())
    b[9] = 255  # OS field: zlib writes 3 (Unix); 255 keeps the bytes identical across platforms
    return bytes(b)


def gunzip_member(buf: bytes) -> bytes:
    d = zlib.decompressobj(31)
    out = d.decompress(buf) + d.flush()
    if not d.eof or d.unused_data:
        raise ValueError("not exactly one gzip member")
    return out


class Writer:
    """Usage:
        w = Writer(path, N, K, meta)
        for i in affinity_order: w.add_top(i, gz_bytes, raw_len)      # any order, each i at most once
        w.begin_modes()
        for j in range(N*K): w.add_block(j, gz_bytes_or_None)          # strictly increasing j; gaps = missing
        info = w.close()
    """

    def __init__(self, path, N, K, meta: dict):
        self.N, self.K = N, K
        mb = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode()
        mb += b" " * (-len(mb) % 8)
        self.meta = mb
        self.meta_off = HEADER_SIZE
        self.ta = self.meta_off + len(mb)
        self.tb = self.ta + N * A_ENTRY.size
        self.sa = self.tb + (N * K + 1) * 8
        self.f = open(path, "wb", buffering=1 << 22)
        self.f.write(b"\0" * self.sa)  # placeholder for header, meta, tables
        self.pos = self.sa
        self.A = [(0, 0)] * N
        self.B = [0] * (N * K + 1)
        self.next_j = 0
        self.sb = None

    def add_top(self, i, gz, raw_len):
        assert self.sb is None and 0 <= i < self.N and self.A[i][1] == 0
        self.f.write(gz)
        self.A[i] = (self.pos, len(gz))
        self.pos += len(gz)

    def begin_modes(self):
        self.sb = self.pos

    def add_block(self, j, gz):
        assert self.sb is not None and j >= self.next_j and j < self.N * self.K
        for jj in range(self.next_j, j + 1):
            self.B[jj] = self.pos
        if gz:
            self.f.write(gz)
            self.pos += len(gz)
        self.next_j = j + 1

    def close(self):
        if self.sb is None:
            self.begin_modes()
        for jj in range(self.next_j, self.N * self.K + 1):
            self.B[jj] = self.pos
        size = self.pos
        hdr = HEADER.pack(MAGIC, VERSION, self.N, self.K, HEADER_SIZE, self.ta, self.tb, self.sa, self.sb, size,
                          self.meta_off, len(self.meta), A_ENTRY.size)
        self.f.seek(0)
        self.f.write(hdr + b"\0" * (HEADER_SIZE - len(hdr)))
        self.f.write(self.meta)
        self.f.write(b"".join(A_ENTRY.pack(*e) for e in self.A))
        self.f.write(struct.pack(f"<{len(self.B)}Q", *self.B))
        self.f.close()
        return dict(file_size=size, header_bytes=self.sa, section_a_bytes=self.sb - self.sa, section_b_bytes=size - self.sb)


class Reader:
    """Independent reader: needs only the file (header + tables), no Parquet."""

    def __init__(self, path):
        self.f = open(path, "rb")
        h = HEADER.unpack(self.f.read(HEADER.size))
        (magic, self.version, self.N, self.K, hs, self.ta, self.tb, self.sa, self.sb, self.size,
         mo, ml, aes) = h
        if magic != MAGIC or self.version != VERSION or aes not in (A_ENTRY.size, A_ENTRY_PROTO.size):
            raise ValueError(f"not a PVATLAS1 v1 file: {path}")
        st = A_ENTRY if aes == A_ENTRY.size else A_ENTRY_PROTO
        self.f.seek(mo)
        self.meta = json.loads(self.f.read(ml))
        self.f.seek(self.ta)
        a = self.f.read(self.N * aes)
        self.A = [st.unpack_from(a, aes * i)[:2] for i in range(self.N)]
        self.f.seek(self.tb)
        self.B = struct.unpack(f"<{self.N * self.K + 1}Q", self.f.read((self.N * self.K + 1) * 8))

    def _read(self, off, ln):
        self.f.seek(off)
        return self.f.read(ln)

    def top_range(self, i):
        return self.A[i]

    def top(self, i):
        off, ln = self.A[i]
        return gunzip_member(self._read(off, ln)) if ln else None

    def block_range(self, i, k):
        j = i * self.K + (k - 1)
        return self.B[j], self.B[j + 1] - self.B[j]

    def modes(self, i, k):
        off, ln = self.block_range(i, k)
        return gunzip_member(self._read(off, ln)) if ln else None

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()
