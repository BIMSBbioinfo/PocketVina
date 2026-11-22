"""``pocketvina.Atlas``, ``pocketvina.io.atlas_source`` and ``pocketvina atlas`` on the fixture release
``tests/data/atlas/v1.1``.

The fixture is read from a folder and over HTTP (the ``atlas_server`` fixture of conftest, which answers Range
requests). Only identifiers of the fixture are written here (UniProt accessions and PVL IDs); drug names, SMILES and
the other contents come from the data at run time, and the checks of text that holds them (search results, command
output) are written so that a failure does not print it.

One test is marked ``network``: it reads the public Atlas over the internet and is skipped unless
``POCKETVINA_NETWORK_TESTS=1``.
"""
import gzip
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from pocketvina import cli, selection, workspace
from pocketvina.atlas import ATLAS_URL, BEST_SCORES_COLUMNS, SEARCH_COLUMNS, SEARCH_TOP, Atlas, _Drugs
from pocketvina.io import atlas_source, posefile, sas
from pocketvina.io.atlas_source import Source, pair_blocks, pose_records, top_record

ABL1 = "P00519"
EGFR = "P00533"
PTGS2 = "P35354"
PROTEINS = (ABL1, EGFR, PTGS2)
DRUG = "PVL00129"        # the drug of the documentation examples (a row of every fixture protein)
REDOCK = "PVL02203"      # its row is a desalted re-docking: pf 1, no stored PoseBusters result
NO_TOP = "PVL02131"      # a row of EGFR without a stored top pose (top_len 0), with mode blocks
NOT_DRUG = "PVL00958"    # a row whose entry was not docked as the drug
AMBIGUOUS = ("PVL00257", "PVL02156")  # two drug pages that share identifiers


@pytest.fixture
def atlas(atlas_dir) -> Atlas:
    return Atlas(atlas_dir)


@pytest.fixture
def local(atlas_dir) -> Source:
    return Source(atlas_dir)


@pytest.fixture
def remote(atlas_server) -> Source:
    return Source(atlas_server.url)


def test_fixture_is_small_and_listed(atlas_dir):
    """The fixture stays under 3 MB, and its MANIFEST.json lists every other file with its size and sha256, but sas/
    and chembl_any/ (as the release's MANIFEST.json): sas/MANIFEST.json and chembl_any/MANIFEST.json list those."""
    files = {p.relative_to(atlas_dir).as_posix(): p for p in atlas_dir.rglob("*") if p.is_file()}
    assert sum(p.stat().st_size for p in files.values()) < 3 * 1024 * 1024
    listed = json.loads((atlas_dir / "MANIFEST.json").read_text())["files"]
    sas_files = {rel for rel in files if rel.startswith("sas/")}
    chembl_files = {rel for rel in files if rel.startswith("chembl_any/")}
    assert set(listed) == set(files) - {"MANIFEST.json"} - sas_files - chembl_files
    chembl_listed = json.loads((atlas_dir / "chembl_any" / "MANIFEST.json").read_text())
    assert {f"chembl_any/{rel}" for rel in chembl_listed} == chembl_files - {"chembl_any/MANIFEST.json"} == {
        f"chembl_any/proteins/{uniprot}.json.gz" for uniprot in PROTEINS} | {"chembl_any/strict.json.gz"}
    for rel, entry in chembl_listed.items():
        data = files[f"chembl_any/{rel}"].read_bytes()
        assert entry == [hashlib.sha256(data).hexdigest(), len(data)], rel
    for rel, entry in listed.items():
        data = files[rel].read_bytes()
        assert entry == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, rel
    sas_listed = json.loads((atlas_dir / "sas" / "MANIFEST.json").read_text())
    assert {f"sas/{name}" for name in sas_listed} == sas_files - {"sas/MANIFEST.json"} == {
        f"sas/{uniprot}.json.gz" for uniprot in PROTEINS}
    for name, entry in sas_listed.items():
        data = files[f"sas/{name}"].read_bytes()
        assert entry == [hashlib.sha256(data).hexdigest(), len(data)], name
    meta = json.loads((atlas_dir / "meta.json").read_text())
    assert meta["subset"] and meta["release"]["status"] == "final"


# --- Source --------------------------------------------------------------------------------------------------------

def test_source_location(atlas_dir, atlas_server):
    assert Source(atlas_dir).location == str(atlas_dir)
    assert Source(atlas_server.url.rstrip("/")).location == atlas_server.url  # a URL without the final "/"
    assert repr(Source(atlas_dir)).startswith("Source(")


@pytest.mark.parametrize("source", ["local", "remote"])
def test_source_json(source, request, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    src = request.getfixturevalue(source)
    assert src.meta["release"]["label"] == "v1.1"
    assert src.json("meta.json") is src.meta
    protein = src.json(f"proteins/{ABL1}.json")  # the release has proteins/<U>.json.gz only
    assert protein["uniprot"] == ABL1
    assert src.json(f"proteins/{ABL1}.json.gz")["uniprot"] == ABL1  # the .gz name works too
    assert set(src.json("ligand_ids.json")) == {"scheme", "status", "ids"}  # a plain JSON file
    assert src.json("search/index.json")["proteins_columns"][0] == "uniprot"


@pytest.mark.parametrize("source", ["local", "remote"])
def test_source_range(source, request, atlas_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    src = request.getfixturevalue(source)
    assert src.range(f"poses/{ABL1}.poses", 0, 8) == posefile.MAGIC
    assert src.range(f"poses/{ABL1}.poses", 0, 0) == b""
    with pytest.raises(ValueError, match="offset >= 0"):
        src.range(f"poses/{ABL1}.poses", -1, 4)
    with pytest.raises(ValueError, match="shorter than"):
        src.range("meta.json", 0, 10_000_000)
    end = (atlas_dir / "poses" / f"{ABL1}.poses").stat().st_size
    with pytest.raises(ValueError, match="shorter than"):  # over HTTP: status 416
        src.range(f"poses/{ABL1}.poses", end, 8)


def test_source_range_without_range_support(atlas_server, tmp_path, monkeypatch):
    """A server that ignores Range and sends the whole file (status 200) is sliced locally."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    atlas_server.ranges = False
    src = Source(atlas_server.url)
    assert src.range(f"poses/{ABL1}.poses", 0, 8) == posefile.MAGIC
    assert atlas_server.requests[-1][1] == "bytes=0-7"  # it did ask for a range


def test_source_file_local(local, atlas_dir):
    assert local.file("meta.json") == str(atlas_dir / "meta.json")
    with pytest.raises(FileNotFoundError, match="no file proteins/XXXXXX.json"):
        local.file("proteins/XXXXXX.json")


def test_source_file_checked(atlas_dir, atlas_server, tmp_path, monkeypatch):
    """Source.file(size=, sha256=): a file of a folder that differs raises ValueError; over HTTP a cached copy that
    differs is fetched again, and a fetched one that differs is not kept."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    rel = f"sas/{ABL1}.json.gz"
    data = (atlas_dir / rel).read_bytes()
    right = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    assert Source(atlas_dir).file(rel, **right) == str(atlas_dir / rel)
    with pytest.raises(ValueError, match="is damaged"):
        Source(atlas_dir).file(rel, size=len(data), sha256="0" * 64)
    remote = Source(atlas_server.url)
    path = Path(remote.file(rel, **right))
    path.write_bytes(b"x" + data[1:])  # damaged in the cache
    asked = len(atlas_server.requests)
    assert remote.file(rel, **right) == str(path) and path.read_bytes() == data
    assert [r for r, _ in atlas_server.requests[asked:]] == [rel]  # fetched again
    with pytest.raises(ValueError, match="is damaged"):
        remote.file(f"sas/{EGFR}.json.gz", sha256="0" * 64)
    assert not path.with_name(f"{EGFR}.json.gz").exists() and not list(tmp_path.glob("**/*.tmp-*"))


def test_source_file_http_is_cached(atlas_server, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    src = Source(atlas_server.url)
    path = Path(src.file("tables/compounds.parquet"))
    cache = tmp_path / "pocketvina" / "atlas"
    assert path.parent.parent.parent == cache and path.parts[-2:] == ("tables", "compounds.parquet")
    # the fixture is a subset of the release (meta.json "subset"), so it gets a folder of its own
    assert path.parent.parent.name.startswith("v1.1-subset-")
    assert path.is_file()
    asked = [rel for rel, _ in atlas_server.requests]
    assert Path(src.file("tables/compounds.parquet")) == path  # the cached copy is used
    assert [rel for rel, _ in atlas_server.requests] == asked
    assert not list(tmp_path.glob("**/*.tmp-*"))


def test_source_http_errors(atlas_server, closed_port, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(atlas_source, "RETRY_DELAYS", (0, 0, 0))  # the retries, without the waits
    src = Source(atlas_server.url)
    with pytest.raises(FileNotFoundError, match="proteins/XXXXXX.json.gz"):
        src.json("proteins/XXXXXX.json")
    with pytest.raises(ConnectionError, match="cannot reach the PocketVina Atlas at .*Atlas\\('/path/to/release'\\)"):
        assert Source(closed_port).meta


def test_source_http_retries_a_dropped_connection(atlas_server, atlas_dir, tmp_path, monkeypatch):
    """An answer cut short by a closed connection is fetched again (a byte range and a whole file)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(atlas_source, "RETRY_DELAYS", (0, 0, 0))
    src = Source(atlas_server.url)
    assert src.meta
    poses = (atlas_dir / "poses" / f"{ABL1}.poses").read_bytes()
    atlas_server.cut = 1
    assert src.range(f"poses/{ABL1}.poses", 0, 4096) == poses[:4096]
    structure = next((atlas_dir / "structures").iterdir())
    atlas_server.cut = 1
    assert Path(src.file(f"structures/{structure.name}")).read_bytes() == structure.read_bytes()
    assert atlas_server.cut == 0 and len(atlas_server.requests) == 5  # meta.json, then each file twice


def test_source_http_reuses_connections(atlas_server, tmp_path, monkeypatch):
    """Requests share a few open connections (not one each), also from several threads."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    atlas = Atlas(atlas_server.url)
    atlas.protein("ABL1")
    atlas.best_scores(ligand=DRUG)
    atlas.select(ABL1, DRUG, pocket="all")
    atlas.show(ABL1, path=tmp_path / "page.html")
    assert len(atlas_server.requests) > 10
    assert atlas_server.connections <= atlas_source.MAX_IDLE < len(atlas_server.requests)


def test_source_http_kept_connection_closed_by_the_server(atlas_server, atlas_dir, tmp_path, monkeypatch):
    """A kept connection that the server has closed in the meantime: the request goes again on a new one, at once
    (not as a retry after a wait)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(atlas_source.time, "sleep", lambda s: pytest.fail("a retry after a wait"))
    src = Source(atlas_server.url)
    atlas_server.hang_up = 1
    assert src.meta  # answered, then the server closes the connection without saying so
    poses = (atlas_dir / "poses" / f"{ABL1}.poses").read_bytes()
    assert src.range(f"poses/{ABL1}.poses", 64, 64) == poses[64:128]
    assert atlas_server.connections == 2


def test_source_http_follows_a_redirect(atlas_server, atlas_dir, tmp_path, monkeypatch):
    """A permanent redirect is followed, and later requests go to the new place directly."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    src = Source(atlas_server.url + "moved/")
    assert src.meta["release"]["label"] == "v1.1"
    poses = (atlas_dir / "poses" / f"{ABL1}.poses").read_bytes()
    assert src.range(f"poses/{ABL1}.poses", 8, 8) == poses[8:16]
    assert atlas_server.requests == [("moved/meta.json", None), ("meta.json", None),
                                     (f"poses/{ABL1}.poses", "bytes=8-15")]
    assert src.location == atlas_server.url + "moved/"  # the release is still named by the URL it was given


def test_source_http_retries_a_server_error(atlas_server, atlas_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(atlas_source, "RETRY_DELAYS", (0, 0, 0))
    src = Source(atlas_server.url)
    assert src.meta
    poses = (atlas_dir / "poses" / f"{ABL1}.poses").read_bytes()
    atlas_server.fail = 2
    assert src.range(f"poses/{ABL1}.poses", 8, 8) == poses[8:16]
    assert len(atlas_server.requests) == 4  # meta.json, then the range three times
    atlas_server.fail = 4
    with pytest.raises(OSError, match="the server answered HTTP 503"):
        src.range(f"poses/{ABL1}.poses", 16, 8)


def test_source_http_errors_through_a_proxy(atlas_server, atlas_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for name in ("no_proxy", "NO_PROXY", "HTTP_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("http_proxy", atlas_server.url)
    src = Source("http://pocketvina-atlas.invalid/")
    with pytest.raises(FileNotFoundError, match="proteins/XXXXXX.json.gz"):
        src.json("proteins/XXXXXX.json")
    end = (atlas_dir / "poses" / f"{ABL1}.poses").stat().st_size
    with pytest.raises(ValueError, match="shorter than"):  # status 416
        src.range(f"poses/{ABL1}.poses", end, 8)


def test_source_http_error_keeps_the_ssl_detail(monkeypatch):
    """A certificate error says why (e.g. a self-signed certificate)."""
    import ssl

    def refuse(url, headers, read, moved=None):
        raise ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
                                              "self-signed certificate")

    monkeypatch.setattr(atlas_source, "_fetch", refuse)
    monkeypatch.setattr(atlas_source, "RETRY_DELAYS", (0, 0, 0))
    with pytest.raises(ConnectionError, match="certificate verify failed: self-signed certificate"):
        assert Source("https://pocketvina-atlas.invalid/").meta


def test_source_https_uses_urllibs_certificate_rule(monkeypatch):
    """HTTPS connections use the context urllib would (ssl._create_default_https_context, which some set to their
    own); after a certificate error the next connection builds it again, so a fix made since then is used."""
    import ssl

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    monkeypatch.setattr(ssl, "_create_default_https_context", lambda: context)
    monkeypatch.setattr(atlas_source, "_ssl_context", None)
    key = ("https", "pocketvina-atlas.invalid", None, os.getpid())
    conn, reused = atlas_source._connection(key, new=True)
    assert conn._context is context and not reused

    def refuse(*args, **kwargs):
        raise ssl.SSLCertVerificationError(1, "certificate verify failed")

    monkeypatch.setattr(atlas_source.http.client.HTTPSConnection, "request", refuse)
    with pytest.raises(ssl.SSLError):
        atlas_source._fetch("https://pocketvina-atlas.invalid/meta.json", {}, lambda r: None)
    fixed = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    monkeypatch.setattr(ssl, "_create_default_https_context", lambda: fixed)
    assert atlas_source._connection(key, new=True)[0]._context is fixed


def test_source_http_drops_connections_idle_too_long(atlas_server, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    src = Source(atlas_server.url)
    assert src.meta
    src.range(f"poses/{ABL1}.poses", 0, 8)
    assert atlas_server.connections == 1  # kept and reused
    monkeypatch.setattr(atlas_source, "IDLE_TIMEOUT", -1)
    src.range(f"poses/{ABL1}.poses", 8, 8)
    assert atlas_server.connections == 2


def test_source_range_cache_is_bounded(atlas_server, atlas_dir, tmp_path, monkeypatch):
    """The ranges kept in memory are at most RANGE_CACHE bytes: the oldest go first."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(atlas_source, "RANGE_CACHE", 1000)
    src = Source(atlas_server.url)
    rel = f"poses/{ABL1}.poses"
    poses = (atlas_dir / rel).read_bytes()
    for offset in (0, 400, 800):
        assert src.range(rel, offset, 250) == poses[offset:offset + 250]  # each 250 bytes: at most 4 kept
    asked = len(atlas_server.requests)
    assert src.range(rel, 800, 250) == poses[800:1050] and len(atlas_server.requests) == asked  # kept
    for offset in (1200, 1600):
        src.range(rel, offset, 250)
    asked = len(atlas_server.requests)
    assert src.range(rel, 0, 250) == poses[:250] and len(atlas_server.requests) == asked + 1  # gone: asked again
    assert src._kept_bytes <= 1000
    for _ in range(2):  # a range above a quarter of RANGE_CACHE is not kept
        assert src.range(rel, 2000, 300) == poses[2000:2300]
    assert len(atlas_server.requests) == asked + 3


def test_source_and_selection_pickle(atlas_server, atlas_dir, tmp_path, monkeypatch):
    """An Atlas, its Source and a Selection can be pickled and copied (e.g. for a process pool), locally and over
    HTTP."""
    import copy
    import pickle

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for location in (atlas_dir, atlas_server.url):
        atlas = Atlas(location)
        pair = atlas.select(ABL1, DRUG)
        again = pickle.loads(pickle.dumps(pair))
        assert again.poses.equals(pair.poses) and copy.deepcopy(atlas).source.location == atlas.source.location
        assert again._source.atlas.source.range(f"poses/{ABL1}.poses", 0, 8) == posefile.MAGIC


def test_source_http_through_a_proxy(atlas_server, atlas_dir, tmp_path, monkeypatch):
    """With a proxy set (http_proxy), requests go through it (urllib reads the proxy settings)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for name in ("no_proxy", "NO_PROXY", "HTTP_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("http_proxy", atlas_server.url)  # the test server answers a proxy request for any host
    src = Source("http://pocketvina-atlas.invalid/")
    assert src.meta["release"]["label"] == "v1.1"
    poses = (atlas_dir / "poses" / f"{ABL1}.poses").read_bytes()
    assert src.range(f"poses/{ABL1}.poses", 8, 8) == poses[8:16]
    assert [rel for rel, _ in atlas_server.requests] == ["meta.json", f"poses/{ABL1}.poses"]


@pytest.mark.parametrize("source", ["local", "remote"])
def test_source_ranges(source, request, atlas_dir, tmp_path, monkeypatch):
    """Source.ranges gives each range as Source.range does; ranges close together come in one request, far apart
    in one each; a range read over HTTP is not asked for again (nor one inside it)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    src = request.getfixturevalue(source)
    rel = f"poses/{ABL1}.poses"
    poses = (atlas_dir / rel).read_bytes()
    far = atlas_source.MERGE_GAP + 1000
    asks = [(rel, 100, 10), (rel, 0, 8), (rel, 50, 0), (rel, 104, 30), (rel, far, 16), ("meta.json", 2, 5)]
    expected = [poses[100:110], poses[:8], b"", poses[104:134], poses[far:far + 16],
                (atlas_dir / "meta.json").read_bytes()[2:7]]
    assert src.ranges(asks) == expected
    if source == "remote":
        server = request.getfixturevalue("atlas_server")
        assert sorted(h for rel_, h in server.requests if h) == ["bytes=0-133", "bytes=2-6",
                                                                  f"bytes={far}-{far + 15}"]
        before = len(server.requests)
        assert src.range(rel, 10, 20) == poses[10:30] and src.ranges(asks) == expected
        assert len(server.requests) == before
    with pytest.raises(ValueError, match="offset >= 0"):
        src.ranges([(rel, -1, 4)])


def test_source_refuses_paths_outside_the_release(local):
    for path in ("../meta.json", "proteins/../../meta.json", ""):
        with pytest.raises(ValueError, match="not a file of a PocketVina Atlas release"):
            local.json(path)


# --- pose files ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ligand", [DRUG, REDOCK, NO_TOP])
@pytest.mark.parametrize("source", ["local", "remote"])
def test_pose_helpers_agree_with_the_reader(ligand, source, request, atlas_dir, tmp_path, monkeypatch):
    """pair_blocks / top_record give what posefile.Reader gives, for a pf 0 and a pf 1 row, locally and over HTTP."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    src = request.getfixturevalue(source)
    atlas = Atlas(atlas_dir)
    protein = EGFR if ligand == NO_TOP else ABL1
    pair = atlas._pair(protein, ligand)
    with posefile.Reader(atlas_dir / pair.ref["file"]) as reader:
        expected = [reader.modes(pair.index, k) for k in range(1, reader.K + 1)]
        top = reader.top(pair.index)
    assert pair_blocks(src, pair.ref, pair.index) == expected
    assert top_record(src, pair.ref["file"], pair.row["top_off"], pair.row["top_len"]) == top
    assert (top is None) == (ligand == NO_TOP)
    record = (pair.ref, pair.index, pair.row["top_off"], pair.row["top_len"])
    assert pose_records(src, [record, record]) == [(expected, top)] * 2


def test_pair_blocks_of_pockets_without_output(tmp_path):
    """A pocket without engine output gives None, a record without any output K times None (a small pose file)."""
    import gzip

    (tmp_path / "poses").mkdir()
    writer = posefile.Writer(str(tmp_path / "poses" / "X.poses"), 2, 3, {})
    writer.begin_modes()
    writer.add_block(0, gzip.compress(b"MODEL 1\n", mtime=0))
    writer.add_block(2, gzip.compress(b"MODEL 3\n", mtime=0))
    writer.close()
    with posefile.Reader(tmp_path / "poses" / "X.poses") as reader:
        ref = {"file": "poses/X.poses", "N": 2, "K": 3, "table_b_offset": reader.tb}
    src = Source(tmp_path)
    assert pair_blocks(src, ref, 0) == [b"MODEL 1\n", None, b"MODEL 3\n"]
    assert pair_blocks(src, ref, 1) == [None, None, None]
    assert top_record(src, ref["file"], 0, 0) is None


def test_pair_blocks_checks_the_record_number(local, atlas):
    pair = atlas._pair(ABL1, DRUG)
    with pytest.raises(IndexError, match="no record"):
        pair_blocks(local, pair.ref, pair.ref["N"])


# --- the Atlas object ----------------------------------------------------------------------------------------------

def test_atlas_repr(atlas, atlas_dir):
    assert "PocketVina Atlas v1.1" in repr(atlas) and str(atlas_dir) in repr(atlas)
    assert "proteins" in repr(atlas) and "drugs" in repr(atlas)
    assert "<div>" in atlas._repr_html_()
    assert atlas.release == "v1.1"


def test_atlas_source_choice(atlas_dir, monkeypatch):
    monkeypatch.setenv("POCKETVINA_ATLAS", str(atlas_dir))
    assert Atlas().source.location == str(atlas_dir)
    assert Atlas(atlas_dir).source.location == str(atlas_dir)
    monkeypatch.delenv("POCKETVINA_ATLAS")
    assert Atlas.__init__.__defaults__ == (None,) and ATLAS_URL.endswith("/v1.1/")


def test_atlas_needs_a_final_release_folder(tmp_path, atlas_dir):
    with pytest.raises(FileNotFoundError, match="is not a PocketVina Atlas release.*meta.json"):
        Atlas(tmp_path)
    for text in (json.dumps({"version": "v1.1"}), "<html>not a release</html>"):
        (tmp_path / "meta.json").write_text(text)
        with pytest.raises(ValueError, match="no release information.*path/to/release"):
            Atlas(tmp_path)
    meta = json.loads((atlas_dir / "meta.json").read_text())
    meta["release"]["status"] = "draft"
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="not a final PocketVina Atlas release.*'draft'"):
        Atlas(tmp_path)


# --- identifiers ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("identifier", [ABL1, "abl1", "ABL1", "ABL1_HUMAN", "AF-P00519-F1-model_v6",
                                        "ENSG00000097007", "25", 25, "HGNC:76"])
def test_protein_identifiers(atlas, identifier):
    assert atlas._uniprot(identifier) == ABL1


@pytest.mark.parametrize("uniprot", PROTEINS)
def test_protein_alphafold_model_id(atlas, uniprot):
    """An AlphaFold model ID finds its protein whatever the protein's gene starts with."""
    assert atlas._uniprot(f"AF-{uniprot}-F1-model_v6") == uniprot


def test_protein_gene_like_an_accession_and_several_hgnc_ids(atlas_dir, tmp_path):
    """A gene written like a UniProt accession (histone genes such as H2BC12) is filed under its first letter, not in
    an accession file; a protein with several HGNC IDs ("HGNC:1;HGNC:2") is found by each (a made-up release)."""
    import gzip

    import pyarrow as pa
    import pyarrow.parquet as pq

    (tmp_path / "meta.json").write_bytes((atlas_dir / "meta.json").read_bytes())
    index = Source(atlas_dir).json("search/index.json")
    row = dict.fromkeys(index["proteins_columns"], "")
    row.update(uniprot="Q00001", gene="H2BC12", entry_page=0)
    (tmp_path / "search").mkdir()
    for name, data in (("index", dict(index, keys={"h": {}})), ("h", {"p": [list(row.values())], "d": []})):
        (tmp_path / "search" / f"{name}.json.gz").write_bytes(gzip.compress(json.dumps(data).encode()))
    (tmp_path / "tables").mkdir()
    pq.write_table(pa.table({"uniprot": ["Q00001"], "hgnc": ["HGNC:900001;HGNC:900002"]}),
                   tmp_path / "tables" / "proteins.parquet")
    atlas = Atlas(tmp_path)
    for identifier in ("H2BC12", "h2bc12", "HGNC:900001", "HGNC:900002"):
        assert atlas._uniprot(identifier) == "Q00001"


def test_protein_identifier_errors(atlas):
    with pytest.raises(KeyError, match="no protein 'NOSUCHGENE'"):
        atlas._uniprot("NOSUCHGENE")
    with pytest.raises(TypeError, match="protein must be an identifier"):
        atlas._uniprot(["ABL1"])
    with pytest.raises(ValueError, match="protein is empty"):
        atlas._uniprot("  ")


def test_protein_ambiguity_lists_the_candidates(atlas, atlas_dir):
    """A gene synonym of several proteins: the error names them all (the fixture keeps those search rows)."""
    index = Source(atlas_dir).json("search/index.json")
    columns = {name: i for i, name in enumerate(index["proteins_columns"])}
    rows = {}
    for key in index["keys"]:
        for row in Source(atlas_dir).json(f"search/{key}.json").get("p") or []:
            rows[row[columns["uniprot"]]] = row
    shared = {}
    for uniprot, row in rows.items():
        for synonym in row[columns["synonyms"]].split():
            shared.setdefault(synonym.casefold(), set()).add(uniprot)
    ambiguous = {s: u for s, u in shared.items() if len(u) > 1 and ABL1 in u}
    assert ambiguous, "the fixture should keep a gene synonym shared by several proteins"
    synonym, candidates = sorted(ambiguous.items())[0]
    with pytest.raises(ValueError) as error:
        atlas._uniprot(synonym)
    assert all(uniprot in str(error.value) for uniprot in candidates)
    assert "UniProt accessions" in str(error.value)


def test_drug_identifiers(atlas):
    drug = atlas.drug(DRUG)
    identifiers = [DRUG, DRUG.lower(), drug.name, drug.name.upper(), drug.chembl_id, drug.inchikey,
                   drug.inchikey[:14], drug.parent_smiles, drug.library_smiles, drug.docked_smiles,
                   *drug.catalogue_numbers, *drug.synonyms[:1]]
    for identifier in identifiers:
        assert atlas._pvl(identifier).target == DRUG, "identifier type not resolved"
    assert atlas._pvl(drug.catalogue_numbers[0]).entry == drug.catalogue_numbers[0]


def test_drug_identifier_smiles_is_canonicalised(atlas):
    """A SMILES of the same molecule written differently (kekulised) still finds the drug."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(atlas.drug(DRUG).parent_smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    other = Chem.MolToSmiles(mol, kekuleSmiles=True)
    assert other != atlas.drug(DRUG).parent_smiles
    assert atlas._pvl(other).target == DRUG


def test_drug_identifier_desalted_entry(atlas):
    cats = [c for c in atlas.drug(REDOCK).catalogue_numbers if c.endswith("-desalted")]
    assert cats, "the fixture should have a desalted entry"
    assert atlas._pvl(cats[0])[:3] == (0, REDOCK, cats[0])  # tier, drug page, the catalogue number named
    assert atlas.drug(cats[0]).catalogue_numbers[0] == cats[0]  # the entry named comes first


def test_drug_chembl_id_prefers_entries_docked_as_the_drug():
    """A ChEMBL ID finds the page of its entries docked as the drug, not the page of a salt entry docked as its
    counterion (the site's rule); with no entry docked as the drug, the pages its entries land on."""
    def table(columns, **values):
        return pd.DataFrame({column: values.get(column, [None] * 3) for column in columns})

    pvls = ["PVL00001", "PVL00002", "PVL00003"]
    drugs = _Drugs(table(_Drugs.ENTRY_COLUMNS, catalog_id=["C1", "C2", "C3"], pvl_id=pvls,
                         docked_as=["drug", "salt_counterion", "salt_counterion"]),
                   table(_Drugs.ANNOTATED_COLUMNS, catalog_id=["C1", "C2", "C3"],
                         chembl_id=["CHEMBL1", "CHEMBL1", "CHEMBL2"]),
                   table(_Drugs.PAGE_COLUMNS, pvl_id=pvls, has_results=[True] * 3))
    assert drugs.resolve("chembl1").target == "PVL00001"
    assert drugs.resolve("CHEMBL2").target == "PVL00003"


def test_drug_identifier_errors(atlas):
    with pytest.raises(KeyError, match="no drug 'nosuchdrug'"):
        atlas._pvl("nosuchdrug")
    with pytest.raises(TypeError, match="ligand must be an identifier"):
        atlas._pvl(3.5)


def test_drug_ambiguity_lists_the_candidates(atlas):
    first, second = (atlas.drug(pvl) for pvl in AMBIGUOUS)
    shared = sorted(set(first.synonyms) & set(second.synonyms))
    assert shared, "the fixture should have two drugs that share a synonym"
    with pytest.raises(ValueError) as error:
        atlas._pvl(shared[0])
    assert all(pvl in str(error.value) for pvl in AMBIGUOUS)
    assert "PVL IDs" in str(error.value)


def test_documentation_examples(atlas):
    """The examples of the docstrings (a gene and a drug name) work on the fixture."""
    assert atlas.protein("ABL1").uniprot == ABL1
    assert atlas.drug("imatinib").pvl == DRUG
    assert len(atlas.best_scores(protein="ABL1", top=20)) == 8
    assert set(atlas.best_scores(ligand="imatinib", top=20)["protein_id"]) == set(PROTEINS)


# --- protein() and drug() --------------------------------------------------------------------------------------

def test_protein_information(atlas):
    info = atlas.protein("ABL1")
    assert (info.uniprot, info.gene) == (ABL1, "ABL1")
    assert info.entry_name == "ABL1_HUMAN" and info.af_id.startswith(f"AF-{ABL1}-F1")
    assert info.uniprot_url.endswith(ABL1) and ABL1 in info.alphafold_url
    assert 0 < info.mean_plddt <= 100 and isinstance(info.protein_class, str)
    assert list(info.pockets.columns) == ["pocket", "score", "probability", "plddt", "residues"]
    assert list(info.pockets["pocket"]) == [1, 2, 3, 4, 5]
    assert list(info.diseases.columns) == ["name", "id", "score"] and len(info.diseases) > 0
    assert info.open_targets_url.startswith("https://platform.opentargets.org/target/ENSG")
    assert ABL1 in repr(info) and "pocket" in repr(info)
    assert "<div>" in info._repr_html_() and "UniProt" in info._repr_html_()


def test_protein_information_html_is_escaped(atlas, monkeypatch):
    data = dict(atlas._protein_json(ABL1), protein_name="<script>x</script>")
    monkeypatch.setattr(atlas, "_protein_json", lambda _: data)
    assert "<script>" not in atlas.protein(ABL1)._repr_html_()


def test_drug_information(atlas):
    drug = atlas.drug(DRUG)
    assert drug.pvl == DRUG and drug.name
    assert drug.chembl_id and drug.chembl_url.endswith(drug.chembl_id)
    assert drug.catalogue_numbers and drug.inchikey.count("-") == 2
    assert drug.parent_smiles and drug.library_smiles and drug.docked_smiles
    assert drug.synonyms and drug.atc and drug.docked_as == "drug" and drug.note is None
    assert DRUG in repr(drug) and "InChIKey" in repr(drug)
    assert "<table>" in drug._repr_html_() and drug.chembl_id in drug._repr_html_()


def test_drug_information_of_an_entry_that_is_not_the_drug(atlas):
    drug = atlas.drug(NOT_DRUG)
    assert drug.pvl == NOT_DRUG and drug.docked_as != "drug" and drug.note
    assert "docked as" in repr(drug)


def test_drug_information_of_an_entry_on_its_drugs_page(atlas_dir, tmp_path):
    """A catalogue number docked as something else than its drug lands on its drug's page, with its own identifiers
    (its record is on the page of the structure docked). In the fixture its drug has no page: a copy gets one."""
    import gzip
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq

    own = Atlas(atlas_dir).drug(NOT_DRUG)
    cat = own.catalogue_numbers[0]
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    entries = pq.read_table(release / "tables" / "compound_entries.parquet").to_pandas().set_index("catalog_id")
    parent = entries.at[cat, "parent_pvl_id"]
    pages = pq.read_table(release / "tables" / "compounds.parquet")
    page = dict.fromkeys(pages.column_names, [None])
    page.update(pvl_id=[parent], name=["its drug"], docked_as=["drug"], has_results=[True])
    pq.write_table(pa.concat_tables([pages, pa.table(page, schema=pages.schema)]),
                   release / "tables" / "compounds.parquet")
    data = {"pvl": parent, "name": "its drug", "docked_as": "drug", "entries": [], "docked_instead_of_drug": []}
    (release / "compounds" / f"{parent}.json.gz").write_bytes(gzip.compress(json.dumps(data).encode()))
    drug = Atlas(release).drug(cat)
    assert drug.pvl == parent and drug.catalogue_numbers[0] == cat
    assert (drug.docked_as, drug.library_smiles, drug.docked_smiles) == (own.docked_as, own.library_smiles,
                                                                        own.docked_smiles)
    assert drug.docked_as != "drug" and drug.note


# --- best_scores -------------------------------------------------------------------------------------------------

def test_best_scores_of_a_protein(atlas):
    scores = atlas.best_scores(protein="ABL1")
    assert list(scores.columns) == list(BEST_SCORES_COLUMNS)
    assert len(scores) == len(atlas._protein_json(ABL1)["rows"]) == 8
    assert list(scores["vina"]) == sorted(scores["vina"])
    assert set(scores["protein_id"]) == {ABL1} and set(scores["gene"]) == {"ABL1"}
    assert DRUG in set(scores["ligand_id"]) and scores["name"].notna().all()
    assert scores["best_pocket"].between(1, 5).all() and (scores["best_mode"] >= 1).all()
    assert (scores["ligand_efficiency"] > 0).all()
    assert list(atlas.best_scores(protein=ABL1, top=3)["ligand_id"]) == list(scores["ligand_id"][:3])
    assert len(atlas.best_scores(protein=ABL1, top=0)) == 0


def test_best_scores_of_a_drug(atlas):
    scores = atlas.best_scores(ligand=DRUG)
    assert list(scores.columns) == list(BEST_SCORES_COLUMNS)
    assert set(scores["protein_id"]) == set(PROTEINS) and set(scores["ligand_id"]) == {DRUG}
    assert list(scores["vina"]) == sorted(scores["vina"])
    assert scores["gene"].notna().all() and scores["name"].notna().all()
    assert len(atlas.best_scores(ligand=DRUG, top=2)) == 2


def test_best_scores_arguments(atlas):
    for kwargs in ({}, {"protein": ABL1, "ligand": DRUG}):
        with pytest.raises(TypeError, match="exactly one of protein= and ligand="):
            atlas.best_scores(**kwargs)
    with pytest.raises(ValueError, match="top must be a number of rows"):
        atlas.best_scores(protein=ABL1, top=-1)


# --- one pair ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("ligand,pf", [(DRUG, 0), (REDOCK, 1)])
def test_pair(atlas, ligand, pf):
    pair = atlas._pair(ABL1, ligand)
    assert pair.protein["uniprot"] == ABL1 and pair.row["pvl"] == ligand
    assert pair.row["pf"] == pf and pair.ref["file"].endswith(f"{ABL1}.poses")
    assert pair.ref["file"].startswith("poses_redock/" if pf else "poses/")
    assert pair.index == pair.row["cidx"] - (pair.ref.get("cidx_offset") or 0)
    assert 0 <= pair.index < pair.ref["N"]


def test_pair_without_a_row(atlas):
    with pytest.raises(ValueError, match=f"no result for {AMBIGUOUS[0]}.* in protein {PTGS2}"):
        atlas._pair(PTGS2, AMBIGUOUS[0])


def test_pair_poses(atlas):
    top, blocks = atlas._pair_poses(ABL1, DRUG)
    assert "V2000" in top and top.rstrip().endswith("$$$$")
    assert sorted(blocks) == [1, 2, 3, 4, 5]
    assert all(block.startswith("MODEL") for block in blocks.values())
    top, blocks = atlas._pair_poses(EGFR, NO_TOP)
    assert top is None and blocks


def test_pair_of_a_named_entry_uses_its_own_run(atlas, atlas_dir):
    """A catalogue number whose own run is in salt_forms/ gives that row, not the drug's canonical row."""
    found = None
    for uniprot in PROTEINS:
        salt = Source(atlas_dir).json(f"salt_forms/{uniprot}.json")
        columns = {name: i for i, name in enumerate(salt["columns"])}
        protein = atlas._protein_json(uniprot)
        shown = {row[protein["columns"].index("pvl")]: row[protein["columns"].index("cat")]
                 for row in protein["rows"]}
        for row in salt["rows"]:
            cat = row[columns["cat"]]
            if shown.get(atlas._pvl(cat).target) not in (None, cat):
                found = (uniprot, cat, row)
                break
    assert found, "the fixture should have a catalogue number with its own run"
    uniprot, cat, row = found
    pair = atlas._pair(uniprot, cat)
    assert pair.row["cat"] == cat and pair.row["vina"] == row[columns["vina"]]
    assert atlas._pair(uniprot, atlas._pvl(cat).target).row["cat"] != cat  # the drug itself: its canonical row


# --- search --------------------------------------------------------------------------------------------------------

def _protein_identifiers(atlas, uniprot):
    """(matched, identifier) of each kind of protein identifier, from the fixture's search files."""
    row = dict(zip(atlas.source.json("search/index.json")["proteins_columns"], atlas._protein_row(uniprot)))
    synonyms = [s for s in row["synonyms"].split() if list(atlas.search(s)["id"]) == [uniprot]]
    return [("gene", row["gene"]), ("gene", row["gene"].lower()), ("UniProt accession", uniprot),
            ("UniProt accession", uniprot.lower()), ("entry name", row["entry"]), ("Entrez ID", row["entrez"]),
            ("Ensembl ID", row["ensembl"]), ("HGNC ID", "HGNC:76"), ("AlphaFold model ID", f"AF-{uniprot}-F1-model_v6"),
            ("gene synonym", synonyms[0]), ("protein name", row["name"]), ("protein name", row["name"].upper())]


def test_search_finds_each_kind_of_protein_identifier(atlas):
    for matched, identifier in _protein_identifiers(atlas, ABL1):
        found = atlas.search(identifier)
        assert list(found.columns) == list(SEARCH_COLUMNS)
        assert (found["kind"][0], found["id"][0], found["matched"][0]) == ("protein", ABL1, matched)
        assert found["value"][0].casefold() == identifier.casefold()


SMILES_FIELDS = {"library SMILES", "parent SMILES", "docked SMILES"}  # one SMILES can be all three


def _drug_identifiers(atlas, pvl):
    """(what may be "matched", identifier) of each kind of drug identifier, from the fixture's tables."""
    from rdkit import Chem

    drug = atlas.drug(pvl)
    mol = Chem.MolFromSmiles(drug.parent_smiles)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    cat = drug.catalogue_numbers[0]
    found = [({"PVL ID"}, pvl), ({"PVL ID"}, pvl.lower()), ({"ChEMBL ID"}, drug.chembl_id),
             ({"catalogue number"}, cat), ({"catalogue number"}, cat.lower()),
             ({"InChIKey", "docked InChIKey"}, drug.inchikey),
             ({"InChIKey first block", "docked InChIKey first block"}, drug.inchikey[:14]),
             (SMILES_FIELDS, drug.parent_smiles), (SMILES_FIELDS, drug.docked_smiles),
             ({f"{s} (canonical)" for s in SMILES_FIELDS}, Chem.MolToSmiles(mol, kekuleSmiles=True)),
             ({"synonym", "name"}, drug.synonyms[0] if drug.synonyms else None),
             ({"name"}, atlas._drug_index().name(pvl).upper())]
    return [(matched, identifier) for matched, identifier in found if identifier]


@pytest.mark.parametrize("pvl", [DRUG, REDOCK])
def test_search_finds_each_kind_of_drug_identifier(atlas, pvl):
    for matched, identifier in _drug_identifiers(atlas, pvl):
        found = atlas.search(identifier)
        assert (found["kind"][0], found["id"][0]) == ("drug", pvl), sorted(matched)
        assert found["matched"][0] in matched, sorted(matched)
        same = found["name"][0] == atlas._drug_index().name(pvl)
        assert same, "the name column is the drug page's name"
    desalted = [c for c in atlas.drug(REDOCK).catalogue_numbers if c.endswith("-desalted")][0]
    assert list(atlas.search(desalted)["id"]) == [REDOCK]


def test_search_docked_smiles_points_at_the_structure_docked(atlas):
    """The docked SMILES of an entry docked as a salt counterion finds the page of that structure."""
    found = atlas.search(atlas.drug(NOT_DRUG).docked_smiles)
    assert found["id"][0] == NOT_DRUG and found["matched"][0].endswith("SMILES")


def test_search_examples_of_the_docstring(atlas):
    assert list(atlas.search("imatinib")["id"]) == [DRUG]
    assert list(atlas.search("ABL1")["id"])[0] == ABL1
    assert list(atlas.search("CHEMBL941")["id"]) == [DRUG]


def test_search_lists_the_start_and_part_of_a_name(atlas):
    name = atlas._drug_index().name(DRUG)
    start, part = atlas.search(name[:4]), atlas.search(name[1:6])
    assert DRUG in set(start["id"]) and start.set_index("id").at[DRUG, "matched"] in ("name", "synonym")
    assert DRUG in set(part["id"])
    assert set(atlas.search(DRUG[:6], top=None)["id"]) >= {DRUG, "PVL00133"}  # the start of a PVL ID


def _made_up_release(atlas_dir, folder: Path) -> Path:
    """A tiny made-up release for the ranking: proteins whose genes, synonyms and names hold "abc" in different ways
    (all with docking results), and drugs whose names and synonyms do."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder.mkdir()
    (folder / "meta.json").write_bytes((atlas_dir / "meta.json").read_bytes())
    index = Source(atlas_dir).json("search/index.json")
    proteins = [("Q00001", "ABC", "", "x"), ("Q00002", "ABCD12", "", "x"), ("Q00003", "ABCD", "", "x"),
                ("Q00004", "XYZ1", "", "Xabc kinase"), ("Q00005", "XYZ2", "ABC", "x"), ("Q00006", "ABCE", "ABC", "x")]
    rows = []
    for uniprot, gene, synonyms, name in proteins:
        row = dict.fromkeys(index["proteins_columns"], "")
        row.update(uniprot=uniprot, gene=gene, synonyms=synonyms, name=name, entry_page=1)
        rows.append(list(row.values()))
    (folder / "search").mkdir()
    for key, data in (("index", dict(index, keys={"a": {}})), ("a", {"p": rows, "d": []})):
        (folder / "search" / f"{key}.json.gz").write_bytes(gzip.compress(json.dumps(data).encode()))
    pvls, names = ["PVL00001", "PVL00002", "PVL00003", "PVL00004"], ["abc", "abcine", "xabcx", "other"]
    cats = ["C1", "C2", "C3", "C4"]

    def table(columns, **values):
        return pa.table({c: values.get(c, [None] * 4) for c in columns})

    (folder / "tables").mkdir()
    pq.write_table(table(_Drugs.ENTRY_COLUMNS, catalog_id=cats, pvl_id=pvls, name=names, docked_as=["drug"] * 4),
                   folder / "tables" / "compound_entries.parquet")
    pq.write_table(table(_Drugs.ANNOTATED_COLUMNS, catalog_id=cats, synonyms=[[], [], [], ["abc"]]),
                   folder / "tables" / "compound_entries_annotated.parquet")
    pq.write_table(table(_Drugs.PAGE_COLUMNS, pvl_id=pvls, name=names, has_results=[True] * 4),
                   folder / "tables" / "compounds.parquet")
    return folder


def test_search_ranks_as_the_atlas_site(atlas_dir, tmp_path):
    """The Atlas site's search-box scores (components/Search.tsx), the first rule that holds: an exact gene (100, + 3
    for a protein with docking results) and drug name (100); a gene that starts with the query (80, less the longer
    the gene); a gene synonym (72; before the gene start rule, so ABCE with the synonym ABC gets 72, not 79); a drug
    synonym (66); part of a drug name (42); part of a protein name (38). The same score: proteins first."""
    atlas = Atlas(_made_up_release(atlas_dir, tmp_path / "release"))
    found = atlas.search("abc")
    assert list(found["id"]) == ["Q00001", "PVL00001", "Q00003", "Q00002", "PVL00002", "Q00005", "Q00006",
                                 "PVL00004", "PVL00003", "Q00004"]
    assert list(found["matched"]) == ["gene", "name", "gene", "gene", "name", "gene synonym", "gene synonym",
                                      "synonym", "name", "protein name"]
    assert list(found["kind"][:2]) == ["protein", "drug"] and found["name"][0] == "ABC: x"
    assert list(atlas.search("ABC", top=3)["id"]) == ["Q00001", "PVL00001", "Q00003"]
    # two characters: the start of a gene (a shorter gene first) or of a drug name; synonyms and parts need three
    assert list(atlas.search("ab")["id"]) == ["Q00001", "Q00003", "Q00006", "PVL00001", "PVL00002", "Q00002"]


def test_search_finds_nothing(atlas):
    found = atlas.search("nosuchthing")
    assert found.empty and list(found.columns) == list(SEARCH_COLUMNS)
    assert atlas.search("x").empty  # one character: only exact matches


def test_search_arguments(atlas):
    assert SEARCH_TOP == 20 and atlas.search.__defaults__ == (SEARCH_TOP,)
    assert len(atlas.search("a", top=None)) >= len(atlas.search("a", top=1))
    assert len(atlas.search(DRUG[:6], top=0)) == 0
    with pytest.raises(ValueError, match="top must be a number of rows"):
        atlas.search(DRUG, top=-1)
    with pytest.raises(ValueError, match="query is empty"):
        atlas.search(" ")
    assert list(atlas.search(25)["id"]) == [ABL1]  # an Entrez ID as a number


# --- download ------------------------------------------------------------------------------------------------------

def _tree(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p for p in folder.rglob("*") if p.is_file()}


def _check_offline(out: Path, proteins, atlas_dir: Path):
    """Atlas(out) does everything for its proteins: search, protein, drug, best_scores, select and show."""
    sub = Atlas(out)
    assert sub.source.path == str(out) and f"{len(proteins)} protein" in repr(sub) and "subset" in repr(sub)
    assert list(sub.search(DRUG)["id"]) == [DRUG] and sub.search("ABL1")["id"][0] == ABL1
    for uniprot in proteins:
        assert sub.protein(uniprot).uniprot == uniprot
        assert len(sub.best_scores(protein=uniprot)) == len(sub._protein_json(uniprot)["rows"])
    assert sub.drug(DRUG).pvl == DRUG and sub.drug(AMBIGUOUS[1]).pvl == AMBIGUOUS[1]  # every drug page
    assert list(sub.best_scores(ligand=DRUG)["protein_id"]) == [
        u for u in Atlas(atlas_dir).best_scores(ligand=DRUG)["protein_id"] if u in proteins]
    pair = sub.select(protein=ABL1, ligand=DRUG, pocket="all")
    assert len(pair.poses) == 5 and Path(pair.export_protein(out.parent / "protein.pdb")).is_file()
    page = Path(sub.show(ABL1, path=out.parent / "page.html")).read_text(encoding="utf-8")
    for uniprot in proteins:  # P2Rank's SAS points, for the page's SAS toggle; the ChEMBL records of "ChEMBL Complete"
        assert sub._sas(uniprot) == sas.decode((atlas_dir / "sas" / f"{uniprot}.json.gz").read_bytes())
        assert sub._chembl(uniprot) == (_chembl_json(atlas_dir, f"proteins/{uniprot}.json.gz"),
                                        _chembl_json(atlas_dir, "strict.json.gz"))
    data = workspace.page_payload(page)["proteins"][0]
    assert data["sas"]["pockets"] and [c["label"] for c in data["topColumns"]] == ["ChEMBL Strict", "ChEMBL Complete"]
    return sub


def _chembl_json(release: Path, rel: str):
    return json.loads(gzip.decompress((release / "chembl_any" / rel).read_bytes()))


def test_download_from_a_folder(atlas, atlas_dir, tmp_path, capsys):
    """A subset of two proteins: their files, every drug page, the search files, four tables and ligand_ids.json,
    byte-identical to the release's; their SAS points with a sas/MANIFEST.json of their own; MANIFEST.json lists them
    all; meta.json says what the subset is."""
    out = tmp_path / "my_atlas"
    assert atlas.download(proteins=["ABL1", "EGFR"], out=out) == str(out)
    line = capsys.readouterr().out
    assert "files" in line and "MB" in line and "2 proteins" in line and str(out) in line
    files, release = _tree(out), json.loads((atlas_dir / "MANIFEST.json").read_text())["files"]
    for rel in set(files) - {"meta.json", "MANIFEST.json", "sas/MANIFEST.json", "chembl_any/MANIFEST.json"}:
        assert files[rel].read_bytes() == (atlas_dir / rel).read_bytes(), rel
    expected = {f"{folder}/{uniprot}{ext}" for uniprot in (ABL1, EGFR) for folder, ext in (
        ("proteins", ".json.gz"), ("salt_forms", ".json.gz"), ("poses", ".poses"), ("poses_redock", ".poses"),
        ("sas", ".json.gz"), ("chembl_any/proteins", ".json.gz"))} | {"chembl_any/strict.json.gz"}
    expected |= {f"structures/AF-{uniprot}-F1-model_v6.pdb.gz" for uniprot in (ABL1, EGFR)}
    expected |= {rel for rel in release if rel.startswith(("search/", "compounds/")) and ".all." not in rel}
    expected |= {f"tables/{name}.parquet" for name in ("compound_entries", "compound_entries_annotated", "compounds",
                                                       "proteins")}
    assert set(files) == expected | {"ligand_ids.json", "meta.json", "MANIFEST.json", "sas/MANIFEST.json",
                                     "chembl_any/MANIFEST.json"}
    release_sas = json.loads((atlas_dir / "sas" / "MANIFEST.json").read_text())
    assert json.loads(files["sas/MANIFEST.json"].read_text()) == {f"{u}.json.gz": release_sas[f"{u}.json.gz"]
                                                                 for u in (ABL1, EGFR)}
    release_chembl = json.loads((atlas_dir / "chembl_any" / "MANIFEST.json").read_text())
    assert json.loads(files["chembl_any/MANIFEST.json"].read_text()) == {
        rel: release_chembl[rel] for rel in (f"proteins/{ABL1}.json.gz", f"proteins/{EGFR}.json.gz", "strict.json.gz")}
    manifest = json.loads(files["MANIFEST.json"].read_text())
    assert set(manifest["files"]) == set(files) - {"MANIFEST.json"}
    for rel, entry in manifest["files"].items():
        data = files[rel].read_bytes()
        assert entry == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, rel
    meta = json.loads(files["meta.json"].read_text())
    assert meta["subset"]["proteins"] == [ABL1, EGFR] and meta["subset"]["of"] == str(atlas_dir)
    assert "best_scores(ligand=...)" in meta["subset"]["note"] and meta["release"] == atlas.meta["release"]
    assert manifest["subset"] == meta["subset"] and not list(tmp_path.glob("**/*.tmp-*"))
    _check_offline(out, [ABL1, EGFR], atlas_dir)


def test_download_over_http_then_offline(atlas_server, atlas_dir, tmp_path, monkeypatch):
    """Downloaded over HTTP; the folder is then read without the server (no request reaches it)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    out = tmp_path / "my_atlas"
    Atlas(atlas_server.url).download(["ABL1"], out)
    for rel, path in _tree(out).items():
        if rel not in ("meta.json", "MANIFEST.json", "sas/MANIFEST.json", "chembl_any/MANIFEST.json"):
            assert path.read_bytes() == (atlas_dir / rel).read_bytes(), rel
    asked = len(atlas_server.requests)
    atlas_server.shutdown()
    _check_offline(out, [ABL1], atlas_dir)
    assert len(atlas_server.requests) == asked


def test_subset_knows_the_other_proteins(atlas, tmp_path):
    """A protein the subset does not have: a clear error naming the subset's proteins; search() and
    best_scores(ligand=) list only the subset's proteins (a download has no compounds/<PVL>.all.json)."""
    out = tmp_path / "my_atlas"
    sub = Atlas(atlas.download(["ABL1"], out))
    assert not list(out.glob("compounds/*.all.json.gz"))
    assert EGFR not in set(sub.search("EGFR")["id"]) and list(sub.search("ABL1")["id"])[:1] == [ABL1]  # its proteins
    assert list(sub.search(DRUG)["id"]) == [DRUG]  # every drug
    expected = f"{EGFR} \\(EGFR\\) is not in this subset .*my_atlas.*its proteins: {ABL1} \\(ABL1\\)"
    with pytest.raises(KeyError, match=expected):
        sub.protein("EGFR")
    with pytest.raises(KeyError, match="not in this subset"):
        sub.select(protein=EGFR, ligand=DRUG)
    assert list(sub.best_scores(ligand=DRUG)["protein_id"]) == [ABL1]
    assert len(atlas.best_scores(ligand=DRUG)) == 3


def test_download_resumes_and_adds_proteins(atlas_server, atlas_dir, tmp_path, monkeypatch):
    """A second download into the same folder copies only what is missing or damaged (sha256), and adds proteins."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    out = tmp_path / "my_atlas"
    Atlas(atlas_server.url).download(["ABL1"], out)
    poses = out / "poses" / f"{ABL1}.poses"
    damaged = bytearray(poses.read_bytes())
    damaged[100] ^= 0xFF  # same size, other bytes
    poses.write_bytes(bytes(damaged))
    page = next(out.glob("compounds/*.json.gz"))
    page.unlink()
    before = len(atlas_server.requests)
    Atlas(atlas_server.url).download(["EGFR"], out)
    fetched = {rel for rel, _ in atlas_server.requests[before:]}
    assert poses.read_bytes() == (atlas_dir / "poses" / f"{ABL1}.poses").read_bytes() and page.is_file()
    assert f"poses/{ABL1}.poses" in fetched and page.relative_to(out).as_posix() in fetched
    assert f"poses/{EGFR}.poses" in fetched and f"poses_redock/{ABL1}.poses" not in fetched
    assert {rel for rel in fetched if rel.startswith("compounds/")} == {page.relative_to(out).as_posix()}
    assert len([rel for rel in fetched if rel.startswith("search/")]) <= 2  # only to find "EGFR"
    assert json.loads((out / "meta.json").read_text())["subset"]["proteins"] == [ABL1, EGFR]
    before = len(atlas_server.requests)
    Atlas(atlas_server.url).download(["ABL1", "EGFR"], out)  # all there: only meta.json and the listings of sas/
    # and chembl_any/ (the server's, not the cache's) are read again, and the size of LICENSE (MANIFEST.json does not
    # list it) is asked for: the fixture has none (HTTP 404)
    assert set(atlas_server.requests[before:]) == {("meta.json", None), ("sas/MANIFEST.json", None),
                                                   ("chembl_any/MANIFEST.json", None), ("LICENSE", "bytes=0-0")}
    _check_offline(out, [ABL1, EGFR], atlas_dir)


@pytest.mark.parametrize("source", ["local", "remote"])
def test_download_refuses_a_damaged_file(source, atlas_dir, tmp_path, http_server, monkeypatch):
    """A file whose sha256 is not the one MANIFEST.json lists is not kept: the download stops with an error."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    page = release / "compounds" / f"{DRUG}.json.gz"
    damaged = bytearray(page.read_bytes())
    damaged[-5] ^= 0xFF
    page.write_bytes(bytes(damaged))
    location = release if source == "local" else http_server(release).url
    with pytest.raises(ValueError, match=f"compounds/{DRUG}.json.gz is damaged .*sha256"):
        Atlas(location).download(["ABL1"], tmp_path / "out")
    assert not (tmp_path / "out" / "compounds" / f"{DRUG}.json.gz").exists()
    assert not (tmp_path / "out" / "meta.json").exists() and not list(tmp_path.glob("out/**/*.tmp-*"))


def test_download_files_the_manifest_does_not_list(atlas_dir, tmp_path, http_server, monkeypatch):
    """salt_forms/ is not in the public release's MANIFEST.json: its files are checked by size only (over HTTP from the
    server's answer), and a release without them (release/v1.1) is downloaded without them."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    manifest = json.loads((release / "MANIFEST.json").read_text())
    manifest["files"] = {rel: e for rel, e in manifest["files"].items() if not rel.startswith("salt_forms/")}
    (release / "MANIFEST.json").write_text(json.dumps(manifest))
    out = Path(Atlas(http_server(release).url).download(["ABL1"], tmp_path / "out"))
    salt = out / "salt_forms" / f"{ABL1}.json.gz"
    assert salt.read_bytes() == (atlas_dir / "salt_forms" / f"{ABL1}.json.gz").read_bytes()
    assert json.loads((out / "MANIFEST.json").read_text())["files"][f"salt_forms/{ABL1}.json.gz"]["bytes"] == \
        salt.stat().st_size
    shutil.rmtree(release / "salt_forms")
    out = Path(Atlas(release).download(["ABL1"], tmp_path / "out2"))
    assert not (out / "salt_forms").exists() and Atlas(out).select(ABL1, DRUG).ligand == DRUG


@pytest.mark.parametrize("source", ["local", "remote"])
def test_download_copies_the_sas_points(source, atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """P2Rank's SAS points of the proteins (sas/<U>.json.gz, checked by sha256 against sas/MANIFEST.json) go into the
    folder with a sas/MANIFEST.json of its proteins, so its pages have the SAS toggle, as does a download of the
    download. A release without sas/ is downloaded without them; a damaged SAS file stops the download, as any other."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    location = release if source == "local" else http_server(release).url
    out = Path(Atlas(location).download(["ABL1"], tmp_path / "out"))
    again = Path(Atlas(out).download(["ABL1"], tmp_path / "again"))  # a download of the download
    listed = json.loads((atlas_dir / "sas" / "MANIFEST.json").read_text())
    for folder in (out, again):
        assert sorted(p.name for p in (folder / "sas").iterdir()) == ["MANIFEST.json", f"{ABL1}.json.gz"]
        assert (folder / "sas" / f"{ABL1}.json.gz").read_bytes() == (atlas_dir / "sas" / f"{ABL1}.json.gz").read_bytes()
        data = (folder / "sas" / "MANIFEST.json").read_bytes()
        assert json.loads(data) == {f"{ABL1}.json.gz": listed[f"{ABL1}.json.gz"]}
        assert json.loads((folder / "MANIFEST.json").read_text())["files"]["sas/MANIFEST.json"] == {
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        page = Path(Atlas(folder).select(ABL1, DRUG).show(path=tmp_path / f"{folder.name}.html")).read_text("utf-8")
        assert workspace.page_payload(page)["proteins"][0]["sas"]["pockets"]
    Atlas(location).download(["EGFR"], out)  # one more protein: sas/MANIFEST.json lists both
    assert json.loads((out / "sas" / "MANIFEST.json").read_text()) == {
        k: listed[k] for k in (f"{ABL1}.json.gz", f"{EGFR}.json.gz")}

    without = Path(shutil.copytree(atlas_dir, tmp_path / "without", ignore=shutil.ignore_patterns("sas")))
    location = without if source == "local" else http_server(without).url
    out = Path(Atlas(location).download(["ABL1"], tmp_path / "out_without"))
    assert not (out / "sas").exists() and not any(rel.startswith("sas/") for rel in json.loads(
        (out / "MANIFEST.json").read_text())["files"])
    page = Path(Atlas(out).select(ABL1, DRUG).show(path=tmp_path / "without.html")).read_text("utf-8")
    assert "sas" not in workspace.page_payload(page)["proteins"][0]

    # a sas/MANIFEST.json that is not JSON: a warning, no sas/; an entry that is not [sha256, bytes]: that file left out
    for k, text in enumerate(["{not json", json.dumps(dict(listed, **{f"{ABL1}.json.gz": ["abc", "12"]}))]):
        broken = Path(shutil.copytree(atlas_dir, tmp_path / f"broken{k}"))
        (broken / "sas" / "MANIFEST.json").write_text(text)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
            out = Path(Atlas(broken if source == "local" else http_server(broken).url).download(
                ["ABL1", "EGFR"], tmp_path / f"out_broken{k}"))
        names = sorted(p.name for p in (out / "sas").iterdir()) if (out / "sas").exists() else []
        assert names == ([] if k == 0 else ["MANIFEST.json", f"{EGFR}.json.gz"]) and (out / "meta.json").is_file()
        assert ("are not copied" if k == 0 else "does not list it as [sha256, bytes]") in caplog.text

    damaged = release / "sas" / f"{ABL1}.json.gz"
    damaged.write_bytes(damaged.read_bytes()[:-1] + b"x")
    location = release if source == "local" else http_server(release).url
    with pytest.raises(ValueError, match=f"sas/{ABL1}.json.gz is damaged .*sha256"):
        Atlas(location).download(["ABL1"], tmp_path / "out_damaged")
    assert not (tmp_path / "out_damaged" / "meta.json").exists()


@pytest.mark.parametrize("source", ["local", "remote"])
def test_download_copies_the_chembl_records(source, atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """The ChEMBL records of "ChEMBL Complete" (chembl_any/proteins/<U>.json.gz of the proteins, and strict.json.gz, the
    strict values' links; checked by sha256 against chembl_any/MANIFEST.json) go into the folder with a
    chembl_any/MANIFEST.json of its own, so its pages have both ChEMBL values, as does a download of the download. Not
    copied: chembl_any/compounds/ (the site's drug pages). A release without chembl_any/ is downloaded without it, and
    its pages have "ChEMBL Strict" only; a damaged file stops the download, as any other."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    listed = json.loads((release / "chembl_any" / "MANIFEST.json").read_text())
    (release / "chembl_any" / "compounds").mkdir()
    (release / "chembl_any" / "compounds" / f"{DRUG}.json.gz").write_bytes(gzip.compress(b"{}", mtime=0))
    data = (release / "chembl_any" / "compounds" / f"{DRUG}.json.gz").read_bytes()
    (release / "chembl_any" / "MANIFEST.json").write_text(json.dumps(dict(listed, **{
        f"compounds/{DRUG}.json.gz": [hashlib.sha256(data).hexdigest(), len(data)]}), indent=0))
    location = release if source == "local" else http_server(release).url
    out = Path(Atlas(location).download(["ABL1"], tmp_path / "out"))
    again = Path(Atlas(out).download(["ABL1"], tmp_path / "again"))  # a download of the download
    for folder in (out, again):
        copied = {rel for rel in _tree(folder) if rel.startswith("chembl_any/")}
        assert copied == {"chembl_any/MANIFEST.json", f"chembl_any/proteins/{ABL1}.json.gz",
                          "chembl_any/strict.json.gz"}
        for rel in (f"proteins/{ABL1}.json.gz", "strict.json.gz"):
            assert (folder / "chembl_any" / rel).read_bytes() == (atlas_dir / "chembl_any" / rel).read_bytes()
        data = (folder / "chembl_any" / "MANIFEST.json").read_bytes()
        assert json.loads(data) == {rel: listed[rel] for rel in (f"proteins/{ABL1}.json.gz", "strict.json.gz")}
        assert json.loads((folder / "MANIFEST.json").read_text())["files"]["chembl_any/MANIFEST.json"] == {
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        page = Path(Atlas(folder).select(ABL1, DRUG).show(path=tmp_path / f"{folder.name}.html")).read_text("utf-8")
        columns = workspace.page_payload(page)["proteins"][0]["topColumns"]
        assert [c["label"] for c in columns] == ["ChEMBL Strict", "ChEMBL Complete"]
    Atlas(location).download(["EGFR"], out)  # one more protein: chembl_any/MANIFEST.json lists both
    assert json.loads((out / "chembl_any" / "MANIFEST.json").read_text()) == {
        rel: listed[rel] for rel in (f"proteins/{ABL1}.json.gz", f"proteins/{EGFR}.json.gz", "strict.json.gz")}

    without = Path(shutil.copytree(atlas_dir, tmp_path / "without", ignore=shutil.ignore_patterns("chembl_any")))
    location = without if source == "local" else http_server(without).url
    out = Path(Atlas(location).download(["ABL1"], tmp_path / "out_without"))
    assert not (out / "chembl_any").exists() and not any(rel.startswith("chembl_any/") for rel in json.loads(
        (out / "MANIFEST.json").read_text())["files"])
    page = Path(Atlas(out).select(ABL1, DRUG).show(path=tmp_path / "without.html")).read_text("utf-8")
    assert [c["label"] for c in workspace.page_payload(page)["proteins"][0]["topColumns"]] == ["ChEMBL Strict"]

    # a chembl_any/MANIFEST.json that is not JSON: a warning, no chembl_any/; an entry that is not [sha256, bytes]:
    # that file left out
    wrong = dict(listed, **{f"proteins/{ABL1}.json.gz": ["abc", "12"]})
    for k, text in enumerate(["{not json", "[]", json.dumps(wrong)]):
        broken = Path(shutil.copytree(atlas_dir, tmp_path / f"broken{k}"))
        (broken / "chembl_any" / "MANIFEST.json").write_text(text)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
            out = Path(Atlas(broken if source == "local" else http_server(broken).url).download(
                ["ABL1", "EGFR"], tmp_path / f"out_broken{k}"))
        names = sorted(rel for rel in _tree(out) if rel.startswith("chembl_any/"))
        assert names == ([] if k < 2 else ["chembl_any/MANIFEST.json", f"chembl_any/proteins/{EGFR}.json.gz",
                                           "chembl_any/strict.json.gz"]) and (out / "meta.json").is_file()
        assert ("are not copied" if k < 2 else "does not list it as [sha256, bytes]") in caplog.text

    damaged = release / "chembl_any" / "proteins" / f"{ABL1}.json.gz"
    damaged.write_bytes(damaged.read_bytes()[:-1] + b"x")
    location = release if source == "local" else http_server(release).url
    with pytest.raises(ValueError, match=f"chembl_any/proteins/{ABL1}.json.gz is damaged .*sha256"):
        Atlas(location).download(["ABL1"], tmp_path / "out_damaged")
    assert not (tmp_path / "out_damaged" / "meta.json").exists()


LICENCE = b"The licence of the data of this test release.\n"


@pytest.mark.parametrize("source", ["local", "remote"])
def test_download_copies_the_licence(source, atlas_dir, tmp_path, http_server, monkeypatch):
    """LICENSE, which the release's MANIFEST.json does not list, is copied as it is (checked by size) and listed in
    the folder's MANIFEST.json, so a download of the download has it too, checked by sha256. A source without it (the
    fixture, an older release, a download made without it, or one whose MANIFEST.json lists it but has lost it) is
    downloaded without it, with no error."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    (release / "LICENSE").write_bytes(LICENCE)
    location = release if source == "local" else http_server(release).url
    out = Path(Atlas(location).download(["ABL1"], tmp_path / "out"))
    again = Path(Atlas(out).download(["ABL1"], tmp_path / "again"))  # a download of the download
    for folder in (out, again):
        assert (folder / "LICENSE").read_bytes() == LICENCE
        assert json.loads((folder / "MANIFEST.json").read_text())["files"]["LICENSE"] == {
            "bytes": len(LICENCE), "sha256": hashlib.sha256(LICENCE).hexdigest()}
    (again / "LICENSE").write_bytes(LICENCE.upper())  # damaged, same size: out's MANIFEST.json has its sha256
    Atlas(out).download(["ABL1"], again)
    assert (again / "LICENSE").read_bytes() == LICENCE  # copied again
    (again / "LICENSE").unlink()  # its MANIFEST.json still lists it
    lost = Path(Atlas(again).download(["ABL1"], tmp_path / "lost"))
    (release / "LICENSE").unlink()
    out = Path(Atlas(location).download(["ABL1"], tmp_path / "without"))
    again = Path(Atlas(out).download(["ABL1"], tmp_path / "without_again"))
    for folder in (lost, out, again):
        assert not (folder / "LICENSE").exists()
        assert "LICENSE" not in json.loads((folder / "MANIFEST.json").read_text())["files"]
        assert Atlas(folder).select(ABL1, DRUG).ligand == DRUG


def test_download_resumes_with_the_licence(atlas_dir, tmp_path, http_server, monkeypatch):
    """A LICENSE in the folder is kept if it has the size of the release's, else copied again (no sha256 is listed
    for it); a folder that has a LICENSE cut short and the folders of a download that stopped is gone on with. A folder
    with only a LICENSE is not a download's: it is left as it is, with an error."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    (release / "LICENSE").write_bytes(LICENCE)
    server = http_server(release)
    out = tmp_path / "out"
    out.mkdir()
    (out / "LICENSE").write_bytes(LICENCE[:5])
    with pytest.raises(ValueError, match=r"is not empty \(it has LICENSE\): download\(\) writes into a new or empty"):
        Atlas(server.url).download(["ABL1"], out)
    assert [p.name for p in out.iterdir()] == ["LICENSE"] and (out / "LICENSE").read_bytes() == LICENCE[:5]
    (out / "proteins").mkdir()  # a download that stopped: the folders of the files it was copying are there
    Atlas(server.url).download(["ABL1"], out)
    assert (out / "LICENSE").read_bytes() == LICENCE
    for extra, copied in ((b"", False), (b"more", True)):
        (out / "LICENSE").write_bytes(LICENCE + extra)
        before = len(server.requests)
        Atlas(server.url).download(["ABL1"], out)
        asked = [request for request in server.requests[before:] if request[0] == "LICENSE"]
        assert asked == [("LICENSE", "bytes=0-0")] + ([("LICENSE", None)] if copied else [])  # its size, the copy
        assert (out / "LICENSE").read_bytes() == LICENCE
    assert not list(tmp_path.glob("out/**/*.tmp-*"))


def test_download_arguments_and_folders(atlas, atlas_dir, tmp_path):
    with pytest.raises(ValueError, match="proteins is empty"):
        atlas.download([], tmp_path / "a")
    with pytest.raises(TypeError, match="proteins must be protein identifiers"):
        atlas.download(3.5, tmp_path / "a")
    with pytest.raises(KeyError, match="no protein 'NOSUCHGENE'"):
        atlas.download(["ABL1", "NOSUCHGENE"], tmp_path / "a")
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "notes.txt").write_text("mine")
    with pytest.raises(ValueError, match="is not empty .*notes.txt.*new or empty folder"):
        atlas.download("ABL1", tmp_path / "busy")
    with pytest.raises(ValueError, match="holds a PocketVina Atlas release it did not write"):
        atlas.download("ABL1", shutil.copytree(atlas_dir, tmp_path / "copy"))
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError, match="is a file, not a folder"):
        atlas.download("ABL1", tmp_path / "file")
    (tmp_path / "empty").mkdir()
    assert Atlas(atlas.download("ABL1", tmp_path / "empty"))._subset_proteins() == [ABL1]  # one identifier
    no_manifest = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    (no_manifest / "MANIFEST.json").unlink()
    with pytest.raises(FileNotFoundError, match="has no MANIFEST.json"):
        Atlas(no_manifest).download(["ABL1"], tmp_path / "b")


def test_source_copy_and_size(atlas_server, atlas_dir, tmp_path, monkeypatch):
    """Source.copy streams a file (a dropped connection starts it again, the progress taken back), Source.size gives
    its size; locally and over HTTP."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(atlas_source, "RETRY_DELAYS", (0, 0, 0))
    rel = f"poses/{ABL1}.poses"
    data = (atlas_dir / rel).read_bytes()
    for src in (Source(atlas_dir), Source(atlas_server.url)):
        assert src.size(rel) == len(data)
        with pytest.raises(FileNotFoundError):
            src.size("proteins/XXXXXX.json.gz")
        seen = []
        assert src.copy(rel, tmp_path / "a" / "x.poses", progress=seen.append) == (
            len(data), hashlib.sha256(data).hexdigest())
        assert (tmp_path / "a" / "x.poses").read_bytes() == data and sum(seen) == len(data)
        with pytest.raises(ValueError, match=f"has {len(data)} bytes, the release says {len(data) + 1}"):
            src.copy(rel, tmp_path / "b" / "x.poses", size=len(data) + 1)
        assert not (tmp_path / "b" / "x.poses").exists()
    atlas_server.cut = 1
    seen = []
    Source(atlas_server.url).copy(rel, tmp_path / "c" / "x.poses", progress=seen.append)
    assert (tmp_path / "c" / "x.poses").read_bytes() == data and sum(seen) == len(data) and min(seen) < 0
    assert not list(tmp_path.glob("**/*.tmp-*"))


# --- the command line ----------------------------------------------------------------------------------------------

def run(*args, ok=True):
    """``pocketvina <args>`` in this process; checks the exit code (0, or 1 with ok=False) without printing the
    output (it holds drug names)."""
    result = CliRunner().invoke(cli.cli, [str(a) for a in args], catch_exceptions=False)
    code = result.exit_code
    assert code == (0 if ok else cli.EXIT_USAGE), f"exit code {code}"
    return result


def test_cli_atlas_search(atlas_dir, monkeypatch):
    out = run("atlas", "search", DRUG, "--atlas", atlas_dir).output
    first, second = out.splitlines()[:2]
    assert first.split() == list(SEARCH_COLUMNS) and second.split()[:2] == ["drug", DRUG]
    assert len(run("atlas", "search", "P", "--top", "1", "--atlas", atlas_dir).output.splitlines()) <= 2
    assert "(0 rows)" in run("atlas", "search", "nosuchthing", "--atlas", atlas_dir).output
    monkeypatch.setenv("POCKETVINA_ATLAS", str(atlas_dir))  # the default, as Atlas()
    assert run("atlas", "search", "ABL1").output.splitlines()[1].split()[:2] == ["protein", ABL1]
    assert cli.ATLAS_SEARCH_TOP == SEARCH_TOP  # cli.py does not import pocketvina.atlas (pandas) to start
    assert cli.ATLAS_EXPORT_FORMATS == selection.EXPORT_FORMATS


def test_cli_atlas_show(atlas_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert run("atlas", "show", "ABL1", "--atlas", atlas_dir).output.strip() == str(tmp_path / f"PV_{ABL1}.html")
    assert (tmp_path / f"PV_{ABL1}.html").is_file()
    out = run("atlas", "show", "ABL1", "--ligand", DRUG, "--pocket", "all", "--mode", "1", "--atlas", atlas_dir).output
    assert out.strip() == str(tmp_path / f"PV_{ABL1}_{DRUG}_pocketall_top.html")
    out = run("atlas", "show", "EGFR", "--ligand", DRUG, "--pocket", "2", "--mode", "all", "--out", "pair.html",
              "--atlas", atlas_dir).output
    assert out.strip() == str(tmp_path / "pair.html") and (tmp_path / "pair.html").is_file()
    for args in (["--pocket", "2"], ["--ligand", DRUG, "--pocket", "x"], ["--ligand", DRUG, "--mode", "0"]):
        assert "Error" in run("atlas", "show", "ABL1", *args, "--atlas", atlas_dir, ok=False).output


def test_cli_atlas_export(atlas_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = run("atlas", "export", "ABL1", DRUG, "--atlas", atlas_dir).output.split()
    assert out == [str(tmp_path / f"PV_{ABL1}_{DRUG}_top.sdf")] and Path(out[0]).is_file()
    out = run("atlas", "export", "ABL1", DRUG, "--pocket", "all", "--mode", "all", "--fmt", "pdbqt", "--protein",
              "--out", "poses", "--atlas", atlas_dir).output.split()
    assert out == [str(tmp_path / "poses" / f"PV_{ABL1}_{DRUG}_pocketall_modes.pdbqt"),
                   str(tmp_path / "poses" / f"PV_{ABL1}.pdb")]
    assert all(Path(p).is_file() for p in out)
    assert run("atlas", "export", "ABL1", DRUG, "--out", "one.sdf", "--atlas", atlas_dir).output.split() == [
        str(tmp_path / "one.sdf")]
    assert "no drug 'nosuchdrug'" in run("atlas", "export", "ABL1", "nosuchdrug", "--atlas", atlas_dir,
                                         ok=False).output


def test_cli_atlas_download(atlas_dir, tmp_path):
    out = tmp_path / "my_atlas"
    result = run("atlas", "download", "ABL1", "EGFR", "--out", out, "--atlas", atlas_dir)
    assert result.output.rstrip().endswith(str(out)) and "files" in result.output  # the path is printed last
    assert json.loads((out / "meta.json").read_text())["subset"]["proteins"] == [ABL1, EGFR]
    assert run("atlas", "search", "EGFR", "--atlas", out).output.splitlines()[1].split()[:2] == ["protein", EGFR]
    said = run("atlas", "show", PTGS2, "--atlas", out, ok=False).output
    assert "not in this subset" in said and "The public release (no --atlas) has every protein" in said
    assert "Missing argument" in run("atlas", "download", "--out", out, "--atlas", atlas_dir, ok=False).output


def test_cli_atlas_errors(atlas_dir, closed_port, tmp_path, monkeypatch):
    """An error is a message and exit code 1; its hints name the command line (in Python they name Python)."""
    monkeypatch.setattr(atlas_source, "RETRY_DELAYS", (0, 0, 0))
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "notes.txt").write_text("mine")
    (tmp_path / "poses").write_text("mine")  # a file: --out poses is a folder
    here = shlex.quote(str(atlas_dir))
    said = {
        ("show", "NOSUCHGENE", "--atlas", atlas_dir): ("no protein 'NOSUCHGENE'", "; pocketvina atlas search "
                                                       f"NOSUCHGENE --atlas {here} finds part of a name"),
        ("export", "ABL1", "no such drug"): ("no drug 'no such drug'", "pocketvina atlas search 'no such drug' finds"),
        ("export", "ABL1", DRUG, "--out", "x.pdbqt", "--atlas", atlas_dir): ("pass --fmt pdbqt for a PDBQT file",),
        ("export", "ABL1", DRUG, "--protein", "--out", "abl1.pdb", "--atlas", atlas_dir): (
            f"{tmp_path / 'abl1.pdb'}: poses are written as SDF; use a .sdf name, or --fmt pdbqt for .pdbqt",),
        ("export", "ABL1", DRUG, "--fmt", "pdbqt", "--out", "x.txt", "--atlas", atlas_dir): (
            f"{tmp_path / 'x.txt'}: poses are written as PDBQT; use a .pdbqt name, or --fmt sdf for .sdf",),
        ("export", "ABL1", DRUG, "--protein", "--out", "poses", "--atlas", atlas_dir): (
            f"Error: {tmp_path / 'poses'} is a file, but a path without an extension is a folder: use e.g. poses.sdf "
            "for a file\n",),
        ("search", "ABL1", "--atlas", closed_port): ("cannot reach the PocketVina Atlas", "use --atlas /path/to/"),
        ("search", "ABL1", "--atlas", tmp_path): ("is not a PocketVina Atlas release", "use --atlas /path/to/release",
                                                  "or no --atlas for the public release"),
        ("download", "ABL1", "--out", tmp_path / "busy", "--atlas", atlas_dir): (
            "pocketvina atlas download writes into a new or empty folder",),
    }
    monkeypatch.setenv("POCKETVINA_ATLAS", str(atlas_dir))
    monkeypatch.chdir(tmp_path)
    for args, parts in said.items():
        out = run("atlas", *args, ok=False).output
        assert all(part in out for part in parts), args
        assert not any(python in out for python in ("atlas.search(", "Atlas(", "download()", "fmt=")), args
    assert not any(tmp_path.glob("*.pdb")) and not (tmp_path / "x.txt").exists()  # nothing written
    assert (tmp_path / "poses").read_text() == "mine"
    with pytest.raises(KeyError, match=r"; atlas\.search\('NOSUCHGENE'\) finds part of a name"):
        Atlas(atlas_dir).protein("NOSUCHGENE")  # Python keeps its own hints


# --- over HTTP -----------------------------------------------------------------------------------------------------

def test_atlas_over_http(atlas_server, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    atlas = Atlas(atlas_server.url)
    assert atlas.release == "v1.1"
    assert atlas.protein("ABL1").uniprot == ABL1
    assert atlas.drug(DRUG).pvl == DRUG
    assert len(atlas.best_scores(protein="ABL1", top=2)) == 2
    top, blocks = atlas._pair_poses(ABL1, DRUG)
    assert top and sorted(blocks) == [1, 2, 3, 4, 5]
    assert any(header for _, header in atlas_server.requests if header)  # the poses came in Range requests


def _no_network(monkeypatch):
    """Every HTTP request fails as without network (at once: no waits between the tries)."""
    def fail(*args, **kwargs):
        raise OSError("Network is unreachable")

    monkeypatch.setattr(atlas_source, "_fetch", fail)
    monkeypatch.setattr(atlas_source, "_urlopen", fail)
    monkeypatch.setattr(atlas_source, "RETRY_DELAYS", (0, 0, 0))


def test_sas_points(atlas_dir, atlas_server, tmp_path, monkeypatch):
    """P2Rank's SAS points of a protein for its page (Atlas._sas): sas/<U>.json.gz, from a folder and over HTTP. Over
    HTTP the file and sas/MANIFEST.json are kept in the cache: later they are read from there (the listing checked with
    the server once per Atlas, HTTP 304), also without network."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    expected = sas.decode((atlas_dir / "sas" / f"{ABL1}.json.gz").read_bytes())
    assert expected["protein"] == ABL1 and list(expected["pockets"]) == ["1", "2", "3", "4", "5"]
    assert Atlas(atlas_dir)._sas(ABL1) == expected
    remote = Atlas(atlas_server.url)
    asked = len(atlas_server.requests)
    assert remote._sas(ABL1) == expected
    assert [r for r, _ in atlas_server.requests[asked:]] == ["sas/MANIFEST.json", f"sas/{ABL1}.json.gz"]
    asked = len(atlas_server.requests)
    again = Atlas(atlas_server.url)
    assert again._sas(ABL1) == expected
    # meta.json is never read from the cache (it names the release); the listing is checked: not changed
    assert [r for r, _ in atlas_server.requests[asked:]] == ["meta.json", "sas/MANIFEST.json"]
    assert atlas_server.conditional == [("sas/MANIFEST.json", 304)]
    again._sas_files = None  # sas/MANIFEST.json is read again: from the cache
    _no_network(monkeypatch)
    assert again._sas(ABL1) == expected


def test_sas_points_missing_or_unreadable(atlas_dir, atlas_server, tmp_path, monkeypatch, caplog):
    """None without SAS points: a release without sas/ (an older release or download), a protein sas/MANIFEST.json does
    not list. None and a warning when they cannot be read: a damaged file, a damaged listing in sas/MANIFEST.json, no
    network (then asked again next time)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    without = Path(shutil.copytree(atlas_dir, tmp_path / "without", ignore=shutil.ignore_patterns("sas")))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    listed = json.loads((release / "sas" / "MANIFEST.json").read_text())
    del listed[f"{EGFR}.json.gz"]
    (release / "sas" / "MANIFEST.json").write_text(json.dumps(listed))
    damaged = release / "sas" / f"{PTGS2}.json.gz"
    damaged.write_bytes(damaged.read_bytes()[:-1] + b"x")
    with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
        assert Atlas(without)._sas(ABL1) is None
        atlas = Atlas(release)
        assert atlas._sas(EGFR) is None and atlas._sas(ABL1)["protein"] == ABL1
        assert not caplog.text
        assert atlas._sas(PTGS2) is None
        assert f"{PTGS2}: the page is shown without P2Rank's SAS points" in caplog.text and "damaged" in caplog.text
        caplog.clear()
        sha256, size = atlas._sas_files[f"{ABL1}.json.gz"]
        for wrong in ([sha256, None], [sha256, str(size)], [sha256], {"sha256": sha256}):  # a damaged listing
            atlas._sas_files[f"{ABL1}.json.gz"] = wrong
            assert atlas._sas(ABL1) is None
            assert "does not list it as [sha256, bytes]" in caplog.text
            caplog.clear()
        remote = Atlas(atlas_server.url)
        _no_network(monkeypatch)
        assert remote._sas(ABL1) is None and "cannot reach the PocketVina Atlas" in caplog.text
        assert remote._sas_files is None  # asked again next time


def test_chembl_records(atlas_dir, atlas_server, tmp_path, monkeypatch):
    """What the ChEMBL values of a page need (Atlas._chembl): the protein's records of "ChEMBL Complete"
    (chembl_any/proteins/<U>.json.gz) and the strict values' links (chembl_any/strict.json.gz), from a folder and over
    HTTP. Over HTTP the files and chembl_any/MANIFEST.json are kept in the cache: later they are read from there (the
    listing checked with the server once per Atlas, HTTP 304), also without network."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    expected = (_chembl_json(atlas_dir, f"proteins/{ABL1}.json.gz"), _chembl_json(atlas_dir, "strict.json.gz"))
    assert expected[0]["uniprot"] == ABL1 and DRUG in expected[0]["d"] and DRUG in expected[1]["pvl"]
    assert list(expected[1]) == ["pvl", "xo", "moa"]
    assert Atlas(atlas_dir)._chembl(ABL1) == expected
    remote = Atlas(atlas_server.url)
    asked = len(atlas_server.requests)
    assert remote._chembl(ABL1) == expected
    assert [r for r, _ in atlas_server.requests[asked:]] == [
        "chembl_any/MANIFEST.json", f"chembl_any/proteins/{ABL1}.json.gz", "chembl_any/strict.json.gz"]
    asked = len(atlas_server.requests)
    assert remote._chembl(EGFR)[0] == _chembl_json(atlas_dir, f"proteins/{EGFR}.json.gz")  # the rest read once
    assert [r for r, _ in atlas_server.requests[asked:]] == [f"chembl_any/proteins/{EGFR}.json.gz"]
    asked = len(atlas_server.requests)
    again = Atlas(atlas_server.url)
    assert again._chembl(ABL1) == expected
    # meta.json is never read from the cache (it names the release); the listing is checked: not changed
    assert [r for r, _ in atlas_server.requests[asked:]] == ["meta.json", "chembl_any/MANIFEST.json"]
    assert atlas_server.conditional == [("chembl_any/MANIFEST.json", 304)]
    again._chembl_files = again._chembl_strict = None  # read again: from the cache
    _no_network(monkeypatch)
    assert again._chembl(ABL1) == expected


def test_chembl_records_missing_or_unreadable(atlas_dir, atlas_server, tmp_path, monkeypatch, caplog):
    """No "ChEMBL Complete" (None) for a release without chembl_any/ (an older release or download); no records
    ({"d": {}}, every value "–", as on the site) for a protein chembl_any/MANIFEST.json does not list. None and a
    warning when they cannot be read: a damaged file or listing, a file of another protein or not as the release writes
    it, no network (then asked again next time). The strict values' links: empty without strict.json.gz; empty and a
    warning when it cannot be read, while "ChEMBL Complete" is still read."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    nothing = {"pvl": {}, "xo": {}}
    strict = _chembl_json(atlas_dir, "strict.json.gz")
    without = Path(shutil.copytree(atlas_dir, tmp_path / "without", ignore=shutil.ignore_patterns("chembl_any")))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    listing = release / "chembl_any" / "MANIFEST.json"
    listed = json.loads(listing.read_text())
    listing.write_text(json.dumps({k: v for k, v in listed.items() if k != f"proteins/{EGFR}.json.gz"}))
    damaged = release / "chembl_any" / "proteins" / f"{PTGS2}.json.gz"
    damaged.write_bytes(damaged.read_bytes()[:-1] + b"x")
    with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
        assert Atlas(without)._chembl(ABL1) == (None, nothing)
        atlas = Atlas(release)
        assert atlas._chembl(EGFR) == ({"d": {}}, strict)
        assert atlas._chembl(ABL1)[0]["uniprot"] == ABL1
        assert not caplog.text
        assert atlas._chembl(PTGS2) == (None, strict)
        assert f'{PTGS2}: the page is shown without the "ChEMBL Complete" values' in caplog.text
        assert "damaged" in caplog.text
        caplog.clear()
        good = atlas._chembl_files[f"proteins/{ABL1}.json.gz"]
        for wrong in ([good[0], None], [good[0], str(good[1])], [good[0]], {"sha256": good[0]}):  # a damaged listing
            atlas._chembl_files[f"proteins/{ABL1}.json.gz"] = wrong
            assert atlas._chembl(ABL1) == (None, strict)
            assert "does not list proteins/" in caplog.text
            caplog.clear()
        atlas._chembl_files[f"proteins/{ABL1}.json.gz"] = good
        record = _chembl_json(atlas_dir, f"proteins/{ABL1}.json.gz")
        some = next(iter(record["d"]))
        for k, bad in enumerate([dict(record, uniprot=EGFR), dict(record, d=[]), dict(record, tc=5),
                                 dict(record, xr="x"), dict(record, d={some: dict(record["d"][some], a="3")}),
                                 dict(record, d={some: dict(record["d"][some], p=True)}),
                                 dict(record, d={some: dict(record["d"][some], par="CHEMBL1")}),
                                 dict(record, d={some: dict(record["d"][some], mn=[])}), ["not", "a", "dict"],
                                 dict(record, d={some: dict(record["d"][some], p=1e300)}),
                                 dict(record, d={some: dict(record["d"][some], p=10 ** 400)})]):
            folder = Path(shutil.copytree(atlas_dir, tmp_path / f"bad{k}"))
            data = gzip.compress(json.dumps(bad).encode(), mtime=0)
            (folder / "chembl_any" / "proteins" / f"{ABL1}.json.gz").write_bytes(data)
            entries = dict(listed, **{f"proteins/{ABL1}.json.gz": [hashlib.sha256(data).hexdigest(), len(data)]})
            (folder / "chembl_any" / "MANIFEST.json").write_text(json.dumps(entries))
            assert Atlas(folder)._chembl(ABL1) == (None, strict), k
            assert "cannot be read" in caplog.text
            assert ("not the file of" if k in (0, 8) else "not as the release") in caplog.text
            caplog.clear()
        for k, text in enumerate(["{not json", "[]"]):  # chembl_any/MANIFEST.json
            folder = Path(shutil.copytree(atlas_dir, tmp_path / f"listing{k}"))
            (folder / "chembl_any" / "MANIFEST.json").write_text(text)
            atlas = Atlas(folder)
            assert atlas._chembl(ABL1) == (None, nothing) and "cannot be read" in caplog.text
            assert atlas._chembl_files is None  # read again next time
            caplog.clear()
        folder = Path(shutil.copytree(atlas_dir, tmp_path / "nostrict"))  # strict.json.gz: not listed, damaged, wrong
        (folder / "chembl_any" / "MANIFEST.json").write_text(
            json.dumps({k: v for k, v in listed.items() if k != "strict.json.gz"}))
        assert Atlas(folder)._chembl(ABL1) == (record, nothing) and not caplog.text
        for k, bad in enumerate([b"not gzip", gzip.compress(b'{"pvl": {"PVL1": "x"}}'), gzip.compress(b"[1]")]):
            (folder / "chembl_any" / "strict.json.gz").write_bytes(bad)
            (folder / "chembl_any" / "MANIFEST.json").write_text(json.dumps(dict(listed, **{
                "strict.json.gz": [hashlib.sha256(bad).hexdigest(), len(bad)]})))
            atlas = Atlas(folder)
            assert atlas._chembl(ABL1) == (record, nothing) and atlas._chembl_strict is None, k
            assert 'the "ChEMBL Strict" values of the page have fewer links and leave out' in caplog.text
            caplog.clear()
        remote = Atlas(atlas_server.url)
        _no_network(monkeypatch)
        assert remote._chembl(ABL1) == (None, nothing) and "cannot reach the PocketVina Atlas" in caplog.text
        assert remote._chembl_files is None  # asked again next time


def _make_chembl_again(release: Path) -> dict:
    """chembl_any/ of ``release`` made again, as the site may do: other records (more activity records, one more
    "other drug") and a new chembl_any/MANIFEST.json. Returns {path in chembl_any/: parsed new file}."""
    folder = release / "chembl_any"
    listed, new = json.loads((folder / "MANIFEST.json").read_text()), {}
    for rel in listed:
        new[rel] = data = _chembl_json(release, rel)
        if rel == "strict.json.gz":
            data["xo"]["PVL99999"] = "ANOTHER DRUG"
        else:
            for rec in data["d"].values():
                rec["a"] += 1000
        raw = gzip.compress(json.dumps(data).encode(), mtime=0)
        (folder / rel).write_bytes(raw)
        listed[rel] = [hashlib.sha256(raw).hexdigest(), len(raw)]
    (folder / "MANIFEST.json").write_text(json.dumps(listed, indent=0))
    return new


def test_chembl_records_made_again(atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """The site may make chembl_any/ again after the cache got its listing and files (kept: the release is final). A
    new Atlas checks the listing with the server before it reads it: its page has the new records, with no warning,
    without download() or clearing the cache, and the listing is asked for once. An Atlas that read the old listing
    before (a session still open) reads what it read until a file does not match its listing: then the listing is
    fetched again, once per Atlas, and the file is read again."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    stale = Atlas(server.url)
    old = stale._chembl(ABL1)  # the cache gets the listing, strict.json.gz and ABL1's records
    new = _make_chembl_again(release)
    with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
        asked = len(server.requests)
        assert stale._chembl(ABL1) == old and len(server.requests) == asked  # read before: from memory and the cache
        atlas = Atlas(server.url)
        asked = len(server.requests)
        page = atlas._page(EGFR, None)["proteins"][0]
        local = Atlas(release)._page(EGFR, None)["proteins"][0]
        assert page["topColumns"] == local["topColumns"] and len(page["topColumns"]) == 2
        assert [r["topValues"] for r in page["rows"]] == [r["topValues"] for r in local["rows"]]
        assert atlas._chembl(EGFR) == (new[f"proteins/{EGFR}.json.gz"], new["strict.json.gz"])
        assert atlas._chembl(ABL1) == (new[f"proteins/{ABL1}.json.gz"], new["strict.json.gz"])
        assert [r for r, _ in server.requests[asked:]].count("chembl_any/MANIFEST.json") == 1
        assert server.conditional == [("chembl_any/MANIFEST.json", 200)]  # checked: the server has another one
        # the session still open: ABL1's file in the cache is the new one now, which its listing does not list, so it
        # fetches the listing again (a plain request) and reads it again
        asked = len(server.requests)
        assert stale._chembl(ABL1) == (new[f"proteins/{ABL1}.json.gz"], new["strict.json.gz"])
        assert server.conditional[1:] == [] and "chembl_any/MANIFEST.json" in [r for r, _ in server.requests[asked:]]
    assert not caplog.text


def _make_sas_again(release: Path) -> dict:
    """sas/ of ``release`` made again, as the site may do: other points (one fewer in no pocket) and a new
    sas/MANIFEST.json. Returns {UniProt: the new points (sas.decode)}."""
    folder = release / "sas"
    listed, new = json.loads((folder / "MANIFEST.json").read_text()), {}
    for name in listed:
        data = json.loads(gzip.decompress((folder / name).read_bytes()))
        data["rest"] = data["rest"][:-1]
        raw = sas.encode(data)
        (folder / name).write_bytes(raw)
        listed[name] = [hashlib.sha256(raw).hexdigest(), len(raw)]
        new[name[:-len(".json.gz")]] = sas.decode(raw)
    (folder / "MANIFEST.json").write_text(json.dumps(listed, indent=0))
    return new


def test_sas_points_made_again(atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """As test_chembl_records_made_again, for sas/ and sas/MANIFEST.json: the page of a new Atlas has the new SAS
    points, with no warning, and the listing is asked for once."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    stale = Atlas(server.url)
    old = stale._sas(ABL1)  # the cache gets sas/MANIFEST.json and ABL1's points
    new = _make_sas_again(release)
    with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
        asked = len(server.requests)
        assert stale._sas(ABL1) == old and len(server.requests) == asked  # read before: from memory and the cache
        atlas = Atlas(server.url)
        asked = len(server.requests)
        page = atlas._page(EGFR, None)["proteins"][0]
        assert page["sas"] == Atlas(release)._page(EGFR, None)["proteins"][0]["sas"]
        assert atlas._sas(EGFR) == new[EGFR] and atlas._sas(ABL1) == new[ABL1]
        assert [r for r, _ in server.requests[asked:]].count("sas/MANIFEST.json") == 1
        assert server.conditional == [("sas/MANIFEST.json", 200)]
        asked = len(server.requests)
        assert stale._sas(ABL1) == new[ABL1]  # its listing is fetched again (a plain request) and read again
        assert server.conditional[1:] == [] and "sas/MANIFEST.json" in [r for r, _ in server.requests[asked:]]
    assert not caplog.text


def test_download_after_listings_made_again(atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """download() fetches sas/MANIFEST.json and chembl_any/MANIFEST.json from the server first (not the ones in the
    cache, which may be older), each once: after the site made sas/ and chembl_any/ again its copy is as the new
    listings say."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    atlas = Atlas(server.url)
    atlas._sas(ABL1), atlas._chembl(ABL1)  # the cache gets the listings
    _make_sas_again(release)
    _make_chembl_again(release)
    with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
        downloading = Atlas(server.url)
        asked = len(server.requests)
        out = Path(downloading.download(["ABL1", "EGFR"], tmp_path / "out"))
    assert _listings_asked(server, asked) == sorted(LISTINGS)  # fetched again, then not checked again
    for folder, names in (("sas", [f"{ABL1}.json.gz", f"{EGFR}.json.gz"]),
                          ("chembl_any", [f"proteins/{ABL1}.json.gz", f"proteins/{EGFR}.json.gz", "strict.json.gz"])):
        listed = json.loads((release / folder / "MANIFEST.json").read_text())
        copied = json.loads((out / folder / "MANIFEST.json").read_text())
        assert copied == {rel: listed[rel] for rel in names}
        for rel in copied:
            assert (out / folder / rel).read_bytes() == (release / folder / rel).read_bytes(), rel
    assert not caplog.text


def test_chembl_records_damaged_after_the_listing_is_fetched_again(atlas_dir, tmp_path, http_server, monkeypatch,
                                                                    caplog):
    """A file that does not match the listing fetched again either is damaged: a warning, as before, and the listing
    is fetched again at most once per Atlas (never a loop; a plain request, also after the check got the server's
    listing). A folder is read as before (nothing is fetched). Without network a cache with the old listing is
    read as before."""
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    for cache in ("offline", "cache"):  # each cache gets the listing, strict.json.gz and ABL1's records
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / cache))
        old = Atlas(server.url)._chembl(ABL1)
    new = _make_chembl_again(release)
    for uniprot in (EGFR, PTGS2):  # damaged on the server: not the files the new listing lists
        damaged = release / "chembl_any" / "proteins" / f"{uniprot}.json.gz"
        damaged.write_bytes(damaged.read_bytes()[:-1] + b"x")
    with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
        atlas = Atlas(server.url)
        asked = len(server.requests)
        assert atlas._chembl(EGFR) == (None, new["strict.json.gz"]) and "damaged" in caplog.text
        assert atlas._chembl(PTGS2) == (None, new["strict.json.gz"])
        assert atlas._chembl(EGFR) == (None, new["strict.json.gz"])
        assert [r for r, _ in server.requests[asked:]].count("chembl_any/MANIFEST.json") == 2
        assert server.conditional == [("chembl_any/MANIFEST.json", 200)]  # the check; then a plain request
        assert atlas._chembl(ABL1) == (new[f"proteins/{ABL1}.json.gz"], new["strict.json.gz"])
        caplog.clear()
        folder = Atlas(release)
        assert folder._chembl(EGFR) == (None, new["strict.json.gz"]) and "damaged" in caplog.text
        assert not folder._fetch_listing_again("chembl_any/MANIFEST.json") and not folder._fetched_again
        caplog.clear()
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "offline"))
        offline = Atlas(server.url)
        assert offline.release == "v1.1"  # meta.json is never read from the cache
        _no_network(monkeypatch)
        assert offline._chembl(ABL1) == old and not caplog.text  # in the cache, as the cached listing says
        assert offline._chembl(EGFR) == (None, old[1]) and "cannot reach the PocketVina Atlas" in caplog.text
        assert not offline._fetched_again  # no network is not a file that does not match: no listing is fetched


def test_sas_points_damaged_after_the_listing_is_fetched_again(atlas_dir, tmp_path, http_server, monkeypatch,
                                                               caplog):
    """As test_chembl_records_damaged_after_the_listing_is_fetched_again, for sas/."""
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    for cache in ("offline", "cache"):  # each cache gets sas/MANIFEST.json and ABL1's points
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / cache))
        old = Atlas(server.url)._sas(ABL1)
    new = _make_sas_again(release)
    for uniprot in (EGFR, PTGS2):  # damaged on the server: not the files the new listing lists
        damaged = release / "sas" / f"{uniprot}.json.gz"
        damaged.write_bytes(damaged.read_bytes()[:-1] + b"x")
    with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
        atlas = Atlas(server.url)
        asked = len(server.requests)
        assert atlas._sas(EGFR) is None and "damaged" in caplog.text
        assert atlas._sas(PTGS2) is None and atlas._sas(EGFR) is None
        assert [r for r, _ in server.requests[asked:]].count("sas/MANIFEST.json") == 2
        assert server.conditional == [("sas/MANIFEST.json", 200)]  # the check; then a plain request
        assert atlas._sas(ABL1) == new[ABL1]
        caplog.clear()
        folder = Atlas(release)
        assert folder._sas(EGFR) is None and "damaged" in caplog.text
        assert not folder._fetch_listing_again("sas/MANIFEST.json") and not folder._fetched_again
        caplog.clear()
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "offline"))
        offline = Atlas(server.url)
        assert offline.release == "v1.1"  # meta.json is never read from the cache
        _no_network(monkeypatch)
        assert offline._sas(ABL1) == old and not caplog.text  # in the cache, as the cached listing says
        assert offline._sas(EGFR) is None and "cannot reach the PocketVina Atlas" in caplog.text
        assert not offline._fetched_again  # no network is not a file that does not match: no listing is fetched


def test_listings_fetched_again_keep_the_cache_on_a_bad_answer(atlas_dir, tmp_path, http_server, monkeypatch,
                                                                caplog):
    """A listing fetched again that is not a listing (e.g. a Wi-Fi login page) never replaces the one in the cache: a
    warning, as for a damaged file, and the files the cached listing lists are still read, in a later session too."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    first = Atlas(server.url)
    old = first._sas(ABL1), first._chembl(ABL1)  # the cache gets both listings and ABL1's files
    page = b"<html><body>Log in to use this network</body></html>"
    for rel in ("sas/MANIFEST.json", "chembl_any/MANIFEST.json", f"sas/{EGFR}.json.gz",
                f"chembl_any/proteins/{EGFR}.json.gz"):
        (release / rel).write_bytes(page)
    with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
        atlas = Atlas(server.url)
        assert atlas._sas(EGFR) is None and atlas._chembl(EGFR) == (None, old[1][1])
        assert caplog.text.count("is not a listing of files (JSON)") == 2
        assert atlas._fetched_again == {"sas/MANIFEST.json", "chembl_any/MANIFEST.json"}
        caplog.clear()
        assert (atlas._sas(ABL1), atlas._chembl(ABL1)) == old
        later = Atlas(server.url)
        assert (later._sas(ABL1), later._chembl(ABL1)) == old
    assert not caplog.text
    for rel in ("sas/MANIFEST.json", "chembl_any/MANIFEST.json"):
        assert next((tmp_path / "cache").rglob(rel)).read_bytes() == (atlas_dir / rel).read_bytes()
    assert not [p for p in (tmp_path / "cache").rglob("*") if ".tmp-" in p.name or ".fetched-" in p.name]


LISTINGS = ("sas/MANIFEST.json", "chembl_any/MANIFEST.json")


def _old_cache(url: str, release: Path) -> None:
    """The cache for ``url`` as a PocketVina older than 0.4.6 left it: the listings of sas/ and chembl_any/ and ABL1's
    files there, as ``release`` has them now, without the headers the server sent with the listings."""
    source = Source(url)
    for rel in LISTINGS + (f"sas/{ABL1}.json.gz", f"chembl_any/proteins/{ABL1}.json.gz", "chembl_any/strict.json.gz"):
        cached = Path(source._cache_path(rel)[0])
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(release / rel, cached)


def _listings_asked(server, since: int) -> list:
    return sorted(rel for rel, _ in server.requests[since:] if rel in LISTINGS)


@pytest.mark.parametrize("etags", [True, False])
def test_listings_checked_once_per_atlas(etags, atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """Over HTTP the listings of sas/ and chembl_any/ in the cache are checked with the server once per Atlas, before
    they are first read (not again for its next pages): a conditional request with the ETag and Last-Modified the
    server sent with the copy (kept next to it in <listing>.headers.json with the copy's sha256; a server without
    ETags: Last-Modified only). HTTP 304 (not changed) has no body: the copy is read as it is, not written again. A
    cache of PocketVina 0.4.5 or older (listings without their headers) gets one plain request, which keeps them. A
    folder asks nothing."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    server.etags = etags
    _old_cache(server.url, release)
    local = Atlas(release)
    expected = local._sas(ABL1), local._chembl(ABL1)
    assert not local._checked and not local._fetched_again  # a folder
    written = {}
    with caplog.at_level(logging.INFO, logger="pocketvina"):
        for k in range(3):
            atlas = Atlas(server.url)
            asked = len(server.requests)
            for n in range(2):  # two pages: the listings are asked for once
                atlas.show(ABL1, path=tmp_path / f"page{k}{n}.html")
            assert (atlas._sas(ABL1), atlas._chembl(ABL1)) == expected
            assert _listings_asked(server, asked) == sorted(LISTINGS) and atlas._checked == set(LISTINGS)
            # the first Atlas: plain requests (no headers kept yet); then conditional ones, the listings not changed
            assert sorted(server.conditional) == sorted((rel, 304) for rel in LISTINGS for _ in range(k))
            for rel in LISTINGS:
                cached = next((tmp_path / "cache").rglob(rel))
                headers = json.loads(cached.with_name(cached.name + ".headers.json").read_text())
                assert headers["sha256"] == hashlib.sha256(cached.read_bytes()).hexdigest()
                assert ("ETag" in headers) == etags and "Last-Modified" in headers
                assert cached.read_bytes() == (release / rel).read_bytes()
                assert written.setdefault(rel, cached.stat().st_mtime_ns) == cached.stat().st_mtime_ns
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING] and "cannot be read" not in caplog.text
    assert not [p for p in (tmp_path / "cache").rglob("*") if ".tmp-" in p.name]


@pytest.mark.parametrize("route", ["direct", "proxy"])
def test_listings_made_again_on_the_server(route, atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """An old cache (the listings and files of sas/ and chembl_any/ as an older PocketVina left them) after the site
    made both folders again: a new Atlas has the new SAS points and ChEMBL values, with no warning, without download()
    or clearing the cache. After the next change its conditional requests get the new listings (HTTP 200), then
    HTTP 304, which is not an error: no info line, and a listing fetched again (as download() does) gets it too. The
    same through a proxy (urllib, which raises HTTP 304 as an error)."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    url = server.url
    if route == "proxy":
        for name in ("no_proxy", "NO_PROXY", "HTTP_PROXY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("http_proxy", server.url)  # the test server answers a proxy request for any host
        url = "http://pocketvina-atlas.invalid/"
    _old_cache(url, release)
    with caplog.at_level(logging.INFO, logger="pocketvina"):
        for k, answers in enumerate([[], [200, 200], [304, 304]]):
            if k < 2:  # the site makes sas/ and chembl_any/ again
                points, chembl = _make_sas_again(release), _make_chembl_again(release)
            atlas = Atlas(url)
            assert atlas.source._proxied == (route == "proxy")
            asked, checked = len(server.requests), len(server.conditional)
            assert atlas._sas(ABL1) == points[ABL1] and atlas._sas(EGFR) == points[EGFR]
            assert atlas._chembl(ABL1) == (chembl[f"proteins/{ABL1}.json.gz"], chembl["strict.json.gz"])
            assert _listings_asked(server, asked) == sorted(LISTINGS)
            assert sorted(status for _, status in server.conditional[checked:]) == answers
        assert atlas._fetch_listing_again(LISTINGS[0])  # as download() does: HTTP 304, then read again
        assert server.conditional[-1] == (LISTINGS[0], 304) and atlas._sas(EGFR) == points[EGFR]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "cannot be" not in caplog.text and "HTTP 304" not in caplog.text


def test_listings_checked_without_an_answer(atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """When the listings in the cache cannot be checked with the server (no network, a server error, an answer that is
    not a listing: a Wi-Fi login page, or JSON that lists no file as [sha256, bytes]), the copies are read as they are:
    no exception and no warning (an info line in the log), and that Atlas does not ask again when it reads them again;
    a later Atlas asks again. An answer that is not a listing never replaces a copy."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(atlas_source, "RETRY_DELAYS", (0, 0, 0))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    first = Atlas(server.url)
    old = first._sas(ABL1), first._chembl(ABL1)  # the cache gets the listings (with their headers) and ABL1's files
    cached = {rel: next((tmp_path / "cache").rglob(rel)) for rel in LISTINGS}
    kept = {rel: (path.read_bytes(), path.with_name(path.name + ".headers.json").read_bytes())
            for rel, path in cached.items()}
    points, chembl = _make_sas_again(release), _make_chembl_again(release)
    good = {rel: (release / rel).read_bytes() for rel in LISTINGS}
    calls = []

    def no_network(m):
        def fail(url, *args, **kwargs):
            calls.append(url)
            raise OSError("Network is unreachable")

        m.setattr(atlas_source, "_fetch", fail)
        m.setattr(atlas_source, "_urlopen", fail)

    def server_error(m):
        server.fail = 2 * (len(atlas_source.RETRY_DELAYS) + 1)  # every try of both listings: HTTP 503

    def login_page(m):
        for rel in LISTINGS:
            (release / rel).write_bytes(b"<html><body>Log in to use this network</body></html>")

    def json_error(m):  # e.g. a proxy or portal that answers in JSON
        for rel in LISTINGS:
            (release / rel).write_text(json.dumps({"error": "busy", "retry": [1, 2]}))

    for cause, why in ((no_network, "Network is unreachable"), (server_error, "HTTP 503"),
                       (login_page, "is not a listing of files"), (json_error, "is not a listing of files")):
        atlas = Atlas(server.url)
        with monkeypatch.context() as m, caplog.at_level(logging.INFO, logger="pocketvina"):
            cause(m)
            assert (atlas._sas(ABL1), atlas._chembl(ABL1)) == old
            assert atlas._checked == set(LISTINGS) and not atlas._fetched_again
            asked = len(server.requests), len(calls)
            atlas._sas_files = atlas._chembl_files = atlas._chembl_strict = None  # read again: not asked again
            assert (atlas._sas(ABL1), atlas._chembl(ABL1)) == old
            assert (len(server.requests), len(calls)) == asked
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING], cause.__name__
        assert caplog.text.count("cannot be checked with the server now") == 2 and why in caplog.text, cause.__name__
        caplog.clear()
        assert server.fail == 0  # the server error: every try was answered with it
        for rel, path in cached.items():
            assert (path.read_bytes(), path.with_name(path.name + ".headers.json").read_bytes()) == kept[rel]
            (release / rel).write_bytes(good[rel])
    assert len(calls) == 2 * (len(atlas_source.RETRY_DELAYS) + 1)  # no network: each listing tried, by one Atlas
    later = Atlas(server.url)  # the server answers again: the new listings
    assert later._sas(ABL1) == points[ABL1]
    assert later._chembl(ABL1) == (chembl[f"proteins/{ABL1}.json.gz"], chembl["strict.json.gz"])
    assert sorted(server.conditional[-2:]) == sorted((rel, 200) for rel in LISTINGS)


def test_listing_headers_of_another_copy_are_not_sent(atlas_dir, tmp_path, http_server, monkeypatch):
    """The headers kept next to a listing are sent only with the copy they came with (its sha256): when another copy
    was written there since (e.g. an older listing, by PocketVina 0.4.5 sharing the cache), the check is a plain
    request, so the server's listing replaces it and the page has the server's SAS points."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    old = (release / LISTINGS[0]).read_bytes()
    points = _make_sas_again(release)
    Atlas(server.url)._sas_listing()  # the cache gets the new listing and its headers
    cached = next((tmp_path / "cache").rglob(LISTINGS[0]))
    cached.write_bytes(old)  # another copy, without headers of its own
    checked = len(server.conditional)
    atlas = Atlas(server.url)
    assert atlas._sas(EGFR) == points[EGFR] and atlas._sas(ABL1) == points[ABL1]
    assert server.conditional[checked:] == []  # a plain request: the headers were not sent
    assert cached.read_bytes() == (release / LISTINGS[0]).read_bytes()
    headers = json.loads(cached.with_name(cached.name + ".headers.json").read_text())
    assert headers["sha256"] == hashlib.sha256(cached.read_bytes()).hexdigest()


def test_listing_fetched_again_after_a_wrong_304(atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """A server without ETags answers If-Modified-Since by the file's time, so a new listing with an older time (e.g.
    copied with its times kept) gets HTTP 304 and the check keeps the older listing. A file that does not match it has
    the listing fetched again with a plain request, which a wrong 304 cannot stop: the page has the new SAS points,
    with no warning, in the same Atlas."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    server.etags = False
    listing = release / LISTINGS[0]
    before = listing.stat().st_mtime
    Atlas(server.url)._sas(ABL1)  # the cache gets the listing, with its Last-Modified
    points = _make_sas_again(release)
    os.utime(listing, (before - 100, before - 100))  # the new listing is older than the copy in the cache
    with caplog.at_level(logging.WARNING, logger="pocketvina"):
        atlas = Atlas(server.url)
        assert atlas._sas(EGFR) == points[EGFR]  # not in the cache: does not match the listing the check kept
        assert atlas._sas(ABL1) == points[ABL1]
    assert server.conditional == [(LISTINGS[0], 304)] and not caplog.text
    assert next((tmp_path / "cache").rglob(LISTINGS[0])).read_bytes() == listing.read_bytes()


def test_listing_fetched_again_after_no_network(atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """A listing that could not be fetched again (no network, e.g. during download()) may be fetched again later by the
    same Atlas: a download() once the network is back copies the files as the server's listing lists them now."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    server = http_server(release)
    atlas = Atlas(server.url)
    atlas._sas(ABL1)  # this Atlas has read the listing
    points = _make_sas_again(release)
    with monkeypatch.context() as m:
        _no_network(m)
        with pytest.raises(OSError, match="cannot reach the PocketVina Atlas"):
            atlas._fetch_listing_again(LISTINGS[0])
    assert LISTINGS[0] not in atlas._fetched_again
    caplog.clear()  # the tries without network (info lines)
    with caplog.at_level(logging.WARNING, logger="pocketvina"):
        out = Path(atlas.download(["ABL1"], tmp_path / "out"))
    assert Atlas(out)._sas(ABL1) == points[ABL1] and not caplog.text


def _put_strict(release: Path, data: dict) -> dict:
    """chembl_any/strict.json.gz of ``release`` written as ``data``, with its entry in chembl_any/MANIFEST.json."""
    raw = gzip.compress(json.dumps(data).encode(), mtime=0)
    (release / "chembl_any" / "strict.json.gz").write_bytes(raw)
    listing = release / "chembl_any" / "MANIFEST.json"
    listing.write_text(json.dumps(dict(json.loads(listing.read_text()), **{
        "strict.json.gz": [hashlib.sha256(raw).hexdigest(), len(raw)]}), indent=0))
    return data


def test_chembl_strict_without_moa(atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """chembl_any/strict.json.gz made before the site added "moa" (19 Nov 2026), in a download made then or in the
    cache: the strict values are the release's flags, with no warning. Over HTTP such a copy in the cache may be older
    than the server's file: the listing is checked with the server before it is read, once per Atlas, and the page
    gets "moa". Without network, or when the server has no "moa" either, the copy is read as it is; after no network,
    download() on the same Atlas still fetches the listing. A folder is read as it is (nothing is fetched)."""
    new = _chembl_json(atlas_dir, "strict.json.gz")
    assert new["moa"]
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    old = _put_strict(release, {k: v for k, v in new.items() if k != "moa"})
    server = http_server(release)
    with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
        folder = Atlas(release)
        assert folder._chembl(ABL1)[1] == old and not folder._fetched_again
        for cache in ("offline", "cache"):  # each cache gets the listing and strict.json.gz without "moa"
            monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / cache))
            atlas = Atlas(server.url)
            asked = len(server.requests)
            assert atlas._chembl(ABL1)[1] == old and atlas._chembl(EGFR)[1] == old  # the server has none either
            assert [r for r, _ in server.requests[asked:]].count("chembl_any/MANIFEST.json") == 1  # read once
        # a listing answer that is not JSON (a Wi-Fi login page) never replaces the cache: the copy is read as it is
        listing = release / "chembl_any" / "MANIFEST.json"
        good = listing.read_bytes()
        listing.write_bytes(b"<html><body>Log in to use this network</body></html>")
        cached = next((tmp_path / "cache").rglob("chembl_any/MANIFEST.json"))
        kept = cached.read_bytes()
        assert Atlas(server.url)._chembl(ABL1)[1] == old and cached.read_bytes() == kept
        listing.write_bytes(good)
        _put_strict(release, new)  # the site adds "moa"
        atlas = Atlas(server.url)
        asked = len(server.requests)
        assert atlas._chembl(ABL1)[1] == new
        assert [r for r, _ in server.requests[asked:]] == ["chembl_any/MANIFEST.json", "chembl_any/strict.json.gz"]
        tip = atlas._page(EGFR, None)["proteins"][0]["topColumns"][0]["tip"]
        assert "(mechanisms also from its salt forms)" in tip
        assert Atlas(server.url)._chembl(ABL1)[1] == new  # from the cache now
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "offline"))
        offline = Atlas(server.url)
        with monkeypatch.context() as m:
            _no_network(m)
            assert offline._chembl(ABL1)[1] == old
        # the network is back: a download with the same Atlas fetches the listing again and copies the new file
        out = Path(offline.download(["ABL1"], tmp_path / "out"))
        assert Atlas(out)._chembl(ABL1)[1] == new
    assert not caplog.text


@pytest.mark.parametrize("source", ["local", "remote"])
def test_download_into_one_made_before_moa(source, atlas_dir, tmp_path, http_server, monkeypatch, caplog):
    """A download made before the site added "moa" to chembl_any/strict.json.gz (PocketVina 0.4.3 before 19 Nov 2026)
    has the release's flags; a new download into the folder replaces its strict.json.gz (its sha256 changed), so its
    pages count the mechanisms on salt forms. Over HTTP also with the older listing in the cache."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    new = _chembl_json(atlas_dir, "strict.json.gz")
    release = Path(shutil.copytree(atlas_dir, tmp_path / "v1.1"))
    old = _put_strict(release, {k: v for k, v in new.items() if k != "moa"})
    location = release if source == "local" else http_server(release).url
    out = Path(Atlas(location).download(["ABL1"], tmp_path / "out"))
    assert Atlas(out)._chembl(ABL1)[1] == old
    _put_strict(release, new)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pocketvina.atlas"):
        Atlas(location).download(["EGFR"], out)
    assert not caplog.text
    strict = "chembl_any/strict.json.gz"
    assert (out / strict).read_bytes() == (release / strict).read_bytes()
    listed = json.loads((release / "chembl_any" / "MANIFEST.json").read_text())
    assert json.loads((out / "chembl_any" / "MANIFEST.json").read_text()) == {
        rel: listed[rel] for rel in (f"proteins/{ABL1}.json.gz", f"proteins/{EGFR}.json.gz", "strict.json.gz")}
    assert Atlas(out)._chembl(ABL1)[1] == new


# --- the public Atlas ------------------------------------------------------------------------------------------

@pytest.mark.network
def test_public_atlas(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("POCKETVINA_ATLAS", raising=False)
    atlas = Atlas()
    assert atlas.source.location == ATLAS_URL and atlas.release == "v1.1"
    assert atlas.protein("ABL1").uniprot == ABL1
    scores = atlas.best_scores(protein="ABL1", top=5)
    assert list(scores.columns) == list(BEST_SCORES_COLUMNS) and len(scores) == 5
    top, blocks = atlas._pair_poses(ABL1, DRUG)
    assert top and all(block.startswith("MODEL") for block in blocks.values())
    points = atlas._sas(ABL1)  # checked against sas/MANIFEST.json
    assert points["protein"] == ABL1 and list(points["pockets"]) == ["1", "2", "3", "4", "5"] and points["rest"]
    records, strict = atlas._chembl(ABL1)  # checked against chembl_any/MANIFEST.json
    assert records["uniprot"] == ABL1 and DRUG in records["d"] and DRUG in strict["pvl"] and DRUG in records["xr"]
    assert any(ABL1 in proteins for proteins in strict["moa"].values())  # the mechanisms on salt forms (19 Nov 2026)


def test_example_script_runs(atlas_dir, tmp_path):
    """example/atlas.py runs as written, on the fixture release ($POCKETVINA_ATLAS), and writes its files into the
    working folder."""
    root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "POCKETVINA_ATLAS": str(atlas_dir), "XDG_CACHE_HOME": str(tmp_path / "cache"),
           "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))}
    run = subprocess.run([sys.executable, str(root / "example" / "atlas.py")], cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=300)
    # only the exception's type on failure: its message may hold drug names (and pytest would show `run` with its
    # output if the assertion named it)
    code, error = run.returncode, (run.stderr.strip().splitlines() or [""])[-1].split(":")[0]
    assert code == 0, error
    for name in ("ABL1_imatinib.html", f"PV_{ABL1}_{DRUG}_pocketall_top.sdf", f"PV_{ABL1}.pdb", "my_atlas/meta.json"):
        assert (tmp_path / name).is_file(), name
