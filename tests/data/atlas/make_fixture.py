"""Build the test fixture ``tests/data/atlas/v1.1/``: a tiny PocketVina Atlas release, laid out like the real one.

Rerun it from a copy of the release (read only; it is never written to):

    python tests/data/atlas/make_fixture.py [RELEASE_DIR] [OUT_DIR]

RELEASE_DIR defaults to the published release folder on this cluster, OUT_DIR to ``tests/data/atlas/v1.1``. The
output is deterministic (gzip without a timestamp, JSON written as it was read, sorted keys in the files this script
builds), so rerunning it on the same release gives the same bytes. It prints counts and sizes only.

What is in it: three proteins (PROTEINS below) with a few drugs each, picked so that the tests see every case: the
three best rows of each protein, the drug the documentation examples use, a desalted re-docking row (pf 1), a row
whose entry was not docked as the drug, rows with and without a stored PoseBusters result, a row without a stored
top pose, two drugs that share identifiers (for the "ambiguous identifier" error), and the rows of the same pairs in
``salt_forms/``.

Every file keeps the layout of the release, so the package reads the fixture exactly as it reads the real Atlas:

  meta.json                     copied, with "subset" describing this fixture
  MANIFEST.json                 only the fixture's files, with their own bytes and sha256
  proteins/<U>.json.gz          the picked rows; the poses / poses_redock reference and the rows' top_off and
  salt_forms/<U>.json.gz        top_len are rewritten for the rebuilt pose files
  compounds/<PVL>.json.gz       rows filtered to the fixture's proteins (same for <PVL>.all.json.gz)
  search/index.json.gz          rebuilt from the kept protein and drug rows, with the release's key rule
  search/<key>.json.gz
  ligand_ids.json, cidx.json    filtered to the fixture's catalogue numbers
  tables/*.parquet              the four small tables the package reads, filtered
  structures/<af_id>.pdb.gz     copied
  poses/<U>.poses               rewritten with posefile.Writer: same N, K and meta, only the fixture's records
  poses_redock/<U>.poses        (the others are empty), so every record index of the release stays valid
"""
import gzip
import hashlib
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from pocketvina.io import posefile  # noqa: E402

RELEASE = "/data/bimsbstatic/public/akalin/asarigun/pocketvina_atlas/data/v1.1"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v1.1")
# the drugs (PVL IDs) kept per protein: the three best rows, the documentation example, a desalted row (pf 1), a row
# not docked as the drug, and for the first protein two drugs that share identifiers
PROTEINS = {
    "P00519": ["PVL00898", "PVL01982", "PVL00250", "PVL00129", "PVL02203", "PVL00958", "PVL00257", "PVL02156"],
    "P00533": ["PVL01579", "PVL00609", "PVL00839", "PVL00129", "PVL02203", "PVL00958", "PVL02131"],
    "P35354": ["PVL00669", "PVL00327", "PVL00133", "PVL00129", "PVL02203", "PVL00958"],
}
SUBSET = ("test fixture of the PocketVina package: 3 proteins and a few drugs each, rows and pose files filtered "
          "(tests/data/atlas/make_fixture.py)")
ACC = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")
WORDS = re.compile(r"[\s,;()\[\]/]+")
TABLES = {"compound_entries": "catalog_id", "compound_entries_annotated": "catalog_id", "compounds": "pvl_id",
          "proteins": "uniprot"}


def read_json(path):
    with open(path, "rb") as f:
        data = f.read()
    return json.loads(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)


def write(rel, data: bytes):
    path = os.path.join(OUT, *rel.split("/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def write_json(rel, obj, sort_keys=False):
    text = json.dumps(obj, separators=(",", ":"), sort_keys=sort_keys, ensure_ascii=False).encode("utf-8")
    return write(rel, gzip.compress(text, compresslevel=9, mtime=0) if rel.endswith(".gz") else text)


def columns(data) -> dict:
    return {name: i for i, name in enumerate(data["columns"])}


def key_of(token):
    """The search key file of one token, as the release's builder picks it."""
    t = str(token).strip()
    if not t:
        return None
    if ACC.match(t.upper()):
        return "acc_" + t[:2].lower()
    tl = t.lower()
    if tl.startswith("ensg") and len(tl) >= 11:
        return "ensg" + tl[9:11]
    return tl[0] if tl[0].isalnum() else "_"


def words(text):
    return [w for w in WORDS.split(text if isinstance(text, str) else "") if w]


# --- pose files ------------------------------------------------------------------------------------------------

def rebuild_poses(release, ref, rows):
    """Write one pose file holding only the records of ``rows`` (dicts with cidx), with the original N, K and meta,
    and return the new reference and {cidx: (top_off, top_len)}."""
    source = os.path.join(release, *ref["file"].split("/"))
    offset = int(ref.get("cidx_offset") or 0)
    wanted = sorted({int(r["cidx"]) - offset for r in rows})
    with posefile.Reader(source) as reader:
        target = write(ref["file"], b"")
        writer = posefile.Writer(target, reader.N, reader.K, reader.meta)
        for i in wanted:
            off, length = reader.A[i]
            if length:
                writer.add_top(i, reader._read(off, length), 0)
        writer.begin_modes()
        for i in wanted:
            for k in range(1, reader.K + 1):
                off, length = reader.block_range(i, k)
                if length:
                    writer.add_block(i * reader.K + (k - 1), reader._read(off, length))
        writer.close()
    with posefile.Reader(target) as new:
        tops = {i + offset: new.A[i] for i in wanted}
        out = dict(ref, table_b_offset=new.tb, bytes=os.path.getsize(target))
    return out, tops


def retop(row, c, tops):
    """A row with its top_off and top_len taken from the rebuilt pose file."""
    row = list(row)
    offset, length = tops[int(row[c["cidx"]])]
    row[c["top_off"]], row[c["top_len"]] = int(offset), int(length)
    return row


# --- the fixture -----------------------------------------------------------------------------------------------

def is_fixture(folder) -> bool:
    """Whether a folder holds an earlier build of this fixture (only that is ever removed)."""
    try:
        return read_json(os.path.join(folder, "meta.json")).get("subset") == SUBSET
    except (OSError, ValueError):
        return False


def build(release):
    if os.path.isdir(OUT) and os.listdir(OUT):
        if not is_fixture(OUT):
            raise SystemExit(f"{OUT} is not empty and not this fixture: give a new or empty output folder")
        shutil.rmtree(OUT)
    meta = read_json(os.path.join(release, "meta.json"))
    entries = read_json(os.path.join(release, "ligand_ids.json"))["ids"]
    pvls = sorted({p for group in PROTEINS.values() for p in group})
    cats = sorted(c for c, pvl in entries.items() if pvl in pvls)
    pairs = {}  # (uniprot, cidx of a pf) -> rewritten (top_off, top_len), to fix the compound files
    af_ids = {}

    for uniprot, keep in PROTEINS.items():
        data = read_json(os.path.join(release, "proteins", f"{uniprot}.json.gz"))
        c = columns(data)
        rows = [r for r in data["rows"] if r[c["pvl"]] in keep]
        if len(rows) != len(keep):
            raise SystemExit(f"{uniprot}: only {len(rows)} of {len(keep)} drugs have a row")
        salt = read_json(os.path.join(release, "salt_forms", f"{uniprot}.json.gz"))
        if salt["columns"] != data["columns"]:
            raise SystemExit(f"{uniprot}: salt_forms has other columns than proteins")
        salt_rows = [r for r in salt["rows"] if entries.get(r[c["cat"]]) in pvls or r[c["pvl"]] in pvls]
        protein = dict(data)
        tops = {}
        for name in ("poses", "poses_redock"):
            ref = data.get(name)
            pf = 0 if name == "poses" else 1
            use = [{"cidx": r[c["cidx"]], "pf": r[c["pf"]]} for r in rows + salt_rows if r[c["pf"]] == pf]
            if not ref or not use:
                continue
            protein[name], new_tops = rebuild_poses(release, ref, use)
            tops[pf] = new_tops
            for cidx, (offset, length) in new_tops.items():
                pairs[(uniprot, pf, cidx)] = (int(offset), int(length))
        protein["rows"] = [retop(r, c, tops[r[c["pf"]]]) for r in rows]
        write_json(f"proteins/{uniprot}.json.gz", protein)
        write_json(f"salt_forms/{uniprot}.json.gz",
                   dict(salt, rows=[retop(r, c, tops[r[c["pf"]]]) for r in salt_rows]))
        af_ids[uniprot] = data["af_id"]
        name = f"{data['af_id']}.pdb.gz"
        write(f"structures/{name}", open(os.path.join(release, "structures", name), "rb").read())
        print(f"{uniprot}: {len(rows)} rows, {len(salt_rows)} salt_forms rows")

    for pvl in pvls:
        for suffix in ("", ".all"):
            data = read_json(os.path.join(release, "compounds", f"{pvl}{suffix}.json.gz"))
            c = columns(data)
            rows = []
            for r in data["rows"]:
                if r[c["uniprot"]] not in PROTEINS:
                    continue
                r = list(r)
                key = (r[c["uniprot"]], int(r[c["pf"]]), int(r[c["cidx"]]))
                if key in pairs:
                    r[c["top_off"]], r[c["top_len"]] = pairs[key]
                rows.append(r)
            out = dict(data, rows=rows, genes={u: g for u, g in (data.get("genes") or {}).items() if u in PROTEINS})
            if "n_targets" in out:
                out["n_targets"] = len(rows)
            if "n_targets_desalted" in out:
                out["n_targets_desalted"] = sum(r[c["pf"]] == 1 for r in rows)
            write_json(f"compounds/{pvl}{suffix}.json.gz", out)
    print(f"compounds: {len(pvls)} drugs")

    search(release, cats)
    write_json("ligand_ids.json", dict(read_json(os.path.join(release, "ligand_ids.json")),
                                       ids={c: entries[c] for c in cats}))
    cidx = read_json(os.path.join(release, "cidx.json"))
    write_json("cidx.json", dict(cidx, cidx={c: v for c, v in sorted(cidx["cidx"].items()) if c in cats}))
    tables(release, cats, pvls)
    write_json("meta.json", dict(meta, subset=SUBSET))
    manifest(meta)


def search(release, cats):
    """The search index of the fixture: the kept protein rows (the fixture's proteins and every protein that shares
    an identifier with one of them, so that an ambiguous identifier stays ambiguous) and drug rows, filed under the
    release's keys."""
    index = read_json(os.path.join(release, "search", "index.json.gz"))
    pc = {name: i for i, name in enumerate(index["proteins_columns"])}
    dc = {name: i for i, name in enumerate(index["drugs_columns"])}
    proteins, drugs = {}, {}
    for key in sorted(index["keys"]):
        data = read_json(os.path.join(release, "search", f"{key}.json.gz"))
        for row in data.get("p") or []:
            proteins[row[pc["uniprot"]]] = row
        for row in data.get("d") or []:
            if row[dc["cat"]] in cats:
                drugs[row[dc["cat"]]] = row

    def tokens(row):
        return {t.casefold() for t in ([row[pc["gene"]], row[pc["entry"]], row[pc["ensembl"]], row[pc["name"]]]
                                       + row[pc["synonyms"]].split() + row[pc["entrez"]].split(";")) if t}

    ours = set()
    for uniprot in PROTEINS:
        ours |= tokens(proteins[uniprot])
    keep = {u: r for u, r in proteins.items() if u in PROTEINS or tokens(r) & ours}
    files = {}
    for row in keep.values():
        fields = [row[pc["uniprot"]], row[pc["gene"]], row[pc["entry"]], *words(row[pc["name"]]),
                  *words(row[pc["synonyms"]]), *row[pc["entrez"]].split(";"), *row[pc["ensembl"]].split(";")]
        for key in {key_of(t) for t in fields} - {None}:
            files.setdefault(key, {"p": [], "d": []})["p"].append(row)
    for row in drugs.values():
        fields = [row[dc["cat"]], *words(row[dc["name"]]), row[dc["chembl"]], *words(row[dc["synonyms"]]),
                  row[dc["pvl"]]]
        for key in {key_of(t) for t in fields} - {None}:
            files.setdefault(key, {"p": [], "d": []})["d"].append(row)
    sizes = {}
    for key, data in sorted(files.items()):
        sizes[key] = os.path.getsize(write_json(f"search/{key}.json.gz", data))
    write_json("search/index.json.gz", dict(
        index, keys={k: {"bytes": b, "proteins": len(files[k]["p"]), "drugs": len(files[k]["d"])}
                     for k, b in sorted(sizes.items())}))
    print(f"search: {len(keep)} proteins ({len(PROTEINS)} with data), {len(drugs)} catalogue numbers, "
          f"{len(files)} key files")


def tables(release, cats, pvls):
    """The four small tables the package reads, with only the fixture's rows."""
    import pyarrow.parquet as pq

    keep = {"compound_entries": cats, "compound_entries_annotated": cats, "compounds": pvls,
            "proteins": sorted(PROTEINS)}
    for name, column in TABLES.items():
        table = pq.read_table(os.path.join(release, "tables", f"{name}.parquet"))
        wanted = set(keep[name])
        mask = [v in wanted for v in table.column(column).to_pylist()]
        rows = table.filter(mask)
        path = os.path.join(OUT, "tables", f"{name}.parquet")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(rows, path, compression="zstd")
        print(f"tables/{name}.parquet: {rows.num_rows} rows, {os.path.getsize(path)} bytes")


def manifest(meta):
    """MANIFEST.json of the fixture: every file written, with its bytes and sha256."""
    files = {}
    for folder, _, names in os.walk(OUT):
        for name in names:
            path = os.path.join(folder, name)
            rel = os.path.relpath(path, OUT).replace(os.sep, "/")
            if rel in ("MANIFEST.json",):
                continue
            with open(path, "rb") as f:
                data = f.read()
            files[rel] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    release = meta.get("release") or {}
    write_json("MANIFEST.json", {"release": release.get("label") or meta.get("version"), "subset": SUBSET,
                                 "files": dict(sorted(files.items()))}, sort_keys=False)
    total = sum(f["bytes"] for f in files.values())
    print(f"MANIFEST.json: {len(files)} files, {total / 1e6:.2f} MB in all")


if __name__ == "__main__":
    release_dir = sys.argv[1] if len(sys.argv) > 1 else RELEASE
    if len(sys.argv) > 2:
        OUT = os.path.abspath(sys.argv[2])
    build(release_dir)
