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
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from pocketvina import cli, selection
from pocketvina.atlas import ATLAS_URL, BEST_SCORES_COLUMNS, SEARCH_COLUMNS, SEARCH_TOP, Atlas, _Drugs
from pocketvina.io import atlas_source, posefile
from pocketvina.io.atlas_source import Source, pair_blocks, top_record

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
    """The fixture stays under 3 MB, and its MANIFEST.json lists every other file with its size and sha256."""
    import hashlib

    files = {p.relative_to(atlas_dir).as_posix(): p for p in atlas_dir.rglob("*") if p.is_file()}
    assert sum(p.stat().st_size for p in files.values()) < 3 * 1024 * 1024
    listed = json.loads((atlas_dir / "MANIFEST.json").read_text())["files"]
    assert set(listed) == set(files) - {"MANIFEST.json"}
    for rel, entry in listed.items():
        data = files[rel].read_bytes()
        assert entry == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}, rel
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
    assert Path(sub.show(ABL1, path=out.parent / "page.html")).is_file()
    return sub


def test_download_from_a_folder(atlas, atlas_dir, tmp_path, capsys):
    """A subset of two proteins: their files, every drug page, the search files, four tables and ligand_ids.json,
    byte-identical to the release's; MANIFEST.json lists them all; meta.json says what the subset is."""
    out = tmp_path / "my_atlas"
    assert atlas.download(proteins=["ABL1", "EGFR"], out=out) == str(out)
    line = capsys.readouterr().out
    assert "files" in line and "MB" in line and "2 proteins" in line and str(out) in line
    files, release = _tree(out), json.loads((atlas_dir / "MANIFEST.json").read_text())["files"]
    for rel in set(files) - {"meta.json", "MANIFEST.json"}:
        assert files[rel].read_bytes() == (atlas_dir / rel).read_bytes(), rel
    expected = {f"{folder}/{uniprot}{ext}" for uniprot in (ABL1, EGFR) for folder, ext in (
        ("proteins", ".json.gz"), ("salt_forms", ".json.gz"), ("poses", ".poses"), ("poses_redock", ".poses"))}
    expected |= {f"structures/AF-{uniprot}-F1-model_v6.pdb.gz" for uniprot in (ABL1, EGFR)}
    expected |= {rel for rel in release if rel.startswith(("search/", "compounds/")) and ".all." not in rel}
    expected |= {f"tables/{name}.parquet" for name in ("compound_entries", "compound_entries_annotated", "compounds",
                                                       "proteins")}
    assert set(files) == expected | {"ligand_ids.json", "meta.json", "MANIFEST.json"}
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
        if rel not in ("meta.json", "MANIFEST.json"):
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
    Atlas(atlas_server.url).download(["ABL1", "EGFR"], out)  # all there: only meta.json is read again
    assert {rel for rel, _ in atlas_server.requests[before:]} == {"meta.json"}
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
    here = shlex.quote(str(atlas_dir))
    said = {
        ("show", "NOSUCHGENE", "--atlas", atlas_dir): ("no protein 'NOSUCHGENE'", "; pocketvina atlas search "
                                                       f"NOSUCHGENE --atlas {here} finds part of a name"),
        ("export", "ABL1", "no such drug"): ("no drug 'no such drug'", "pocketvina atlas search 'no such drug' finds"),
        ("export", "ABL1", DRUG, "--out", "x.pdbqt", "--atlas", atlas_dir): ("pass --fmt pdbqt for a PDBQT file",),
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
