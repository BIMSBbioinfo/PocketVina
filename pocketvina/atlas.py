"""The PocketVina Atlas from Python: docking results of drugs against the human proteome, computed in advance.

    from pocketvina import Atlas
    atlas = Atlas()                                # the public release (ATLAS_URL); Atlas("my_atlas"): a local copy
    atlas.search("imatinib")                       # proteins and drugs that match, best first
    atlas.protein("ABL1")                          # protein information (in a notebook: a small card)
    atlas.drug("imatinib")                         # drug identifiers
    atlas.best_scores(protein="ABL1", top=20)      # drugs on a protein, highest-affinity first
    atlas.best_scores(ligand="imatinib", top=20)   # proteins of a drug, highest-affinity first
    pair = atlas.select(protein="ABL1", ligand="imatinib")   # a Selection, as Results.select gives
    pair.show(); pair.bust(); pair.export(); pair.export(protein=True)
    atlas.show(protein="ABL1")                     # the protein's page: its highest-affinity drugs, in 3D
    atlas.download(proteins=["ABL1", "EGFR"], out="my_atlas")   # a subset that Atlas("my_atlas") reads offline

``protein=`` and ``ligand=`` take any identifier the Atlas has. A protein: its UniProt accession ("P00519"), entry
name, gene ("ABL1"), gene synonyms, Entrez, HGNC or Ensembl gene ID, or protein name. A drug: its name, synonyms,
PVL ID (the Atlas ID of a docked structure), ChEMBL ID, catalogue number (also "<catalogue number>-desalted"),
InChIKey (or its first block) or SMILES (as in the library, docked, or of the parent drug; any SMILES of the same
molecule matches). Matching is exact and ignores case. An identifier of several proteins or drugs raises an error
that lists them; ``atlas.search`` also finds part of a name.
"""
import gzip
import hashlib
import html
import json
import logging
import numbers
import operator
import os
import re
import threading
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, NamedTuple, Optional, Tuple

import pandas as pd

from . import __version__, pdbqt
from .io import sas
from .io.atlas_source import Source, pose_records, top_record
from .selection import Selection, _check_mode, _check_pocket, _pick_poses, _short, name_part

logger = logging.getLogger(__name__)

# the public release; a later release goes beside it (data/v1.2/), so this address does not change
ATLAS_URL = "https://bimsbstatic.mdc-berlin.de/akalin/asarigun/pocketvina_atlas/data/v1.1/"
RELEASES = ("v1.1",)  # the releases this version of PocketVina reads
BEST_SCORES_COLUMNS = ("protein_id", "gene", "ligand_id", "name", "vina", "best_pocket", "best_mode",
                       "ligand_efficiency")
MAX_LIGANDS = 25  # drugs listed in Top hits of a page (highest-affinity first); the poses of each are fetched
THREADS = 8  # files copied at the same time by download
PAGE_THREADS = 16  # requests sent at the same time for a page (Atlas.show)
NON_DRUG = 8  # bit of a row's flags: docked as a salt counterion, solvent or fragment (meta.json "flags"), not a drug
# Top hits of a page, in the Atlas site's words (components/ProteinWorkspace.tsx)
TOP_HITS = ("Highest Vina affinities among PoseBusters-valid poses (non-drug structures and poses not yet checked "
            "left out). Pick one to see its poses.")
UNIPROT_URL = "https://www.uniprot.org/uniprotkb/{}"
ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/entry/{}"
OPEN_TARGETS_URL = "https://platform.opentargets.org/target/{}/associations"
OPEN_TARGETS_EVIDENCE_URL = "https://platform.opentargets.org/evidence/{}/{}"
CHEMBL_URL = "https://www.ebi.ac.uk/chembl/explore/compound/{}"
# what a download (atlas.download) holds besides the files of its proteins: every drug page, the search files, these
# tables (the drug identifiers and the HGNC IDs) and these files of the release's folder (LICENSE, the data's licence:
# if the release has one; its MANIFEST.json does not list it)
SUBSET_TABLES = ("compound_entries", "compound_entries_annotated", "compounds", "proteins")
SUBSET_FILES = ("ligand_ids.json", "LICENSE")
SUBSET_NOTE = ("A part of the PocketVina Atlas, written by pocketvina's Atlas.download(): for these proteins their "
               "files of proteins/, salt_forms/, poses/, poses_redock/, structures/ and sas/ (with sas/MANIFEST.json "
               "listing them), and from the release every "
               "drug page (compounds/<PVL>.json.gz), the search files (search/), the tables compound_entries, "
               "compound_entries_annotated, compounds and proteins (tables/), ligand_ids.json and LICENSE (the "
               "data's licence, if the release has one), all as released (byte-identical copies, checked against "
               "the release's MANIFEST.json). Not copied: compounds/<PVL>.all.json.gz (every protein of a drug): "
               "best_scores(ligand=...) lists the proteins of this subset only, from their proteins/<UniProt>.json "
               "rows, and search() lists them and every drug. The identifiers of every protein and drug of the "
               "release are known (a protein outside the subset: an error that says so).")
_SUBSET_FOLDERS = ("proteins", "salt_forms", "poses", "poses_redock", "structures", "sas", "compounds", "search",
                   "tables")
_SHOWN = 10  # candidates listed in an "ambiguous identifier" error
# identifiers that go into a link of a page (the site's checks, lib/keys.ts RX and app/protein/[id]/page.tsx)
_CHEMBL_ID = re.compile(r"^CHEMBL\d{1,9}$")
_ENSG = re.compile(r"^ENSG\d{11}$")
_DISEASE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9]+$")
_HINT = "Atlas('/path/to/release/v1.1') (a folder with meta.json), or Atlas() for the public release"
# UniProt accession format (site lib/client.ts; search/index.json.gz "rule")
_ACC = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")
_AF_ID = re.compile(r"^AF-([A-Z0-9]{6,10})-F\d{1,2}(?:-MODEL_V\d+)?$")
_HGNC = re.compile(r"^HGNC:\d+$", re.IGNORECASE)
_PVL = re.compile(r"^PVL\d{5,7}$", re.IGNORECASE)
_INCHIKEY_BLOCK = re.compile(r"^[A-Za-z]{14}$")
_WORDS = re.compile(r"[\s,;()\[\]/]+")  # how the release splits names into search tokens
SEARCH_COLUMNS = ("kind", "id", "name", "matched", "value")
SEARCH_TOP = 20  # rows atlas.search returns by default
# P2Rank's SAS points of each protein are in sas/<U>.json.gz (pocketvina.io.sas); this file lists them with their
# sha256 and size, {"<U>.json.gz": [sha256, bytes]} (the release's MANIFEST.json does not list sas/)
SAS_MANIFEST = "sas/MANIFEST.json"
# atlas.search ranks a match by the score the Atlas website's search box gives it (components/Search.tsx), so that a
# query lists proteins and drugs in the same order on both. Proteins: _protein_score. Drugs, per identifier: the score
# of an exact match, (score, shortest query) of a match at its start and of one inside it (None: not matched that way);
# a synonym also matches at the start of one of its words. Identifiers the site's search box does not have (HGNC and
# AlphaFold model IDs, InChIKeys, SMILES) score as an exact identifier, a little less for an InChIKey's first block
# and for a SMILES of the same molecule written another way.
_DRUG_SCORES = {
    "PVL ID": (102, (54, 5), None), "catalogue number": (101, (55, 3), None), "name": (100, (80, 2), (42, 3)),
    "page name": (100, (79, 2), (41, 3)), "ChEMBL ID": (100, None, None), "InChIKey": (100, None, None),
    "docked InChIKey": (100, None, None), "synonym": (None, (66, 3), (34, 4)),
}
_SMILES_SCORE, _BLOCK_SCORE, _CANONICAL_SCORE = 100, 95, 90  # exact SMILES, first block of an InChIKey, same molecule
_DOCKED_BONUS = 3  # added to the score of a protein that has docking results (as on the site)


# --- identifiers ---------------------------------------------------------------------------------------------------

def _key_of(token: str) -> Optional[str]:
    """The search key file of one token (the release's rule): "acc_" + 2 characters for a UniProt accession, "ensg" +
    characters 10-11 for an Ensembl gene ID, else the first character (lower case; "_" if not a letter or digit)."""
    t = token.strip()
    if not t:
        return None
    if _ACC.match(t.upper()):
        return "acc_" + t[:2].lower()
    tl = t.lower()
    if tl.startswith("ensg") and len(tl) >= 11:
        return "ensg" + tl[9:11]
    return tl[0] if tl[0].isalnum() else "_"


def _search_keys(query: str) -> List[str]:
    """The search key files a query needs, as the site's searchKeys() picks them (lib/client.ts), plus the file of its
    first word (where a name that starts with a bracket is filed)."""
    q = query.strip()
    ql = q.lower()
    if not q:
        return []
    keys = []
    if _ACC.match(q.upper()):
        keys.append("acc_" + ql[:2])
    elif ql.startswith("ensg") and len(ql) >= 11:
        keys.append("ensg" + ql[9:11])
    else:
        if re.match(r"^[a-z][0-9]", ql):
            keys.append("acc_" + ql[:2])
        keys.append(ql[0] if ql[0].isalnum() else "_")
        words = [w for w in _WORDS.split(q) if w]
        if words:
            keys.append(_key_of(words[0]))
    m = re.match(r"^(?:PV-)?([A-Z0-9]{6,10})[-_]", q.upper())
    if m and _ACC.match(m.group(1)):
        keys.append("acc_" + m.group(1)[:2].lower())
    return list(dict.fromkeys(k for k in keys if k))


def _query(value, what: str, example: str) -> str:
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        value = str(value)  # an Entrez ID given as a number
    if not isinstance(value, str):
        raise TypeError(f"{what} must be an identifier (a string) such as {example}, not {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{what} is empty: give an identifier such as {example}")
    return text


def _split(text, seps: str = ";") -> List[str]:
    if not isinstance(text, str):
        return []
    return [t.strip() for t in re.split(f"[{seps}]", text) if t.strip()]


def _str(value) -> Optional[str]:
    """A text field of the release (None for null, NaN or "")."""
    return value if isinstance(value, str) and value else None


def _strings(value) -> List[str]:
    """A list field of the release (a JSON list or a Parquet list column) as a list of str."""
    if value is None or isinstance(value, (str, float)):
        return []
    return [str(v) for v in value if v is not None and str(v)]


def _int(value) -> Optional[int]:
    """An int column of a row (n_failed, pb_mask, pb may be JSON floats); None for null."""
    return None if value is None else int(value)


def _choices(items: List[Tuple[str, str]], kind: str) -> str:
    """'ID (label), ...' for an error message, at most _SHOWN of them."""
    shown = ", ".join(f"{i} ({label})" if label else i for i, label in items[:_SHOWN])
    more = f", ... ({len(items)} in all)" if len(items) > _SHOWN else ""
    return f"{len(items)} {kind}: {shown}{more}"


def _protein_title(row: list, c: Dict[str, int]) -> str:
    """"gene: protein name" of a row of the search files (the name column of atlas.search)."""
    gene, name = _str(row[c["gene"]]), _str(row[c["name"]])
    return f"{gene}: {name}" if gene and name else gene or name or row[c["uniprot"]]


def _docked_bonus(row: list, c: Dict[str, int]) -> int:
    """The bonus of a protein with docking results (its entry_page in the search files), as the site's search box."""
    return _DOCKED_BONUS if "entry_page" not in c or row[c["entry_page"]] else 0


def _protein_score(row: list, c: Dict[str, int], q: str) -> Optional[Tuple[int, str, str]]:
    """(score, which identifier, its text) of a protein of the search files for a query (``q``, case-folded) as the
    Atlas website's search box scores it (components/Search.tsx: the first rule that holds), without the bonus for
    docking results; None if no rule holds. Each gene of a protein that has several ("A; B") also counts as its gene,
    as for ``protein=``."""
    gene, uniprot, name, ensembl = (row[c[k]] or "" for k in ("gene", "uniprot", "name", "ensembl"))
    synonyms = (row[c["synonyms"]] or "").split()
    exact = ([("gene", gene)] + [("gene", g) for g in _split(gene, ";/,")]
             + [("UniProt accession", uniprot), ("Ensembl ID", ensembl), ("entry name", row[c["entry"]] or "")]
             + [("Entrez ID", e) for e in _split(row[c["entrez"]], ";")])
    for field, value in exact:
        if value and value.casefold() == q:
            return 100, field, value
    n = len(q)
    for synonym in synonyms:
        if synonym.casefold() == q:
            return 72, "gene synonym", synonym
    if n >= 2 and gene.casefold().startswith(q):
        return 80 - min(20, len(gene) - n), "gene", gene
    if n >= 3 and uniprot.casefold().startswith(q):
        return 60, "UniProt accession", uniprot
    for synonym in synonyms:
        if n >= 3 and synonym.casefold().startswith(q):
            return 50, "gene synonym", synonym
    if n >= 3 and q in name.casefold():
        return (45 if name.casefold().startswith(q) else 38), "protein name", name
    if n >= 6 and ensembl.casefold().startswith(q):
        return 40, "Ensembl ID", ensembl
    return None


class _Match(NamedTuple):
    tier: int  # 0: the IDs themselves, 1: other identifiers, 2: synonyms and partial / converted identifiers
    target: Optional[str]  # UniProt or PVL; None: a drug entry without docking results
    entry: Optional[str] = None  # the catalogue number, when the query was one
    field: str = ""  # which identifier it is, e.g. "synonym" (atlas.search's "matched")
    value: Optional[str] = None  # the identifier as the release writes it


def _best(matches: List[_Match], query: str, kind: str, label) -> _Match:
    """The match of the first tier that has one; ValueError if that tier names more than one protein / drug."""
    found = [m for m in matches if m.target]
    tier = min(m.tier for m in found)
    targets = list(dict.fromkeys(m.target for m in found if m.tier == tier))
    if len(targets) > 1:
        what = "proteins" if kind == "protein" else "drugs"
        ids = "UniProt accessions" if kind == "protein" else "PVL IDs"
        raise ValueError(f"{query!r} matches {_choices([(t, label(t)) for t in targets], what)}; pass one of these "
                         f"{ids} instead")
    entries = {m.entry for m in found if m.tier == tier and m.target == targets[0] and m.entry}
    return _Match(tier, targets[0], entries.pop() if len(entries) == 1 else None)


def _drug_score(field: str, text: str, q: str) -> Optional[int]:
    """The score of a drug identifier (``text``, case-folded) of kind ``field`` for the query ``q`` (case-folded), as
    atlas.search ranks it (``_DRUG_SCORES``); None if it does not match."""
    exact, start, inside = _DRUG_SCORES[field]
    if exact is not None and text == q:
        return exact
    if start and len(q) >= start[1] and (text.startswith(q) or (field == "synonym" and f" {q}" in text)):
        return start[0]
    if inside and len(q) >= inside[1] and q in text:
        return inside[0]
    return None


def _top(top) -> Optional[int]:
    if top is not None and (isinstance(top, bool) or operator.index(top) < 0):
        raise ValueError(f"top must be a number of rows (0, 1, ...) or None, got {top!r}")
    return top


def _canonical_smiles(smiles: str) -> Optional[str]:
    """RDKit's canonical SMILES of a molecule, or None if the text is not a SMILES RDKit reads."""
    from rdkit import Chem, rdBase

    block = rdBase.BlockLogs()  # no parser messages
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else None
    finally:
        del block


class _Drugs:
    """The drug identifiers of a release, loaded once per Atlas: one row per catalogue number
    (tables/compound_entries.parquet and compound_entries_annotated.parquet) and one per drug page, i.e. per PVL ID
    with docking results (tables/compounds.parquet).

    A catalogue number lands on the page of its drug (``landing``, the site's landingPvl in lib/keys.ts): an entry
    docked as something else than the drug (a salt counterion, a solvent, a product component) on the page of its
    parent drug when that page exists, else on the page of the structure it was docked as."""

    ENTRY_COLUMNS = ["catalog_id", "pvl_id", "name", "input_smiles", "docked_smiles", "docked_inchikey",
                     "parent_smiles", "docked_as", "parent_pvl_id"]
    ANNOTATED_COLUMNS = ["catalog_id", "chembl_id", "synonyms", "atc", "parent_inchikey", "parent_smiles"]
    PAGE_COLUMNS = ["pvl_id", "name", "docked_as", "label", "note", "canonical_catalog_id", "docked_inchikey",
                    "docked_smiles", "has_results"]

    def __init__(self, entries: pd.DataFrame, annotated: pd.DataFrame, pages: pd.DataFrame):
        pages = pages[pages["has_results"].fillna(False).astype(bool)]
        self.pages = pages.drop(columns="has_results").set_index("pvl_id", drop=False)
        annotated = annotated.rename(columns={"parent_smiles": "parent_smiles_annotated"})
        entries = entries.merge(annotated, on="catalog_id", how="left")
        entries["landing"] = [self._landing(p, d, q) for p, d, q in
                              zip(entries["pvl_id"], entries["docked_as"], entries["parent_pvl_id"])]
        self.entries = entries.set_index("catalog_id", drop=False)
        self._text: Dict[str, List[_Match]] = defaultdict(list)  # identifier (case-folded) -> matches
        self._smiles: Dict[str, List[_Match]] = defaultdict(list)  # SMILES as written -> matches
        self._blocks: Dict[str, List[_Match]] = defaultdict(list)  # first block of an InChIKey -> matches
        self._canonical: Optional[Dict[str, List[_Match]]] = None
        for pvl, name in zip(self.pages["pvl_id"], self.pages["name"]):
            self._add(0, pvl, pvl, field="PVL ID")
            self._add(1, name, pvl, field="page name")
        chembl: Dict[str, List[Tuple[bool, Optional[str]]]] = defaultdict(list)
        for e in self.entries.itertuples(index=False):
            own = e.pvl_id if e.pvl_id in self.pages.index else e.landing  # the page of the docked structure
            self._add(0, e.catalog_id, e.landing, e.catalog_id, field="catalogue number")
            self._add(1, e.name, e.landing, field="name")
            if _str(e.chembl_id):
                chembl[e.chembl_id].append((e.docked_as == "drug" and e.landing is not None, e.landing))
            for key, target, field in ((e.parent_inchikey, e.landing, "InChIKey"),
                                       (e.docked_inchikey, own, "docked InChIKey")):
                self._add(1, key, target, field=field)
                if isinstance(key, str) and len(key) > 14:
                    self._blocks[key[:14].upper()].append(_Match(2, target, field=f"{field} first block",
                                                                 value=key[:14]))
            for smiles, target, field in self._smiles_of(e):
                self._smiles[smiles].append(_Match(1, target, field=field, value=smiles))
            for synonym in _strings(e.synonyms):
                self._add(2, synonym, e.landing, field="synonym")
        # a ChEMBL ID stands for the pages of its entries docked as the drug, else for wherever its entries land (as on
        # the site); atlas.search lists every page its entries land on (tier 2), as the site's search box does
        for chembl_id, found in chembl.items():
            as_drug = {landing for drug, landing in found if drug}
            for landing in dict.fromkeys(landing for _, landing in found):
                self._add(1 if not as_drug or landing in as_drug else 2, chembl_id, landing, field="ChEMBL ID")

    def _landing(self, pvl, docked_as, parent) -> Optional[str]:
        pages = self.pages.index
        parent = _str(parent)
        if docked_as != "drug" and parent in pages:
            return parent
        return pvl if pvl in pages else parent if parent in pages else None

    def _smiles_of(self, e) -> List[Tuple[str, Optional[str], str]]:
        """(SMILES, the page it points to, which SMILES) of an entry: the library and parent SMILES -> its drug; the
        docked SMILES -> the page of the docked structure."""
        own = e.pvl_id if e.pvl_id in self.pages.index else e.landing
        out = [(e.input_smiles, e.landing, "library SMILES"), (e.parent_smiles, e.landing, "parent SMILES"),
               (e.parent_smiles_annotated, e.landing, "parent SMILES"), (e.docked_smiles, own, "docked SMILES")]
        return [(s, t, field) for s, t, field in out if _str(s)]

    def _add(self, tier: int, identifier, target: Optional[str], entry: Optional[str] = None, *, field: str):
        if _str(identifier):
            text = identifier.strip()
            self._text[text.casefold()].append(_Match(tier, target, entry, field, text))

    def canonical(self) -> Dict[str, List[_Match]]:
        """RDKit-canonical SMILES of every library, parent and docked SMILES -> matches (computed on first use)."""
        if self._canonical is None:
            out: Dict[str, List[_Match]] = defaultdict(list)
            done: Dict[str, Optional[str]] = {}
            for e in self.entries.itertuples(index=False):
                for smiles, target, field in self._smiles_of(e):
                    if smiles not in done:
                        done[smiles] = _canonical_smiles(smiles)
                    if done[smiles]:
                        out[done[smiles]].append(_Match(2, target, field=f"{field} (canonical)", value=smiles))
            self._canonical = out
        return self._canonical

    def name(self, pvl: str) -> Optional[str]:
        return _str(self.pages["name"].get(pvl)) if pvl in self.pages.index else None

    def resolve(self, query: str) -> _Match:
        """The drug page (PVL ID) of an identifier, and the catalogue number when the query was one."""
        matches = list(self._text.get(query.casefold(), [])) + list(self._smiles.get(query, []))
        if _INCHIKEY_BLOCK.match(query):
            matches += self._blocks.get(query.upper(), [])
        if _PVL.match(query) and not matches and query.upper() in set(self.entries["pvl_id"]):
            matches.append(_Match(0, None))  # a docked structure without results
        if not any(m.target and m.tier < 2 for m in matches):
            canonical = _canonical_smiles(query) if not _PVL.match(query) else None
            if canonical:
                matches += self.canonical().get(canonical, [])
        if not any(m.target for m in matches):
            if matches:
                raise ValueError(f"{query!r} is in the library of the PocketVina Atlas but has no docking results")
            raise KeyError(f"no drug {query!r} in the PocketVina Atlas: give its name, a synonym, PVL ID, ChEMBL ID, "
                           f"catalogue number, InChIKey or SMILES; atlas.search({query!r}) finds part of a name")
        return _best(matches, query, "drug", self.name)

    def search(self, query: str) -> List[Tuple[int, str, str, str]]:
        """Every match of a query among the drug identifiers (atlas.search): (score, PVL ID of the drug page, which
        identifier, its text). Names, synonyms, PVL IDs and catalogue numbers also match in part (``_drug_score``);
        ChEMBL IDs, InChIKeys (or their first block) and SMILES match whole (a SMILES also as the same molecule)."""
        q = query.casefold()
        hits = []
        for text, matches in self._text.items():
            if q not in text:
                continue
            for m in matches:
                score = _drug_score(m.field, text, q) if m.target else None
                if score:
                    hits.append((score, m.target, "name" if m.field == "page name" else m.field, m.value))
        whole = [(m, _SMILES_SCORE) for m in self._smiles.get(query, [])]
        if _INCHIKEY_BLOCK.match(query):
            whole += [(m, _BLOCK_SCORE) for m in self._blocks.get(query.upper(), [])]
        if not whole and q not in self._text and not _PVL.match(query):
            canonical = _canonical_smiles(query)
            if canonical:
                whole += [(m, _CANONICAL_SCORE) for m in self.canonical().get(canonical, [])]
        return hits + [(score, m.target, m.field, m.value) for m, score in whole if m.target]

    def catalogue_numbers(self, pvl: str, first: Optional[str] = None) -> List[str]:
        """The catalogue numbers of a drug page: those that land on it and those docked as its structure; ``first``
        (default: the page's canonical entry) first."""
        e = self.entries
        cats = sorted(set(e.index[(e["landing"] == pvl) | (e["pvl_id"] == pvl)]))
        first = first or _str(self.pages["canonical_catalog_id"].get(pvl))
        return ([first] if first in cats else []) + [c for c in cats if c != first]


# --- information cards ---------------------------------------------------------------------------------------------

def _link(url: Optional[str], text: str) -> str:
    e = html.escape
    return f"<a href='{e(url)}' target='_blank' rel='noopener'>{e(text)}</a>" if url else e(text)


class ProteinInfo:
    """A protein of the PocketVina Atlas (``Atlas.protein``). Attributes: ``uniprot``, ``gene``, ``name``,
    ``entry_name``, ``af_id`` (the AlphaFold model the drugs were docked into), ``uniprot_url``, ``alphafold_url``
    (its AlphaFold DB page), ``mean_plddt`` (AlphaFold's mean confidence, 0-100), ``protein_class`` (ChEMBL target
    class, e.g. "Enzyme > Kinase"; None if it has none), ``pockets`` (table: pocket, score, probability, plddt,
    residues), ``diseases`` (Open Targets associations: name, id, score; empty if none) and ``open_targets_url``."""

    def __init__(self, data: dict):
        self.uniprot: str = data["uniprot"]
        self.gene: Optional[str] = _str(data.get("gene"))
        self.name: Optional[str] = _str(data.get("protein_name"))
        self.entry_name: Optional[str] = _str(data.get("entry_name"))
        self.af_id: Optional[str] = _str(data.get("af_id"))
        self.uniprot_url = UNIPROT_URL.format(self.uniprot)
        self.alphafold_url = ALPHAFOLD_URL.format(self.uniprot)
        plddt = data.get("mean_plddt")
        self.mean_plddt: Optional[float] = float(plddt) if isinstance(plddt, (int, float)) else None
        self.protein_class: Optional[str] = " > ".join(_strings(data.get("chembl_class"))) or None
        self.pockets = pd.DataFrame(
            [(int(p["rank"]), p.get("score"), p.get("probability"), p.get("plddt"),
              " ".join(_strings(p.get("residues")))) for p in sorted(data.get("pockets") or [],
                                                                     key=lambda p: p["rank"])],
            columns=["pocket", "score", "probability", "plddt", "residues"])
        ot = data.get("open_targets") or {}
        self.diseases = pd.DataFrame([(d.get("name"), d.get("id"), d.get("score")) for d in ot.get("diseases") or []],
                                     columns=["name", "id", "score"])
        self.open_targets_url: Optional[str] = (OPEN_TARGETS_URL.format(ot["ensembl_id"])
                                                if _str(ot.get("ensembl_id")) else None)

    def _title(self) -> str:
        gene = f" ({self.gene})" if self.gene else ""
        return f"{self.uniprot}{gene}" + (f": {self.name}" if self.name else "")

    def _facts(self) -> List[str]:
        facts = []
        if self.af_id:
            plddt = f", mean pLDDT {self.mean_plddt:.1f}" if self.mean_plddt is not None else ""
            facts.append(f"AlphaFold model {self.af_id}{plddt}")
        if self.protein_class:
            facts.append(f"class {self.protein_class}")
        facts.append(f"{len(self.pockets)} pocket{'s' if len(self.pockets) != 1 else ''} (P2Rank)")
        if len(self.diseases):
            facts.append(f"{len(self.diseases)} associated diseases (Open Targets)")
        return facts

    def __repr__(self):
        return "\n".join([f"ProteinInfo {self._title()}"] + [f"  {f}" for f in self._facts()])

    def _repr_html_(self) -> str:
        e = html.escape
        links = [_link(self.uniprot_url, "UniProt"), _link(self.alphafold_url, "AlphaFold DB")]
        if self.open_targets_url:
            links.append(_link(self.open_targets_url, "Open Targets"))
        return (f"<div><b>{e(self._title())}</b><br>{'<br>'.join(e(f) for f in self._facts())}<br>"
                f"<small>{' &middot; '.join(links)}</small></div>")


class DrugInfo:
    """A drug of the PocketVina Atlas (``Atlas.drug``): one docked structure, its PVL ID. Attributes: ``pvl``,
    ``name``, ``chembl_id`` and ``chembl_url`` (None if it has none), ``catalogue_numbers`` (library entries docked as
    this structure or whose drug it is; the entry the other identifiers are of comes first), ``inchikey``,
    ``parent_smiles`` (the drug), ``library_smiles`` (as screened, e.g. with its salt), ``docked_smiles`` (the
    molecule docked), ``synonyms``, ``atc`` (ATC codes), ``docked_as`` ("drug", or "salt_counterion", "solvent",
    "fragment" when not the drug was docked) and ``note`` (the release's explanation in that case, else None)."""

    def __init__(self, *, pvl: str, name: Optional[str], chembl_id: Optional[str], catalogue_numbers: List[str],
                 inchikey: Optional[str], parent_smiles: Optional[str], library_smiles: Optional[str],
                 docked_smiles: Optional[str], synonyms: List[str], atc: List[str], docked_as: str,
                 note: Optional[str]):
        self.pvl, self.name = pvl, name
        self.chembl_id = chembl_id
        self.chembl_url = CHEMBL_URL.format(chembl_id) if chembl_id else None
        self.catalogue_numbers = list(catalogue_numbers)
        self.inchikey = inchikey
        self.parent_smiles, self.library_smiles, self.docked_smiles = parent_smiles, library_smiles, docked_smiles
        self.synonyms, self.atc = list(synonyms), list(atc)
        self.docked_as, self.note = docked_as, note

    def _rows(self) -> List[Tuple[str, str]]:
        rows = [("PVL ID", self.pvl), ("ChEMBL ID", self.chembl_id),
                ("catalogue numbers", ", ".join(self.catalogue_numbers)), ("InChIKey", self.inchikey),
                ("parent SMILES", self.parent_smiles), ("library SMILES", self.library_smiles),
                ("docked SMILES", self.docked_smiles), ("synonyms", ", ".join(self.synonyms)),
                ("ATC", ", ".join(self.atc))]
        if self.docked_as != "drug":
            rows.append(("docked as", self.docked_as.replace("_", " ") + (f": {self.note}" if self.note else "")))
        return [(k, v) for k, v in rows if v]

    def __repr__(self):
        return "\n".join([f"DrugInfo {self.pvl}: {self.name or ''}"] + [f"  {k}: {v}" for k, v in self._rows()[1:]])

    def _repr_html_(self) -> str:
        e = html.escape
        cells = []
        for k, v in self._rows():
            value = _link(self.chembl_url, v) if k == "ChEMBL ID" else e(v)
            mono = " style='font-family:monospace;word-break:break-all'" if "SMILES" in k or k == "InChIKey" else ""
            cells.append(f"<tr><th style='text-align:left'>{e(k)}</th><td{mono}>{value}</td></tr>")
        return f"<div><b>{e(self.name or self.pvl)}</b><table>{''.join(cells)}</table></div>"


class _Pair(NamedTuple):
    """Where a protein x drug pair is: the protein's JSON, the pair's row (a dict of the protein's columns), the pose
    file (the ``poses`` or ``poses_redock`` entry of the JSON) and the pair's record index in it."""

    protein: dict
    row: dict
    ref: dict
    index: int


# --- the Atlas -------------------------------------------------------------------------------------------------

class Atlas:
    """The PocketVina Atlas: docking results of every drug of the screening library against human proteins, read
    from the public release or from a local copy.

        atlas = Atlas()                              # the public release (ATLAS_URL), read over HTTP
        atlas = Atlas("/data/pocketvina_atlas/v1.1")  # a release folder (one with meta.json)
        atlas = Atlas("my_atlas")                    # a folder written by atlas.download (works offline)

    source: a release folder or the base URL of one; None (default): ``$POCKETVINA_ATLAS`` if it is set, else the
        public release. Files read over HTTP are kept in ``~/.cache/pocketvina/atlas/<release>/``.
    """

    def __init__(self, source=None):
        location = source if source is not None else (os.environ.get("POCKETVINA_ATLAS") or ATLAS_URL)
        self.source = Source(location)
        self.meta = self._read_meta()
        release = self.meta.get("release") or {}
        self.release: str = str(release.get("label") or self.meta.get("version") or "")
        if self.release not in RELEASES:
            logger.warning("%s is release %r; this version of PocketVina was written for %s", self.source.location,
                           self.release or "unknown", ", ".join(RELEASES))
        self._drugs: Optional[_Drugs] = None
        self._proteins: Dict[str, dict] = {}  # UniProt -> proteins/<U>.json
        self._keys: Dict[str, List[list]] = {}  # search key file -> its protein rows
        self._key_columns: Optional[Dict[str, int]] = None
        self._hgnc: Optional[Dict[str, List[str]]] = None
        self._page_names: Optional[Dict[str, str]] = None
        self._sas_files: Optional[Dict[str, list]] = None  # sas/MANIFEST.json, read on first use

    def _read_meta(self) -> dict:
        where = self.source.location
        try:
            meta = self.source.meta
        except FileNotFoundError:
            raise FileNotFoundError(f"{where} is not a PocketVina Atlas release: it has no meta.json; use "
                                    f"{_HINT}") from None
        except ValueError:  # meta.json is not JSON (e.g. a web page)
            meta = None
        if not isinstance(meta, dict) or "screen" not in meta or not isinstance(meta.get("release"), dict):
            raise ValueError(f"{where} is not a PocketVina Atlas release: its meta.json has no release information; "
                             f"use {_HINT}")
        status = meta["release"].get("status")
        if status != "final":
            raise ValueError(f"{where} is not a final PocketVina Atlas release (its meta.json says status "
                             f"{status!r}): its files may still change; use {_HINT}")
        return meta

    # --- what it is ------------------------------------------------------------------------------------------------

    def _summary(self) -> str:
        screen = self.meta.get("screen") or {}
        counts = [(screen.get("proteins"), "proteins"), (screen.get("drugs"), "drugs"), (screen.get("pairs"), "pairs")]
        subset = self._subset_proteins()
        if subset is not None:  # a download: its proteins, and every drug
            counts = [(len(subset), "proteins" if len(subset) != 1 else "protein"), counts[1]]
        return ", ".join(f"{n:,} {what}" for n, what in counts if isinstance(n, int))

    def _subset_proteins(self) -> Optional[List[str]]:
        """The proteins (UniProt accessions) of a subset written by download(); None for any other release."""
        subset = self.meta.get("subset")
        if isinstance(subset, dict) and isinstance(subset.get("proteins"), list):
            return [str(u) for u in subset["proteins"]]
        return None

    def __repr__(self):
        subset = " (a subset)" if self.meta.get("subset") else ""
        return f"PocketVina Atlas {self.release}{subset}: {self._summary()} [{self.source.location}]"

    def _repr_html_(self) -> str:
        e = html.escape
        subset = " (a subset)" if self.meta.get("subset") else ""
        return (f"<div><b>PocketVina Atlas {e(self.release)}</b>{subset}: {e(self._summary())}<br>"
                f"<small>{e(self.source.location)}</small></div>")

    # --- proteins --------------------------------------------------------------------------------------------------

    def _search_rows(self, key: str) -> List[list]:
        """The protein rows of one search key file ([] if the release has no file for that key)."""
        if key not in self._keys:
            index = self.source.json("search/index.json")
            if self._key_columns is None:
                self._key_columns = {c: i for i, c in enumerate(index["proteins_columns"])}
            rows = self.source.json(f"search/{key}.json").get("p") or [] if key in index["keys"] else []
            self._keys[key] = rows
        return self._keys[key]

    def _hgnc_ids(self) -> Dict[str, List[str]]:
        """HGNC ID -> UniProt accessions, from tables/proteins.parquet (the only file of the release that has HGNC
        IDs; a protein can have several, "HGNC:1;HGNC:2"); read on the first HGNC query."""
        if self._hgnc is None:
            import pyarrow.parquet as pq

            table = pq.read_table(self.source.file("tables/proteins.parquet"), columns=["uniprot", "hgnc"])
            hgnc: Dict[str, List[str]] = defaultdict(list)
            for u, h in zip(table.column("uniprot").to_pylist(), table.column("hgnc").to_pylist()):
                for one in _split(h, ";"):
                    hgnc[one.upper()].append(str(u))
            self._hgnc = dict(hgnc)
        return self._hgnc

    def _protein_matches(self, query: str) -> List[_Match]:
        """Every protein of the query's search key files whose identifiers hold the query: the accession and the
        AlphaFold model ID first, then entry name, gene, Entrez and Ensembl, then gene synonyms and protein name."""
        af = _AF_ID.match(query.upper())
        accession = af.group(1) if af else None  # an AlphaFold model ID: its accession's file
        matches = [m for key in _search_keys(accession or query) for m in self._key_matches(key, query, accession)]
        if not matches and not af and _ACC.match(query.upper()):
            # a gene written like an accession (e.g. H2BC12) is filed under its first character, not in the accession
            # file
            matches = self._key_matches(query[0].lower(), query, None)
        if _HGNC.match(query):
            matches += [_Match(1, uniprot) for uniprot in self._hgnc_ids().get(query.upper(), [])]
        return matches

    def _key_matches(self, key: str, query: str, accession: Optional[str]) -> List[_Match]:
        """The proteins of one search key file that have the query as an identifier (``accession``: the accession of
        an AlphaFold model ID)."""
        q = query.casefold()
        matches = []
        for row in self._search_rows(key):
            c = self._key_columns
            uniprot = row[c["uniprot"]]
            if uniprot.casefold() == q or uniprot == accession:
                matches.append(_Match(0, uniprot))
                continue
            exact = [row[c["entry"]], row[c["ensembl"]]] + _split(row[c["gene"]], ";/,") + _split(row[c["entrez"]], ";")
            if any(isinstance(v, str) and v.casefold() == q for v in exact):
                matches.append(_Match(1, uniprot))
            elif any(v.casefold() == q for v in row[c["synonyms"]].split()) or row[c["name"]].casefold() == q:
                matches.append(_Match(2, uniprot))
        return matches

    def _uniprot(self, protein) -> str:
        """The UniProt accession of any protein identifier."""
        query = _query(protein, "protein", '"ABL1", "P00519" or "EGFR_HUMAN"')
        matches = self._protein_matches(query)
        if not matches:
            raise KeyError(f"no protein {query!r} in the PocketVina Atlas: give its gene, UniProt accession, entry "
                           f"name, Entrez, HGNC or Ensembl gene ID, or protein name; atlas.search({query!r}) finds "
                           f"part of a name")
        return _best(matches, query, "protein", lambda u: self._label(u)).target

    def _label(self, uniprot: str) -> str:
        """gene or protein name of a protein, for an error message."""
        row = self._protein_row(uniprot)
        return (row[self._key_columns["gene"]] or row[self._key_columns["name"]]) if row else ""

    def _protein_row(self, uniprot: str) -> Optional[list]:
        """The protein's row of the search files (in its accession's file), or None."""
        for key in _search_keys(uniprot):
            for row in self._search_rows(key):
                if row[self._key_columns["uniprot"]] == uniprot:
                    return row
        return None

    def _protein_json(self, uniprot: str) -> dict:
        if uniprot not in self._proteins:
            try:
                self._proteins[uniprot] = self.source.json(f"proteins/{uniprot}.json")
            except FileNotFoundError:
                if not self.meta.get("subset"):
                    raise
                # a subset (e.g. a download) knows the identifiers of every protein but has the files of some
                label = self._label(uniprot)
                subset = self._subset_proteins()
                has = ""
                if subset is not None:
                    shown = ", ".join(f"{u} ({self._label(u)})" for u in subset[:_SHOWN])
                    has = f"; its proteins: {shown}{f', ... ({len(subset)} in all)' if len(subset) > _SHOWN else ''}"
                raise KeyError(f"{uniprot}{f' ({label})' if label else ''} is not in this subset of the PocketVina "
                               f"Atlas ({self.source.location}){has}. Atlas() reads every protein") from None
        return self._proteins[uniprot]

    def protein(self, protein) -> ProteinInfo:
        """What the Atlas knows about a protein: gene, name, the AlphaFold model, its pockets and the diseases it is
        associated with (``ProteinInfo``; in a notebook a small card).

            atlas.protein("ABL1")          # also "P00519", "ABL1_HUMAN", "ENSG00000097007", 25, "HGNC:76"
            atlas.protein("ABL1").pockets  # a table: pocket, score, probability, plddt, residues

        protein: any identifier of the protein (see the module docstring).
        """
        return ProteinInfo(self._protein_json(self._uniprot(protein)))

    def _protein_hits(self, query: str) -> List[Tuple[int, str, str, str, str]]:
        """The proteins of the query's search key files (as the site's searchKeys picks them, in their order) whose
        identifiers match it (``_protein_score``), and an HGNC ID's protein: (score, UniProt accession, gene and protein
        name, which identifier, its text)."""
        q = query.casefold()
        af = _AF_ID.match(query.upper())
        accession = af.group(1) if af else None  # an AlphaFold model ID: its accession's file
        rows: Dict[str, list] = {}
        for key in _search_keys(accession or query):
            for row in self._search_rows(key):
                rows.setdefault(row[self._key_columns["uniprot"]], row)
        if not af and _ACC.match(query.upper()) and query.upper() not in rows:
            # a gene written like an accession (e.g. H2BC12) is filed under its first character
            for row in self._search_rows(query[0].lower()):
                rows.setdefault(row[self._key_columns["uniprot"]], row)
        c = self._key_columns
        hits = []
        for uniprot, row in rows.items():
            hit = (100, "AlphaFold model ID", query) if uniprot == accession else _protein_score(row, c, q)
            if hit:
                hits.append((hit[0] + _docked_bonus(row, c), uniprot, _protein_title(row, c), *hit[1:]))
        if _HGNC.match(query):
            for uniprot in self._hgnc_ids().get(query.upper(), []):
                row = self._protein_row(uniprot)
                hits.append((100 + (_docked_bonus(row, c) if row else 0), uniprot,
                             _protein_title(row, c) if row else uniprot, "HGNC ID", query.upper()))
        return hits

    def search(self, query, top: Optional[int] = SEARCH_TOP) -> pd.DataFrame:
        """Proteins and drugs whose identifiers match a query, best match first, as the Atlas website's search box
        ranks them: an identifier itself (a gene, UniProt accession, drug name, ChEMBL ID, catalogue number, ...)
        first, then for example a gene synonym, the start of a gene or of a drug's name, the start of a synonym, part of
        a name (case is ignored). One row per protein or drug, with kind ("protein" or "drug"), id (UniProt accession
        or PVL ID), name (gene and protein name, or the drug's name), matched (the identifier that matched, e.g. "gene",
        "synonym", "ChEMBL ID") and value (its text). An empty table if nothing matches. In a download
        (``Atlas("my_atlas")``) only its proteins are listed.

            atlas.search("imatinib")    # a drug's name (or the start or part of one)
            atlas.search("ABL1")        # a gene
            atlas.search("CHEMBL941")   # a ChEMBL ID

        query: any identifier ``protein=`` and ``ligand=`` take (see the module docstring), or the start or part of a
            name, gene, synonym, UniProt accession, PVL ID or catalogue number.
        top: only the first N rows (default 20); None: all.
        """
        query = _query(query, "query", '"imatinib", "ABL1" or "CHEMBL941"')
        top = _top(top)
        subset = self._subset_proteins()
        subset = set(subset) if subset is not None else None
        found: Dict[Tuple[str, str], tuple] = {}  # (kind, id) -> (score, name, matched, value) of its best match

        def keep(kind: str, ident: str, score: int, name: str, field: str, value: str):
            if (kind, ident) not in found or score > found[kind, ident][0]:
                found[kind, ident] = (score, name, field, value)

        for score, uniprot, name, field, value in self._protein_hits(query):
            if subset is None or uniprot in subset:  # a download: its proteins only
                keep("protein", uniprot, score, name, field, value)
        drugs = self._drug_index()
        for score, pvl, field, value in sorted(drugs.search(query), key=lambda hit: hit[1]):
            keep("drug", pvl, score, drugs.name(pvl) or pvl, field, value)
        # best score first; the same score: proteins (in the order of the search files, as on the site), then drugs (by
        # PVL ID)
        ranked = sorted(found.items(), key=lambda item: (-item[1][0], item[0][0] != "protein"))
        rows = [(kind, ident, *rest) for (kind, ident), (_, *rest) in ranked[:top]]
        return pd.DataFrame(rows, columns=list(SEARCH_COLUMNS))

    # --- drugs -----------------------------------------------------------------------------------------------------

    def _drug_index(self) -> _Drugs:
        """The drug identifiers of the release (read once per Atlas: three small tables of ``tables/``)."""
        if self._drugs is None:
            import pyarrow.parquet as pq

            def read(name: str, columns: List[str]) -> pd.DataFrame:
                return pq.read_table(self.source.file(f"tables/{name}.parquet"), columns=columns).to_pandas()

            self._drugs = _Drugs(read("compound_entries", _Drugs.ENTRY_COLUMNS),
                                 read("compound_entries_annotated", _Drugs.ANNOTATED_COLUMNS),
                                 read("compounds", _Drugs.PAGE_COLUMNS))
        return self._drugs

    def _pvl(self, ligand) -> _Match:
        """(PVL ID, the catalogue number the query named) of any drug identifier."""
        query = _query(ligand, "ligand", '"imatinib", "PVL00129" or a catalogue number')
        return self._drug_index().resolve(query)

    def _compound_json(self, pvl: str, every_target: bool = False) -> dict:
        return self.source.json(f"compounds/{pvl}{'.all' if every_target else ''}.json")

    def drug(self, ligand) -> DrugInfo:
        """The identifiers of a drug of the Atlas: its name, PVL ID, ChEMBL ID, catalogue numbers, InChIKey, SMILES,
        synonyms and ATC codes (``DrugInfo``; in a notebook a small card).

            atlas.drug("imatinib")       # also a synonym, its PVL ID, ChEMBL ID, catalogue number, InChIKey or SMILES
            atlas.drug("imatinib").pvl   # its PVL ID, "PVL00129"

        ligand: any identifier of the drug (see the module docstring).
        """
        match = self._pvl(ligand)
        return self._drug_info(match.target, match.entry)

    def _drug_info(self, pvl: str, entry: Optional[str] = None, data: Optional[dict] = None) -> DrugInfo:
        """The DrugInfo of a drug page (PVL ID) with the identifiers of one of its catalogue numbers (``entry``; None:
        the page's canonical entry); ``data``: the page's compounds/<PVL>.json if read already."""
        data = page = data if data is not None else self._compound_json(pvl)  # page: the one with the entry's record
        chosen = _form(data, entry)
        entries = self._drug_index().entries
        if chosen is None and entry and entry in entries.index:
            # an entry docked as something else than its drug (a salt counterion, a solvent, a fragment) lands on its
            # drug's page; its own record is on the page of the structure it was docked as
            docked = _str(entries.at[entry, "pvl_id"])
            if docked != pvl and docked in self._drug_index().pages.index:
                page = self._compound_json(docked)
                chosen = _form(page, entry)
        if chosen is None:
            page, forms = data, _forms(data)
            chosen = _form(data, data.get("canonical_cat")) or (forms[0] if forms else {})
        note = _str(page.get("note")) or _str(chosen.get("parent_text"))
        return DrugInfo(pvl=data["pvl"], name=_str(data.get("name")), chembl_id=_str(chosen.get("chembl_id")),
                        catalogue_numbers=self._drug_index().catalogue_numbers(data["pvl"], _str(chosen.get("cat"))),
                        inchikey=_str(chosen.get("inchikey")) or _str(data.get("docked_inchikey")),
                        parent_smiles=_str(chosen.get("parent_smiles")),
                        library_smiles=_str(chosen.get("smiles_docked_input")),
                        docked_smiles=_str(chosen.get("docked_smiles")) or _str(page.get("docked_smiles")),
                        synonyms=_strings(chosen.get("synonyms")), atc=_strings(chosen.get("atc")),
                        docked_as=_str(chosen.get("docked_as")) or _str(page.get("docked_as")) or "drug", note=note)

    def _names(self) -> Dict[str, str]:
        """PVL ID -> drug name, from tables/compounds.parquet (200 KB) or from the drug identifiers if those are
        loaded already."""
        if self._drugs is not None:
            return dict(zip(self._drugs.pages["pvl_id"], self._drugs.pages["name"]))
        if self._page_names is None:
            import pyarrow.parquet as pq

            table = pq.read_table(self.source.file("tables/compounds.parquet"), columns=["pvl_id", "name"])
            self._page_names = dict(zip(table.column("pvl_id").to_pylist(), table.column("name").to_pylist()))
        return self._page_names

    # --- scores ----------------------------------------------------------------------------------------------------

    def best_scores(self, protein=None, ligand=None, top: Optional[int] = None) -> pd.DataFrame:
        """Lowest Vina score of each pair of a protein or of a drug, highest-affinity first: one row per pair with
        protein_id (UniProt), gene, ligand_id (PVL), name (the drug), vina (kcal/mol), best_pocket, best_mode and
        ligand_efficiency (-vina / heavy atoms). No PoseBusters columns (as in a results directory).

            atlas.best_scores(protein="ABL1", top=20)     # the 20 highest-affinity drugs on this protein
            atlas.best_scores(ligand="imatinib", top=20)  # the 20 proteins with this drug's lowest Vina scores

        protein: any identifier of a protein; the drugs docked into it.
        ligand: any identifier of a drug; the proteins it was docked into (in a download: its proteins). Give exactly
            one of protein and ligand.
        top: only the first N rows, e.g. 20; None (default): all rows.
        """
        if (protein is None) == (ligand is None):
            raise TypeError("give exactly one of protein= and ligand=, e.g. atlas.best_scores(protein='ABL1') or "
                            "atlas.best_scores(ligand='imatinib')")
        top = _top(top)
        names = self._names()
        if protein is not None:
            data = self._protein_json(self._uniprot(protein))
            c = {name: i for i, name in enumerate(data["columns"])}
            gene = _str(data.get("gene")) or ""
            rows = [(data["uniprot"], gene, r[c["pvl"]], names.get(r[c["pvl"]]), r[c["vina"]], r[c["pocket"]],
                     r[c["mode"]], r[c["le"]]) for r in data["rows"][:top]]
        elif self._subset_proteins() is not None:
            # a download has no compounds/<PVL>.all.json (every protein of a drug): its proteins' rows of the drug
            pvl = self._pvl(ligand).target
            rows = []
            for uniprot in self._subset_proteins():
                data = self._protein_json(uniprot)
                c = {name: i for i, name in enumerate(data["columns"])}
                rows += [(uniprot, _str(data.get("gene")) or "", pvl, names.get(pvl), r[c["vina"]], r[c["pocket"]],
                          r[c["mode"]], r[c["le"]]) for r in data["rows"] if r[c["pvl"]] == pvl]
            rows = sorted(rows, key=lambda r: (r[4] is None, r[4] or 0, r[0]))[:top]
        else:
            match = self._pvl(ligand)
            data = self._compound_json(match.target, every_target=True)
            c = {name: i for i, name in enumerate(data["columns"])}
            genes, name = data.get("genes") or {}, names.get(match.target)
            rows = [(r[c["uniprot"]], genes.get(r[c["uniprot"]], ""), data["pvl"], name, r[c["vina"]], r[c["pocket"]],
                     r[c["mode"]], r[c["le"]]) for r in data["rows"][:top]]
        df = pd.DataFrame(rows, columns=list(BEST_SCORES_COLUMNS))
        for column in ("best_pocket", "best_mode"):
            df[column] = df[column].astype("int64")
        return df

    # --- one pair (used by atlas.select and atlas.show) -------------------------------------------------------------

    def _pair(self, protein, ligand) -> _Pair:
        """Where one protein x drug pair is: its protein JSON, its row, its pose file and its record index there."""
        uniprot, match = self._uniprot(protein), self._pvl(ligand)
        data = self._protein_json(uniprot)
        c = {name: i for i, name in enumerate(data["columns"])}
        row = next((r for r in data["rows"] if r[c["pvl"]] == match.target), None)
        if match.entry and (row is None or row[c["cat"]] != match.entry):
            row = self._entry_row(uniprot, match.entry, c) or row  # the named entry's own run, if it has one
        if row is None:
            raise ValueError(f"the PocketVina Atlas has no result for {match.target} "
                             f"({self._names().get(match.target) or 'this drug'}) in protein {uniprot}")
        row = {name: row[i] for name, i in c.items()}
        return _Pair(data, row, *_pose_ref(data, row))

    def _entry_row(self, uniprot: str, cat: str, columns: Dict[str, int]) -> Optional[list]:
        """The row of one catalogue number's own run (salt_forms/<U>.json), if the release has that file."""
        try:
            data = self.source.json(f"salt_forms/{uniprot}.json")
        except FileNotFoundError:
            return None
        c = {name: i for i, name in enumerate(data["columns"])}
        if c != columns:
            return None
        return next((r for r in data["rows"] if r[c["cat"]] == cat), None)

    def _pair_poses(self, protein, ligand) -> Tuple[Optional[str], Dict[int, str]]:
        """The stored poses of one pair: (the best pose as SDF text, or None if the release has none; the engine
        output (every mode) per pocket as PDBQT text, only pockets that have one)."""
        pair = self._pair(protein, ligand)
        blocks, top = self._poses([(pair.ref, pair.index, pair.row)])[0]
        return (_text(top) if top else None,
                {k: _text(block) for k, block in enumerate(blocks, 1) if block})

    def _sas(self, uniprot: str) -> Optional[dict]:
        """P2Rank's SAS points of a protein (sas/<U>.json.gz, ``io.sas.decode``), checked against sas/MANIFEST.json
        (sha256 and size); over HTTP the file is kept in the cache, so it is read from there afterwards. None when the
        release has none (an older release or download) or they cannot be read now, e.g. without network, or
        sas/MANIFEST.json lists them wrongly (a warning): the page is shown without them."""
        try:
            entry = self._sas_listing().get(f"{uniprot}.json.gz")
            if entry is None:
                return None
            if not _sas_entry(entry):
                raise ValueError(f"{SAS_MANIFEST} does not list it as [sha256, bytes]")
            path = self.source.file(f"sas/{uniprot}.json.gz", size=entry[1], sha256=entry[0])
            with open(path, "rb") as f:
                return sas.decode(f.read())
        except FileNotFoundError:
            return None
        except (OSError, ValueError, EOFError, zlib.error) as e:
            logger.warning("%s: the page is shown without P2Rank's SAS points, which cannot be read: %s", uniprot, e)
            return None

    def _sas_listing(self) -> dict:
        """sas/MANIFEST.json, {"<U>.json.gz": [sha256, bytes]}, read once; {} when the release has no sas/."""
        if self._sas_files is None:
            try:
                listed = self.source.json(SAS_MANIFEST)
            except FileNotFoundError:
                listed = {}
            self._sas_files = listed if isinstance(listed, dict) else {}
        return self._sas_files

    def _structure(self, data: dict) -> str:
        """PDB text of the AlphaFold model the drugs were docked into (structures/<AF id>.pdb.gz of the release, or
        .pdb); ``data``: the protein's JSON."""
        name = f"structures/{data['af_id']}.pdb"
        try:
            path = self.source.file(name + ".gz")
        except FileNotFoundError:
            path = self.source.file(name)
        with open(path, "rb") as f:
            raw = f.read()
        return (gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw).decode("utf-8", "replace")

    # --- select and show (the Selection and the page of a results directory) ---------------------------------------

    def select(self, protein, ligand, pocket=None, mode=1) -> Selection:
        """The poses of one protein x drug pair, to show, check or save: a ``Selection``, as ``Results.select`` gives
        for docking results (the same pocket and mode choices, the same methods). By default the pair's
        highest-affinity pose.

            pair = atlas.select(protein="ABL1", ligand="imatinib")               # the highest-affinity pose
            pair = atlas.select(protein="ABL1", ligand="imatinib", pocket="all")  # that pose in each pocket
            pair = atlas.select(protein="ABL1", ligand="imatinib", pocket=2, mode="all")  # every mode of pocket 2
            pair.show(); pair.bust(); pair.export(); pair.export(protein=True)

        protein: any identifier of the protein, e.g. "ABL1" or "P00519".
        ligand: any identifier of the drug, e.g. "imatinib", its PVL ID or a catalogue number (that entry's own run,
            if the release has one).
        pocket: a pocket number (P2Rank rank), e.g. 2; None (default): the pocket of the pair's highest-affinity pose;
            "all": every pocket that has poses.
        mode: 1 (default): the highest-affinity mode of the pocket; 2, 3, ...: that mode; "all": every mode of the
            pocket(s).

        The selection's ``protein`` is the UniProt accession and its ``ligand`` the PVL ID, so files are named like
        PV_P00519_PVL00129_top.sdf. Unknown proteins and drugs raise ``KeyError``; a pair without results, a pocket
        without poses or a mode out of range raise ``ValueError``.
        """
        pocket, mode = _check_pocket(pocket), _check_mode(mode)
        pair = self._pair(protein, ligand)
        uniprot, row = pair.protein["uniprot"], pair.row
        all_blocks = pose_records(self.source, [(pair.ref, pair.index, 0, 0)])[0][0]  # the top pose: only if needed
        blocks = {k: block for k, block in enumerate(all_blocks, 1) if block and pdbqt.parse_modes(block)}
        docked = sorted(blocks)
        best_pocket, best_mode = int(row["pocket"]), int(row["mode"])
        if best_pocket not in blocks:
            raise ValueError(f"ligand {row['pvl']!r} has no docked pose in protein {uniprot!r} (the PocketVina Atlas "
                             f"has no poses of this pair in its pose file)")
        poses = _pick_poses(blocks.get, docked, pocket, mode, best_pocket, best_mode, uniprot, row["pvl"])
        has_best = any((p.pocket, p.mode) == (best_pocket, best_mode) for p in poses)
        top = top_record(self.source, pair.ref["file"], int(row["top_off"]), int(row["top_len"])) if has_best else None
        library, docked_smiles = self._entry_smiles(row["cat"], row["pvl"])
        return Selection(protein=uniprot, ligand=row["pvl"], ligand_idx=int(row["cidx"]),
                         smiles=library or docked_smiles, smiles_docked=docked_smiles or library, pocket=pocket,
                         mode=mode, best_pocket=best_pocket, best_mode=best_mode, vina=_short(row["vina"]),
                         docked_pockets=docked, poses=poses, top_sdf=_text(top) if top else None,
                         source=_PairSource(self, pair))

    def _poses(self, rows: List[Tuple[dict, int, dict]],
               map=map) -> List[Tuple[List[Optional[bytes]], Optional[bytes]]]:
        """(mode blocks per pocket, top pose) of rows of a protein's JSON, each given as (pose file ref, record index,
        row), read together (``pose_records``); ``map``: how the requests of a round run (default: one after another,
        on one kept connection)."""
        return pose_records(self.source, [(ref, index, int(row["top_off"]), int(row["top_len"]))
                                          for ref, index, row in rows], map)

    def _entry_smiles(self, cat: str, pvl: str) -> Tuple[Optional[str], Optional[str]]:
        """(library SMILES as screened, SMILES of the molecule docked) of a catalogue number; the drug page's docked
        SMILES if the release does not list the catalogue number."""
        entries = self._drug_index().entries
        if cat in entries.index:
            return _str(entries.at[cat, "input_smiles"]), _str(entries.at[cat, "docked_smiles"])
        pages = self._drug_index().pages
        return None, _str(pages.at[pvl, "docked_smiles"]) if pvl in pages.index else None

    def show(self, protein, *, path=None, molstar: str = "inline", theme: str = "auto", height: int = 760,
             max_ligands: Optional[int] = MAX_LIGANDS):
        """The protein's page, as ``Results.show`` draws docking results: the 3D view (pocket chips, modes, legend,
        contacts, Save image) beside the tabs Top hits (its highest-affinity drugs with a PoseBusters-valid pose,
        as on the Atlas site; pick one there), Selected pair, Pockets and Download, with the protein's information,
        its associated diseases and each drug's identifiers. In Jupyter, Colab or VSCode it is shown below the cell;
        elsewhere an HTML file is written (open it in a browser), e.g. PV_P00519.html, and its path returned.
        ``pair.show()`` shows it with that pair selected (also when the pair is not among Top hits).

            atlas.show(protein="ABL1")
            atlas.show(protein="ABL1", max_ligands=50, path="ABL1.html")

        protein: any identifier of the protein, e.g. "ABL1" or "P00519".
        path: write the page to this HTML file instead, e.g. "ABL1.html"; returns the path.
        molstar: "inline" (default): the 3D viewer's code is in the page (works offline); "url": loaded from the
            PocketVina Atlas website when the page opens (a much smaller page; needs internet).
        theme: "auto" (default: as the browser), "light" or "dark".
        height: height of the view in a notebook, in pixels, e.g. 900.
        max_ligands: drugs listed in Top hits, highest-affinity first, e.g. 50 (default 25: the poses of each are
            fetched); None: all.
        """
        return self._show(self._uniprot(protein), path=path, molstar=molstar, theme=theme, height=height,
                          max_ligands=max_ligands)

    def _show(self, uniprot: str, *, selection: Optional[Selection] = None, pair: Optional[_Pair] = None, path=None,
              molstar: str = "inline", theme: str = "auto", height: int = 760,
              max_ligands: Optional[int] = MAX_LIGANDS):
        """atlas.show and pair.show (``selection`` and its ``pair`` given: the page with that pair selected)."""
        from . import workspace

        height, max_ligands = workspace._check_options(molstar, theme, height, max_ligands)
        data = self._page(uniprot, max_ligands, pair)
        protein = data["proteins"][0]
        gene = protein["info"]["title"]
        if selection is not None:
            data["initial"] = workspace._initial(selection, uniprot, protein["docked"])
            drug = data["ligands"][str(int(selection.ligand_idx))]["id"]
            name, title = selection.name, f"PocketVina Atlas: {gene} x {drug}"
        else:
            name, title = f"PV_{name_part(uniprot)}", f"PocketVina Atlas: {gene}"
        page = workspace.workspace_html(data, molstar=molstar, theme=theme, title=title)
        if len(page) > workspace.LARGE_PAGE:
            logger.warning("the PocketVina view is %.0f MB (%d drugs): a browser or notebook may be slow with it; a "
                           "smaller max_ligands makes it smaller", len(page) / 1e6, len(protein["rows"]))
        return workspace._output(page, name, path, height)

    def _page(self, uniprot: str, max_ligands: Optional[int], pair: Optional[_Pair] = None) -> dict:
        """The page's data (``workspace.workspace_data``'s format): the protein (its AlphaFold model, pockets with
        pLDDT, information, P2Rank's SAS points when the release has them) and its Top hits as the Atlas site lists
        them (its rows of drugs whose highest-affinity pose is PoseBusters-valid, best first; ``max_ligands`` of them),
        and ``pair``'s row if it is not among them, each with its poses in every pocket, its top pose (SDF), its stored
        PoseBusters result and its drug's identifiers. As on the site, ``rank`` and ``n`` count the protein's rows of
        drugs (a row docked as something else than a drug, or a catalogue number's own run, has no rank). The rows'
        poses (``pose_records``: two rounds of reads), their drug pages, the structure and the SAS points are fetched
        at the same time (``PAGE_THREADS``)."""
        from .workspace import _initial, _page_sas, ligand_svg

        data = self._protein_json(uniprot)
        rows = [dict(zip(data["columns"], r)) for r in data["rows"]]  # best first
        drugs = [row for row in rows if not int(row["flags"] or 0) & NON_DRUG]
        rank_of = {row["cidx"]: n for n, row in enumerate(drugs, 1)}
        hits = [row for row in drugs if _int(row["pb"]) == 0]
        listed = list(hits[:max_ligands] if max_ligands is not None else hits)
        if pair is not None and all(row["cidx"] != pair.row["cidx"] for row in listed):
            listed.append(pair.row)
        canonical = {row["cidx"] for row in rows}  # any other row: a catalogue number's own run (salt_forms/)
        title = _str(data.get("gene")) or uniprot
        top_note = TOP_HITS
        if max_ligands is not None and len(hits) > max_ligands:
            top_note += f" The top {max_ligands} of {len(hits):,} on {title} (max_ligands={max_ligands})."
            logger.info("%s: Top hits lists the top %d of %d drugs with a PoseBusters-valid pose (max_ligands=%d)",
                        uniprot, max_ligands, len(hits), max_ligands)
        self._drug_index()  # loaded once, before the threads use it

        def drug(row: dict):
            page = self._compound_json(row["pvl"])
            return self._drug_info(row["pvl"], row["cat"], page), _across(page, uniprot)

        with ThreadPoolExecutor(max_workers=PAGE_THREADS) as pool:
            later: list = []  # the structure and the drug pages: asked for after the first round of pose reads

            def ask_later():
                if not later:
                    later.append(pool.submit(self._structure, data))
                    later.append(pool.submit(self._sas, uniprot))
                    later.extend(pool.submit(drug, row) for row in listed)

            def first_round_first(fn, items):
                futures = [pool.submit(fn, item) for item in items]
                ask_later()
                return [future.result() for future in futures]

            poses = self._poses([(*_pose_ref(data, row), row) for row in listed], first_round_first)
            ask_later()
            structure, points = later[0].result(), later[1].result()
            drugs_of_rows = [future.result() for future in later[2:]]
        checks, statuses = list(self.meta.get("pb_checks") or []), list(self.meta.get("pb_status") or [])
        items, ligands = [], {}
        for row, (blocks, top), (info, across) in zip(listed, poses, drugs_of_rows):
            pockets = []
            for k, block in enumerate(blocks, 1):
                modes = pdbqt.parse_modes(block) if block else []
                if modes:
                    pockets.append({"rank": k, "pdbqt": _text(block),
                                    "modes": [[_num(m.affinity), _num(m.rmsd_lb), _num(m.rmsd_ub)] for m in modes]})
            items.append({"lig": int(row["cidx"]), "vina": _num(row["vina"]), "pocket": int(row["pocket"]),
                          "mode": int(row["mode"]), "le": _num(row["le"], 3), "rank": rank_of.get(row["cidx"]),
                          # as on the site, not for an entry's own run (the drug page ranks the drug's runs)
                          "across": across if row["cidx"] in canonical else None,
                          "sdf": _text(top) if top else None, "pockets": pockets, "pb": _pb(row, checks, statuses)})
            smiles = info.library_smiles or info.docked_smiles
            ligands[str(int(row["cidx"]))] = {
                "id": info.name or row["pvl"], "smiles": smiles, "docked": info.docked_smiles,
                "inchikey": info.inchikey, "svg": ligand_svg(info.docked_smiles or smiles),
                "name": name_part(row["pvl"]), "ids": _ligand_ids(info, self.meta)}
        pockets = sorted(data.get("pockets") or [], key=lambda p: p["rank"])
        docked = [int(p["rank"]) for p in pockets]
        protein = {"id": uniprot, "name": name_part(uniprot), "key": f"structures/{data['af_id']}.pdb",
                   "pdb": structure, "docked": docked, "n": len(drugs), "rows": items, "topNote": top_note,
                   "pockets": [{"rank": int(p["rank"]), "score": _num(p.get("score"), 2),
                                "probability": _num(p.get("probability"), 3),
                                "center": [_num(x, 2) for x in p.get("center") or []],
                                "residues": _strings(p.get("residues")), "docked": True,
                                "plddt": _num(p.get("plddt"), 1)} for p in pockets],
                   "info": _protein_info(data, self.meta)}
        if points is not None:
            protein["sas"] = _page_sas(points)
        return {"format": 1, "proteins": [protein], "ligands": ligands, "maxLigands": max_ligands,
                "initial": _initial(None, uniprot, docked)}

    # --- download --------------------------------------------------------------------------------------------------

    def download(self, proteins, out) -> str:
        """Download what the Atlas has on some proteins into a folder: a part of the release that ``Atlas(out)``
        reads offline, with every method (search, protein, drug, best_scores, select, show) for these proteins.
        Most of it is the proteins' pose files, which are large: before it starts, it says how many files and MB it
        copies; a progress bar follows.

            atlas.download(proteins=["ABL1", "EGFR"], out="my_atlas")
            Atlas("my_atlas").best_scores(protein="ABL1")   # offline

        proteins: identifiers of proteins, e.g. ["ABL1", "EGFR"], or one, "ABL1".
        out: the folder to write, e.g. "my_atlas": a new or empty folder, or one that download() wrote before (its
            proteins stay; files there that are complete and correct are kept, so a download that stopped goes on
            where it stopped).

        Returns the folder's path. Every file is a byte-identical copy of the release's file, checked against the
        release's MANIFEST.json (size and sha256; salt_forms/ and LICENSE, which it does not list, by size; the SAS
        points sas/<U>.json.gz against sas/MANIFEST.json, which the folder gets for its proteins). In the folder,
        best_scores(ligand=...) and search() list its proteins only (and every drug).
        The folder carries the release's LICENSE (the licence of the data), if the release has one.
        """
        uniprots = self._protein_list(proteins)
        out = os.path.abspath(os.path.expanduser(os.fspath(out)))
        uniprots = list(dict.fromkeys(_earlier_download(out, self.release) + uniprots))
        files = self._subset_files(uniprots)
        total = sum(size or 0 for size, _ in files.values())
        there = sum(1 for rel, (size, _) in files.items() if _has_size(os.path.join(out, *rel.split("/")), size))
        print(f"PocketVina Atlas {self.release}: {len(files):,} files, {total / 1e6:,.1f} MB, for {len(uniprots)} "
              f"protein{'s' if len(uniprots) != 1 else ''} into {out}"
              + (f" ({there:,} of them there already: kept if their sha256 is right)" if there else ""), flush=True)
        copied = self._copy_files(files, out, total)
        # sas/MANIFEST.json of the folder: its proteins' SAS files, as the release lists them (Atlas._sas reads it)
        sas_listed = {rel[4:]: [digest, n] for rel, (n, digest) in copied.items() if rel.startswith("sas/")}
        sas_manifest = json.dumps(dict(sorted(sas_listed.items())), indent=0).encode("utf-8") if sas_listed else None
        if sas_manifest:
            _write_bytes(os.path.join(out, *SAS_MANIFEST.split("/")), sas_manifest)
        subset = {"proteins": uniprots, "of": self.source.location, "release": self.release,
                  "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "pocketvina": __version__,
                  "note": SUBSET_NOTE}
        meta = (json.dumps(dict(self.meta, subset=subset), indent=1, ensure_ascii=False) + "\n").encode("utf-8")
        listed = {rel: {"bytes": n, "sha256": digest} for rel, (n, digest) in copied.items()}
        listed["meta.json"] = {"bytes": len(meta), "sha256": hashlib.sha256(meta).hexdigest()}
        if sas_manifest:
            listed[SAS_MANIFEST] = {"bytes": len(sas_manifest), "sha256": hashlib.sha256(sas_manifest).hexdigest()}
        manifest = {"release": self.release, "subset": subset, "files": dict(sorted(listed.items()))}
        _write_bytes(os.path.join(out, "MANIFEST.json"), (json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
        _write_bytes(os.path.join(out, "meta.json"), meta)  # last: the folder is a release once it has meta.json
        logger.info("PocketVina Atlas %s in %s: %s, every file checked", self.release, out, ", ".join(uniprots))
        return out

    def _protein_list(self, proteins) -> List[str]:
        """UniProt accessions of one protein identifier or a list of them."""
        if isinstance(proteins, (str, numbers.Integral)):
            proteins = [proteins]
        try:
            items = list(proteins)
        except TypeError:
            raise TypeError(f"proteins must be protein identifiers, e.g. ['ABL1', 'EGFR'], not "
                            f"{type(proteins).__name__}") from None
        if not items:
            raise ValueError("proteins is empty: give protein identifiers, e.g. ['ABL1', 'EGFR']")
        return list(dict.fromkeys(self._uniprot(p) for p in items))

    def _subset_files(self, uniprots: List[str]) -> Dict[str, Tuple[Optional[int], Optional[str]]]:
        """The files of a download of these proteins: path -> (bytes, sha256) as the release's MANIFEST.json lists
        them; for a file it does not list (salt_forms/, LICENSE), its size and None. salt_forms/ and LICENSE are left
        out if the source does not have them (LICENSE even if its MANIFEST.json lists it, as a download's does). The
        proteins' SAS points (sas/<U>.json.gz) as sas/MANIFEST.json lists them, if the source has them."""
        try:
            listed = self.source.json("MANIFEST.json")["files"]
        except FileNotFoundError:
            raise FileNotFoundError(f"{self.source.location} has no MANIFEST.json (the list of the release's files "
                                    f"with their size and sha256), which download() needs") from None
        names = []
        for uniprot in uniprots:
            data = self._protein_json(uniprot)
            structure = f"structures/{data['af_id']}.pdb"  # the form the client reads: .pdb.gz, else .pdb
            names += [f"proteins/{uniprot}.json.gz", f"salt_forms/{uniprot}.json.gz", data["poses"]["file"]]
            names += [data["poses_redock"]["file"]] if data.get("poses_redock") else []
            names.append(structure + ".gz" if structure + ".gz" in listed or structure not in listed else structure)
        names += sorted(rel for rel in listed if rel.startswith("compounds/") and rel.endswith(".json.gz")
                        and not rel.endswith(".all.json.gz"))
        names += sorted(rel for rel in listed if rel.startswith("search/"))
        names += [f"tables/{name}.parquet" for name in SUBSET_TABLES] + list(SUBSET_FILES)
        files = {}
        for rel in dict.fromkeys(names):
            if rel in listed and rel != "LICENSE":  # LICENSE: its size is asked even when listed (is it there?)
                files[rel] = (int(listed[rel]["bytes"]), listed[rel]["sha256"])
                continue
            try:
                files[rel] = (self.source.size(rel), (listed.get(rel) or {}).get("sha256"))
            except FileNotFoundError:
                # salt_forms/ is in the public copy of the release only; an older release or download has no LICENSE
                if not (rel.startswith("salt_forms/") or rel == "LICENSE"):
                    raise
        try:
            sas_files = self._sas_listing()
        except ValueError as e:  # not JSON: the download goes on without them
            logger.warning("the SAS points (sas/) are not copied: %s cannot be read: %s", SAS_MANIFEST, e)
            sas_files = {}
        for uniprot in uniprots:
            entry = sas_files.get(f"{uniprot}.json.gz")
            if entry is not None and _sas_entry(entry):
                files[f"sas/{uniprot}.json.gz"] = (entry[1], entry[0])
            elif entry is not None:
                logger.warning("%s: not copied, %s does not list it as [sha256, bytes]", f"sas/{uniprot}.json.gz",
                               SAS_MANIFEST)
        return files

    def _copy_files(self, files: Dict[str, Tuple[Optional[int], Optional[str]]], out: str,
                    total: int) -> Dict[str, Tuple[int, str]]:
        """Copy the files into ``out`` (``THREADS`` at a time, with a progress bar); a copy already there that is as
        the release says (size, sha256) is kept. Returns path -> (bytes, sha256) of every file."""
        from tqdm import tqdm

        lock, stop = threading.Lock(), threading.Event()
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=f"PocketVina Atlas {self.release}")  # MB, as said

        def progress(n: int):
            if stop.is_set():
                raise _Stopped()
            with lock:
                bar.update(n)

        def one(rel: str) -> Tuple[str, Tuple[int, str]]:
            size, sha256 = files[rel]
            target = os.path.join(out, *rel.split("/"))
            kept = _kept(target, size, sha256)
            if kept:
                progress(kept[0])
                return rel, kept
            return rel, self.source.copy(rel, target, size=size, sha256=sha256, progress=progress)

        try:
            with ThreadPoolExecutor(max_workers=THREADS) as pool:
                futures = [pool.submit(one, rel) for rel in files]
                try:
                    return dict(future.result() for future in as_completed(futures))
                except BaseException:
                    stop.set()  # the copies under way stop too
                    for future in futures:
                        future.cancel()
                    raise
        finally:
            bar.close()


# --- download helpers ----------------------------------------------------------------------------------------------

class _Stopped(Exception):
    """A copy of a download stops because another one failed."""


def _earlier_download(out: str, release: str) -> List[str]:
    """The proteins of an earlier download of this release into ``out`` ([] for a new or empty folder); ValueError if
    ``out`` holds something else (download() writes only into a folder of its own)."""
    rule = f"download() writes into a new or empty folder, or into one it wrote before from release {release}"
    meta_path = os.path.join(out, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "rb") as f:
                meta = json.load(f)
        except ValueError:
            meta = None
        meta = meta if isinstance(meta, dict) else {}
        about = meta.get("release") if isinstance(meta.get("release"), dict) else {}
        subset, label = meta.get("subset"), about.get("label")
        if isinstance(subset, dict) and isinstance(subset.get("proteins"), list) and label == release:
            return [str(u) for u in subset["proteins"]]
        what = f"release {label}" if label and label != release else "a PocketVina Atlas release it did not write"
        raise ValueError(f"{out} holds {what}: {rule}")
    if os.path.isdir(out):
        names = [name for name in os.listdir(out) if ".tmp-" not in name]
        # a download that stopped: files of its own. A LICENSE alone is someone's own, not one (it would be replaced):
        # a download starts copying LICENSE after every other file, so their folders are there with it
        others = sorted(name for name in names if name not in _SUBSET_FOLDERS + SUBSET_FILES + ("MANIFEST.json",)
                        or names == ["LICENSE"])
        if others:
            raise ValueError(f"{out} is not empty (it has {', '.join(others[:3])}{', ...' if len(others) > 3 else ''})"
                             f": {rule}")
    elif os.path.lexists(out):
        raise ValueError(f"{out} is a file, not a folder: {rule}")
    return []


def _sas_entry(entry) -> bool:
    """Whether an entry of sas/MANIFEST.json is [sha256, bytes]."""
    return (isinstance(entry, list) and len(entry) == 2 and isinstance(entry[0], str) and isinstance(entry[1], int)
            and not isinstance(entry[1], bool))


def _has_size(path: str, size: Optional[int]) -> bool:
    """Whether the file is there with this size (any size when the release does not say)."""
    try:
        n = os.path.getsize(path)
    except OSError:
        return False
    return size is None or n == size


def _kept(path: str, size: Optional[int], sha256: Optional[str]) -> Optional[Tuple[int, str]]:
    """(bytes, sha256) of a file already in a download folder if it is as the release says (its size, and its
    sha256 when the release's MANIFEST.json lists it), else None (it is copied again)."""
    if not os.path.isfile(path) or not _has_size(path, size):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    return (os.path.getsize(path), digest) if sha256 is None or digest == sha256 else None


def _write_bytes(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class _PairSource:
    """Where a ``Selection`` of the Atlas gets its protein and its page (the source hook of ``Selection``): the
    protein is the AlphaFold model of the release, the page is ``Atlas.show``'s with the pair selected."""

    def __init__(self, atlas: Atlas, pair: _Pair):
        self.atlas, self.pair = atlas, pair

    def protein_pdb(self, protein: str) -> str:
        return self.atlas._structure(self.pair.protein)

    def show(self, selection: Selection, *, path=None, molstar: str = "inline", theme: str = "auto",
             height: int = 760, max_ligands: Optional[int] = MAX_LIGANDS):
        return self.atlas._show(self.pair.protein["uniprot"], selection=selection, pair=self.pair, path=path,
                                molstar=molstar, theme=theme, height=height, max_ligands=max_ligands)


# --- the page's fields (Atlas.show) --------------------------------------------------------------------------------

def _pose_ref(protein: dict, row: dict) -> Tuple[dict, int]:
    """The pose file of a row of a protein's JSON (``poses``, or ``poses_redock`` when pf is 1) and the row's record
    index there (cidx, less the file's cidx_offset)."""
    ref = protein["poses_redock"] if row["pf"] == 1 else protein["poses"]
    if not ref:
        raise ValueError(f"{protein['uniprot']}: the release has no pose file for this pair (pf={row['pf']})")
    return ref, int(row["cidx"]) - int(ref.get("cidx_offset") or 0)


def _num(x, digits: Optional[int] = None):
    """A JSON number of the page (None for null or NaN)."""
    if not isinstance(x, (int, float)) or isinstance(x, bool) or x != x:
        return None
    return round(float(x), digits) if digits is not None else float(x)


def _pb(row: dict, checks: List[str], statuses: List[str]) -> Optional[dict]:
    """The stored PoseBusters result of a row's highest-affinity pose as the page shows it (``workspace._posebusters``:
    status, valid, passed, total, failed); None if PoseBusters was not run on it (pb null). ``checks``: meta.json
    pb_checks (bit i of pb_mask: check i passed); ``statuses``: meta.json pb_status (the pb codes' names)."""
    code = _int(row.get("pb"))
    if code is None:
        return None
    status = statuses[code] if 0 <= code < len(statuses) else "error"
    valid = {0: True, 1: False}.get(code)
    mask = _int(row.get("pb_mask"))
    if mask is None or not checks:
        return {"status": status, "valid": valid, "passed": 0, "total": 0, "failed": []}
    failed = [name for i, name in enumerate(checks) if not (mask >> i) & 1]
    return {"status": status, "valid": valid, "passed": len(checks) - len(failed), "total": len(checks),
            "failed": failed}


def _across(compound: dict, uniprot: str) -> Optional[list]:
    """[rank of the protein among the drug's docked proteins (best first), their number] from the drug's page
    (compounds/<PVL>.json: its best targets); [None, their number, the number of best targets] if the protein is not
    among them (the page then says so, as the site does)."""
    columns = {name: i for i, name in enumerate(compound.get("columns") or [])}
    if "uniprot" not in columns or "vina" not in columns:
        return None
    vina = columns["vina"]
    rows = sorted(compound.get("rows") or [], key=lambda r: (r[vina] is None, r[vina] or 0))
    rank = next((n for n, r in enumerate(rows, 1) if r[columns["uniprot"]] == uniprot), None)
    total = int(compound.get("n_targets") or len(rows))
    if rank is None:
        return [None, total, len(rows)] if total > len(rows) else None
    return [rank, total] if total > 1 else None


_ID_TIPS = {
    "PVL ID": "PocketVina Atlas identifier for this docked structure (PVL = PocketVina ligand)",
    "catalogue numbers": "Library entries (catalogue numbers) docked as this structure or whose drug it is",
    "library SMILES": "As screened: the library entry, e.g. with its salt",
    "docked SMILES": "The molecule docked",
}
_FEW = 3  # synonyms shown before the list is opened


def _ligand_ids(info: DrugInfo, meta: dict) -> List[dict]:
    """A drug's identifiers as the page lists them (rows of label, value and an optional link, tooltip and short
    text for a collapsed row): its name, then the rows of ``DrugInfo`` (atlas.drug) in their order, empty ones left
    out."""
    docked_as = meta.get("docked_as") or {}
    out = [{"label": "Name", "value": info.name}] if info.name else []
    for label, value in info._rows():
        row = {"label": label[:1].upper() + label[1:], "value": value}
        if label in _ID_TIPS:
            row["tip"] = _ID_TIPS[label]
        if label == "ChEMBL ID" and _CHEMBL_ID.match(info.chembl_id or ""):
            row["href"] = info.chembl_url
        elif label == "synonyms" and len(info.synonyms) > _FEW:
            row["short"] = f"{', '.join(info.synonyms[:_FEW])} +{len(info.synonyms) - _FEW}"
        elif label == "docked as":
            what = (docked_as.get("labels") or {}).get(info.docked_as) or info.docked_as.replace("_", " ")
            row["value"] = what + (f": {info.note}" if info.note else "")
            tip = (docked_as.get("tooltips") or {}).get(info.docked_as)
            if tip:
                row["tip"] = tip
        out.append(row)
    return out


def _protein_info(data: dict, meta: dict) -> dict:
    """The protein's information as the Atlas site's protein page shows it (app/protein/[id]/page.tsx): the gene (or
    the accession) as title, the protein's name, links to UniProt and AlphaFold DB, the ChEMBL class and the model
    (length, mean pLDDT, compartment, pockets) under Annotations, and the diseases Open Targets associates with it
    (components/OpenTargets.tsx; None if the release has none)."""
    info = ProteinInfo(data)
    links = [{"label": "UniProt", "value": info.uniprot, "href": info.uniprot_url}]
    if info.af_id:
        links.append({"label": "AlphaFold", "value": re.sub(r"-model_v\d+$", "", info.af_id),
                      "href": info.alphafold_url})
    facts = []
    protein_class = (_str(data.get("chembl_class_paths")) or "").split(" | ")[0] or info.protein_class
    if protein_class:
        facts.append({"label": "ChEMBL class", "value": protein_class})
    docked = ((meta.get("stats") or {}).get("pockets_per_protein") or {}).get("value")
    model = []
    if isinstance(data.get("length"), (int, float)):
        model.append(f"{int(data['length']):,} aa")
    if info.mean_plddt is not None:
        model.append(f"mean pLDDT {info.mean_plddt:.1f}")
    if _str(data.get("compartment")):
        model.append(data["compartment"])
    if isinstance(data.get("n_p2rank_pockets"), int):
        model.append(f"{data['n_p2rank_pockets']} P2Rank pockets" + (f", top {docked} docked" if docked else ""))
    if model:
        facts.append({"label": "Model", "value": " · ".join(model)})
    return {"title": info.gene or info.uniprot, "name": info.name, "links": links, "facts": facts,
            "diseases": _diseases(data.get("open_targets"))}


def _diseases(ot: Optional[dict]) -> Optional[dict]:
    """The protein's associated diseases (Open Targets, as the release stores them: the top ones with their scores)
    for the page, with links to Open Targets; None without a valid Ensembl ID and disease IDs (they go into links)."""
    ot = ot or {}
    ensembl, diseases = _str(ot.get("ensembl_id")), ot.get("diseases") or []
    if not ensembl or not _ENSG.match(ensembl) or not diseases or \
            not all(isinstance(d.get("id"), str) and _DISEASE_ID.match(d["id"]) for d in diseases):
        return None
    association = f" ({ot['association']})" if _str(ot.get("association")) else ""
    release = f" {ot['release']}" if ot.get("release") else ""
    return {"title": "Associated diseases (Open Targets)",
            "tip": "Target–disease association score from Open Targets (0–1). Not related to docking results.",
            "items": [{"name": _str(d.get("name")) or d["id"], "score": _num(d.get("score"), 3),
                       "href": OPEN_TARGETS_EVIDENCE_URL.format(ensembl, d["id"])} for d in diseases],
            "all": {"text": "View all associations on Open Targets", "href": OPEN_TARGETS_URL.format(ensembl)},
            "note": f"Open Targets Platform{release} · overall association score{association}"}


def _forms(data: dict) -> List[dict]:
    """The catalogue numbers of a drug page (compounds/<PVL>.json): its entries and those docked instead of a drug."""
    return list(data.get("entries") or []) + list(data.get("docked_instead_of_drug") or [])


def _form(data: dict, cat: Optional[str]) -> Optional[dict]:
    """The record of one catalogue number on a drug page, or None."""
    return next((f for f in _forms(data) if cat and f.get("cat") == cat), None)


def _text(data: bytes) -> str:
    # surrogateescape: text.encode("utf-8", "surrogateescape") gives the stored bytes back, as in io.results
    return data.decode("utf-8", "surrogateescape")


__all__ = ["Atlas", "ProteinInfo", "DrugInfo", "ATLAS_URL"]
