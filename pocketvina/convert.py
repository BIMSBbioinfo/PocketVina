"""Docked PDBQT pose -> RDKit molecule -> SDF, in memory (no files, no OpenBabel).

Atoms and coordinates come from the pose; bond orders, charges and hydrogens come from the docked SMILES
(template matching as in RDKit's ``AssignBondOrdersFromTemplate``); stereo comes from the pose's 3D coordinates.
Poses of ligands prepared with Meeko (``REMARK SMILES`` lines) are rebuilt with Meeko's ``RDKitMolCreate`` when
Meeko is installed and rebuilds the molecule of the SMILES. The result is checked against the SMILES by InChIKey
(``same_molecule``).
"""
import contextlib
import io
import logging
import re
from typing import Dict, List, Optional, Tuple, Union

from rdkit import Chem, rdBase
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Geometry import Point3D

logger = logging.getLogger(__name__)

# AutoDock 4 / Vina / Meeko atom types (case-sensitive: "NA" is an acceptor nitrogen, "Na" sodium)
AD_ELEMENTS = {
    "C": "C", "A": "C", "N": "N", "NA": "N", "NS": "N", "NX": "N", "O": "O", "OA": "O", "OS": "O",
    "S": "S", "SA": "S", "H": "H", "HD": "H", "HS": "H", "P": "P", "F": "F", "Cl": "Cl", "CL": "Cl",
    "Br": "Br", "BR": "Br", "I": "I", "Si": "Si", "B": "B", "Se": "Se", "Mg": "Mg", "MG": "Mg", "Ca": "Ca",
    "Mn": "Mn", "Fe": "Fe", "Zn": "Zn", "Na": "Na", "K": "K", "Cu": "Cu", "Co": "Co", "Ni": "Ni",
}
_MEEKO_GLUE_CARBON = re.compile(r"^CG\d$")  # Meeko macrocycle ring-closure carbons
_MEEKO_PSEUDO = re.compile(r"^(G\d|W)$")  # Meeko ring-closure dummies and hydrated-docking waters: not atoms
_MEEKO_SMILES = re.compile(r"^REMARK SMILES\s", re.M)
_ISOTOPE = re.compile(r"\[\d")
_ELEMENTS = frozenset(Chem.GetPeriodicTable().GetElementSymbol(z) for z in range(1, 119))
_MAX_TEMPLATE_MATCHES = 1000  # as AssignBondOrdersFromTemplate (GetSubstructMatches default)
_MAX_STEREO_MATCHES = 256
_UNSPECIFIED_BOND = (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
_PERIODIC_TABLE = Chem.GetPeriodicTable()
# proximity bonding joins atoms up to their covalent radii + 0.45 A apart (C-C 1.97 A with RDKit's current radii); a
# pair the SMILES does not bond is a 1,3 ring contact only if it is farther apart than a bond (bond radii + margin)
_CONTACT_MARGIN = 0.2


def _remove_hs_params(**flags) -> "Chem.RemoveHsParameters":
    ps = Chem.RemoveHsParameters()
    ps.showWarnings = False
    for name, value in flags.items():
        setattr(ps, name, value)
    return ps


# pose: every hydrogen (their coordinates are kept aside); SMILES: deuterium and stereo-defining H too
_POSE_HS = _remove_hs_params(removeDegreeZero=True, removeHigherDegrees=True, removeIsotopes=True)
_SMILES_HS = _remove_hs_params(removeIsotopes=True, removeDefiningBondStereo=True, removeWithWedgedBond=True)


class ConversionError(Exception):
    """A PDBQT pose could not be turned into the molecule of its SMILES."""


@contextlib.contextmanager
def _quiet():
    """Silence RDKit's log (PDB/SMILES parser messages, InChI stereo warnings) inside the block."""
    block = rdBase.BlockLogs()
    try:
        yield
    finally:
        del block


def _as_text(model: Union[bytes, str]) -> str:
    if isinstance(model, (bytes, bytearray, memoryview)):
        return bytes(model).decode("utf-8", errors="replace")
    if isinstance(model, str):
        return model
    raise ConversionError(f"expected PDBQT bytes or str, got {type(model).__name__}")


def _first_model(text: str) -> str:
    """``text`` up to and including the first ``ENDMDL`` (all of it if there is none)."""
    m = re.search(r"^ENDMDL.*$", text, re.M)
    return text[:m.end()] + "\n" if m else text


def _element(ad_type: str, name: str) -> Optional[str]:
    """Element symbol of an AutoDock atom type (atom name as fallback); None for Meeko pseudo-atoms."""
    if ad_type in AD_ELEMENTS:
        return AD_ELEMENTS[ad_type]
    if _MEEKO_GLUE_CARBON.match(ad_type):
        return "C"
    if _MEEKO_PSEUDO.match(ad_type):
        return None
    for cand in (ad_type, ad_type.capitalize(), re.sub(r"[^A-Za-z]", "", name)[:1]):
        if cand in _ELEMENTS:
            return cand
    raise ConversionError(f"unknown AutoDock atom type {ad_type!r} (atom {name.strip()!r})")


def _pose_atoms(text: str) -> List[Tuple[str, str, Tuple[float, float, float]]]:
    """(atom name, element, xyz) of the ligand atoms of one PDBQT model; flexible residues are skipped."""
    atoms = []
    in_flex = False
    for line in text.splitlines():
        if line.startswith("BEGIN_RES"):
            in_flex = True
        elif line.startswith("END_RES"):
            in_flex = False
        elif line.startswith("ENDMDL"):
            break
        elif line.startswith(("ATOM", "HETATM")) and not in_flex:
            name = line[12:16]
            try:
                xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            except ValueError:
                raise ConversionError(f"bad coordinates in PDBQT line {line!r}") from None
            tokens = line[54:].split()  # occupancy, B-factor, partial charge, AutoDock type
            ad_type = tokens[-1] if tokens and tokens[-1][0].isalpha() else ""
            element = _element(ad_type, name)
            if element is not None:
                atoms.append((name, element, xyz))
    if not atoms:
        raise ConversionError("no ATOM/HETATM records in the PDBQT model")
    return atoms


def _pdb_block(atoms) -> str:
    """PDB text with proper element columns, all atoms in one non-standard residue (no residue templates)."""
    lines = []
    for serial, (name, element, (x, y, z)) in enumerate(atoms, 1):
        lines.append(f"HETATM{serial:5d} {name:<4} UNL A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          "
                     f"{element.upper():>2}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _heavy_pose(text: str) -> Tuple[Chem.Mol, Dict[int, list]]:
    """Heavy-atom graph of the pose (single bonds, no charges) + the pose's H coordinates per heavy atom.

    Heavy atoms carry their PDB index in the private atom property ``_pvAtom``.
    """
    atoms = _pose_atoms(text)
    with _quiet():
        pdb = Chem.MolFromPDBBlock(_pdb_block(atoms), removeHs=False, sanitize=False, proximityBonding=True)
    if pdb is None:
        raise ConversionError("RDKit could not read the pose")
    pdb.UpdatePropertyCache(strict=False)
    pose_hs = {}
    for atom in pdb.GetAtoms():
        atom.SetIntProp("_pvAtom", atom.GetIdx())
        if atom.GetAtomicNum() > 1:
            hs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if hs:
                pose_hs[atom.GetIdx()] = [atoms[i][2] for i in hs]
    heavy = Chem.RemoveHs(pdb, _POSE_HS, sanitize=False)
    if heavy.GetNumAtoms() != heavy.GetNumHeavyAtoms():
        raise ConversionError("the pose has hydrogens RDKit could not remove")
    for bond in heavy.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in heavy.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(False)
        atom.SetNumExplicitHs(0)
    return heavy, pose_hs


def _composition(mol: Chem.Mol) -> List[int]:
    return sorted(a.GetAtomicNum() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _clean(mol: Chem.Mol, isotopes: bool = True) -> Chem.Mol:
    """``mol`` (a copy if changed) without explicit hydrogens, deuterium included, and without isotope labels."""
    if mol.GetNumAtoms() != mol.GetNumHeavyAtoms():
        mol = Chem.RemoveHs(mol, _SMILES_HS)
    if isotopes and any(a.GetIsotope() for a in mol.GetAtoms()):
        mol = Chem.Mol(mol)
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
    return mol


def _fragments(smiles: str, like: Chem.Mol) -> List[Chem.Mol]:
    """Fragments of ``smiles`` (cleaned) with the same heavy atoms as ``like``, in SMILES order, then the whole."""
    with _quiet():
        ref = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles else None
    if ref is None:
        raise ConversionError(f"invalid SMILES {smiles!r}")
    frags = Chem.GetMolFrags(ref, asMols=True) if "." in smiles else (ref,)
    if len(frags) > 1:
        frags = (*frags, ref)  # a pose (or molecule) that kept every fragment of a salt
    want = _composition(like)
    isotopes = bool(_ISOTOPE.search(smiles))
    out = []
    for frag in frags:
        if frag.GetNumHeavyAtoms() == len(want):
            frag = _clean(frag, isotopes)
            if _composition(frag) == want:
                out.append(frag)
    return out


def _add_hydrogens(mol: Chem.Mol, pose_hs) -> Chem.Mol:
    """``AddHs(addCoords=True)``; H of heavy atoms that carry as many H in the pose get the pose coordinates, the
    other added H are relaxed (``_relax_hydrogens``)."""
    fixed = set(range(mol.GetNumAtoms()))  # AddHs appends the hydrogens
    mol = Chem.AddHs(mol, addCoords=True)
    conf = mol.GetConformer()
    for atom in mol.GetAtoms():
        if not atom.HasProp("_pvAtom"):
            continue
        coords = pose_hs.get(atom.GetIntProp("_pvAtom"))
        atom.ClearProp("_pvAtom")
        if coords:
            hs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if len(hs) == len(coords):
                for i, xyz in zip(hs, coords):
                    conf.SetAtomPosition(i, Point3D(*xyz))
                    fixed.add(i)
    _relax_hydrogens(mol, fixed)
    return mol


def _relax_hydrogens(mol: Chem.Mol, fixed: set) -> None:
    """MMFF94 minimisation of the atoms not in ``fixed`` (the hydrogens without pose coordinates), all others fixed.

    ``AddHs`` puts each H by its own atom only, so on a docked conformation H of neighbouring groups can overlap
    (H-H < 1 A), which fails PoseBusters' internal energy check. MMFF, not UFF: UFF's trigonal angle term lets the two
    H of a =CH2 collapse onto each other. Left as placed if MMFF has no parameters for the molecule (e.g. metals).
    """
    if len(fixed) == mol.GetNumAtoms():
        return
    try:
        with _quiet():
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
            if props is None:
                return
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props)
            for i in sorted(fixed):
                ff.AddFixedPoint(i)
            ff.Minimize()
    except Exception as e:  # RDKit raises several types; the hydrogens keep their AddHs coordinates
        logger.debug("hydrogen positions not relaxed: %s", e)


def _ring_contacts(tmpl: Chem.Mol, heavy: Chem.Mol, match) -> Optional[List[Tuple[int, int]]]:
    """Pose bonds that ``match`` maps no template bond onto, if all are 1,3 contacts, else None.

    A 1,3 contact joins two atoms with a common template neighbour that are farther apart than a bond: the bridgeheads
    of a bicyclo[1.1.1]pentane (1.85 A), the diagonal of an oxetane. Any other extra bond is a clash, or a bond the
    SMILES lacks (a cyclopropane pose with the SMILES of propane).
    """
    image = {frozenset((match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()])) for b in tmpl.GetBonds()}
    inverse = {p: t for t, p in enumerate(match)}
    conf = heavy.GetConformer()
    contacts = []
    for bond in heavy.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if frozenset((i, j)) in image:
            continue
        a, b = tmpl.GetAtomWithIdx(inverse[i]), tmpl.GetAtomWithIdx(inverse[j])
        if not {n.GetIdx() for n in a.GetNeighbors()} & {n.GetIdx() for n in b.GetNeighbors()}:
            return None
        # Rb0 (C 0.77 A), not Rcovalent: the latter changed between RDKit releases
        bond_length = sum(_PERIODIC_TABLE.GetRb0(x.GetAtomicNum()) for x in (a, b)) + _CONTACT_MARGIN
        if (conf.GetAtomPosition(i) - conf.GetAtomPosition(j)).Length() <= bond_length:
            return None
        contacts.append((i, j))
    return contacts


def _template_matches(tmpl: Chem.Mol, heavy: Chem.Mol, pose_h: Optional[List[int]] = None) -> tuple:
    """Matches of ``tmpl`` (as single bonds, no charges) on the pose graph ``heavy``.

    With ``pose_h`` (H count per pose atom) only matches that put every non-carbon template atom on a pose atom with
    as many H: isotope labels on copies steer the matcher, so the cap on matches cannot cut those off.
    """
    query = Chem.Mol(tmpl)
    for bond in query.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in query.GetAtoms():
        atom.SetFormalCharge(0)
    target = heavy
    if pose_h is not None:
        target = Chem.Mol(heavy)
        for q, t in zip(query.GetAtoms(), tmpl.GetAtoms()):
            q.SetIsotope(t.GetTotalNumHs() + 1 if t.GetAtomicNum() != 6 else 0)
        for atom, n in zip(target.GetAtoms(), pose_h):
            atom.SetIsotope(n + 1 if atom.GetAtomicNum() != 6 else 0)
    return target.GetSubstructMatches(query, uniquify=False, maxMatches=_MAX_TEMPLATE_MATCHES)


def _assign_bond_orders(tmpl: Chem.Mol, heavy: Chem.Mol, pose_hs) -> Chem.Mol:
    """Bond orders, charges and H counts of ``tmpl`` on the pose graph ``heavy`` (sanitized copy).

    Same matching as RDKit's ``AssignBondOrdersFromTemplate``, but of several symmetric matches it takes one that
    puts the template's polar H where the pose has them (RDKit takes the first: the OH of a carboxylic acid could
    become the C=O and its docked H would move to the other oxygen). Extra pose bonds must be 1,3 ring contacts
    (``_ring_contacts``); they are removed.
    """
    pose_h = [len(pose_hs.get(a.GetIntProp("_pvAtom"), ())) for a in heavy.GetAtoms()]
    error = "the pose's bond graph does not match the SMILES"
    contacts = {}
    for constrained in (pose_h, None):  # pose H as in the SMILES first; else (other protonation, no H) any match
        matches = _template_matches(tmpl, heavy, constrained)
        if matches and heavy.GetNumBonds() > tmpl.GetNumBonds():
            contacts = {}
            for m in matches:
                extra = _ring_contacts(tmpl, heavy, m)
                if extra is not None:
                    contacts[m] = extra
            matches = list(contacts)
            if not matches:
                error = "the pose has atoms closer than a bond that the SMILES does not bond (clash)"
        if matches:
            break
    else:
        raise ValueError(error)
    if len(matches) > 1:
        polar = [(a.GetIdx(), a.GetTotalNumHs()) for a in tmpl.GetAtoms() if a.GetAtomicNum() != 6]
        match = max(matches, key=lambda m: sum(pose_h[m[i]] == n for i, n in polar))
    else:
        match = matches[0]
    mol = Chem.RWMol(heavy)
    for i, j in contacts.get(match, ()):
        mol.RemoveBond(i, j)
    mol = mol.GetMol()
    for bond in tmpl.GetBonds():
        target = mol.GetBondBetweenAtoms(match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()])
        target.SetBondType(bond.GetBondType())
        target.SetIsAromatic(bond.GetIsAromatic())
    for atom in tmpl.GetAtoms():
        target = mol.GetAtomWithIdx(match[atom.GetIdx()])
        target.SetHybridization(atom.GetHybridization())
        target.SetIsAromatic(atom.GetIsAromatic())
        target.SetNoImplicit(atom.GetNoImplicit())
        target.SetNumExplicitHs(atom.GetNumExplicitHs())
        target.SetFormalCharge(atom.GetFormalCharge())
        target.SetNumRadicalElectrons(atom.GetNumRadicalElectrons())
    Chem.SanitizeMol(mol)
    return mol


def _from_template(text: str, smiles: str) -> Tuple[Chem.Mol, Chem.Mol, List[Chem.Mol]]:
    """(molecule with H, its heavy-atom molecule, the SMILES fragments with the pose's atoms, the one used first).

    Fragments that differ only in stereo (``A.ent-A``) fit the same pose graph: the identity check needs them all.
    """
    heavy, pose_hs = _heavy_pose(text)
    templates = _fragments(smiles, heavy)
    if not templates:
        raise ConversionError(f"no fragment of {smiles!r} has the pose's {heavy.GetNumAtoms()} heavy atoms")
    errors = []
    for tmpl in templates:
        if tmpl.GetNumBonds() > heavy.GetNumBonds():
            errors.append(f"pose has {heavy.GetNumBonds()} bonds, SMILES fragment {tmpl.GetNumBonds()}")
            continue
        try:
            mol = _assign_bond_orders(tmpl, heavy, pose_hs)
        except ValueError as e:  # no match, or the matched pose does not sanitize
            errors.append(str(e))
            continue
        Chem.AssignStereochemistryFrom3D(mol)
        return _add_hydrogens(mol, pose_hs), mol, [tmpl] + [t for t in templates if t is not tmpl]
    raise ConversionError(f"pose does not match {smiles!r}: {'; '.join(errors)}")


def _meeko_mol(text: str) -> Optional[Chem.Mol]:
    """The pose rebuilt by Meeko from its ``REMARK SMILES`` lines; None if Meeko is missing or fails."""
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
    except Exception as e:  # not installed, or broken by its own dependencies (numpy, rdkit versions)
        if not isinstance(e, ImportError):
            logger.debug("Meeko could not be imported (%s); using the SMILES template", e)
        return None
    try:
        with _quiet():
            pmol = PDBQTMolecule(text, is_dlg=False, skip_typing=True)
            mols = [m for m in RDKitMolCreate.from_pdbqt_mol(pmol) if m is not None]
        if len(mols) != 1 or mols[0].GetNumConformers() == 0:
            raise ValueError(f"{len(mols)} molecules")
        mol = Chem.Mol(mols[0])
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
    except Exception as e:  # Meeko raises many types; the SMILES template path still applies
        logger.debug("Meeko could not rebuild the pose (%s); using the SMILES template", e)
        return None
    for name in list(mol.GetPropNames()):
        mol.ClearProp(name)
    return mol


def _meeko_match(text: str, smiles: str) -> Optional[Chem.Mol]:
    """Meeko's rebuild of a pose with ``REMARK SMILES`` lines if it is the molecule of ``smiles``, else None.

    Another molecule (e.g. other charges than ``smiles``) goes to the SMILES template, as without Meeko.
    """
    if not _MEEKO_SMILES.search(text):
        return None
    mol = _meeko_mol(text)
    if mol is None or same_molecule(mol, smiles):
        return mol
    logger.debug("Meeko rebuilt another molecule than %r; using the SMILES template", smiles)
    return None


def pdbqt_to_mol(model: Union[bytes, str], smiles: str) -> Chem.Mol:
    """RDKit molecule (all hydrogens, 3D) of one docked PDBQT model (``MODEL … ENDMDL``; the first if several).

    ``smiles`` is the docked SMILES; if it has several fragments, the one with the pose's heavy atoms is used.
    Polar hydrogens keep their docked coordinates; the others are added by RDKit and relaxed with MMFF94, every
    other atom fixed (SMILES template path). Raises ``ConversionError`` if the pose does not fit the SMILES.
    """
    text = _first_model(_as_text(model))
    mol = _meeko_match(text, smiles)
    return mol if mol is not None else _from_template(text, smiles)[0]


def mol_to_sdf(mol: Chem.Mol, props: Optional[dict] = None) -> bytes:
    """One SDF record (ending in ``$$$$``) of ``mol``; ``props`` become data fields (``str(value)``).

    A ``"_Name"`` key in ``props`` sets the title line.
    """
    mol = Chem.Mol(mol)
    for key, value in (props or {}).items():
        mol.SetProp(str(key), "" if value is None else str(value))
    if mol.HasProp("_Name"):  # a line break in the title would make the record unreadable
        mol.SetProp("_Name", " ".join(mol.GetProp("_Name").splitlines()))
    out = io.StringIO()
    writer = Chem.SDWriter(out)
    try:
        writer.write(mol)
    finally:
        writer.close()
    return out.getvalue().encode()


def _inchikey(mol: Chem.Mol) -> str:
    with _quiet():
        try:
            return Chem.MolToInchiKey(mol) or ""
        except (ValueError, RuntimeError):
            return ""


def _same_as_fragment(probe: Chem.Mol, frag: Chem.Mol) -> bool:
    """InChIKey identity of ``probe`` (heavy atoms, stereo tags, no conformer) and a cleaned SMILES fragment."""
    key = _inchikey(frag)
    if not key:
        return False
    if _inchikey(probe) == key:
        return True
    # drop the pose's stereo where the SMILES leaves it unspecified (a symmetric graph has several mappings)
    seen = set()
    for match in probe.GetSubstructMatches(frag, useChirality=False, uniquify=False,
                                           maxMatches=_MAX_STEREO_MATCHES):
        atoms = frozenset(match[a.GetIdx()] for a in frag.GetAtoms()
                          if a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED)
        bonds = frozenset(probe.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()]).GetIdx()
                          for b in frag.GetBonds()
                          if b.GetBondType() == Chem.BondType.DOUBLE and b.GetStereo() in _UNSPECIFIED_BOND)
        if (atoms, bonds) in seen:
            continue
        seen.add((atoms, bonds))
        m = Chem.Mol(probe)
        for i in atoms:
            m.GetAtomWithIdx(i).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        for i in bonds:
            m.GetBondWithIdx(i).SetStereo(Chem.BondStereo.STEREONONE)
        if _inchikey(m) == key:
            return True
    return False


def _stereo_probe(mol: Chem.Mol) -> Chem.Mol:
    """Copy of ``mol`` for InChI: no conformer (InChI would read stereo from coordinates instead of the tags)."""
    probe = Chem.Mol(mol)
    probe.RemoveAllConformers()
    return probe


def same_molecule(mol: Chem.Mol, smiles: str) -> bool:
    """True if ``mol`` is ``smiles`` or one of its fragments: same standard InChIKey, isotope labels ignored.

    Stereo of ``mol`` comes from its 3D coordinates when it has them. Stereo that ``smiles`` leaves unspecified is
    not compared; specified stereo must match.
    """
    if mol is None or not smiles:
        return False
    try:
        probe = Chem.Mol(mol)
        if probe.GetNumConformers() and probe.GetConformer().Is3D():
            Chem.AssignStereochemistryFrom3D(probe)
        probe = _stereo_probe(_clean(probe))
        frags = _fragments(smiles, probe)
    except (ConversionError, ValueError, RuntimeError):
        return False
    return any(_same_as_fragment(probe, f) for f in frags)


def pose_to_sdf(model: Union[bytes, str], smiles: str, props: Optional[dict] = None) -> Tuple[Optional[bytes], str]:
    """(SDF bytes, "ok") for a docked model that converts to the molecule of ``smiles``, else (None, "convert_failed").

    Same as ``pdbqt_to_mol`` + ``same_molecule`` + ``mol_to_sdf``; never raises for a bad pose or SMILES.
    """
    try:
        text = _first_model(_as_text(model))
        mol = _meeko_match(text, smiles)
        if mol is None:
            mol, heavy, templates = _from_template(text, smiles)
            probe = _stereo_probe(heavy)
            if not any(_same_as_fragment(probe, t) for t in templates):
                raise ConversionError("the converted pose has a different InChIKey than the SMILES")
        return mol_to_sdf(mol, props), "ok"
    except Exception as e:  # RDKit and Meeko raise many types; the caller only records the status
        logger.debug("pose conversion failed for %r: %s", smiles, e)
        return None, "convert_failed"
