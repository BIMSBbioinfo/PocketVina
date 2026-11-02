"""pocketvina.convert: docked PDBQT -> RDKit -> SDF in memory, checked by InChIKey against the docked SMILES."""
import builtins
import io
import os
import re
import sys
import tempfile

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pocketvina import convert
from pocketvina.convert import ConversionError, mol_to_sdf, pdbqt_to_mol, pose_to_sdf, same_molecule

MODEL_RE = re.compile(rb"^MODEL[^\n]*\n.*?^ENDMDL[^\n]*\n", re.S | re.M)
TOL = 1e-3

# name: (SMILES given to convert, fragment that is docked)
GENERATED = {
    "protonated_piperazine": ("C[NH+]1CCN(CC1)c1ccc(F)cc1", None),
    "nitro": ("NC(=O)c1ccc(cc1)[N+](=O)[O-]", None),
    "carboxylate": ("[O-]C(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", None),
    "stereocentres": ("C[C@H](N)[C@@H](O)c1ccccc1", None),
    "salt": ("C[NH+](C)CCC=C1c2ccccc2CCc2ccccc12.[Cl-]", "C[NH+](C)CCC=C1c2ccccc2CCc2ccccc12"),
    # both fragments have the pose's heavy atoms (C7O): only the second one fits
    "same_size_fragments": ("Oc1ccccc1C.COc1ccccc1", "COc1ccccc1"),
}


def _models(path) -> list:
    return MODEL_RE.findall(path.read_bytes())


def _atoms(model: bytes) -> list:
    """(is_hydrogen, xyz) of the ATOM/HETATM lines of a PDBQT model."""
    out = []
    for line in model.decode().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            out.append((line.split()[-1] in ("H", "HD", "HS"), xyz))
    return out


def _close(a, b) -> bool:
    return all(abs(p - q) <= TOL for p, q in zip(a, b))


def _xyz(mol, idx):
    p = mol.GetConformer().GetAtomPosition(idx)
    return (p.x, p.y, p.z)


def _key(smiles: str) -> str:
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def _read_sdf(sdf: bytes) -> Chem.Mol:
    assert isinstance(sdf, bytes) and sdf.endswith(b"$$$$\n")
    mols = list(Chem.ForwardSDMolSupplier(io.BytesIO(sdf), removeHs=False))
    assert len(mols) == 1 and mols[0] is not None
    return mols[0]


def _check_coordinates(mol, model: bytes):
    """Every pose atom, polar hydrogens included, has an atom of ``mol`` at its coordinates."""
    atoms = _atoms(model)
    out = {True: [], False: []}
    for a in mol.GetAtoms():
        out[a.GetAtomicNum() == 1].append(_xyz(mol, a.GetIdx()))
    assert len(out[False]) == sum(1 for is_h, _ in atoms if not is_h)
    for is_h, xyz in atoms:
        assert any(_close(xyz, p) for p in out[is_h]), f"pose {'H' if is_h else 'heavy'} atom at {xyz} moved"


def _docked_pose(smiles: str, seed: int = 0xC0FFEE):
    """(PDBQT model bytes, embedded RDKit mol with H): RDKit ETKDG + OpenBabel PDBQT writer, in memory."""
    pybel = pytest.importorskip("openbabel.pybel")
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    pdbqt = pybel.readstring("mol", Chem.MolToMolBlock(mol)).write("pdbqt")
    model = f"MODEL 1\nREMARK VINA RESULT:      -7.1      0.000      0.000\n{pdbqt}ENDMDL\n"
    return model.encode(), mol


def _meeko_pose(smiles: str, seed: int = 7) -> bytes:
    meeko = pytest.importorskip("meeko")
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    text, ok, err = meeko.PDBQTWriterLegacy.write_string(meeko.MoleculePreparation().prepare(mol)[0])
    assert ok, err
    return f"MODEL 1\nREMARK VINA RESULT:      -6.0      0.000      0.000\n{text}ENDMDL\n".encode()


@pytest.fixture(scope="module")
def generated():
    return {name: _docked_pose(docked or smiles) for name, (smiles, docked) in GENERATED.items()}


# --- real engine outputs (tests/data) ---------------------------------------------------------------------------

def test_fixture_poses_convert(real_run):
    smiles = real_run.smiles()
    n = 0
    for name in real_run.docked:
        for pocket in real_run.pockets:
            models = _models(real_run.output(name, pocket))
            assert len(models) == real_run.n_modes
            for mode, model in enumerate(models, 1):
                props = {"_Name": name, "pocket": pocket, "mode": mode}
                sdf, status = pose_to_sdf(model, smiles[name], props)
                assert status == "ok", (name, pocket, mode)
                mol = _read_sdf(sdf)
                assert mol.GetProp("_Name") == name and mol.GetProp("mode") == str(mode)
                assert same_molecule(mol, smiles[name])
                assert Chem.MolToInchiKey(mol)[:14] == _key(smiles[name])[:14]
                _check_coordinates(mol, model)
                n += 1
    assert n == len(real_run.docked) * len(real_run.pockets) * real_run.n_modes


def test_fixture_carboxylic_acid(real_run):
    """molecule2: carboxylic acid, OpenBabel wrote part of it as a GLY residue (no residue templates must apply)."""
    smi = real_run.smiles()["molecule2"]
    acid = Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]")
    for model in _models(real_run.output("molecule2", 1)):
        mol = pdbqt_to_mol(model, smi)
        assert Chem.MolToInchiKey(mol) == _key(smi)
        assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == 0
        # the OH is the oxygen that carries the docked H, so the C=O is the shorter C-O bond of the pose
        c, o_double, o_h = mol.GetSubstructMatch(acid)
        pos = mol.GetConformer().GetAtomPosition
        assert (pos(c) - pos(o_double)).Length() < (pos(c) - pos(o_h)).Length()
        _check_coordinates(mol, model)


def test_fixture_stereo_from_pose(real_run):
    """molecule5: E double bond kept; the stereocentre left open by the SMILES gets the pose's configuration."""
    smi = real_run.smiles()["molecule5"]
    mol = pdbqt_to_mol(_models(real_run.output("molecule5", 2))[0], smi)
    docked = Chem.MolToSmiles(Chem.RemoveHs(mol))
    assert len(re.findall("@+", docked)) == 1
    assert same_molecule(mol, smi)
    assert same_molecule(mol, docked)
    mirror = docked.replace("@@", "!").replace("@", "@@").replace("!", "@")
    assert not same_molecule(mol, mirror)
    assert not same_molecule(mol, smi.replace("C/C=C/C", "C/C=C\\C"))


def test_ligand_input_pdbqt_without_model_lines(real_run):
    """An engine input file (no MODEL/ENDMDL) with a Si atom converts too."""
    smi = real_run.smiles()["molecule10"]
    sdf, status = pose_to_sdf(real_run.ligand_pdbqt("molecule10").read_bytes(), smi)
    assert status == "ok"
    mol = _read_sdf(sdf)
    assert any(a.GetSymbol() == "Si" for a in mol.GetAtoms())
    assert Chem.MolToInchiKey(mol) == _key(smi)


def test_first_model_is_used(real_run):
    path = real_run.output("molecule5", 1)
    smi = real_run.smiles()["molecule5"]
    first, second = _models(path)[:2]
    mol = pdbqt_to_mol(path.read_text(), smi)  # whole file, str
    _check_coordinates(mol, first)
    with pytest.raises(AssertionError):
        _check_coordinates(mol, second)


# --- generated poses: RDKit embedding + OpenBabel PDBQT -----------------------------------------------------------

@pytest.mark.parametrize("name", list(GENERATED))
def test_generated_pose(generated, name):
    smiles, docked = GENERATED[name]
    docked = docked or smiles
    model, ref = generated[name]
    sdf, status = pose_to_sdf(model, smiles, {"_Name": name, "vina": -7.1})
    assert status == "ok"
    mol = _read_sdf(sdf)
    assert mol.GetProp("vina") == "-7.1"
    assert Chem.MolToInchiKey(mol) == _key(docked)  # full key: charges, protonation and specified stereo
    assert same_molecule(mol, smiles)
    assert mol.GetNumAtoms() == ref.GetNumAtoms()
    assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == Chem.GetFormalCharge(Chem.MolFromSmiles(docked))
    _check_coordinates(mol, model)
    ref_heavy = [(a.GetAtomicNum(), _xyz(ref, a.GetIdx())) for a in ref.GetAtoms() if a.GetAtomicNum() > 1]
    for a in mol.GetAtoms():
        if a.GetAtomicNum() > 1:
            assert any(z == a.GetAtomicNum() and _close(_xyz(mol, a.GetIdx()), xyz) for z, xyz in ref_heavy)


def test_charged_groups(generated):
    piperazine = pdbqt_to_mol(generated["protonated_piperazine"][0], GENERATED["protonated_piperazine"][0])
    assert piperazine.HasSubstructMatch(Chem.MolFromSmarts("[NX4H1+]"))
    nitro = pdbqt_to_mol(generated["nitro"][0], GENERATED["nitro"][0])
    assert nitro.HasSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]"))
    carboxylate = pdbqt_to_mol(generated["carboxylate"][0], GENERATED["carboxylate"][0])
    assert carboxylate.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[O-]"))
    assert not same_molecule(carboxylate, "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl")  # neutral acid is another InChIKey


def test_symmetric_acid_keeps_docked_oh():
    """More symmetric template matches than the matcher's cap (2 CF3 + 2 tBu + ring + COOH: 10368), acid atoms first:
    the OH is still the oxygen with the docked H and the C=O the shorter C-O bond."""
    smiles = "OC(=O)C(C(F)(F)F)(C(F)(F)F)c1cc(C(C)(C)C)cc(C(C)(C)C)c1"
    model, _ = _docked_pose(smiles)
    mol = pdbqt_to_mol(model, smiles)
    _check_coordinates(mol, model)
    c, o_double, o_h = mol.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=[OX1])[OX2H1]"))
    pos = mol.GetConformer().GetAtomPosition
    assert (pos(c) - pos(o_double)).Length() < (pos(c) - pos(o_h)).Length()
    assert pose_to_sdf(model, smiles)[1] == "ok"


def test_stereocentres_must_match(generated):
    model = generated["stereocentres"][0]
    smi = GENERATED["stereocentres"][0]
    mol = pdbqt_to_mol(model, smi)
    assert same_molecule(mol, smi)
    assert same_molecule(mol, "CC(N)C(O)c1ccccc1")  # unspecified: not compared
    assert same_molecule(mol, "C[C@H](N)C(O)c1ccccc1")  # partly specified
    assert not same_molecule(mol, "C[C@@H](N)[C@H](O)c1ccccc1")  # enantiomer
    assert not same_molecule(mol, "C[C@H](N)[C@H](O)c1ccccc1")  # diastereomer
    assert not same_molecule(mol, "C[C@@H](N)C(O)c1ccccc1")
    enantiomer = "C[C@@H](N)[C@H](O)c1ccccc1"
    assert pose_to_sdf(model, enantiomer) == (None, "convert_failed")
    # a racemate written as two fragments with the same graph: the pose is the second one
    assert same_molecule(mol, f"{enantiomer}.{smi}")
    assert pose_to_sdf(model, f"{enantiomer}.{smi}")[1] == "ok"
    assert pose_to_sdf(model, f"{smi}.{enantiomer}")[1] == "ok"


def test_salt_uses_matching_fragment(generated):
    model = generated["salt"][0]
    mol = pdbqt_to_mol(model, GENERATED["salt"][0])
    assert len(Chem.GetMolFrags(mol)) == 1
    assert same_molecule(mol, GENERATED["salt"][0])
    assert same_molecule(mol, "[Cl-].C[NH+](C)CCC=C1c2ccccc2CCc2ccccc12")  # fragment order does not matter
    model = generated["same_size_fragments"][0]
    assert pose_to_sdf(model, "COc1ccccc1.Oc1ccccc1C")[1] == "ok"
    with pytest.raises(ConversionError):
        pdbqt_to_mol(model, "Oc1ccccc1C")  # same atoms, other molecule


def test_salt_pose_with_counterion(generated):
    """A pose that kept the counterion converts to the whole salt, which same_molecule accepts."""
    salt, cation = GENERATED["salt"]
    model = generated["salt"][0].decode()
    x = max(float(ln[30:38]) for ln in model.splitlines() if ln.startswith("ATOM")) + 4.0
    cl = f"ATOM     99 CL   UNL     1    {x:8.3f}   0.000   0.000  1.00  0.00    -1.000 Cl\n"
    model = model.replace("ENDMDL", cl + "ENDMDL").encode()
    sdf, status = pose_to_sdf(model, salt)
    assert status == "ok"
    mol = _read_sdf(sdf)
    assert len(Chem.GetMolFrags(mol)) == 2 and Chem.MolToInchiKey(mol) == _key(salt)
    assert same_molecule(mol, salt) and not same_molecule(mol, cation)
    _check_coordinates(mol, model)
    assert pose_to_sdf(model, cation) == (None, "convert_failed")


def test_unspecified_double_bond_is_not_compared():
    model, _ = _docked_pose("C/C=C/C(=O)N")
    assert pose_to_sdf(model, "CC=CC(=O)N")[1] == "ok"
    assert pose_to_sdf(model, "C/C=C/C(=O)N")[1] == "ok"
    assert pose_to_sdf(model, "C/C=C\\C(=O)N") == (None, "convert_failed")


def test_wrong_or_broken_input(generated):
    model = generated["nitro"][0]
    for smi in ["CCO", "not a smiles", "", "NC(=O)c1ccc(cc1)C(=O)O"]:
        with pytest.raises(ConversionError):
            pdbqt_to_mol(model, smi)
        assert pose_to_sdf(model, smi) == (None, "convert_failed")
    for bad in [b"", b"REMARK nothing here\n", "MODEL 1\nENDMDL\n"]:
        with pytest.raises(ConversionError):
            pdbqt_to_mol(bad, GENERATED["nitro"][0])
        assert pose_to_sdf(bad, GENERATED["nitro"][0]) == (None, "convert_failed")
    assert pose_to_sdf(None, GENERATED["nitro"][0]) == (None, "convert_failed")
    assert pose_to_sdf(model, None) == (None, "convert_failed")


def test_clashing_pose_is_rejected(generated):
    """The heavy atom farthest from the first one moved next to it gives the wrong bond graph: not converted."""
    model, _ = generated["nitro"]
    lines = model.decode().splitlines(keepends=True)
    heavy = [i for i, ln in enumerate(lines) if ln.startswith("ATOM") and ln.split()[-1] not in ("H", "HD")]
    xyz = {i: (float(lines[i][30:38]), float(lines[i][38:46]), float(lines[i][46:54])) for i in heavy}
    first = heavy[0]
    far = max(heavy, key=lambda i: sum((p - q) ** 2 for p, q in zip(xyz[i], xyz[first])))
    x, y, z = xyz[first]
    lines[far] = f"{lines[far][:30]}{x + 1.0:8.3f}{y:8.3f}{z:8.3f}{lines[far][54:]}"
    assert pose_to_sdf("".join(lines), GENERATED["nitro"][0]) == (None, "convert_failed")


def test_ring_contacts_are_not_bonds(generated):
    """1,3 contacts closer than RDKit's proximity-bonding cut-off (bicyclo[1.1.1]pentane bridgeheads at ~1.85 A, an
    oxetane diagonal in an unminimised ETKDG pose) are not bonds; a 3-ring bond the SMILES lacks still is."""
    bcp, oxetane = "NC(=O)C12CC(c3ccccc3)(C1)C2", "CC1(COC1)c1ccc(NC(=O)C)cc1"
    for smiles in (bcp, oxetane):
        model, ref = _docked_pose(smiles)
        if smiles == bcp and Chem.GetPeriodicTable().GetRcovalent(6) >= 0.75:
            # RDKit's current radii (older releases: C 0.68 A, cut-off 1.81 A) bond the bridgeheads
            heavy, _ = convert._heavy_pose(model.decode())
            assert heavy.GetNumBonds() == Chem.MolFromSmiles(bcp).GetNumBonds() + 1
        sdf, status = pose_to_sdf(model, smiles)
        assert status == "ok", smiles
        mol = _read_sdf(sdf)
        # InChI reads (meaningless) stereo of BCP bridgeheads from 3D: compare with the embedded molecule
        assert Chem.MolToInchiKey(mol) == Chem.MolToInchiKey(ref)
        assert same_molecule(mol, smiles)
        _check_coordinates(mol, model)
    model, _ = _docked_pose("NC(=O)C1CC1c1ccccc1")
    assert pose_to_sdf(model, "NC(=O)C1CC1c1ccccc1")[1] == "ok"
    assert pose_to_sdf(model, "NC(=O)CCCc1ccccc1") == (None, "convert_failed")  # same atoms, open chain
    # a counterion overlapping the cation (1.9 A from a carbon) is a clash, not a contact
    salt = GENERATED["salt"][0]
    text = generated["salt"][0].decode()
    c = next(ln for ln in text.splitlines() if ln.startswith("ATOM") and ln.split()[-1] in ("C", "A"))
    x, y, z = float(c[30:38]) + 1.9, float(c[38:46]), float(c[46:54])
    cl = f"ATOM     99 CL   UNL     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    -1.000 Cl\n"
    assert pose_to_sdf(text.replace("ENDMDL", cl + "ENDMDL"), salt) == (None, "convert_failed")


def test_flexible_residues_are_skipped(generated):
    model, _ = generated["nitro"]
    flex = ("BEGIN_RES TYR A 100\n"
            "ATOM      1  CA  TYR A 100       9.000   9.000   9.000  1.00  0.00     0.180 C \n"
            "END_RES TYR A 100\n")
    model = model.replace(b"ENDMDL", flex.encode() + b"ENDMDL")
    assert pose_to_sdf(model, GENERATED["nitro"][0])[1] == "ok"


# a docked menthol pose of the PocketVina FDA screen (P00505, pocket 1, mode 1), whose AddHs hydrogens overlap
MENTHOL = ("CC(C)C1CCC(C)CC1O", """MODEL 1
ROOT
ATOM      1  C   UNL A   1      -1.730   5.521 -23.796  0.00  0.00    +0.020 C
ATOM      2  C   UNL A   1      -3.070   6.097 -24.025  0.00  0.00    +0.006 C
ATOM      3  C   UNL A   1      -3.686   6.787 -22.840  0.00  0.00    +0.004 C
ATOM      4  C   UNL A   1      -3.677   5.874 -21.664  0.00  0.00    -0.012 C
ATOM      5  C   UNL A   1      -4.132   6.689 -20.456  0.00  0.00    +0.007 C
ATOM      6  C   UNL A   1      -2.359   5.259 -21.340  0.00  0.00    +0.035 C
ATOM      7  C   UNL A   1      -1.492   4.877 -22.479  0.00  0.00    +0.118 C
ENDROOT
BRANCH   1   9
ATOM      8  C   UNL A   1      -1.100   3.165 -24.388  0.00  0.00    +0.007 C
ATOM      9  C   UNL A   1      -1.422   4.549 -24.920  0.00  0.00    -0.012 C
ATOM     10  C   UNL A   1      -2.563   4.375 -25.884  0.00  0.00    +0.007 C
ENDBRANCH   1   9
BRANCH   7  11
ATOM     11  O   UNL A   1      -0.170   5.132 -22.093  0.00  0.00    -0.392 OA
ATOM     12  H   UNL A   1       0.122   4.598 -21.312  0.00  0.00    +0.210 HD
ENDBRANCH   7  11
TORSDOF 2
ENDMDL
""")


def _closest_hh(mol) -> float:
    """Shortest distance between two hydrogens of ``mol`` (two H of the same atom included)."""
    pos = mol.GetConformer().GetPositions()
    hs = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    return min(((pos[i] - pos[j]) ** 2).sum() ** 0.5 for n, i in enumerate(hs) for j in hs[n + 1:])


def test_added_hydrogens_do_not_overlap(monkeypatch):
    """Hydrogens without pose coordinates are relaxed (MMFF, the pose atoms fixed): AddHs alone puts two H of this
    pose 1.09 A apart, which fails PoseBusters' internal energy check."""
    smiles, model = MENTHOL
    sdf, status = pose_to_sdf(model, smiles)
    mol = _read_sdf(sdf)
    assert status == "ok" and same_molecule(mol, smiles)
    _check_coordinates(mol, model.encode())  # heavy atoms and the docked OH hydrogen did not move
    assert mol.GetNumAtoms() == 11 + 20 and _closest_hh(mol) > 1.6
    monkeypatch.setattr(convert, "_relax_hydrogens", lambda mol, fixed: None)
    assert _closest_hh(pdbqt_to_mol(model, smiles)) < 1.2  # what the relaxation fixes


def test_atom_types():
    for ad, element in [("A", "C"), ("C", "C"), ("NA", "N"), ("Na", "Na"), ("OA", "O"), ("SA", "S"), ("HD", "H"),
                        ("Cl", "Cl"), ("CL", "Cl"), ("Br", "Br"), ("Si", "Si"), ("CG0", "C"), ("Zn", "Zn")]:
        assert convert._element(ad, " X1 ") == element
    assert convert._element("G1", " G1 ") is None  # Meeko ring-closure dummy
    assert convert._element("", " N3 ") == "N"  # no type column: from the atom name
    with pytest.raises(ConversionError):
        convert._element("Xx", " Q1 ")


def test_mol_to_sdf():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    sdf = mol_to_sdf(mol, {"_Name": "ethanol", "affinity": -3.25, "pocket": 2, "note": None})
    back = _read_sdf(sdf)
    assert sdf.count(b"$$$$") == 1
    assert back.GetProp("_Name") == "ethanol"
    assert back.GetProp("affinity") == "-3.25" and back.GetProp("pocket") == "2" and back.GetProp("note") == ""
    assert back.GetNumAtoms() == mol.GetNumAtoms()
    assert not mol.HasProp("affinity")  # input left alone
    assert mol_to_sdf(mol).startswith(b"\n")  # no title
    assert _read_sdf(mol_to_sdf(mol, {"_Name": "two\r\nline name"})).GetProp("_Name") == "two line name"


def test_same_molecule_edge_cases():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](N)C(=O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    assert same_molecule(mol, "C[C@H](N)C(=O)O")
    assert same_molecule(mol, "C[13C@H](N)C(=O)O")  # isotope labels are ignored
    assert not same_molecule(mol, "C[C@@H](N)C(=O)O")
    assert not same_molecule(mol, "garbage(")
    assert not same_molecule(mol, "")
    assert not same_molecule(None, "CCO")
    no_conf = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
    assert same_molecule(no_conf, "C[C@H](N)C(=O)O")  # 2D/0D: stereo from the tags


# --- Meeko-prepared ligands ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("smiles", ["C[NH+]1CCN(c2ccccc2)CC1", "C[C@H](N)[C@@H](O)c1ccccc1"])
def test_meeko_pose(smiles, monkeypatch):
    model = _meeko_pose(smiles)
    assert b"REMARK SMILES " in model
    assert convert._meeko_mol(model.decode()) is not None
    sdf, status = pose_to_sdf(model, smiles)
    assert status == "ok"
    mol = _read_sdf(sdf)
    assert Chem.MolToInchiKey(mol) == _key(smiles)
    _check_coordinates(mol, model)
    # without Meeko the SMILES template path handles the same file
    monkeypatch.setitem(sys.modules, "meeko", None)
    assert convert._meeko_mol(model.decode()) is None
    mol = pdbqt_to_mol(model, smiles)
    assert Chem.MolToInchiKey(mol) == _key(smiles)
    _check_coordinates(mol, model)


def test_meeko_pose_follows_the_smiles():
    """Meeko's molecule is used only if it is the SMILES' molecule; otherwise the SMILES template, as without Meeko."""
    model = _meeko_pose("C[NH+]1CCN(c2ccccc2)CC1")
    with pytest.raises(ConversionError):
        pdbqt_to_mol(model, "CCO")
    assert pose_to_sdf(model, "CCO") == (None, "convert_failed")
    neutral = "CN1CCN(c2ccccc2)CC1"
    assert Chem.MolToInchiKey(pdbqt_to_mol(model, neutral)) == _key(neutral)
    sdf, status = pose_to_sdf(model, neutral)
    assert status == "ok" and Chem.MolToInchiKey(_read_sdf(sdf)) == _key(neutral)


# --- nothing touches the disk -------------------------------------------------------------------------------------

def test_no_files_written(real_run, generated, tmp_path, monkeypatch):
    work, tmp = tmp_path / "cwd", tmp_path / "tmp"
    work.mkdir()
    tmp.mkdir()
    jobs = [(m, real_run.smiles()[n]) for n in real_run.docked for m in _models(real_run.output(n, 1))[:2]]
    jobs += [(generated[n][0], GENERATED[n][0]) for n in GENERATED]
    try:
        jobs.append((_meeko_pose("C[NH+]1CCN(c2ccccc2)CC1"), "C[NH+]1CCN(c2ccccc2)CC1"))
    except pytest.skip.Exception:
        pass
    monkeypatch.chdir(work)
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    real_open, real_os_open = builtins.open, os.open

    def no_write_open(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            raise AssertionError(f"opened {file!r} for writing")
        return real_open(file, mode, *args, **kwargs)

    def no_write_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            raise AssertionError(f"os.open({path!r}) for writing")
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", no_write_open)
    monkeypatch.setattr(os, "open", no_write_os_open)
    results = [pose_to_sdf(model, smi, {"_Name": "x"}) for model, smi in jobs]
    monkeypatch.undo()
    assert [status for _, status in results] == ["ok"] * len(jobs)
    assert list(work.iterdir()) == [] and list(tmp.iterdir()) == []
