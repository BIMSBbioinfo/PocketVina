"""pocketvina.ligands: ligand table reading, largest fragment, seeded standard preparation, legacy path, ligand dir."""
import dataclasses
import hashlib
import os
import subprocess
import sys

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from pocketvina import ligands
from pocketvina.ligands import (
    LigandRecord,
    find_obabel,
    largest_fragment,
    prepare_ligand,
    prepare_ligands,
    read_smiles_file,
    truncate_after_torsdof,
    write_ligand_dir,
)

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"
PARACETAMOL_KEY = "RZVAJINKPMORJF-UHFFFAOYSA-N"
LIGANDS_COLUMNS = ["ligand_idx", "ligand_id", "smiles_input", "smiles_docked", "inchikey", "n_heavy", "prep_method",
                   "seed", "status", "error", "pdbqt"]

needs_obabel = pytest.mark.skipif(find_obabel() is None, reason="no working obabel CLI (OpenBabel) found")


def atom_lines(pdbqt):
    return [line for line in pdbqt.splitlines() if line.startswith(("ATOM", "HETATM"))]


def heavy_atoms(pdbqt):
    return [line for line in atom_lines(pdbqt) if line.split()[-1] not in ("H", "HD", "HS")]


def topology(pdbqt):
    """The PDBQT without coordinates, occupancy and B-factor (those change with the conformer / OpenBabel version)."""
    return [(line[:30], line[66:]) if line.startswith(("ATOM", "HETATM")) else line for line in pdbqt.splitlines()]


def assert_engine_input(pdbqt):
    lines = pdbqt.splitlines()
    assert lines.count("ROOT") == 1 and lines.count("ENDROOT") == 1
    assert lines[-1].startswith("TORSDOF") and sum(line.startswith("TORSDOF") for line in lines) == 1
    assert pdbqt.endswith("\n")
    types = {line.split()[-1] for line in atom_lines(pdbqt)}
    assert types <= {"C", "A", "N", "NA", "OA", "S", "SA", "P", "F", "Cl", "Br", "I", "HD"}, types


# --- read_smiles_file ---------------------------------------------------------------------------------------------

def test_read_smiles_file_csv_and_txt(tmp_path, example_dir):
    text = "molecule_name,smiles\nmol_a,CCO\n\n  mol_b , c1ccccc1 \nmol_c,CC(=O)[O-].[Na+]\n"
    expected = [("mol_a", "CCO"), ("mol_b", "c1ccccc1"), ("mol_c", "CC(=O)[O-].[Na+]")]
    for ext in (".csv", ".txt"):
        path = tmp_path / f"ligands{ext}"
        path.write_text(text)
        assert read_smiles_file(path) == expected
        assert read_smiles_file(str(path)) == expected

    rows = read_smiles_file(example_dir / "ligands.txt")
    assert len(rows) == 9
    assert rows[0] == ("molecule1", "CNC(=O)C1=CC=CC=C1SC2=CC3=C(C=C2)C(=N[NH]3)\\C=C\\C4=CC=CC=N4")
    assert [name for name, _ in rows] == [f"molecule{i}" for i in range(1, 10)]


def test_read_smiles_file_columns_by_header_name(tmp_path):
    path = tmp_path / "ligands.csv"
    path.write_text("\ufeffSMILES,Molecule_Name,mw\nCCO,ethanol,46.07\n\"C(=O)O\",formic acid,46.03\n",
                    encoding="utf-8")
    assert read_smiles_file(path) == [("ethanol", "CCO"), ("formic acid", "C(=O)O")]

    path.write_text("name,smi\nethanol,CCO\n")  # unknown header: first two columns, as before
    assert read_smiles_file(path) == [("ethanol", "CCO")]

    path.write_bytes("molecule_name,smiles\r\nácido fórmico,C(=O)O\r\nβ-alanine,NCCC(=O)O\r\n".encode("utf-8"))
    assert read_smiles_file(path) == [("ácido fórmico", "C(=O)O"), ("β-alanine", "NCCC(=O)O")]


def test_read_smiles_file_bad_rows(tmp_path):
    path = tmp_path / "ligands.csv"
    path.write_text("molecule_name,smiles\nok,CCO\nonly_a_name\n")
    with pytest.raises(ValueError, match="line 3"):
        read_smiles_file(path)
    path.write_text("")
    assert read_smiles_file(path) == []


# --- largest_fragment ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("smiles, parent", [
    ("CC(=O)Nc1ccc(O)cc1.Cl", PARACETAMOL),
    ("Cl.OC1=CC=C(NC(C)=O)C=C1", PARACETAMOL),
    ("CC(=O)[O-].[Na+]", "CC(=O)[O-]"),
    ("[Cl-].C[NH+](C)CCc1ccccc1", "C[NH+](C)CCc1ccccc1"),
    ("C.[O-]S(=O)(=O)[O-]", "C"),  # organic fragments win over larger inorganic ones
    ("OC1=CC=C(NC(C)=O)C=C1", PARACETAMOL),  # single fragment: canonical SMILES
])
def test_largest_fragment(smiles, parent):
    assert largest_fragment(smiles) == parent


@pytest.mark.parametrize("smiles", ["C1CC", "", "C(C)(C)(C)(C)C"])
def test_largest_fragment_invalid(smiles):
    with pytest.raises(ValueError, match="invalid SMILES"):
        largest_fragment(smiles)


# --- standard preparation -----------------------------------------------------------------------------------------

def test_record_fields_are_the_ligands_columns():
    assert [f.name for f in dataclasses.fields(LigandRecord)] == LIGANDS_COLUMNS


def test_record_fields_match_tables_schema():
    tables = pytest.importorskip("pocketvina.io.tables")
    assert [f.name for f in dataclasses.fields(LigandRecord)] == list(tables.LIGANDS.names)


def test_salt_docks_largest_fragment():
    rec = prepare_ligand(3, "paracetamol_hcl", "CC(=O)Nc1ccc(O)cc1.Cl")
    assert rec.status == "ok" and rec.error is None
    assert (rec.ligand_idx, rec.ligand_id, rec.smiles_input) == (3, "paracetamol_hcl", "CC(=O)Nc1ccc(O)cc1.Cl")
    assert (rec.smiles_docked, rec.inchikey, rec.n_heavy) == (PARACETAMOL, PARACETAMOL_KEY, 11)
    assert (rec.prep_method, rec.seed) == ("standard", 42)
    assert_engine_input(rec.pdbqt)
    assert len(heavy_atoms(rec.pdbqt)) == 11
    assert not any(line.split()[-1] == "Cl" for line in atom_lines(rec.pdbqt))

    # the same molecule written differently (no salt, salt first, other atom order) gives the same engine input
    for smiles in (PARACETAMOL, "Cl.OC1=CC=C(NC(C)=O)C=C1"):
        assert prepare_ligand(0, "other", smiles).pdbqt == rec.pdbqt


def test_counter_ion_first_is_dropped():
    rec = prepare_ligand(0, "salt", "[Cl-].C[NH+](C)CCc1ccccc1")
    assert rec.status == "ok"
    assert rec.smiles_docked == "C[NH+](C)CCc1ccccc1" and rec.n_heavy == 11
    assert len(heavy_atoms(rec.pdbqt)) == 11


def test_unsupported_counter_ion_is_dropped():
    rec = prepare_ligand(0, "sodium acetate", "CC(=O)[O-].[Na+]")
    assert rec.status == "ok" and rec.smiles_docked == "CC(=O)[O-]"


def test_seed_reproducibility(real_run):
    smiles = real_run.smiles()["molecule5"]  # unspecified stereocentre: the seed also picks the enantiomer
    a = prepare_ligand(0, "molecule5", smiles, seed=7)
    b = prepare_ligand(0, "molecule5", smiles, seed=7)
    c = prepare_ligand(0, "molecule5", smiles, seed=8)
    assert a.status == b.status == c.status == "ok"
    assert a.pdbqt.encode() == b.pdbqt.encode()
    assert a.seed == 7 and c.seed == 8
    assert c.pdbqt != a.pdbqt
    assert topology(c.pdbqt) == topology(a.pdbqt)  # only the conformer changes


def test_seed_reproducibility_across_workers_and_processes(real_run, repo_root, tmp_path):
    rows = sorted(real_run.smiles().items()) + [("paracetamol", PARACETAMOL), ("ethanol", "CCO")]
    serial = prepare_ligands(rows, seed=42, n_jobs=1)
    threaded = prepare_ligands(rows, seed=42, n_jobs=4)
    assert [r.pdbqt for r in threaded] == [r.pdbqt for r in serial]
    assert [r.status for r in serial].count("ok") == 4

    code = ("import hashlib, sys\n"
            "from pocketvina.ligands import prepare_ligand\n"
            "rec = prepare_ligand(0, 'x', sys.argv[1], seed=int(sys.argv[2]))\n"
            "print(hashlib.sha256(rec.pdbqt.encode()).hexdigest())\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(repo_root), os.environ.get("PYTHONPATH")])))
    smiles = real_run.smiles()["molecule2"]
    proc = subprocess.run([sys.executable, "-c", code, smiles, "42"], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    expected = next(r for r in serial if r.ligand_id == "molecule2").pdbqt
    assert proc.stdout.strip() == hashlib.sha256(expected.encode()).hexdigest()


def test_threaded_salts_print_no_rdkit_log_lines(repo_root, tmp_path):
    """LargestFragmentChooser info lines go to stdout whenever another thread is inside EmbedMolecule; a threaded
    run must print none of them and give the serial result. A subprocess, so RDKit's buffered C++ stdout is seen."""
    smiles = ["CC(=O)Nc1ccc(O)cc1.Cl", "[Cl-].C[NH+](C)CCc1ccccc1", "CC(=O)[O-].[Na+]",
              "CCCCCCCCCCCCCCCCCC(=O)Nc1ccc(O)cc1"]
    rows = [(str(i), s) for i, s in enumerate(smiles * 8)]
    code = ("import hashlib, sys\n"
            "from pocketvina.ligands import prepare_ligands\n"
            "rows = [(str(i), s) for i, s in enumerate(sys.argv[1:] * 8)]\n"
            "for rec in prepare_ligands(rows, n_jobs=4):\n"
            "    print(rec.status, rec.smiles_docked, hashlib.sha256(rec.pdbqt.encode()).hexdigest())\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(repo_root), os.environ.get("PYTHONPATH")])))
    proc = subprocess.run([sys.executable, "-c", code, *smiles], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "LargestFragmentChooser" not in proc.stdout + proc.stderr
    expected = [f"{r.status} {r.smiles_docked} {hashlib.sha256(r.pdbqt.encode()).hexdigest()}"
                for r in prepare_ligands(rows, n_jobs=1)]
    assert proc.stdout.splitlines() == expected


def test_ctrl_c_still_works_after_preparation(repo_root, tmp_path):
    """RDKit's embedding (2024.09) installs its own SIGINT handler and leaves it there: Ctrl-C was then swallowed
    once (a docking run went on) and the next one killed the process without cleanup. A subprocess, because a
    failure would kill the test session."""
    code = ("import os, signal, sys, time\n"
            "from pocketvina.ligands import prepare_ligand, prepare_ligands\n"
            "calls = {'one': lambda: prepare_ligand(0, 'a', 'CCO'),\n"
            "         'many': lambda: prepare_ligands([('a', 'CCO'), ('b', 'CCN'), ('c', 'CCC')], n_jobs=2),\n"
            "         'legacy': lambda: prepare_ligands([('a', 'CCO')], method='legacy')}\n"
            "for name in sys.argv[1:]:\n"
            "    calls[name]()\n"
            "    try:\n"
            "        os.kill(os.getpid(), signal.SIGINT)\n"
            "        time.sleep(5)\n"
            "        print(name, 'SIGINT swallowed', flush=True)\n"
            "    except KeyboardInterrupt:\n"
            "        print(name, 'KeyboardInterrupt', flush=True)\n")
    names = ["one", "many"] + (["legacy"] if find_obabel() else [])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(repo_root), os.environ.get("PYTHONPATH")])))
    proc = subprocess.run([sys.executable, "-c", code, *names], cwd=tmp_path, env=env, capture_output=True,
                          text=True, timeout=300)
    assert proc.stdout.splitlines() == [f"{name} KeyboardInterrupt" for name in names], proc.stdout + proc.stderr
    assert proc.returncode == 0, proc.stderr


def test_seed_none_and_invalid_seeds():
    rec = prepare_ligand(0, "ethanol", "CCO", seed=None)
    assert rec.status == "ok" and rec.seed is None
    for bad in (-1, 2**31):
        with pytest.raises(ValueError, match="seed"):
            prepare_ligand(0, "ethanol", "CCO", seed=bad)
        with pytest.raises(ValueError, match="seed"):
            prepare_ligands([("ethanol", "CCO")], seed=bad)
    with pytest.raises(TypeError):
        prepare_ligand(0, "ethanol", "CCO", seed=1.5)


def test_unsupported_element(real_run):
    smiles = real_run.smiles()["molecule10"]
    rec = prepare_ligand(9, "molecule10", smiles)
    assert rec.status == "unsupported_element"
    assert "Si" in rec.error
    assert rec.pdbqt is None
    assert rec.smiles_docked == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    assert rec.n_heavy == Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()
    assert rec.inchikey and rec.seed == 42


@pytest.mark.parametrize("smiles, element", [("OB(O)c1ccccc1", "B"), ("C[Se]C", "Se"), ("*CC", "*"),
                                             ("[Na+].[Cl-]", "Cl")])
def test_unsupported_element_list(smiles, element):
    rec = prepare_ligand(0, "x", smiles)
    if element == "Cl":  # no organic fragment: the largest ion is kept and it is a supported element
        assert rec.status == "ok" and rec.smiles_docked == "[Cl-]"
        return
    assert rec.status == "unsupported_element"
    assert element in rec.error and rec.pdbqt is None


@pytest.mark.parametrize("smiles", ["[H+]", "[H][H]", "[H]", "[H+].[H+]"])
def test_no_heavy_atoms_status(smiles):
    rec = prepare_ligand(0, "proton", smiles)
    assert rec.status == "no_heavy_atoms" and rec.pdbqt is None and rec.n_heavy == 0


@pytest.mark.parametrize("smiles, reason", [
    ("C1CC", "syntax"),
    ("not a smiles", "syntax"),
    ("C(C)(C)(C)(C)C", "valence"),
    ("", "empty"),
    ("   ", "empty"),
])
def test_invalid_smiles_status(smiles, reason):
    rec = prepare_ligand(5, "bad", smiles)
    assert rec.status == "invalid_smiles"
    assert reason in rec.error.lower()
    assert rec.pdbqt is None and rec.smiles_docked is None and rec.n_heavy is None
    assert (rec.ligand_idx, rec.ligand_id, rec.smiles_input) == (5, "bad", smiles)


def test_embed_failure_status(monkeypatch):
    monkeypatch.setattr(ligands.AllChem, "EmbedMolecule", lambda *args, **kwargs: -1)
    rec = prepare_ligand(0, "paracetamol", PARACETAMOL)
    assert rec.status == "embed_failed" and rec.pdbqt is None
    assert rec.smiles_docked == PARACETAMOL


def test_prepare_ligands_order_and_statuses(real_run):
    rows = [("paracetamol", PARACETAMOL), ("bad", "C1CC"), ("si", real_run.smiles()["molecule10"]), ("etoh", "CCO")]
    records = prepare_ligands(rows, n_jobs=3)
    assert [r.ligand_idx for r in records] == [0, 1, 2, 3]
    assert [r.ligand_id for r in records] == ["paracetamol", "bad", "si", "etoh"]
    assert [r.status for r in records] == ["ok", "invalid_smiles", "unsupported_element", "ok"]
    assert {r.status for r in records} <= set(ligands.STATUSES)
    assert prepare_ligands([]) == []


def test_bad_arguments():
    with pytest.raises(ValueError, match="method"):
        prepare_ligand(0, "x", "CCO", method="meeko")
    with pytest.raises(ValueError, match="method"):
        prepare_ligands([("x", "CCO")], method="meeko")
    with pytest.raises(ValueError, match="n_jobs"):
        prepare_ligands([("x", "CCO")], n_jobs=0)


def test_truncate_after_torsdof():
    first = ("REMARK  Name = a\nROOT\n"
             "ATOM      1  C   UNL     1       0.000   0.000   0.000  1.00  0.00    +0.000 C \n"
             "ENDROOT\nTORSDOF 0\n")
    second = first.replace("Name = a", "Name = b")
    assert truncate_after_torsdof(first + second) == first
    assert truncate_after_torsdof(first.replace("\n", "\r\n")) == first
    assert truncate_after_torsdof("REMARK no torsdof\nROOT\n") is None
    assert truncate_after_torsdof("") is None


# --- legacy -------------------------------------------------------------------------------------------------------

@needs_obabel
def test_legacy_reproduces_fixture_ligands(real_run, tmp_path, monkeypatch):
    """Same atoms, names, residues, charges, types and torsion tree as the original run; only coordinates differ."""
    monkeypatch.chdir(tmp_path)
    smiles = real_run.smiles()
    for i, name in enumerate(real_run.ligands):
        rec = prepare_ligand(i, name, smiles[name], method="legacy", seed=123)
        assert rec.status == "ok", rec.error  # no element check: molecule10 (Si) is left to the engine, as before
        assert (rec.prep_method, rec.seed, rec.smiles_input) == ("legacy", None, smiles[name])
        reference = real_run.ligand_pdbqt(name).read_text()
        assert topology(rec.pdbqt) == topology(reference)
        assert rec.n_heavy == len(heavy_atoms(reference))
    assert os.listdir(tmp_path) == []  # temp files live in their own temp dir


@needs_obabel
def test_legacy_keeps_first_fragment():
    """The old salt behaviour: obabel writes one molecule per fragment and the truncation keeps the first."""
    rec = prepare_ligand(0, "salt", "[Cl-].C[NH+](C)CCc1ccccc1", method="legacy")
    assert rec.status == "ok"
    assert rec.smiles_docked == "[Cl-]" and rec.n_heavy == 1
    assert len(heavy_atoms(rec.pdbqt)) == 1

    rec = prepare_ligand(0, "salt", "CC(=O)[O-].[Na+]", method="legacy")
    assert rec.smiles_docked == "CC(=O)[O-]" and len(heavy_atoms(rec.pdbqt)) == 4


@needs_obabel
def test_standard_conversion_matches_legacy_cli(monkeypatch):
    """For the same conformer the in-memory conversion is byte-identical to the legacy mol2 + obabel CLI step."""
    def seeded_etkdg():
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        return params

    monkeypatch.setattr(ligands.AllChem, "ETKDG", seeded_etkdg)
    for smiles in (PARACETAMOL, "C[NH+]1CCN(c2ccccc2)CC1", "O=C(O)c1ccccc1[N+](=O)[O-]"):
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))  # standard embeds the canonical atom order
        standard = prepare_ligand(0, "x", smiles, seed=42)
        legacy = prepare_ligand(0, "x", smiles, method="legacy")
        assert standard.status == legacy.status == "ok"
        assert standard.pdbqt == legacy.pdbqt
        assert any(float(line[70:76]) != 0 for line in atom_lines(standard.pdbqt))  # Gasteiger charges are set


@needs_obabel
def test_legacy_failure_statuses(monkeypatch):
    assert prepare_ligand(0, "bad", "C1CC", method="legacy").status == "invalid_smiles"
    records = prepare_ligands([("etoh", "CCO"), ("bad", "C1CC")], method="legacy", seed=42, n_jobs=2)
    assert [(r.status, r.seed, r.prep_method) for r in records] == [("ok", None, "legacy"),
                                                                    ("invalid_smiles", None, "legacy")]
    monkeypatch.setattr(ligands.AllChem, "EmbedMolecule", lambda *args, **kwargs: -1)
    assert prepare_ligand(0, "etoh", "CCO", method="legacy").status == "embed_failed"


@needs_obabel
def test_legacy_with_relative_obabel_path(tmp_path, monkeypatch):
    """A relative $POCKETVINA_OBABEL (or PATH entry) is made absolute: obabel runs with its cwd in a temp dir."""
    link = tmp_path / "bin" / "obabel"
    link.parent.mkdir()
    try:
        os.symlink(find_obabel(), link)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create a symlink here")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("POCKETVINA_OBABEL", os.path.join("bin", "obabel"))
    found = find_obabel()
    assert os.path.isabs(found) and os.path.samefile(found, link)
    rec = prepare_ligand(0, "etoh", "CCO", method="legacy")
    assert rec.status == "ok", rec.error


def test_legacy_without_obabel_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("POCKETVINA_OBABEL", str(tmp_path / "no-such-obabel"))
    assert find_obabel() is None
    with pytest.raises(RuntimeError, match="obabel"):
        prepare_ligand(0, "etoh", "CCO", method="legacy")
    with pytest.raises(RuntimeError, match="obabel"):
        prepare_ligands([("etoh", "CCO")], method="legacy")
    assert prepare_ligand(0, "etoh", "CCO").status == "ok"  # standard needs no CLI


# --- write_ligand_dir ---------------------------------------------------------------------------------------------

def test_write_ligand_dir_naming(tmp_path, real_run):
    rows = [("paracetamol", PARACETAMOL), ("bad", "C1CC"), ("si", real_run.smiles()["molecule10"]), ("etoh", "CCO")]
    records = prepare_ligands(rows, n_jobs=2)
    out = tmp_path / "nested" / "ligands"
    stems = write_ligand_dir(records, str(out))
    assert stems == {"L0000000": 0, "L0000003": 3}
    assert sorted(os.listdir(out)) == ["L0000000.pdbqt", "L0000003.pdbqt"]
    for stem, idx in stems.items():
        assert (out / f"{stem}.pdbqt").read_bytes() == records[idx].pdbqt.encode()

    # records rebuilt from table rows work too; indices beyond 7 digits just get longer
    big = LigandRecord(**dict(dataclasses.asdict(records[0]), ligand_idx=12345678))
    assert write_ligand_dir([big], out / "big") == {"L12345678": 12345678}
    assert write_ligand_dir([], out / "empty") == {} and os.path.isdir(out / "empty")


def test_write_ligand_dir_rejects_bad_records_before_writing(tmp_path):
    ok = prepare_ligand(0, "paracetamol", PARACETAMOL)
    other = dataclasses.replace(ok, ligand_idx=1)
    bad = {
        "duplicate": [ok, other, ok],
        "no PDBQT": [ok, dataclasses.replace(other, pdbqt=None)],  # e.g. a table row whose pdbqt was dropped
        "negative": [ok, dataclasses.replace(other, ligand_idx=-1)],
    }
    for match, records in bad.items():
        out = tmp_path / match.replace(" ", "_")
        with pytest.raises(ValueError, match=match):
            write_ligand_dir(records, out)
        assert not out.exists()  # nothing half-written
    # a failed record without PDBQT is simply skipped
    failed = dataclasses.replace(other, status="embed_failed", pdbqt=None)
    assert write_ligand_dir([ok, failed], tmp_path / "ok") == {"L0000000": 0}
