"""pocketvina.io.importer: a legacy PocketVina output_dir (tests/data/legacy_run) -> results directory (atlas)."""
import csv
import gzip
import hashlib
import io
import json
import logging
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from pocketvina import convert, pdbqt
from pocketvina.io import posefile
from pocketvina.io.importer import _natural_key, import_legacy
from pocketvina.io.tables import read_table
from pocketvina.io.writer import ResultsDir

PID = "AF-Q96LA5-F1-model_v4"
NAMES = ("molecule2", "molecule5", "molecule10")


def digests(root) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(root).rglob("*") if p.is_file()}


def table(results, name):
    path = Path(results) / "tables" / (f"{name}.parquet" if name == "ligands" else f"{name}/{PID}.parquet")
    return read_table(path)


def read_sdf(sdf: bytes):
    mols = list(Chem.ForwardSDMolSupplier(io.BytesIO(sdf), removeHs=False))
    assert len(mols) == 1 and mols[0] is not None
    return mols[0]


def combined_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_config(root, base, pocket, size=(20, 20, 20), batch=1):
    folder = Path(root) / "configs"
    folder.mkdir(exist_ok=True)
    text = (f"receptor = /old/run/proteins/structures/{base}.pdbqt\nligand_directory = /old/run/batch_{batch}\n"
            f"opencl_binary_path = /old/pkg/QuickVina2-GPU-2-1\ncenter_x = 1.0\ncenter_y = 2.0\ncenter_z = 3.0\n"
            f"size_x = {size[0]}\nsize_y = {size[1]}\nsize_z = {size[2]}\nthread = 8000\npocket_rank = {pocket}\n")
    (folder / f"{base}_pocketrank_{pocket}_config_batch_{batch}.txt").write_text(text)


def write_smiles(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["molecule_name", "smiles"])
        w.writerows(rows)
    return str(path)


@pytest.fixture(scope="module")
def imported(real_run, tmp_path_factory):
    """The fixture run imported with its SMILES, straight from the read-only tests/data folder."""
    base = tmp_path_factory.mktemp("import")
    tmp = base / "tmp"
    results = import_legacy(str(real_run.root), str(base / "results"), str(real_run.smiles_csv), tmp_dir=str(tmp))
    return Path(results)


@pytest.fixture(scope="module")
def imported_no_smiles(real_run, tmp_path_factory):
    return Path(import_legacy(str(real_run.root), str(tmp_path_factory.mktemp("import") / "results")))


# --- the fixture run -----------------------------------------------------------------------------------------------

def test_layout(imported, tmp_path_factory):
    files = sorted(digests(imported))
    assert files == sorted([
        "MANIFEST.json", f"manifest/{PID}.json", f"poses/{PID}.poses", f"structures/{PID}.pdbqt.gz",
        "tables/ligands.parquet", *(f"tables/{t}/{PID}.parquet"
                                    for t in ("proteins", "pockets", "pocket_runs", "modes", "pairs"))])
    rd = ResultsDir(imported)
    assert rd.done_proteins() == {PID} and rd.verify(PID) == []
    manifest = rd.protein_manifest(PID)
    run_level = ("MANIFEST.json", f"manifest/{PID}.json", "tables/ligands.parquet")
    assert sorted(manifest["files"]) == [f for f in files if f not in run_level]
    assert manifest["summary"]["source"] == "import_legacy" and manifest["summary"]["n_outputs"] == 4
    assert rd.run_manifest()["params"] == {"source": "import_legacy", "ligand_prep": "legacy"}
    assert not any(imported.parent.joinpath("tmp").iterdir())  # the temp folder was removed


def test_ligands_table(imported, real_run):
    lig = table(imported, "ligands")
    smiles = real_run.smiles()
    assert list(lig["ligand_idx"]) == [0, 1, 2] and tuple(lig["ligand_id"]) == NAMES  # smiles.csv order
    assert list(lig["smiles_input"]) == [smiles[n] for n in NAMES]
    assert list(lig["pdbqt"]) == [real_run.ligand_pdbqt(n).read_text() for n in NAMES]
    assert set(lig["status"]) == {"ok"} and set(lig["prep_method"]) == {"legacy"}
    assert lig["seed"].isna().all() and lig["error"].isna().all()
    for row in lig.itertuples():
        mol = Chem.MolFromSmiles(smiles[row.ligand_id])
        assert row.smiles_docked == Chem.MolToSmiles(mol)
        assert row.inchikey == Chem.MolToInchiKey(mol)
        assert row.n_heavy == mol.GetNumHeavyAtoms()


def test_pocket_runs(imported, real_run):
    runs = table(imported, "pocket_runs")
    assert len(runs) == len(NAMES) * len(real_run.pockets)
    for row in runs.itertuples():
        assert row.protein_id == PID and row.ligand_id == NAMES[row.ligand_idx]
        if row.ligand_id in real_run.failed:
            assert (row.status, row.n_modes, row.length, row.raw_length) == ("no_output", 0, 0, 0)
            assert math.isnan(row.best_affinity) and row.sha256 is None and "no engine output" in row.error
            continue
        raw = real_run.output(row.ligand_id, row.pocket).read_bytes()
        assert (row.status, row.n_modes, row.raw_length) == ("ok", real_run.n_modes, len(raw))
        assert row.sha256 == hashlib.sha256(raw).digest()
        assert row.best_affinity == np.float32(min(m.affinity for m in pdbqt.parse_modes(raw)))
        assert row.error is None and math.isnan(row.seed) and math.isnan(row.runtime_s)


def test_pose_blocks_byte_identical(imported, real_run):
    runs = table(imported, "pocket_runs")
    with posefile.Reader(imported / "poses" / f"{PID}.poses") as reader:
        assert (reader.N, reader.K) == (3, 2)
        for row in runs.itertuples():
            block = reader.modes(row.ligand_idx, row.pocket)
            assert reader.block_range(row.ligand_idx, row.pocket) == (row.offset, row.length)
            if row.ligand_id in real_run.failed:
                assert block is None
            else:
                assert block == real_run.output(row.ligand_id, row.pocket).read_bytes()


def test_modes_match_combined_csv(imported, real_run):
    modes = table(imported, "modes")
    assert len(modes) == len(real_run.docked) * len(real_run.pockets) * real_run.n_modes
    expected = {(r["protein"], r["molecule_name"], int(r["mode"])): r for r in combined_rows(real_run.combined_csv)}
    assert len(expected) == len(modes)
    for row in modes.itertuples():
        name = NAMES[row.ligand_idx]
        e = expected[(f"{PID}-pocket{row.pocket}", name, row.mode)]
        assert int(e["pocket rank"]) == row.pocket
        assert row.affinity == np.float32(e["affinity (kcal/mol)"])
        assert row.rmsd_lb == np.float32(e["dist from rmsd l.b."])
        assert row.rmsd_ub == np.float32(e["dist from rmsd u.b."])
        rec = pdbqt.parse_modes(real_run.output(name, row.pocket).read_bytes())[row.mode - 1]
        assert (row.raw_offset, row.raw_length) == (rec.raw_offset, rec.raw_length)
        assert row.uniprot == "Q96LA5"


def test_pairs_and_best_sdf(imported, real_run):
    pairs = table(imported, "pairs").set_index("ligand_id")
    assert (pairs.loc["molecule2", ["best_pocket", "best_mode"]] == [2, 1]).all()
    assert pairs.loc["molecule2", "vina"] == np.float32(-6.3)
    assert (pairs.loc["molecule5", ["best_pocket", "best_mode"]] == [1, 1]).all()
    assert pairs.loc["molecule5", "vina"] == np.float32(-5.2)
    assert pairs.loc["molecule10", "sdf_status"] == "no_pose" and pairs.loc["molecule10", "best_pocket"] == -1
    smiles = real_run.smiles()
    with posefile.Reader(imported / "poses" / f"{PID}.poses") as reader:
        assert reader.top(2) is None
        for name in real_run.docked:
            row = pairs.loc[name]
            assert row["sdf_status"] == "ok"
            sdf = reader.top(int(row["ligand_idx"]))
            assert reader.top_range(int(row["ligand_idx"])) == (row["top_offset"], row["top_length"])
            assert (len(sdf), hashlib.sha256(sdf).digest()) == (row["top_raw_length"], row["top_sha256"])
            mol = read_sdf(sdf)
            assert mol.GetProp("_Name") == name and mol.GetProp("protein_id") == PID
            assert int(mol.GetProp("ligand_idx")) == row["ligand_idx"]
            assert int(mol.GetProp("pocket")) == row["best_pocket"] and mol.GetProp("mode") == "1"
            assert np.float32(mol.GetProp("affinity")) == row["vina"]
            assert convert.same_molecule(mol, smiles[name])
            # same coordinates as the best MODEL of the engine output
            model = pdbqt.model(real_run.output(name, int(row["best_pocket"])).read_bytes(), 1)
            first_atom = next(line for line in model.decode().splitlines() if line.startswith("ATOM"))
            xyz = [float(first_atom[30:38]), float(first_atom[38:46]), float(first_atom[46:54])]
            assert any(np.allclose(list(mol.GetConformer().GetAtomPosition(i)), xyz, atol=1e-3)
                       for i in range(mol.GetNumAtoms()))


def test_pockets_and_protein(imported, real_run):
    pockets = table(imported, "pockets")
    assert list(pockets["pocket"]) == list(range(1, real_run.n_pockets + 1))
    with open(real_run.pockets_csv, newline="") as f:
        source = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f, skipinitialspace=True)]
    for row, src in zip(pockets.itertuples(), source):
        assert row.name == src["name"] and row.residue_ids == src["residue_ids"]
        for col in ("score", "probability", "center_x", "center_y", "center_z"):
            assert getattr(row, col) == np.float32(src[col])
    assert pockets[["size_x", "size_y", "size_z"]].isna().all().all()  # no configs/ in the fixture

    protein = table(imported, "proteins").iloc[0]
    receptor = real_run.receptor.read_bytes()
    assert protein["protein_id"] == PID and protein["uniprot"] == "Q96LA5"
    assert (protein["n_pockets"], protein["n_pockets_docked"], protein["status"]) == (22, 2, "ok")
    assert protein["receptor_sha256"] == hashlib.sha256(receptor).digest()
    assert protein["source_path"] == str(real_run.receptor)
    assert protein["structure_file"] == f"structures/{PID}.pdbqt.gz"
    assert gzip.decompress((imported / protein["structure_file"]).read_bytes()) == receptor


def test_without_smiles(imported_no_smiles, real_run):
    lig = table(imported_no_smiles, "ligands")
    assert tuple(lig["ligand_id"]) == NAMES  # natural name order
    assert set(lig["smiles_input"]) == {""} and lig["smiles_docked"].isna().all() and lig["inchikey"].isna().all()
    assert set(lig["status"]) == {"ok"}
    assert list(lig["n_heavy"]) == [Chem.MolFromSmiles(real_run.smiles()[n]).GetNumHeavyAtoms() for n in NAMES]
    pairs = table(imported_no_smiles, "pairs").set_index("ligand_id")
    assert list(pairs["sdf_status"]) == ["convert_failed", "convert_failed", "no_pose"]
    assert (pairs["top_length"] == 0).all() and pairs["top_sha256"].isna().all()
    assert pairs.loc["molecule2", "vina"] == np.float32(-6.3)
    with posefile.Reader(imported_no_smiles / "poses" / f"{PID}.poses") as reader:
        assert all(reader.top(i) is None for i in range(3))
        for name in real_run.docked:
            for k in real_run.pockets:
                assert reader.modes(NAMES.index(name), k) == real_run.output(name, k).read_bytes()


# --- resume, options, odd folders ----------------------------------------------------------------------------------

def test_legacy_dir_is_only_read(legacy_run_dir, tmp_results, tmp_path, real_run):
    before = digests(legacy_run_dir)
    import_legacy(legacy_run_dir, tmp_results, str(real_run.smiles_csv), tmp_dir=str(tmp_path / "tmp"))
    assert digests(legacy_run_dir) == before
    assert list((tmp_path / "tmp").iterdir()) == []


def test_resume_and_conflicts(legacy_run_dir, tmp_results, real_run, caplog):
    import_legacy(legacy_run_dir, tmp_results, str(real_run.smiles_csv))
    manifest = Path(tmp_results, "manifest", f"{PID}.json")
    first = manifest.read_bytes()
    with caplog.at_level(logging.INFO, logger="pocketvina.io.importer"):
        assert import_legacy(legacy_run_dir, tmp_results, str(real_run.smiles_csv)) == tmp_results
    assert manifest.read_bytes() == first and "already imported" in caplog.text
    with pytest.raises(ValueError, match="other ligands"):
        import_legacy(legacy_run_dir, tmp_results)  # no SMILES: other ligand table
    import_legacy(legacy_run_dir, tmp_results, force=True)
    assert manifest.read_bytes() != first
    pairs = read_table(Path(tmp_results, "tables", "pairs", f"{PID}.parquet"))
    assert set(pairs["sdf_status"]) == {"convert_failed", "no_pose"}
    assert ResultsDir(tmp_results).verify(PID) == []


def test_configs_give_box_size_and_docked_pockets(legacy_run_dir, tmp_results):
    for k in (1, 2, 3):  # pocket 3 was docked too, but no ligand got an output there
        write_config(legacy_run_dir, PID, k, size=(22, 24, 26))
    import_legacy(legacy_run_dir, tmp_results)
    pockets = read_table(Path(tmp_results, "tables", "pockets", f"{PID}.parquet"))
    assert (pockets[["size_x", "size_y", "size_z"]].to_numpy() == [22, 24, 26]).all()  # one size for the run
    runs = read_table(Path(tmp_results, "tables", "pocket_runs", f"{PID}.parquet"))
    assert sorted(set(runs["pocket"])) == [1, 2, 3] and len(runs) == 9
    assert set(runs.loc[runs["pocket"] == 3, "status"]) == {"no_output"}
    protein = read_table(Path(tmp_results, "tables", "proteins", f"{PID}.parquet")).iloc[0]
    assert protein["n_pockets_docked"] == 3
    with posefile.Reader(Path(tmp_results, "poses", f"{PID}.poses")) as reader:
        assert reader.K == 3 and reader.meta["pockets"] == [1, 2, 3]


def test_configs_with_different_sizes(legacy_run_dir, tmp_results):
    write_config(legacy_run_dir, PID, 1, size=(20, 20, 20))
    write_config(legacy_run_dir, PID, 2, size=(30, 30, 30))
    Path(legacy_run_dir, "configs", "notes.txt").write_text("not a config\n")
    import_legacy(legacy_run_dir, tmp_results)
    sizes = read_table(Path(tmp_results, "tables", "pockets", f"{PID}.parquet"))[["size_x", "size_y", "size_z"]]
    assert (sizes.iloc[0] == 20).all() and (sizes.iloc[1] == 30).all()
    assert sizes.iloc[2:].isna().all().all()  # no single size to fall back to


def test_pockets_ranked_above_int8_are_left_out(legacy_run_dir, tmp_results, real_run, caplog):
    """A legacy run docks every P2Rank pocket; ranks above 127 must not fail the whole import."""
    shutil.copy(real_run.output("molecule2", 1),
                Path(legacy_run_dir, "batch_1_out", f"{PID}-pocket130-molecule2_cleaned_out.pdbqt"))
    write_config(legacy_run_dir, PID, 140)
    with caplog.at_level(logging.WARNING, logger="pocketvina.io.importer"):
        import_legacy(legacy_run_dir, tmp_results)
    assert "ranked above 127" in caplog.text
    runs = read_table(Path(tmp_results, "tables", "pocket_runs", f"{PID}.parquet"))
    assert sorted(set(runs["pocket"])) == [1, 2] and len(runs) == 6
    protein = read_table(Path(tmp_results, "tables", "proteins", f"{PID}.parquet")).iloc[0]
    assert (protein["status"], protein["n_pockets_docked"]) == ("ok", 2)
    assert protein["error"].endswith("(int8 pocket columns): 130, 140")
    assert ResultsDir(tmp_results).protein_manifest(PID)["summary"]["n_outputs"] == 4


def test_outputs_of_unknown_receptors_are_skipped(legacy_run_dir, tmp_results, real_run, caplog):
    src = real_run.output("molecule2", 1)
    shutil.copy(src, Path(legacy_run_dir, "batch_1_out", "OTHER-pocket1-molecule2_cleaned_out.pdbqt"))
    with caplog.at_level(logging.WARNING, logger="pocketvina.io.importer"):
        import_legacy(legacy_run_dir, tmp_results)
    assert "OTHER-pocket1-molecule2_cleaned_out.pdbqt" in caplog.text
    assert ResultsDir(tmp_results).done_proteins() == {PID}
    assert len(read_table(Path(tmp_results, "tables", "pocket_runs", f"{PID}.parquet"))) == 6


def test_duplicate_outputs(legacy_run_dir, tmp_results, real_run):
    second = Path(legacy_run_dir, "batch_2_out")
    shutil.copytree(Path(legacy_run_dir, "batch_1_out"), second)
    import_legacy(legacy_run_dir, tmp_results)  # identical copies are one output
    assert len(read_table(Path(tmp_results, "tables", "modes", f"{PID}.parquet"))) == 36
    name = real_run.output("molecule5", 2).name
    (second / name).write_bytes((second / name).read_bytes().replace(b"-5.1", b"-9.9", 1))
    with pytest.raises(ValueError, match="two different engine outputs"):
        import_legacy(legacy_run_dir, str(Path(tmp_results).with_name("other")))


def test_unreadable_output_is_kept(legacy_run_dir, tmp_results, real_run):
    path = Path(legacy_run_dir, "batch_1_out", real_run.output("molecule2", 2).name)
    broken = path.read_bytes()[:2000]  # cut inside MODEL 1: no ENDMDL
    path.write_bytes(broken)
    import_legacy(legacy_run_dir, tmp_results, str(real_run.smiles_csv))
    runs = read_table(Path(tmp_results, "tables", "pocket_runs", f"{PID}.parquet")).set_index(["ligand_id", "pocket"])
    assert runs.loc[("molecule2", 2), "status"] == "parse_error" and runs.loc[("molecule2", 2), "n_modes"] == 0
    with posefile.Reader(Path(tmp_results, "poses", f"{PID}.poses")) as reader:
        assert reader.modes(0, 2) == broken  # stored anyway, byte for byte
    pairs = read_table(Path(tmp_results, "tables", "pairs", f"{PID}.parquet")).set_index("ligand_id")
    assert pairs.loc["molecule2", "best_pocket"] == 1 and pairs.loc["molecule2", "vina"] == np.float32(-5.6)


def test_smiles_file_and_run_disagree(legacy_run_dir, tmp_results, real_run, tmp_path, caplog):
    smiles = real_run.smiles()
    path = write_smiles(tmp_path / "ligands.csv", [("molecule5", smiles["molecule5"]), ("extra", "CCO"),
                                                   ("broken", "C1CC"), ("molecule2", smiles["molecule2"])])
    with caplog.at_level(logging.WARNING, logger="pocketvina.io.importer"):
        import_legacy(legacy_run_dir, tmp_results, path)
    assert "molecule10" in caplog.text
    lig = read_table(Path(tmp_results, "tables", "ligands.parquet")).set_index("ligand_id")
    assert list(lig.index) == ["molecule5", "extra", "broken", "molecule2", "molecule10"]
    assert lig.loc["extra", "status"] == "error" and "preparation failed" in lig.loc["extra", "error"]
    assert lig.loc["broken", "status"] == "invalid_smiles" and lig.loc["broken", "smiles_input"] == "C1CC"
    assert lig.loc["molecule10", "status"] == "ok" and lig.loc["molecule10", "smiles_input"] == ""
    runs = read_table(Path(tmp_results, "tables", "pocket_runs", f"{PID}.parquet"))
    assert len(runs) == 5 * 2  # every ligand x docked pocket, as pipeline.run_atlas records them
    skipped = runs[runs["ligand_id"].isin(["extra", "broken"])]
    assert len(skipped) == 4 and set(skipped["status"]) == {"skipped"} and (skipped["length"] == 0).all()
    assert skipped["error"].str.startswith("ligand not prepared: ").all()
    assert set(runs.loc[~runs.index.isin(skipped.index), "status"]) == {"ok", "no_output"}
    pairs = read_table(Path(tmp_results, "tables", "pairs", f"{PID}.parquet")).set_index("ligand_id")
    assert pairs.loc["molecule5", "sdf_status"] == "ok" and pairs.loc["molecule5", "ligand_idx"] == 0
    assert list(pairs.loc[["extra", "broken"], "sdf_status"]) == ["no_pose", "no_pose"]
    assert list(pairs.loc[["extra", "broken"], "best_pocket"]) == [-1, -1]


def test_salt_smiles_describes_the_docked_fragment(legacy_run_dir, tmp_results, real_run, tmp_path):
    smiles = dict(real_run.smiles())
    smiles["molecule2"] += ".[Na+]"  # the legacy prep docks the fragment OpenBabel kept (here the first one)
    path = write_smiles(tmp_path / "salt.csv", list(smiles.items()))
    import_legacy(legacy_run_dir, tmp_results, path)
    lig = read_table(Path(tmp_results, "tables", "ligands.parquet")).set_index("ligand_id")
    plain = Chem.MolFromSmiles(real_run.smiles()["molecule2"])
    assert lig.loc["molecule2", "smiles_input"].endswith(".[Na+]")
    assert lig.loc["molecule2", "smiles_docked"] == Chem.MolToSmiles(plain)
    assert lig.loc["molecule2", "n_heavy"] == plain.GetNumHeavyAtoms()
    pairs = read_table(Path(tmp_results, "tables", "pairs", f"{PID}.parquet")).set_index("ligand_id")
    assert pairs.loc["molecule2", "sdf_status"] == "ok"


def test_unbatched_ligands_and_missing_pockets_csv(legacy_run_dir, tmp_results, real_run, caplog):
    for p in Path(legacy_run_dir, "batch_1").iterdir():
        p.rename(Path(legacy_run_dir, p.name))  # as before split_files_into_batches
    Path(legacy_run_dir, "batch_1").rmdir()
    Path(legacy_run_dir, "proteins", "pockets", f"{PID}.csv").unlink()
    with caplog.at_level(logging.WARNING, logger="pocketvina.io.importer"):
        import_legacy(legacy_run_dir, tmp_results)
    assert "no P2Rank predictions" in caplog.text
    lig = read_table(Path(tmp_results, "tables", "ligands.parquet"))
    assert list(lig["pdbqt"]) == [real_run.ligand_pdbqt(n).read_text() for n in NAMES]
    assert len(read_table(Path(tmp_results, "tables", "pockets", f"{PID}.parquet"))) == 0
    protein = read_table(Path(tmp_results, "tables", "proteins", f"{PID}.parquet")).iloc[0]
    assert (protein["n_pockets"], protein["n_pockets_docked"]) == (0, 2)


def test_odd_file_names(legacy_run_dir, tmp_results):
    """Every folder listing is sorted by a natural key; "²" is a digit to str.isdigit() but not to int()."""
    assert sorted(["mol10", "mol2", "mol1²2"], key=_natural_key) == ["mol1²2", "mol2", "mol10"]
    Path(legacy_run_dir, "batch_1_out", "notes1²2.txt").write_text("not an output\n")
    import_legacy(legacy_run_dir, tmp_results)
    assert ResultsDir(tmp_results).done_proteins() == {PID}


def test_protein_without_outputs(legacy_run_dir, tmp_results):
    shutil.copy(Path(legacy_run_dir, "proteins", "structures", f"{PID}.pdbqt"),
                Path(legacy_run_dir, "proteins", "structures", "empty_receptor.pdbqt"))
    import_legacy(legacy_run_dir, tmp_results)
    rd = ResultsDir(tmp_results)
    assert rd.done_proteins() == {PID, "empty_receptor"}
    protein = read_table(Path(tmp_results, "tables", "proteins", "empty_receptor.parquet")).iloc[0]
    assert (protein["status"], protein["n_pockets_docked"]) == ("no_output", 0)
    assert len(read_table(Path(tmp_results, "tables", "pocket_runs", "empty_receptor.parquet"))) == 0
    assert rd.protein_manifest("empty_receptor")["summary"]["n_runs"] == 0


def test_errors(legacy_run_dir, tmp_path, real_run):
    with pytest.raises(FileNotFoundError):
        import_legacy(str(tmp_path / "missing"), str(tmp_path / "r1"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="proteins/structures"):
        import_legacy(str(tmp_path / "empty"), str(tmp_path / "r2"))
    dup = write_smiles(tmp_path / "dup.csv", [("molecule2", "CCO"), ("molecule2", "CCN")])
    with pytest.raises(ValueError, match="listed twice"):
        import_legacy(legacy_run_dir, str(tmp_path / "r3"), dup)
    structures = Path(legacy_run_dir, "proteins", "structures")
    shutil.copy(structures / f"{PID}.pdbqt", structures / "a b.pdbqt")
    shutil.copy(structures / f"{PID}.pdbqt", structures / "a_b.pdbqt")
    with pytest.raises(ValueError, match="both give protein_id 'a_b'"):
        import_legacy(legacy_run_dir, str(tmp_path / "r4"))
    assert not (tmp_path / "r4").exists()


def test_manifest_is_json_with_provenance(imported, real_run):
    summary = json.loads((imported / "manifest" / f"{PID}.json").read_text())["summary"]
    assert summary["legacy_output_dir"] == str(real_run.root)
    assert summary["receptor"] == str(real_run.receptor)
    assert summary["runs_by_status"] == {"ok": 4, "no_output": 2}
    assert summary["sdf_by_status"] == {"ok": 2, "no_pose": 1}
