import sys, os
import glob
import subprocess
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from statistics import median
from contextlib import contextmanager

# RDKit imports for molecule handling
from rdkit import Chem
from rdkit.Chem import AllChem

# PocketVina and configuration classes
from pocketvina import PocketVinaGPU, PocketVinaConfig, P2RankConfig, PocketConfig

from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from benchmark.utils.utils import TopologyBuilder

# Parse command-line arguments.
parser = argparse.ArgumentParser(description='Search Parameters.')
parser.add_argument('--run_number', type=int, default=1, help='Run number.')
parser.add_argument('--box_length', type=int, default=29, help='Box length.')
parser.add_argument('--vina_threads', type=int, default=8000, help='Vina threads.')
parser.add_argument('--p2rank_threads', type=int, default=8, help='P2Rank threads.')
parser.add_argument('--p2rank_model', type=str, default="alphafold", help='P2Rank model.')
parser.add_argument('--results_dir', type=str, default="FACTORY/RESULTS/PocketVina_Benchmark/benchmark_updated", help='Results directory.')
args = parser.parse_args()

###############################################
#           LIGAND CONVERSION MODULE          #
###############################################

def read_ligand_list(file_path: str):
    """Read ligand IDs from a text file, one per line."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ligand list file not found: {file_path}")
    with open(file_path, 'r') as f:
        ligands = [line.strip() for line in f if line.strip()]
    return ligands

def load_molecule(pdb_file: str):
    """
    Load a molecule from a PDB file using RDKit (removes Hs initially).
    (Not used in this revised version.)
    """
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=True)
    if mol is None:
        raise ValueError(f"Unable to load molecule from {pdb_file}")
    return mol

def fix_molecule_obabel(input_sdf: str, output_sdf: str) -> str:
    """
    Use Open Babel to convert the SDF file to MOL and then back to SDF.
    This conversion (with 3D generation and no stereo correction)
    can sometimes fix explicit valence errors and prevent long delays.
    """
    try:
        temp_mol = input_sdf.replace(".sdf", "_temp.mol")
        # Convert SDF to MOL with 3D generation and disable stereo correction.
        subprocess.run(["obabel", input_sdf, "-omol", "-O", temp_mol, "--gen3d", "--noStereo"],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Convert MOL back to SDF with 3D generation and disable stereo correction.
        subprocess.run(["obabel", temp_mol, "-osdf", "-O", output_sdf, "--gen3d", "--noStereo"],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(temp_mol)
        print(f"Fixed SDF written to: {output_sdf}")
        return output_sdf
    except Exception as e:
        raise ValueError(f"Open Babel fix failed for {input_sdf}: {e}")

def load_molecule_from_sdf(file_path: str):
    """
    Load a molecule from an SDF file using RDKit with improved error handling.
    First, attempt normal sanitization.
    If that fails, try a fallback that skips kekulization.
    If that also fails, attempt to fix the SDF using Open Babel.
    """
    supplier = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=False)
    mol = next(iter(supplier), None)
    if mol is None:
        raise ValueError(f"Unable to load molecule from {file_path}")
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
        return mol
    except Exception as e:
        print(f"Warning: Sanitization failed for {file_path}: {e}. Attempting fallback without kekulization.")
        try:
            mol_block = Chem.MolToMolBlock(mol, kekulize=False)
            mol = Chem.MolFromMolBlock(mol_block, removeHs=False, sanitize=False)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            return mol
        except Exception as e2:
            print(f"Fallback without kekulization failed for {file_path}: {e2}")
            fixed_file = file_path.replace(".sdf", "_fixed.sdf")
            try:
                fix_molecule_obabel(file_path, fixed_file)
                supplier_fixed = Chem.SDMolSupplier(fixed_file, removeHs=False, sanitize=False)
                mol_fixed = next(iter(supplier_fixed), None)
                if mol_fixed is None:
                    raise ValueError(f"Unable to load molecule from fixed file {fixed_file}")
                Chem.SanitizeMol(mol_fixed, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
                return mol_fixed
            except Exception as e3:
                raise ValueError(f"Unable to fix molecule from {file_path}: {e3}")

def load_molecule_soft(file_path: str):
    """
    Attempt to load the molecule without full sanitization.
    This soft load may bypass valence errors but can yield chemically inconsistent molecules.
    """
    supplier = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=False)
    mol = next(iter(supplier), None)
    if mol is None:
        raise ValueError(f"Unable to soft-load molecule from {file_path}")
    try:
        mol_noH = Chem.RemoveHs(mol, sanitize=False)
        mol_fixed = Chem.AddHs(mol_noH)
        err = Chem.SanitizeMol(mol_fixed, catchErrors=True)
        if err != Chem.SANITIZE_NONE:
            print(f"Soft sanitization produced error code {err} for {file_path}; proceeding with caution.")
        return mol_fixed
    except Exception as e:
        raise ValueError(f"Soft load failed for {file_path}: {e}")

def generate_3d_conformation(mol: Chem.Mol) -> Chem.Mol:
    """Generate a 3D conformation: add Hs, embed with ETKDG, then remove Hs."""
    mol_with_h = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_with_h, AllChem.ETKDG()) != 0:
        raise ValueError("3D embedding failed")
    return Chem.RemoveHs(mol_with_h)

def write_sdf(mol: Chem.Mol, sdf_file: str):
    """
    Write a molecule to an SDF file. If the default SDWriter fails (e.g. due to kekulization issues),
    fall back to writing the MolBlock with kekulization disabled.
    """
    try:
        writer = Chem.SDWriter(sdf_file)
        writer.write(mol)
        writer.close()
    except Exception as e:
        print(f"Warning: SDWriter failed for {sdf_file}: {e}. Attempting manual write without kekulization.")
        try:
            mol_block = Chem.MolToMolBlock(mol, kekulize=False)
            with open(sdf_file, "w") as f:
                f.write(mol_block)
                f.write("\n$$$$\n")
        except Exception as e2:
            raise ValueError(f"Failed to write molecule to {sdf_file}: {e2}")

def convert_sdf_to_pdbqt(sdf_file: str, pdbqt_file: str):
    """Convert an SDF file to a PDBQT file using Open Babel."""
    subprocess.run(["obabel", sdf_file, "-opdbqt", "-O", pdbqt_file, "--gen3d", "--torsions"], check=True)

# New conversion function that replaces the obabel-based method.
def convert_mol_to_pdbqt_with_topology(mol: Chem.Mol, pdbqt_file: str):
    """
    Convert an RDKit molecule to a pdbqt file using TopologyBuilder.
    This replaces the obabel conversion in the ligand conversion module.
    """
    # Create an instance of TopologyBuilder using the molecule.
    topology_builder = TopologyBuilder(mol)
    # Build the molecular graph (assigns atom types, computes charges, fragments, etc.)
    topology_builder.build_molecular_graph()
    # Write the pdbqt file to the specified path.
    topology_builder.write_pdbqt_file(pdbqt_file)
    print(f"Converted molecule to PDBQT using TopologyBuilder: {pdbqt_file}")

# Modify the ligand processing function to use the new conversion.
def process_ligand(ligand: str, input_dir: str, base_output_dir: str):
    """
    Process a single ligand:
      - Create output directories.
      - Load the ligand molecule from an SDF file (named "{ligand}_ligand.sdf").
      - Convert the molecule to PDBQT using TopologyBuilder.
    """
    ligand_output_dir = os.path.join(base_output_dir, ligand)
    os.makedirs(ligand_output_dir, exist_ok=True)
    batch_dir = os.path.join(ligand_output_dir, "batch_1")
    os.makedirs(batch_dir, exist_ok=True)
    
    sdf_input_file = os.path.join(input_dir, f"{ligand}/{ligand}_ligand.sdf")
    pdbqt_file = os.path.join(batch_dir, f"{ligand}_cleaned.pdbqt")
    
    if not os.path.exists(sdf_input_file):
        print(f"❌ Missing file: {sdf_input_file}, skipping...")
        return
    
    print(f"🔄 Processing ligand: {ligand} ...")
    try:
        mol = load_molecule_from_sdf(sdf_input_file)
    except Exception as e:
        print(f"Normal loading failed for {ligand}: {e}")
        try:
            print(f"Attempting soft load for {ligand}...")
            mol = load_molecule_soft(sdf_input_file)
        except Exception as e2:
            print(f"❌ Soft load failed for {ligand}: {e2}. Skipping ligand.")
            return

    try:
        # Instead of writing a temporary SDF and calling obabel,
        # convert the RDKit molecule to PDBQT using TopologyBuilder.
        convert_mol_to_pdbqt_with_topology(mol, pdbqt_file)
        print(f"✅ Converted {ligand} to {pdbqt_file}")
    except Exception as e:
        print(f"❌ Error processing {ligand} during conversion: {e}")

def main_conversion(testset_file: str):
    """Main routine for ligand conversion."""
    try:
        ligand_list = read_ligand_list(testset_file)
    except Exception as e:
        print(f"Error reading ligand list: {e}")
        return
    
    base_dir = "/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF"
    sub_output_dir = f"{args.results_dir}/dockgen/{args.p2rank_model}_prank_thread{args.p2rank_threads}/{args.box_length}_Angst/run_{args.run_number}/results"
    base_output_dir = os.path.join(base_dir, sub_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    
    for ligand in ligand_list:
        process_ligand(ligand, os.path.join(base_dir, "PROTEIN_DB/DockGen/processed_files"), base_output_dir)
    
    print("🎯 Ligand conversion complete!")

###############################################
#             DOCKING & RMSD MODULE           #
###############################################

def load_molecule_from_sdf_for_docking(file_path: str):
    """Load a molecule from an SDF file for docking (as reference), with error handling."""
    supplier = Chem.SDMolSupplier(file_path, removeHs=False, sanitize=False)
    mol = next(iter(supplier), None)
    if mol is None:
        print(f"Error: Could not read molecule from {file_path}")
        return None
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    except (Chem.KekulizeException, Chem.AtomValenceException):
        print(f"Warning: Issue sanitizing {file_path}.")
    try:
        mol = Chem.RemoveHs(mol, implicitOnly=False, updateExplicitCount=True, sanitize=True)
    except Exception as e:
        print(f"Error removing hydrogens from {file_path}: {e}")
    return mol

def compute_symmetry_rmsd(ref_file: str, cand_file: str, protein_file: str):
    """
    Compute the symmetry-corrected RMSD between two SDF molecules.
    Returns a tuple: (rmsd, None, None)
    """
    try:
        ref_rdmol = load_molecule_from_sdf_for_docking(ref_file)
        cand_rdmol = load_molecule_from_sdf_for_docking(cand_file)
        if ref_rdmol is None or cand_rdmol is None:
            print(f"Skipping RMSD for {cand_file} due to loading failure.")
            return None
        ref_mol = Molecule.from_rdkit(ref_rdmol)
        cand_mol = Molecule.from_rdkit(cand_rdmol)
        ref_mol.strip()
        cand_mol.strip()
        rmsd_value = rmsdwrapper(ref_mol, cand_mol, symmetry=True, minimize=False)
        return rmsd_value[0], None, None
    except Exception as e:
        print(f"Error computing RMSD for {cand_file}: {e}")
    return None

def convert_pdbqt_to_sdf(pdbqt_file: str, sdf_output: str):
    """Convert a pdbqt file to SDF using Open Babel."""
    subprocess.run(["obabel", pdbqt_file, "-osdf", "-O", sdf_output], check=True)
    print(f"Converted {pdbqt_file} to {sdf_output}")

def separate_sdf_conformations(sdf_file: str, sdf_dir: str):
    """
    Separate multi-conformation SDF files into individual files.
    Returns a list of SDF file paths.
    """
    new_sdf_files = []
    try:
        with open(sdf_file, "r") as f_in:
            content = f_in.read()
    except Exception as e:
        print(f"Error reading {sdf_file}: {e}")
        return []
    records = [r.strip() for r in content.split("$$$$") if r.strip()]
    if len(records) > 1:
        base_name = os.path.splitext(os.path.basename(sdf_file))[0]
        for i, record in enumerate(records, start=1):
            new_sdf = os.path.join(sdf_dir, f"{base_name}_conf{i}.sdf")
            with open(new_sdf, "w") as f_out:
                f_out.write(record + "\n$$$$\n")
            new_sdf_files.append(new_sdf)
            print(f"Created separate SDF: {new_sdf}")
        os.remove(sdf_file)
        print(f"Removed combined SDF file: {sdf_file}")
    else:
        new_sdf_files.append(sdf_file)
        print(f"File {sdf_file} contains a single conformation.")
    return new_sdf_files

def process_single_pdb_id(pdb_id: str) -> list:
    """
    Process a single pdb_id:
      - Set up directories and file paths.
      - Create a PocketVinaGPU configuration and execute docking.
      - Convert resulting pdbqt files to SDF.
      - Separate multi-conformation SDF files.
      - Compute RMSD.
    Returns a list of tuples:
      (Candidate_SDF, Reference_SDF, pdb_id, RMSD)
    """
    results = []
    base_dir = f"/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/PROTEIN_DB/DockGen/processed_files/{pdb_id}"
    ref_sdf = os.path.join(base_dir, f"{pdb_id}_ligand.sdf")
    smiles_path = os.path.join(base_dir, f"{pdb_id}_ligand.smiles")
    protein_txt_path = os.path.join(base_dir, f"{pdb_id}_protein_path.txt")
    protein_file = os.path.join(base_dir, f"{pdb_id}_protein_processed.pdb")
    
    common_base_dir = "/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF"
    sub_output_dir = f"{args.results_dir}/dockgen/{args.p2rank_model}_prank_thread{args.p2rank_threads}/{args.box_length}_Angst/run_{args.run_number}/results"
    output_dir = os.path.join(common_base_dir, sub_output_dir, pdb_id)
    
    config = PocketVinaConfig(
        protein_paths=protein_txt_path,
        smiles_file=smiles_path,
        output_dir=output_dir,
        batch_size=10,
        p2rank_config=P2RankConfig(
            threads=args.p2rank_threads,
            visualizations=False,
            vis_copy_proteins=False,
            output_dir=output_dir,
            model=args.p2rank_model
        ),
        pocket_config=PocketConfig(
            size_x=args.box_length,
            size_y=args.box_length,
            size_z=args.box_length,
            thread=args.vina_threads
        )
    )
    
    print(f"Running docking for {pdb_id} ...")
    pocketvina = PocketVinaGPU(config)
    pocketvina.execute_docking()
    
    batch_out_dir = os.path.join(output_dir, "batch_1_out")
    sdf_dir = os.path.join(batch_out_dir, "sdf_files")
    os.makedirs(sdf_dir, exist_ok=True)
    
    pdbqt_files = glob.glob(os.path.join(batch_out_dir, "*.pdbqt"))
    for pdbqt in pdbqt_files:
        base_name = os.path.splitext(os.path.basename(pdbqt))[0]
        sdf_output = os.path.join(sdf_dir, base_name + ".sdf")
        try:
            convert_pdbqt_to_sdf(pdbqt, sdf_output)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {pdbqt} to SDF: {e}")
            continue
        
        candidate_sdf_files = separate_sdf_conformations(sdf_output, sdf_dir)
        for cand_sdf in candidate_sdf_files:
            # Here we do *not* update connectivity,
            # instead computing RMSD on the candidate molecule as is.
            res = compute_symmetry_rmsd(ref_sdf, cand_sdf, protein_file)
            if res is not None:
                rmsd_val, _, _ = res
                print(f"RMSD for {os.path.basename(cand_sdf)} vs {os.path.basename(ref_sdf)}: {rmsd_val:.3f} Å")
                results.append((os.path.basename(cand_sdf), os.path.basename(ref_sdf), pdb_id, rmsd_val))
    return results

def process_testset(testset_file: str, limit: int = None) -> list:
    """Read pdb_ids from a testset file and process each; return combined results."""
    all_results = []
    try:
        with open(testset_file, "r") as f:
            pdb_ids = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading testset file {testset_file}: {e}")
        return all_results
    
    for count, pdb_id in enumerate(pdb_ids):
        if limit is not None and count >= limit:
            break
        print(f"Processing pdb_id: {pdb_id}")
        results = process_single_pdb_id(pdb_id)
        all_results.extend(results)
    return all_results

def default_converter(o):
    if isinstance(o, (np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.float64, np.float32)):
        return float(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def append_summary_to_csv(csv_filename: str, summary_dict: dict, header_length: int):
    """Append a summary row to the CSV file."""
    summary_str = json.dumps(summary_dict, default=default_converter)
    row = [""] * header_length
    row[0] = "Summary"
    row[-1] = summary_str
    with open(csv_filename, "a") as f:
        f.write("\n" + ",".join(row))

def save_results_to_csv(results: list, csv_filename: str):
    """Save results to a CSV file and append a summary dictionary at the end."""
    if not results:
        print("No results to save.")
        return
    header = ["Candidate_SDF", "Reference_SDF", "pdb_id", "RMSD"]
    df = pd.DataFrame(results, columns=header)
    df_sorted = df.sort_values(by="RMSD", ascending=True)
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    df_sorted.to_csv(csv_filename, index=False)
    print(f"Results saved to CSV: {csv_filename}")
    
    grouped = df.groupby("pdb_id")["RMSD"]
    best = grouped.min()
    total_cases = len(best)
    best_le1 = (best <= 1).sum()
    best_le2 = (best <= 2).sum()
    best_median = best.median() if total_cases > 0 else None
    best_pct_le1 = (best_le1 / total_cases * 100) if total_cases > 0 else 0
    best_pct_le2 = (best_le2 / total_cases * 100) if total_cases > 0 else 0


    summary_dict = {
        "Config": {
            "run_number": args.run_number,
            "box_length": args.box_length,
            "vina_threads": args.vina_threads,
            "p2rank_threads": args.p2rank_threads,
            "p2rank_model": args.p2rank_model
        },
        "Unique Number of pdb_id": total_cases,
        "Best RMSD <= 1 (%)": f"{best_le1} ({best_pct_le1:.1f}%)",
        "Best RMSD <= 2 (%)": f"{best_le2} ({best_pct_le2:.1f}%)",
        "Best median RMSD": best_median
    }
    
    append_summary_to_csv(csv_filename, summary_dict, len(header))

def main_docking(testset_file: str):
    """Main routine for docking evaluation using RMSD only."""
    results = process_testset(testset_file, limit=190)
    common_base_dir = "/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF"
    sub_output_dir = f"{args.results_dir}/dockgen/{args.p2rank_model}_prank_thread{args.p2rank_threads}/{args.box_length}_Angst/run_{args.run_number}/results"
    csv_filename = os.path.join(common_base_dir, sub_output_dir, "results.csv")
    save_results_to_csv(results, csv_filename)

###############################################
#                MAIN PIPELINE                #
###############################################

def main():
    testset_file = "/fast/AG_Akalin/asarigun/Arcas_Stage_1/ROOF/PROTEIN_DB/DockGen/split_test.txt"  # File containing pdb_ids, one per line.
    print("Starting ligand conversion phase...")
    main_conversion(testset_file)
    print("Starting docking evaluation phase...")
    main_docking(testset_file)

if __name__ == '__main__':
    main()
