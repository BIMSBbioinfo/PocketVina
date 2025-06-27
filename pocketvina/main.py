import os
import csv
import json
import time
import shutil
import logging
import subprocess
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel

@dataclass
class P2RankConfig:
    """Configuration for P2Rank processor"""
    threads: int = 4
    visualizations: bool = False
    vis_copy_proteins: bool = False
    output_dir: str = "output"
    model: str = "alphafold_conservation_hmm"

@dataclass
class PocketConfig:
    size_x: int = 20
    size_y: int = 20
    size_z: int = 20
    thread: int = 8000

@dataclass
class Paths:
    database_path: Optional[str] = None
    protein_paths: Optional[str] = None
    output_dir: str = "output"  # Add this
    pdb_structures: Optional[str] = None 
    uniprot_structures: Optional[str] = None
    pdb_pockets: Optional[str] = None
    uniprot_pockets: Optional[str] = None
    structures_dir: Optional[str] = None
    pockets_dir: Optional[str] = None

    def __post_init__(self):
        if self.database_path:
            # Database path initialization
            self.pdb_structures = f"{self.database_path}/PDB_conservation/structures"
            self.uniprot_structures = f"{self.database_path}/AF_conservation/structures"
            self.pdb_pockets = f"{self.database_path}/PDB_conservation/pockets"
            self.uniprot_pockets = f"{self.database_path}/AF_conservation/pockets"
        elif self.protein_paths:
            # Direct protein paths initialization
            self.structures_dir = os.path.join(self.output_dir, "proteins", "structures")
            self.pockets_dir = os.path.join(self.output_dir, "proteins", "pockets")

@dataclass
class PocketVinaConfig:
    """Configuration for PocketVina"""
    database_path: Optional[str] = None
    protein_paths: Optional[str] = None
    protein_list: Optional[str] = None  # For database workflow
    smiles_file: Optional[str] = None  # Can be .csv or .txt
    output_dir: Optional[str] = None   # Will default to "output"
    batch_size: Optional[int] = None   # Will default to 10
    p2rank_config: Optional[P2RankConfig] = None
    pocket_config: PocketConfig = field(default_factory=PocketConfig) 

    def __post_init__(self):
        if not self.smiles_file:
            raise ValueError("SMILES file is required")
        if not (self.database_path or self.protein_paths):
            raise ValueError("Either database_path or protein_paths must be provided")
        if self.database_path and not self.protein_list:
            raise ValueError("protein_list is required when using database_path")
        if self.output_dir is None:
            self.output_dir = "output"
        if self.batch_size is None:
            self.batch_size = 10
        self.output_dir = os.path.abspath(self.output_dir)


class P2RankProcessor:
    def __init__(self, config: P2RankConfig = None):
        """Initialize P2Rank processor with configuration"""
        self.config = config or P2RankConfig()
        self.setup_directories()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger('P2RankProcessor')
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(f'{self.config.output_dir}/p2rank_processor.log')
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)
    
    def setup_directories(self):
        """Setup required directories"""

        os.makedirs(self.config.output_dir, exist_ok=True)

        self.pockets_dir = os.path.join(self.config.output_dir, "proteins", "pockets")
        self.structures_dir = os.path.join(self.config.output_dir, "proteins", "structures")
        os.makedirs(self.pockets_dir, exist_ok=True)
        os.makedirs(self.structures_dir, exist_ok=True)
    
    def run_p2rank(self, pdb_file: str) -> bool:
        """Run P2Rank prediction on a single PDB file"""
        self.logger.info(f"Running P2Rank on {pdb_file}")
        
        try:
            # Run P2Rank
            result = subprocess.run([
                'prank', 'predict',
                '-f', pdb_file,
                '-o', self.config.output_dir,
                '-visualizations', str(self.config.visualizations).lower(),
                '-threads', str(self.config.threads),
                '-vis_copy_proteins', str(self.config.vis_copy_proteins).lower(),
                '-model', self.config.model,  # Ensure this model exists in your setup
            ], check=True, capture_output=True, text=True)
            
            # Handle prediction file
            input_filename = os.path.basename(pdb_file)
            base_name = os.path.splitext(input_filename)[0]
            predictions_file = os.path.join(self.config.output_dir, f"{input_filename}_predictions.csv")
            target_file = os.path.join(self.pockets_dir, f"{base_name}.csv")
            
            if os.path.exists(predictions_file):
                shutil.move(predictions_file, target_file)
                self._cleanup_temp_files(self.config.output_dir, input_filename)
                self.logger.info(f"Successfully processed {pdb_file}")
                return True
                
            self.logger.error(f"No predictions file found at {predictions_file}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error in P2Rank execution: {str(e)}")
            return False
    
    def process_pdb_file(self, input_file: str) -> Optional[str]:
        """Process PDB file and convert to PDBQT"""
        self.logger.info(f"Processing PDB file: {input_file}")
        
        try:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            cleaned_pdb = os.path.join(self.structures_dir, f"{base_name}_cleaned.pdb")
            temp_pdbqt = os.path.join(self.structures_dir, f"{base_name}_temp.pdbqt")
            final_pdbqt = os.path.join(self.structures_dir, f"{base_name}.pdbqt")
            
            # Clean PDB file - keep only ATOM and HETATM lines, remove HOH
            with open(input_file) as infile, open(cleaned_pdb, 'w') as outfile:
                for line in infile:
                    if (line.startswith(('ATOM', 'HETATM')) and 'HOH' not in line):
                        outfile.write(line)
            
            # Convert to PDBQT
            subprocess.run(
                ['obabel', cleaned_pdb, '-O', temp_pdbqt],
                check=True, capture_output=True, text=True
            )
            
            # Filter PDBQT to keep only ATOM lines
            with open(temp_pdbqt, 'r') as infile, open(final_pdbqt, 'w') as outfile:
                for line in infile:
                    if line.startswith('ATOM'):
                        outfile.write(line)
            
            # Cleanup temporary files
            os.remove(cleaned_pdb)
            os.remove(temp_pdbqt)
            
            if os.path.exists(final_pdbqt):
                self.logger.info(f"Successfully converted {input_file} to PDBQT")
                return final_pdbqt
                
            self.logger.error(f"PDBQT file was not created at {final_pdbqt}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing PDB file: {str(e)}")
            return None
    
    def process_file_list(self, file_list_path: str) -> Dict[str, bool]:
        """Process multiple PDB files from a list"""
        self.logger.info(f"Processing file list from: {file_list_path}")
        results = {}

        file_list_parent = os.path.dirname(os.path.abspath(file_list_path))
        try:
            # Read file paths
            with open(file_list_path, 'r') as f:
                pdb_files = [line.strip() for line in f if line.strip()]
            
            # Process each file
            for pdb_file in pdb_files:
                if os.path.abspath(pdb_file) != pdb_file:
                    pdb_file = os.path.abspath(os.path.relpath(os.path.join(file_list_parent, pdb_file), '.'))
                base_name = os.path.splitext(os.path.basename(pdb_file))[0]
                pdbqt_file = os.path.join(self.structures_dir, f"{base_name}.pdbqt")
                pockets_csv = os.path.join(self.pockets_dir, f"{base_name}.csv")
                
                # Check if both files already exist
                if os.path.exists(pdbqt_file) and os.path.exists(pockets_csv):
                    self.logger.info(f"Skipping {base_name}: files already exist")
                    results[pdb_file] = True
                    continue
                
                self.logger.info(f"Processing: {pdb_file}")
                
                if not self.run_p2rank(pdb_file):
                    results[pdb_file] = False
                    continue
                    
                if not self.process_pdb_file(pdb_file):
                    results[pdb_file] = False
                    continue
                    
                results[pdb_file] = True
                
        except Exception as e:
            self.logger.error(f"Error processing file list: {str(e)}")
            
        self._print_summary(results)
        return results
    
    def _cleanup_temp_files(self, output_dir: str, input_filename: str):
        """Clean up temporary files after processing"""
        temp_files = [
            f"{input_filename}_residues.csv",
            "params.txt",
            "run.log"
        ]
        for temp_file in temp_files:
            temp_path = os.path.join(output_dir, temp_file)
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def _print_summary(self, results: Dict[str, bool]):
        """Print processing summary"""
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        self.logger.info("\nProcessing Summary:")
        self.logger.info("-" * 50)
        self.logger.info(f"Successfully processed: {successful}/{total} files")
        
        if successful != total:
            self.logger.info("\nFailed files:")
            for file_path, success in results.items():
                if not success:
                    self.logger.info(f"- {file_path}")


class PocketVinaGPU:
    def __init__(self, config: PocketVinaConfig, htvs_mode = False):
        """Initialize PocketVina with configuration"""
        self.htvs_mode = htvs_mode
        self.config = config
        self.paths = Paths(
            database_path=config.database_path,
            protein_paths=config.protein_paths,
            output_dir=config.output_dir
        )
        self.pocket_config = config.pocket_config
        self.batch_size = config.batch_size
        self.output_dir = config.output_dir
        self.config_dir = os.path.join(self.output_dir, 'configs')
        self.ligand_output_dir = self.output_dir
        
        # Initialize P2Rank if using direct protein paths
        if config.protein_paths:
            p2rank_config = config.p2rank_config or P2RankConfig()
            p2rank_config.output_dir = config.output_dir
            self.p2rank_processor = P2RankProcessor(p2rank_config)
        
        package_dir = os.path.dirname(os.path.abspath(__file__))
        self.opencl_binary_path = os.path.join(package_dir, "QuickVina2-GPU-2-1")
        self._setup_logging()

    def process_proteins(self) -> Dict[str, Dict]:
        """Process proteins either from database or direct paths"""
        if self.config.protein_paths:
            return self._process_protein_paths()
        elif self.config.database_path:
            return self._process_database_proteins()
        else:
            self.logger.error("Neither protein paths nor database path provided")
            return {}
        
    def _process_protein_paths(self) -> Dict[str, Dict]:
        """Process proteins from direct paths using P2Rank"""
        results = {}
        
        # Read protein paths from file
        with open(self.config.protein_paths, 'r') as f:
            protein_paths = [line.strip() for line in f if line.strip()]
        
        for protein_path in protein_paths:
            base_name = os.path.splitext(os.path.basename(protein_path))[0]
            pdbqt_path = os.path.join(self.paths.structures_dir, f"{base_name}.pdbqt")
            pockets_csv = os.path.join(self.paths.pockets_dir, f"{base_name}.csv")
            
            # Check if both files already exist
            if os.path.exists(pdbqt_path) and os.path.exists(pockets_csv):
                self.logger.info(f"Found existing files for {base_name}, skipping P2Rank processing")
                try:
                    df = pd.read_csv(pockets_csv)
                    df.columns = df.columns.str.strip().str.lower()
                    results[protein_path] = {
                        'pdbqt_path': os.path.abspath(pdbqt_path),
                        'pockets_csv': os.path.abspath(pockets_csv),
                        'pockets': df.to_dict('records')
                    }
                    continue
                except Exception as e:
                    self.logger.error(f"Error reading existing files for {base_name}: {e}")
        
        # Process remaining proteins with P2Rank
        remaining_proteins = [p for p in protein_paths if p not in results]
        if remaining_proteins:
            p2rank_results = self.p2rank_processor.process_file_list(remaining_proteins)
            
            for protein_path, success in p2rank_results.items():
                if not success:
                    continue
                
                base_name = os.path.splitext(os.path.basename(protein_path))[0]
                protein_data = {
                    'pdbqt_path': os.path.abspath(os.path.join(self.paths.structures_dir, f"{base_name}.pdbqt")),
                    'pockets_csv': os.path.abspath(os.path.join(self.paths.pockets_dir, f"{base_name}.csv"))
                }
                
                try:
                    df = pd.read_csv(protein_data['pockets_csv'])
                    df.columns = df.columns.str.strip().str.lower()
                    protein_data['pockets'] = df.to_dict('records')
                    results[protein_path] = protein_data
                except Exception as e:
                    self.logger.error(f"Error reading pockets for {protein_path}: {e}")
        
        return results
    
    def _process_database_proteins(self) -> Dict[str, Dict]:
        """Process proteins from database"""
        results = {}
        
        try:
            with open(self.config.protein_list, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                protein_ids = [(row[0], row[1]) for row in reader]

            for protein_db, protein_id in protein_ids:
                protein_file = self._get_database_protein_file(protein_db, protein_id)
                if not protein_file:
                    continue

                pockets = self._read_database_pockets(protein_db, protein_id)
                if not pockets:
                    continue

                results[f"{protein_db}_{protein_id}"] = {
                    'pdbqt_path': protein_file,
                    'pockets': pockets
                }

        except Exception as e:
            self.logger.error(f"Error processing database proteins: {e}")

        return results

    def _setup_logging(self) -> None:
        """Setup logging configuration with performance considerations"""
        logging.basicConfig(
            level=logging.INFO,  # Only log INFO and above
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                # Buffer writes to file for better performance
                logging.FileHandler(f'{self.ligand_output_dir}/pocketvina_summary.log', 'a', encoding='utf-8', delay=True),
                # Optional: comment out StreamHandler if you don't need console output
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def truncate_after_torsdof(self, file_path: str, output_path: str) -> None:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            truncated_lines = []
            found_torsdof = False

            for line in lines:
                truncated_lines.append(line)
                if line.startswith("TORSDOF"):
                    found_torsdof = True
                    break

            if found_torsdof:
                with open(output_path, 'w') as file:
                    file.writelines(truncated_lines)
            else:
                print(f"No TORSDOF line found in {file_path}")
        except Exception as e:
            self.logger.error(f"Error truncating file {file_path}: {e}")

    def _create_molecule(self, smiles: str) -> Chem.Mol:
        """Create RDKit molecule from SMILES"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
            raise ValueError(f"3D embedding failed for: {smiles}")
        return mol

    def _convert_and_save_molecule(self, mol: Chem.Mol, molecule_name: str, output_dir: str) -> None:
        """Convert molecule to PDBQT format and save"""
        mol2_file = os.path.join(output_dir, f"{molecule_name}.mol2")
        pdbqt_file = os.path.join(output_dir, f"{molecule_name}.pdbqt")
        cleaned_pdbqt = os.path.join(output_dir, f"{molecule_name}_cleaned.pdbqt")

        try:
            ob_mol = pybel.readstring("mol", Chem.MolToMolBlock(mol))
            ob_mol.write("mol2", mol2_file, overwrite=True)

            subprocess.run(['obabel', mol2_file, '-O', pdbqt_file], check=True, text=True)
            os.remove(mol2_file)
            self.truncate_after_torsdof(pdbqt_file, cleaned_pdbqt)
            os.remove(pdbqt_file)
        except Exception as e:
            raise RuntimeError(f"Error converting molecule: {e}")

    def process_smiles(self, row: Tuple[str, str], output_dir: str) -> None:
        """Process single SMILES entry"""
        molecule_name, smiles = row
        try:
            mol = self._create_molecule(smiles)
            self._convert_and_save_molecule(mol, molecule_name, output_dir)
        except Exception as e:
            self.logger.error(f"Error processing {molecule_name}: {e}")

    def smiles_to_pdbqt(self) -> None:
        """Convert SMILES to PDBQT format"""
        try:
            os.makedirs(self.ligand_output_dir, exist_ok=True)
            with open(self.config.smiles_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                rows = list(reader)

            with ThreadPoolExecutor() as executor:
                list(executor.map(
                    lambda row: self.process_smiles(row, self.ligand_output_dir), 
                    rows
                ))
        except Exception as e:
            self.logger.error(f"Error in SMILES conversion: {e}")

    def split_files_into_batches(self, source_folder: str) -> List[Tuple[str, int]]:
        """Split files into batches for processing"""
        try:
            destination_folder = source_folder
            files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f)) and f.endswith('.pdbqt')]

            # Remove batch_size parameter since it's now an instance variable
            existing_batch_folders = [
                (os.path.join(destination_folder, folder), int(folder.split('_')[1]))
                for folder in os.listdir(destination_folder)
                if folder.startswith("batch_") and os.path.isdir(os.path.join(destination_folder, folder))
            ]

            if not files and existing_batch_folders:
                print("No `.pdbqt` files left to process. All files have already been moved to batch folders.")
                return sorted(existing_batch_folders, key=lambda x: x[1])

            if not files:
                print("No `.pdbqt` files found and no batch folders exist.")
                return []

            print(f"Found {len(files)} '.pdbqt' files in the source folder.")

            batch_folders = existing_batch_folders.copy()

            for i in range(0, len(files), self.batch_size):
                batch_number = len(batch_folders) + 1
                batch_folder = os.path.join(destination_folder, f"batch_{batch_number}")

                if os.path.exists(batch_folder):
                    print(f"Batch folder '{batch_folder}' already exists.")
                else:
                    print(f"Creating new batch folder: '{batch_folder}'")
                    os.makedirs(batch_folder, exist_ok=True)

                batch_folders.append((batch_folder, batch_number))

                batch_files = files[i:i + self.batch_size]
                print(f"Moving {len(batch_files)} files to '{batch_folder}'.")
                for file in batch_files:
                    shutil.move(os.path.join(source_folder, file), os.path.join(batch_folder, file))

            print("Files have been split into batches and moved.")
            return sorted(batch_folders, key=lambda x: x[1])
        except Exception as e:
            self.logger.error(f"Error splitting files into batches: {e}")
            return []

    def get_database_protein_file(self) -> Optional[str]:
        """Get protein file path based on database type"""
        try:
            if self.protein_db == "PDB":
                return os.path.join(self.paths.pdb_structures, f"pdb{self.protein_id}_filtered.pdbqt")
            elif self.protein_db == "UniProt":
                return os.path.join(self.paths.uniprot_structures, f"AF-{self.protein_id}-F1-model_v1_filtered.pdbqt")
            else:
                raise ValueError(f"Unsupported protein database: {self.protein_db}")
        except Exception as e:
            self.logger.error(f"Error getting protein file: {e}")
            return None

    def read_database_pockets(self) -> Optional[List[Dict]]:
        """Read pocket information from CSV file"""
        try:
            if self.protein_db == "PDB":
                csv_file = os.path.join(self.paths.pdb_pockets, f"pdb{self.protein_id}.ent.gz_predictions.csv")
            elif self.protein_db == "UniProt":
                csv_file = os.path.join(self.paths.uniprot_pockets, f"AF-{self.protein_id}-F1-model_v1.pdb.gz_predictions.csv")
            else:
                raise ValueError(f"Unsupported protein database: {self.protein_db}")

            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            df = df.rename(columns=lambda x: x.strip().lower())

            required_columns = {'rank', 'center_x', 'center_y', 'center_z'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Missing required columns in CSV file: {required_columns - set(df.columns)}")

            return df.to_dict('records')

        except Exception as e:
            self.logger.error(f"Error reading pockets: {e}")
            return None

    def generate_config(self, protein_file: str, ligand_directory: str, 
                       pocket: Dict, batch: int) -> str:
        """Generate configuration file for docking"""
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            filename_without_extension = os.path.splitext(os.path.basename(protein_file))[0]
            config_file = os.path.join(
                self.config_dir, 
                f"{filename_without_extension}_pocketrank_{pocket['rank']}_config_batch_{batch}.txt"
            )

            with open(config_file, 'w') as file:
                file.write(f"receptor = {os.path.abspath(protein_file)}\n")
                file.write(f"ligand_directory = {os.path.abspath(ligand_directory)}\n")
                file.write(f"opencl_binary_path = {self.opencl_binary_path}\n")
                file.write(f"center_x = {pocket['center_x']}\n")
                file.write(f"center_y = {pocket['center_y']}\n")
                file.write(f"center_z = {pocket['center_z']}\n")
                file.write(f"size_x = {self.pocket_config.size_x}\n")
                file.write(f"size_y = {self.pocket_config.size_y}\n")
                file.write(f"size_z = {self.pocket_config.size_z}\n")
                file.write(f"thread = {self.pocket_config.thread}\n")
                file.write(f"pocket_rank = {pocket['rank']}\n")
            return config_file
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None

    def execute_docking(self) -> None:
        """Main docking execution with better error handling"""
        try:
            
            # Prepare directories and convert SMILES if needed
            os.makedirs(self.ligand_output_dir, exist_ok=True)
            batch_folders = [f for f in os.listdir(self.ligand_output_dir) 
                            if f.startswith('batch_') and os.path.isdir(os.path.join(self.ligand_output_dir, f))]

            if batch_folders:
                self.logger.info(f"Found {len(batch_folders)} existing batch folders.")
            else:
                self.logger.info("No batch folders found. Proceeding with conversion and batch splitting.")
                self.smiles_to_pdbqt()

            # Use self.batch_size here
            lig_batch_folders = self.split_files_into_batches(self.ligand_output_dir)
            lig_batch_folders = [(folder, number) for folder, number in lig_batch_folders 
                                if not folder.endswith('_out')]

            start_time = time.time()
            self.logger.info("Starting docking execution...")
            if self.config.database_path:
                # Database workflow
                try:
                    with open(self.config.protein_list, 'r') as file:
                        reader = csv.reader(file)
                        next(reader)  # Skip header
                        protein_ids = [(row[0], row[1]) for row in reader]

                    for protein_db, protein_id in protein_ids:
                        try: 
                            self.protein_db = protein_db
                            self.protein_id = protein_id
                            
                            protein_file = self.get_database_protein_file()
                            if not protein_file:
                                self.logger.warning(f"Skipping {self.protein_db}_{self.protein_id}: protein file not found")
                                continue

                            pockets = self.read_database_pockets()
                            if not pockets:
                                self.logger.warning(f"Skipping {self.protein_db}_{self.protein_id}: no pockets found")
                                continue

                            self._process_protein_docking(protein_file, pockets, lig_batch_folders)
                        except Exception as e:
                            self.logger.error(f"Error processing {protein_db}_{protein_id}: {e}")
                            continue
                except Exception as e:
                    self.logger.error(f"Error processing database proteins: {e}")
                    raise
            else:
                # Direct protein paths workflow
                protein_results = self.p2rank_processor.process_file_list(self.config.protein_paths)
                for protein_path, success in protein_results.items():
                    if not success:
                        continue
                        
                    base_name = os.path.splitext(os.path.basename(protein_path))[0]
                    pdbqt_path = os.path.join(self.paths.structures_dir, f"{base_name}.pdbqt")
                    pockets_csv = os.path.join(self.paths.pockets_dir, f"{base_name}.csv")
                    
                    try:
                        df = pd.read_csv(pockets_csv)
                        df.columns = df.columns.str.strip().str.lower()
                        pockets = df.to_dict('records')
                        self._process_protein_docking(pdbqt_path, pockets, lig_batch_folders)
                    except Exception as e:
                        self.logger.error(f"Error processing {protein_path}: {e}")
                        continue
                

            self.logger.info(f"Docking execution completed in {time.time() - start_time:.2f} seconds")

            # Process results
            self._process_results()

        except Exception as e:
            self.logger.error(f"Docking execution failed: {e}")
            raise

    def _process_protein_docking(self, protein_file: str, pockets: List[Dict], 
                            batch_folders: List[Tuple[str, int]]) -> None:
        """Process docking for a single protein"""
        package_dir = os.path.dirname(os.path.abspath(__file__))
        boost_lib_path = os.path.join(package_dir, "boost_1_77_0/stage/lib")
        opencl_path = os.path.join(package_dir, "OpenCL")
        
        # Set environment variables
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{boost_lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
        env["OPENCL_ROOT"] = opencl_path

        pockets_to_process = pockets[:5] if getattr(self, "htvs_mode", False) else pockets
        
        for pocket in pockets_to_process:
            for folder, batch_num in batch_folders:
                try:
                    config_file = self.generate_config(
                        protein_file=protein_file,
                        ligand_directory=folder,
                        pocket=pocket,
                        batch=batch_num
                    )
                    binary_path = os.path.join(package_dir, 'PocketVina-GPU')
                    
                    # Run the command from the package directory with OpenCL path
                    subprocess.run([binary_path, '--config', config_file],
                                check=True, env=env, cwd=package_dir)
                except Exception as e:
                    self.logger.error(f"Error in docking: {e}")

    def _process_results(self) -> None:
        """Process docking results"""
        try:
            output_json = os.path.join(self.ligand_output_dir, "combined_results.json")
            output_csv = os.path.join(self.ligand_output_dir, "combined_results.csv")
            
            # Initialize empty lists for combined data
            combined_data = []
            
            # Process each batch output folder
            for folder in os.listdir(self.ligand_output_dir):
                folder_path = os.path.join(self.ligand_output_dir, folder)
                
                if folder.endswith('_out') and os.path.isdir(folder_path):
                    self.logger.info(f"Processing folder: {folder}")
                    
                    for file in os.listdir(folder_path):
                        if file.endswith('.json'):
                            file_path = os.path.join(folder_path, file)
                            
                            with open(file_path, 'r') as f:
                                try:
                                    data = json.load(f)
                                    combined_data.append(data)
                                except json.JSONDecodeError:
                                    self.logger.warning(f"Skipping invalid JSON file: {file_path}")
            
            # Save combined JSON
            with open(output_json, 'w') as f:
                json.dump(combined_data, f, indent=4)
            self.logger.info(f"Combined JSON file created at: {output_json}")
            
            # Convert to CSV
            self.json_to_csv(output_json, output_csv)
        except Exception as e:
            self.logger.error(f"Error processing results: {e}")

    def combine_json_files(self, base_dir: str, output_file: str) -> None:
        """
        Combines all JSON files from folders ending with '_out' into a single JSON file.

        Args:
            base_dir (str): The base directory containing the folders.
            output_file (str): The path to the output JSON file.
        """
        combined_data = []

        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)

            if folder.endswith('_out') and os.path.isdir(folder_path):
                self.logger.info(f"Processing folder: {folder}")

                for file in os.listdir(folder_path):
                    if file.endswith('.json'):
                        file_path = os.path.join(folder_path, file)

                        with open(file_path, 'r') as f:
                            try:
                                data = json.load(f)
                                combined_data.append(data)
                            except json.JSONDecodeError:
                                self.logger.warning(f"Skipping invalid JSON file: {file_path}")

        with open(output_file, 'w') as f:
            json.dump(combined_data, f, indent=4)

        self.logger.info(f"Combined JSON file created at: {output_file}")

    def json_to_csv(self, json_file: str, csv_file: str) -> None:
        """
        Converts a nested JSON file into a flattened CSV format.

        Args:
            json_file (str): Path to the input JSON file.
            csv_file (str): Path to the output CSV file.
        """
        # Read the JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Prepare the CSV rows
        csv_rows = []
        for entry in tqdm(data):
            for complex_name, results in entry.items():
                for result in results:
                    # Add the complex name to each row
                    row = {
                        "complex": complex_name,
                        **result  # Merge the rest of the result fields
                    }
                    csv_rows.append(row)

        # Extract headers from the keys of the first row
        if not csv_rows:
            self.logger.warning("No data to convert to CSV")
            return
            
        headers = csv_rows[0].keys()

        # Write to CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()  # Write the header row
            writer.writerows(csv_rows)  # Write all rows

        self.logger.info(f"CSV file created at: {csv_file}")
        self.split_complex_column(csv_file, csv_file)

    def split_complex_column(self, input_csv: str, output_csv: str) -> None:
        """Split the complex column into protein and molecule_name columns"""
        try:
            df = pd.read_csv(input_csv)
            
            # Handle cases where complex might be numeric or NaN
            df['complex'] = df['complex'].fillna('').astype(str)
            
            def split_complex(x):
                try:
                    # Remove '_cleaned_out' and '-results' if present
                    x = x.replace('_cleaned_out', '').replace('-results', '')
                    
                    # Find the last occurrence of 'pocket' in the string
                    import re
                    pocket_match = re.search(r'(.*pocket\d+)-(.*)', x)
                    if pocket_match:
                        protein = pocket_match.group(1)  # Everything up to and including pocketX
                        molecule = pocket_match.group(2)  # Everything after the hyphen
                        
                        # Remove '_filtered-pocket*' part from protein
                        protein = re.sub(r'_filtered-pocket\d+', '', protein)
                        
                        return [protein, molecule]
                    return ['', x]
                except:
                    return ['', x]
            
            # Apply the split function
            split_results = df['complex'].apply(split_complex)
            df['protein'] = split_results.str[0]
            df['molecule_name'] = split_results.str[1]
            
            # Reorder columns
            columns = ['protein', 'molecule_name']
            for col in df.columns:
                if col not in ['protein', 'molecule_name', 'complex']:
                    columns.append(col)
            
            df = df[columns]
            
            # Save the modified DataFrame to a new CSV file
            df.to_csv(output_csv, index=False)
            
            self.logger.info(f"Modified CSV file saved at: {output_csv}")
        except Exception as e:
            self.logger.error(f"Error splitting complex column: {e}")

