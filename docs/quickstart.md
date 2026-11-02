# Getting started

This page shows you how to dock the example ligands of the repository into its two example proteins. Run the steps
in the `PocketVina` folder after the [installation](installation.md). The run should finish within one to two minutes on one
GPU.

## Input files

You need a protein list and a ligand file.

### Proteins

The protein list is a text file with one structure file per line. PDB, mmCIF and PDBQT files work, also gzipped.
A relative path is relative to the folder of the list. The file name without extension is the protein id, such as
`1fbl`.

```text
/path/to/protein1.pdb
/path/to/protein2.pdb
```

### Ligands

The ligand file is a CSV file with the header `molecule_name,smiles`. The molecule name is the ligand id, such as
`molecule1`, and must be unique.

```text
molecule_name,smiles
molecule1,CC1=CC=C(C=C1)C(=O)CCCN1CCC(O)(CC1)c1ccc(Cl)cc1
molecule2,CC(C)(C)NCC(O)c1ccc(O)c(CO)c1
```

## Quick Start

### Make the protein list

Activate the environment and list the example proteins:

```bash
conda activate pocketvina
ls example/proteins/*.pdb > proteins.txt    # one protein structure file per line
```

### Dock

Set up the run and start it:

```python
from pocketvina import PocketVinaGPU, PocketVinaConfig, P2RankConfig, PocketConfig

config = PocketVinaConfig(
    protein_paths="proteins.txt",        # the proteins
    smiles_file="example/ligands.txt",   # the ligands: a CSV file with molecule_name,smiles
    output_dir="results",
    output_format="atlas",
    batch_size=10,
    p2rank_config=P2RankConfig(
        threads=8,
        visualizations=False,
        vis_copy_proteins=False,
        output_dir="results",
        model="default",                 # "alphafold" for AlphaFold models
    ),
    pocket_config=PocketConfig(
        size_x=20,                       # docking box in Angstrom
        size_y=20,
        size_z=20,
        thread=8000,
    ),
)

pocketvina = PocketVinaGPU(config, htvs_mode=True)   # docks the 5 best pockets of each protein
pocketvina.execute_docking()
```

`execute_docking()` finds the pockets, docks every ligand and then shows the best score of each protein and ligand.

### Look at the results

Show the best scores, the 3D page, or the poses of one protein and ligand:

```python
pocketvina.best_scores()                        # the best Vina score of each protein x ligand
pocketvina.best_scores(protein="1fbl", top=5)   # the 5 best ligands of one protein
pocketvina.show()                               # all results in 3D
pocketvina.show(path="results.html")            # the same, as a web page

pair = pocketvina.select(protein="1fbl", ligand="molecule1")                    # the best pose
pair = pocketvina.select(protein="1fbl", ligand="molecule1", pocket=2, mode=3)  # mode 3 in pocket 2
pair = pocketvina.select(protein="1fbl", ligand="molecule1", pocket="all")      # the best pose in each pocket
pair = pocketvina.select(protein="1fbl", ligand="molecule1", mode="all")        # all modes of the best pocket
pair.show()             # 3D view
pair.bust()             # PoseBusters checks
pair.export()           # saves PV_1fbl_molecule1_pk1_modes.sdf
pair.export_protein()   # saves PV_1fbl.pdb
```

In Jupyter, Colab or VS Code, `show()` draws the 3D page below the cell. In a script it writes an HTML file. See
[3D views](visualisation.md) and [PoseBusters checks](posebusters.md) for more.

## GPU

A run uses one GPU, the first visible one. It docks as many pockets at the same time as the GPU memory holds, about
6.4 GB each: 3 on a 24 GB P40, 5 on a 40 GB A100. For several GPUs, run one job per GPU, for example a SLURM array
over parts of the protein list. See [GPUs and GPU sharing](gpu-sharing.md).

## Results of an earlier run

You can open the results of an earlier session, or of a run on a cluster, with `Results`:

```python
from pocketvina import Results

res = Results("results")      # the output_dir of the run
res.best_scores()
res.show()
pair = res.select(protein="1fbl", ligand="molecule1")
```

`Results` has the same `best_scores()`, `show()` and `select()` as the run.

## Choose the pockets

Set one of these in `PocketVinaConfig`, and create the run with `PocketVinaGPU(config)`:

| Setting | Docks |
| --- | --- |
| `top_pockets=7` | the 7 best-ranked pockets of each protein |
| `pockets="10-12"` | only the pockets ranked 10 to 12. `pockets=10` and `pockets=[1, 3, 10]` work too |
| neither | every pocket P2Rank finds |

`PocketVinaGPU(config, htvs_mode=True)` docks the 5 best pockets. Do not combine it with `top_pockets` or
`pockets`.

## P2Rank first, docking later

You can find the pockets of a whole proteome once, then dock one or more ligand libraries into them:

```python
from pocketvina import P2RankConfig, P2RankProcessor, PocketVinaConfig, PocketVinaGPU

# 1. the pockets of every protein in proteins.txt, saved as pockets/proteins/pockets/<protein>.csv
p2rank = P2RankProcessor(P2RankConfig(output_dir="pockets", model="default", threads=8))  # "alphafold" for AF models
p2rank.process_file_list("proteins.txt")

# 2. docking into these pockets: P2Rank is not run again
config = PocketVinaConfig(
    protein_paths="proteins.txt",
    smiles_file="example/ligands.txt",
    output_dir="results_library1",
    output_format="atlas",
    pockets_dir="pockets/proteins/pockets",
)
PocketVinaGPU(config, htvs_mode=True).execute_docking()
```

## Output folder

The run writes this folder:

```text
results/
├── MANIFEST.json              # the run's settings
├── manifest/<protein>.json    # written when a protein is done: running again skips it
├── poses/<protein>.poses      # every docked pose, compressed
├── structures/                # the input structures
└── tables/                    # Parquet tables: ligands, proteins, pockets, pocket_runs, modes, pairs
```

If you run the same code again, it skips the proteins that are done. So a stopped run resumes. You can read the
tables with pandas or DuckDB, see [Results format](output-format.md). Without `output_format="atlas"`, PocketVina writes the
output folder of version 0.1.

## Command line

You can run the same steps from the command line:

```bash
pocketvina dock --proteins example/proteins --ligands example/ligands.txt --out results --top-pockets 2
pocketvina info --results results
pocketvina export --results results --what best --format sdf --out best_poses
pocketvina --help
```

`dock` docks the ligands, `info` shows the settings and counts of a run, and `export` writes the best poses as SDF
files. A results folder holds one run with one set of settings. If `results/` is left from the Python run, use
another `--out`.

Two more commands you will often use:

```bash
pocketvina query best --results results --top 10      # the 10 best protein x ligand pairs
pocketvina show --results results --protein 1fbl --ligand molecule1 --out pose.html
```

`pocketvina dock --help` lists all options. The [Command line reference](cli.md) describes every command, and how to keep the settings
in a file.
