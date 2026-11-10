# Getting started

This page shows you how to dock the example ligands of the repository into its two example proteins. The run should
finish within one to two minutes on one GPU. After the [installation](installation.md), download the repository for its
example files and run the steps in its folder:

```bash
git clone https://github.com/BIMSBbioinfo/PocketVina
cd PocketVina
```

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

Activate the environment:

```bash
conda activate pocketvina
```

List the example proteins in `proteins.txt`, one structure file per line:

```bash
ls example/proteins/*.pdb > proteins.txt
```

### Dock

Set up the run: the proteins, the ligands, the results folder and the settings of P2Rank and of the docking box:

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
```

Create the run. With `htvs_mode=True`, it docks the 5 best pockets of each protein:

```python
pocketvina = PocketVinaGPU(config, htvs_mode=True)
```

Start it. `execute_docking()` finds the pockets, docks every ligand and then shows the lowest Vina score of each
protein and ligand:

```python
pocketvina.execute_docking()
```

### Look at the results

`best_scores()` shows the lowest Vina score of each protein and ligand:

```python
pocketvina.best_scores()
```

With `protein` and `top`, it shows the 5 ligands with the lowest Vina score on one protein:

```python
pocketvina.best_scores(protein="1fbl", top=5)
```

`show()` draws all results in 3D. In Jupyter, Colab or VS Code the page appears below the cell. In a script it is
saved as an HTML file:

```python
pocketvina.show()
```

With `path`, it saves the page to that file, to open in a browser or to share:

```python
pocketvina.show(path="results.html")
```

### One protein and ligand

`select()` takes the poses of one protein and ligand. Without more, it takes the highest-affinity pose:

```python
pair = pocketvina.select(protein="1fbl", ligand="molecule1")
```

With `pocket` and `mode`, it takes mode 3 in pocket 2:

```python
pair = pocketvina.select(protein="1fbl", ligand="molecule1", pocket=2, mode=3)
```

With `pocket="all"`, it takes the highest-affinity pose in each pocket:

```python
pair = pocketvina.select(protein="1fbl", ligand="molecule1", pocket="all")
```

With `mode="all"`, it takes all modes of the pocket of the highest-affinity pose:

```python
pair = pocketvina.select(protein="1fbl", ligand="molecule1", mode="all")
```

The next steps work on this last selection. `show()` draws it in 3D:

```python
pair.show()
```

`bust()` runs the PoseBusters checks on its poses:

```python
pair.bust()
```

`export()` saves its poses as an SDF file, such as `PV_1fbl_molecule1_pk1_modes.sdf`:

```python
pair.export()
```

With `protein=True`, it also saves the protein as `PV_1fbl.pdb`:

```python
pair.export(protein=True)
```

See [3D views](visualisation.md) and [PoseBusters checks](posebusters.md) for more.

## GPU

A run uses one GPU, the first visible one. It docks as many pockets at the same time as the GPU memory holds, about
6.4 GB each: 3 on a 24 GB P40, 5 on a 40 GB A100. For several GPUs, run one job per GPU, for example a SLURM array
over parts of the protein list. See [GPUs and GPU sharing](gpu-sharing.md).

## Results of an earlier run

You can open the results of an earlier session, or of a run on a cluster, with `Results`. Give it the `output_dir`
of the run:

```python
from pocketvina import Results

res = Results("results")
```

`Results` has the same `best_scores()`, `show()` and `select()` as the run. The lowest Vina score of each protein and
ligand:

```python
res.best_scores()
```

All results in 3D:

```python
res.show()
```

The highest-affinity pose of one protein and ligand:

```python
pair = res.select(protein="1fbl", ligand="molecule1")
```

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

You can find the pockets of a whole proteome once, then dock one or more ligand libraries into them.

Set up P2Rank. Use `model="alphafold"` for AlphaFold models:

```python
from pocketvina import P2RankConfig, P2RankProcessor

p2rank = P2RankProcessor(P2RankConfig(output_dir="pockets", model="default", threads=8))
```

Find the pockets of every protein in `proteins.txt`. They are saved as `pockets/proteins/pockets/<protein>.csv`:

```python
p2rank.process_file_list("proteins.txt")
```

Set up the docking into these pockets. With `pockets_dir`, P2Rank is not run again:

```python
from pocketvina import PocketVinaConfig, PocketVinaGPU

config = PocketVinaConfig(
    protein_paths="proteins.txt",
    smiles_file="example/ligands.txt",
    output_dir="results_library1",
    output_format="atlas",
    pockets_dir="pockets/proteins/pockets",
)
```

Start the docking:

```python
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

You can run the same steps from the command line. A results folder holds one run with one set of settings, so if
`results/` is left from the Python run, use another `--out`.

`dock` docks the ligands, here into the 2 best-ranked pockets of each protein:

```bash
pocketvina dock --proteins example/proteins --ligands example/ligands.txt --out results --top-pockets 2
```

`info` shows the settings and counts of a run:

```bash
pocketvina info --results results
```

`export` writes the highest-affinity poses as SDF files:

```bash
pocketvina export --results results --what best --format sdf --out best_poses
```

`--help` lists all commands:

```bash
pocketvina --help
```

Two more commands you will often use. `query best` lists the 10 highest-affinity protein x ligand pairs:

```bash
pocketvina query best --results results --top 10
```

`show` saves the 3D view of the highest-affinity pose of a protein and ligand as a web page:

```bash
pocketvina show --results results --protein 1fbl --ligand molecule1 --out pose.html
```

`pocketvina dock --help` lists all options. The [Command line reference](cli.md) describes every command, and how to keep the settings
in a file.
