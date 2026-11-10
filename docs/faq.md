# FAQ

## Installation

### The engine finds no GPU, or P2Rank does not start

See [Troubleshooting](installation.md#troubleshooting) on the installation page.

### Can I use PocketVina without a GPU?

Docking needs an NVIDIA GPU. Reading results works on any computer, for example on a laptop with a copy of the
results folder. This covers `Results`, `show()`, `bust()` and the commands `info`, `query`, `show` and `export`.

## Runs

### How do I resume a run that stopped?

Run the same command or code again. Proteins that are done are skipped, and all others are docked.

### The run stops with "... holds another run"

A results folder holds one run with one set of ligands and settings. Use a new `--out` folder, or `--force` to
replace the run.

### The engine runs out of GPU memory

PocketVina retries a failed call with one engine less on that GPU. If the error `CL_MEM_OBJECT_ALLOCATION_FAILURE`
or `CL_OUT_OF_RESOURCES` remains, lower `--thread` to no less than 1000, or use `--procs-per-gpu 1`.

### Some ligands were not docked

This query lists the ligands that could not be prepared, and the reason:

```bash
pocketvina query --results results "SELECT ligand_id, status, error FROM ligands WHERE status <> 'ok'"
```

The engine docks only molecules made of C, N, O, S, P, F, Cl, Br and I.

### Why do two runs give different poses?

With the same seed, ligands, settings and engine, a run gives the same poses again, on any number of GPUs. Changing
`--seed`, `--thread`, the box or `--ligands-per-call` changes the poses. Different RDKit versions also build
different 3D starting structures, so a pip and a conda environment can give different scores.

### Which pocket was a ligand docked into?

The column `best_pocket` of `best_scores()` is the P2Rank rank of the pocket of the highest-affinity pose. The
`pockets` table has its centre, box size and residues ([Results format](output-format.md)).

### Do I need to clear a cache after an update?

No. After a driver or PocketVina update, the engine compiles its GPU code into a new folder under
`~/.cache/pocketvina/kernels/`. You can delete the old folders. A folder set with `POCKETVINA_KERNEL_CACHE` is the
exception: empty it yourself.

## Viewing and analysis

### The 3D page shows "Run this cell again to see the PocketVina 3D view."

Jupyter removes the page when a saved notebook is opened again. Run the cell again.

### A pocketvina.viz view stays empty

These views load Mol\* from the internet. Without internet access, use `backend="py3dmol"` or the 3D page of
`show()` ([3D views](visualisation.md)).

### Every pose is invalid in PoseBusters

The column `failed_checks` names the checks that failed. Docked poses often fail on clashes with the protein or on
the energy of the ligand. `select(..., mode="all").bust()` checks every mode of a pair, so you can find the modes
that pass ([PoseBusters checks](posebusters.md)).

### Can I still get combined_results.csv?

Yes. This command writes it from a results folder:

```bash
pocketvina export --results results --what all --format csv --out tables
```
