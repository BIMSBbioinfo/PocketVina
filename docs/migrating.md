# Migrating from PocketVina 0.1

## Upgrade an installation

The pip package is now called `pocketvina`. Version 0.1 was `pocketvina-gpu`. Remove the old one before you install
the new one:

```bash
pip uninstall pocketvina-gpu
pip install pocketvina
```

If you installed 0.2 over 0.1 already, run `pip uninstall pocketvina-gpu` and then
`pip install --force-reinstall --no-deps pocketvina`.

Installing PocketVina no longer downloads Boost or compiles the engine. P2Rank is downloaded at the first run, and the
P2Rank of an older installation in `pocketvina/p2rank_2.5/` is still found. You no longer need to set
`OCL_ICD_VENDORS` or other environment variables.

## Update your scripts

- `PocketVinaGPU`, `PocketVinaConfig`, `P2RankConfig` and `PocketConfig` keep their names and defaults.
- The Python default is still `output_format="files"`, the 0.1 output folder. Add `output_format="atlas"` to get the
  new results folder and the methods `best_scores()`, `show()` and `select()` ([Getting started](quickstart.md)).
- Ligands are now prepared from their largest fragment with a fixed seed, `seed=42`, so a run gives the same poses
  again.
- To regenerate published 0.1 results, set `ligand_prep="legacy"`. On the command line it is
  `--ligand-prep legacy`.

## Import 0.1 results

`pocketvina import-legacy` converts a 0.1 output folder into a [results folder](output-format.md). You can then
query, view, check and export old runs like new ones:

```bash
pocketvina import-legacy output --results results --smiles ligands.csv
```

`output` is the `output_dir` of the old run, and `--smiles` its ligand CSV. The old folder is not changed. You can
try it with the test data of the repository:

```bash
pocketvina import-legacy tests/data/legacy_run --results legacy_results --smiles tests/data/smiles.csv
pocketvina query best --results legacy_results
```

The import reads these files:

```text
output/
  batch_<n>/<name>_cleaned.pdbqt                               ligand inputs of the engine
  batch_<n>_out/<receptor>-pocket<k>-<name>_cleaned_out.pdbqt   engine outputs
  proteins/structures/<receptor>.pdbqt                         receptors
  proteins/pockets/<receptor>.csv                              P2Rank predictions
  configs/<receptor>_pocketrank_<k>_config_batch_<n>.txt       engine configs with the box sizes, if kept
```

- The engine outputs are kept unchanged. `pocketvina export --results results --what all --format csv --out tables`
  writes the same `combined_results.csv` again.
- Without `--smiles`, the highest-affinity poses are not converted to SDF. The viewers then show the PDBQT.
- Running the import again resumes it.

The same in Python:

```python
from pocketvina.io.importer import import_legacy

import_legacy("output", "results", smiles_file="ligands.csv")
```
