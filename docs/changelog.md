# Changelog

## 0.2.0

### Added

- `output_format="atlas"` writes one resumable results folder per run, with Parquet tables and one pose file per
  protein. Several jobs can write into the same folder ([Results format](output-format.md)).
- `execute_docking()` shows the best scores at the end. `best_scores()`, `show()` and `select()` work on the run,
  and `Results` gives the same methods for an earlier run ([Getting started](quickstart.md)).
- `show()` draws the results like the protein page of the PocketVina Atlas ([3D views](visualisation.md)).
- PoseBusters checks of selected poses, of a whole run, or during docking ([PoseBusters checks](posebusters.md)).
- Several engines per GPU, and several GPUs in one run ([GPUs and GPU sharing](gpu-sharing.md)).
- The `pocketvina` command with `dock`, `bust`, `show`, `export`, `info`, `query`, `import-legacy`,
  `setup p2rank` and `bench gpu` ([Command line reference](cli.md)).
- Seeded runs: the same seed gives the same poses again.
- `top_pockets` and `pockets` choose the pockets. `pockets_dir` docks into pockets that P2Rank found before.
- mmCIF and PDBQT structures, also gzipped.
- A Colab notebook, these pages and a conda recipe.

### Changed

- `pip install .` installs everything and compiles nothing. P2Rank is downloaded at the first run, and no
  environment variables need to be set ([Installation](installation.md)).
- The pip package is now `pocketvina`. Uninstall `pocketvina-gpu` first ([Migrating from PocketVina 0.1](migrating.md)).
- Ligands are docked from their largest fragment. `ligand_prep="legacy"` restores the preparation of 0.1.
- The engine compiles its GPU code on first use into `~/.cache/pocketvina` and never writes into the installation.
- Importing `pocketvina` no longer changes environment variables.
