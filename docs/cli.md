# Command line reference

You can run every step from the command line with `pocketvina`. `python -m pocketvina` runs the same commands.
Every command has `--help`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | everything docked |
| 1 | usage or configuration error |
| 2 | some proteins or engine calls failed. Run the same command again to retry them |
| 3 | nothing docked |

## Settings in a file

You can keep the settings of `pocketvina dock` in a TOML file. The keys are the option names in snake_case:

```toml
# run.toml
proteins = "example/proteins"
ligands = "example/ligands.txt"
out = "results"
top_pockets = 2
box = [20, 20, 20]
seed = 42
```

```bash
pocketvina dock --config run.toml
```

Options given on the command line override the file. Relative paths are relative to the folder of the file.

## The PocketVina Atlas

`pocketvina atlas` reads the [PocketVina Atlas](atlas.md), docking results of drugs against the human proteome
computed in advance, as `pocketvina.Atlas` does in Python. It reads the public release, or the folder or URL given
with `--atlas`:

```bash
pocketvina atlas search imatinib
pocketvina atlas show ABL1 --ligand imatinib --out ABL1_imatinib.html
pocketvina atlas export ABL1 imatinib --pocket all --fmt sdf --protein --out poses
pocketvina atlas download ABL1 EGFR --out my_atlas
pocketvina atlas search ABL1 --atlas my_atlas
```

## Commands

```{eval-rst}
.. click:: pocketvina.cli:cli
   :prog: pocketvina
   :nested: full
```
