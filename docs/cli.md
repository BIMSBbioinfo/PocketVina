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

## Commands

```{eval-rst}
.. click:: pocketvina.cli:cli
   :prog: pocketvina
   :nested: full
```
