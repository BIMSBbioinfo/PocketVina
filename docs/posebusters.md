# PoseBusters checks

[PoseBusters](https://github.com/maabuu/posebusters) tests whether a docked pose is physically plausible. It checks
bond lengths and angles, flat rings, stereochemistry, clashes, the ligand's energy and its contacts with the protein.
PoseBusters is installed with PocketVina. Docking does not run it unless you ask for it.

## One pair

`bust()` checks the poses you picked with `select()`. Pick the highest-affinity pose of a protein and ligand:

```python
pair = pocketvina.select(protein="1fbl", ligand="molecule1")
```

Check it:

```python
pair.bust()
```

To check all modes of the pocket of the highest-affinity pose, pick them with `mode="all"`:

```python
pair = pocketvina.select(protein="1fbl", ligand="molecule1", mode="all")
```

Check them, with 4 processes at a time:

```python
checks = pair.bust(workers=4)
```

The modes that pass every check:

```python
checks[checks.pb_valid]
```

`bust()` returns a table with one row per selected pose. Nothing is written to disk. The main columns:

- `pb_valid`: true if every check passed
- `pb_status`: `valid`, `invalid`, or why the pose was not checked
- `failed_checks`: the names of the checks that failed
- then one column per check: passed, failed, or empty if it does not apply

## A whole run

`pocketvina bust` checks the poses of a whole results folder. Without `--select`, it checks the highest-affinity
pose of every ligand:

```bash
pocketvina bust --results results
```

With `--select top_k --top-k 3`, it checks the 3 highest-affinity modes of every ligand. `--overwrite` checks the
proteins again:

```bash
pocketvina bust --results results --select top_k --top-k 3 --overwrite
```

With `--select per_pocket --top-k 1 --protein 1fbl`, it checks the highest-affinity mode of every ligand in each
pocket, only in protein 1fbl:

```bash
pocketvina bust --results results --select per_pocket --top-k 1 --protein 1fbl --overwrite
```

The checks are saved in the results folder, in `tables/posebusters/`. Proteins that were checked already are
skipped, so you can run `bust` again after docking more proteins.

| Option | Meaning | Default |
| --- | --- | --- |
| `--select best` | the highest-affinity pose of each ligand | `best` |
| `--select top_k --top-k K` | the K highest-affinity modes of each ligand, over all pockets | |
| `--select per_pocket --top-k K` | the K highest-affinity modes of each ligand in each pocket | |
| `--config dock` | PoseBusters configuration: `dock`, `dock_fast`, ... | `dock` |
| `--workers N` | worker processes | the CPUs, at most 16 |
| `--timeouts 30,300` | seconds per pose for each try. A pose that takes longer is tried once more, then saved as `timeout` | `30,300` |
| `--overwrite` | check proteins again | |
| `--protein P` | only this protein. Can be given more than once | all |

You can also check while docking. Add `--posebusters`, or set `posebusters=True` in `PocketVinaConfig`. The CPUs
then check each protein while the GPU docks the next one:

```bash
pocketvina dock --proteins example/proteins --ligands example/ligands.txt --out results --posebusters
```

In Python, `bust_results` checks a whole run. It returns the protein ids checked:

```python
from pocketvina import bust

bust.bust_results("results", select="best", workers=4)
```

Outside Linux, put this code under `if __name__ == "__main__":` in a script.

`posebusters` reads the saved checks of one protein, here with a few of their columns:

```python
from pocketvina.io.results import Results

pb = Results("results").posebusters("1fbl")
pb[["ligand_id", "pocket", "mode", "pb_status", "failed_checks"]]
```

## Saved results

| `pb_status` | Meaning |
| --- | --- |
| `valid` | every check passed |
| `invalid` | at least one check failed. `failed_checks` names them |
| `timeout` | still running after the last timeout |
| `error` | PoseBusters failed. The `error` column has the message |
| `convert_failed` | the pose could not be turned into a molecule |
| `no_pose` | the ligand has no pose in this protein |

The columns `pb_mask` and `pb_na_mask` store each check as one bit: passed, or no result.
`Results("results").posebusters_checks("1fbl")` returns the check names in bit order.

You can query the checks with DuckDB. The number of poses with each status:

```bash
pocketvina query --results results "SELECT pb_status, count(*) AS n FROM posebusters GROUP BY pb_status"
```

The poses that failed a check, with the names of the checks they failed (at most 20):

```bash
pocketvina query --results results "SELECT protein_id, ligand_id, pocket, mode, failed_checks FROM posebusters WHERE NOT pb_valid LIMIT 20"
```
