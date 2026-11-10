# Results format

A run with `output_format="atlas"` writes one results folder. This is the default of `pocketvina dock`. The folder
has Parquet tables with the scores and one compressed pose file per protein, as in the PocketVina Atlas.

## Results directory

A run writes this folder:

```text
results/
  MANIFEST.json                      the run: PocketVina version, settings, checksum of the ligand list
  manifest/<protein_id>.json         written last for each protein: the protein is done
  poses/<protein_id>.poses           highest-affinity pose per ligand as SDF, every engine output as PDBQT
  structures/<protein_id>.pdb.gz     the input structure, gzipped
  tables/
    ligands.parquet                  one table for the whole run
    proteins/<protein_id>.parquet    the other tables have one part per protein
    pockets/<protein_id>.parquet
    pocket_runs/<protein_id>.parquet
    modes/<protein_id>.parquet
    pairs/<protein_id>.parquet
    posebusters/<protein_id>.parquet  after PoseBusters checks
```

A protein is done when its `manifest/<protein_id>.json` exists. A new run skips the done proteins.

A folder holds one run. `MANIFEST.json` keeps the settings that change the results, such as box, seed, pockets and
P2Rank model. Docking into the folder with other settings or ligands is refused. Use a new folder, or `--force` to
replace the run. You can change the GPUs, the temp folder and the timeouts.

## Reading results

### pandas

`Results` gives you the tables as pandas DataFrames. Open a results folder:

```python
from pocketvina.io.results import Results

res = Results("results")
```

The protein ids, sorted:

```python
res.proteins()
```

The highest-affinity pose of each protein and ligand:

```python
pairs = res.pairs()
```

The 10 of them with the lowest Vina score:

```python
pairs.sort_values("vina").head(10)
```

The P2Rank pockets of one protein, with their score, probability and centre:

```python
res.pockets("1fbl")[["pocket", "score", "probability", "center_x", "center_y", "center_z"]]
```

Every pose of one protein with its Vina score (`affinity`). `columns` reads only these columns:

```python
res.modes("1fbl", columns=["ligand_idx", "pocket", "mode", "affinity"])
```

The ligands, with the SMILES docked and whether their preparation worked (`status`):

```python
res.ligands(columns=["ligand_idx", "ligand_id", "smiles_docked", "status"])
```

The SDF text of the highest-affinity pose of a protein and ligand:

```python
res.top_sdf("1fbl", "molecule1")
```

The input structure of a protein, as bytes:

```python
res.structure("1fbl")
```

`Results` reads only the proteins that are done.

### DuckDB

`Results.duckdb()` gives you a DuckDB connection with one view per table:

```python
con = res.duckdb()
```

Run SQL over these views. `.df()` gives the rows as a pandas DataFrame, here the 10 protein x ligand pairs with the
lowest Vina score:

```python
con.sql("""
    SELECT p.protein_id, p.ligand_id, p.vina, round(p.ligand_efficiency, 3) AS le, l.smiles_docked
    FROM pairs p JOIN ligands l USING (ligand_idx)
    ORDER BY p.vina LIMIT 10
""").df()
```

You can run SQL from the command line too, here the lowest Vina score of each ligand over all proteins, lowest first:

```bash
pocketvina query --results results "SELECT ligand_id, min(vina) AS best FROM pairs GROUP BY ligand_id ORDER BY best"
```

With `--csv`, the rows are saved as a CSV file, here every pose with a Vina score below -9:

```bash
pocketvina query --results results --csv strong.csv "SELECT * FROM modes WHERE affinity < -9"
```

## Exports

`pocketvina export` writes poses and tables as plain files. With `--what best --format sdf`, it writes the
highest-affinity pose of each protein and ligand, one SDF file per protein x ligand:

```bash
pocketvina export --results results --what best --format sdf --out best_poses
```

With `--what all --format pdbqt`, it writes every engine output:

```bash
pocketvina export --results results --what all --format pdbqt --out outputs
```

With `--what best --format csv`, it writes the highest-affinity pose of each protein and ligand as a table,
`best_poses.csv`:

```bash
pocketvina export --results results --what best --format csv --out tables
```

With `--what all --format csv`, it writes `combined_results.csv`, as in v0.1:

```bash
pocketvina export --results results --what all --format csv --out tables
```

In Python, use `res.export(out_dir, what=..., fmt=...)` and `res.to_combined_csv(path)`.

## Tables

All tables are Parquet files. "null" marks columns that can be empty. Vina scores are in kcal/mol, distances in Å.

### Ids

- `protein_id`: the structure file name without extension. Characters other than `A-Za-z0-9._-` become `_`.
- `ligand_id`: the `molecule_name` of the input. `ligand_idx` is its row in `tables/ligands.parquet`, from 0.
- `pocket`: the P2Rank rank of the pocket, where 1 is the best.
- `mode`: the rank of a pose within one pocket, where 1 is the highest-affinity pose.

The highest-affinity pose of a ligand has the lowest Vina score over all docked pockets. Ties go to the lower pocket,
then the lower mode.

### ligands

The prepared ligands.

| Column | Type | Meaning |
| --- | --- | --- |
| `ligand_idx`, `ligand_id` | int32, string | |
| `smiles_input` | string | the SMILES as given |
| `smiles_docked` | string, null | the SMILES docked: the largest fragment of a salt or mixture |
| `inchikey`, `n_heavy` | string null, int16 null | InChIKey and heavy atoms of the docked molecule |
| `prep_method`, `seed` | string, int64 null | `standard` or `legacy`, and the seed of the 3D structure |
| `status`, `error` | string, string null | `ok`, or why preparation failed, such as `invalid_smiles` or `unsupported_element` |
| `pdbqt` | string, null | the input of the engine |

### proteins

One row per protein.

| Column | Type | Meaning |
| --- | --- | --- |
| `protein_id`, `uniprot` | string, string null | `uniprot` is set for AlphaFold DB names such as `AF-P00533-F1-model_v4` |
| `source_path`, `structure_file` | string | the input file, and its copy in the results folder |
| `receptor_sha256` | binary(32) | checksum of the receptor given to the engine |
| `n_pockets`, `n_pockets_docked` | int16 | pockets found by P2Rank, and pockets docked |
| `status`, `error` | string, string null | `ok`, or `no_pockets` if P2Rank found none |

### pockets

Every P2Rank pocket, docked or not.

| Column | Type | Meaning |
| --- | --- | --- |
| `protein_id`, `pocket`, `name` | string, int16, string | the P2Rank rank and name, such as `pocket1` |
| `score`, `probability` | float32, float32 null | P2Rank score and probability |
| `center_x`, `center_y`, `center_z` | float32 | pocket centre, the centre of the docking box |
| `size_x`, `size_y`, `size_z` | float32 | size of the docking box |
| `residue_ids` | string, null | residues lining the pocket, such as `A_101 A_103` |

### pocket_runs

One row per ligand and docked pocket.

| Column | Type | Meaning |
| --- | --- | --- |
| `protein_id`, `ligand_idx`, `ligand_id`, `pocket` | string, int32, string, int8 | |
| `status` | string | `ok`, `no_output`, `parse_error`, `engine_error` or `skipped` |
| `n_modes`, `best_affinity` | int8, float32 null | poses found, and the lowest Vina score in this pocket |
| `offset`, `length`, `raw_length`, `sha256` | | where the engine output is in the pose file |
| `seed`, `runtime_s`, `error` | int64 null, float32 null, string null | engine seed, seconds per ligand, error message |

### modes

One row per pose.

| Column | Type | Meaning |
| --- | --- | --- |
| `protein_id`, `uniprot`, `ligand_idx`, `pocket`, `mode` | | |
| `affinity` | float32 | Vina score |
| `rmsd_lb`, `rmsd_ub` | float32 | RMSD lower and upper bound to mode 1, as Vina reports them |
| `block_offset`, `block_length`, `raw_offset`, `raw_length` | | where the pose is in the pose file |

### pairs

The highest-affinity pose of each ligand in each protein.

| Column | Type | Meaning |
| --- | --- | --- |
| `protein_id`, `uniprot`, `ligand_idx`, `ligand_id` | | |
| `vina` | float32 | lowest Vina score. NaN without a pose |
| `best_pocket`, `best_mode` | int8 | where the highest-affinity pose is. -1 without a pose |
| `rmsd_lb`, `rmsd_ub` | float32 | of that mode |
| `pb_status` | string, null | PoseBusters status of the highest-affinity pose |
| `top_offset`, `top_length`, `top_raw_length`, `top_sha256` | | where its SDF is in the pose file |
| `sdf_status` | string | `ok`, `convert_failed` or `no_pose` |

`Results.pairs()` and the DuckDB view `pairs` add `ligand_efficiency`, which is `-vina / n_heavy`. With PoseBusters
results they also add `pb_valid`.

### posebusters

Written by `pocketvina bust` or `posebusters=True`. See [PoseBusters checks](posebusters.md).

| Column | Type | Meaning |
| --- | --- | --- |
| `protein_id`, `uniprot`, `ligand_idx`, `ligand_id`, `pocket`, `mode` | | the pose checked. -1: no pose |
| `pb_config`, `pb_version` | string | PoseBusters configuration and version |
| `pb_status`, `pb_valid` | string, bool null | `valid`, `invalid`, `timeout`, `error`, `convert_failed` or `no_pose` |
| `pb_mask`, `pb_na_mask` | int32 | bit *b* set: check *b* passed, or gave no result |
| `failed_checks` | string, null | names of the failed checks, comma-separated |
| `n_heavy_docked`, `runtime_s`, `error` | int16 null, float32 null, string null | |

## Pose files

`poses/<protein_id>.poses` holds all poses of one protein, in the PVATLAS1 format. Part A has the highest-affinity
pose of every ligand as SDF, highest-affinity first. Part B has every engine output as PDBQT, with all modes. The
offset columns of the tables point into this file. `Results` reads the poses for you with `top_sdf()`, `pose_block()`
and `pose_model()`.

## The "files" format

`pocketvina dock --format files` writes the output folder of PocketVina 0.1. It is also the default of
`PocketVinaConfig`, for scripts written for 0.1. The folder has one PDBQT and JSON file per ligand and pocket, and
`combined_results.csv`. This format does not resume and has no PoseBusters step. `pocketvina import-legacy`
converts it into a results folder ([Migrating from PocketVina 0.1](migrating.md)).
