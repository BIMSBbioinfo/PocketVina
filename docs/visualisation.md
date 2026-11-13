# 3D views

## The 3D page

`show()` draws the results of a run in 3D. Here `pocketvina` is the run from the [quick start](quickstart.md). For an
earlier run, `Results("results")` has the same `show()` and `select()`.

Every protein of the run, with a protein menu:

```python
pocketvina.show()
```

This is the page that `pocketvina.show()` writes for the quick start run. It is interactive: rotate the protein,
choose a ligand in *Top hits* or switch pockets.

<iframe src="_static/examples/results.html" title="The 3D page of the quick start run" width="100%" height="800" loading="lazy"
        style="border: 1px solid var(--color-background-border); border-radius: 4px;"></iframe>

One protein:

```python
pocketvina.show(protein="1fbl")
```

With `path`, the page is saved as an HTML file, to open in a browser or to share:

```python
pocketvina.show(path="results.html")
```

`select` takes the poses of one protein and ligand, here every mode of pocket 2 (see
[Selecting poses](#selecting-poses)):

```python
pair = pocketvina.select(protein="1fbl", ligand="molecule1", pocket=2, mode="all")
```

`pair.show()` draws the page of 1fbl with these poses:

```python
pair.show()
```

<iframe src="_static/examples/pair.html" title="The 3D page of one protein and ligand" width="100%" height="800" loading="lazy"
        style="border: 1px solid var(--color-background-border); border-radius: 4px;"></iframe>

The page has the layout of the protein page of the
[PocketVina Atlas](https://pocketvina-atlas.mdc-berlin.de/). In Jupyter, Colab or
VS Code it appears below the cell. In a script it is saved as an HTML file, such as `PV_1fbl.html`, and the path is
printed. The page contains all its data and works offline.
`Atlas.show()` draws a protein of the PocketVina Atlas on the same page ([PocketVina Atlas](atlas.md)).

| Part | What it shows |
| --- | --- |
| 3D view | the protein, the pocket surfaces, the pose, the residues within 4 Å and their contacts with the ligand |
| Protein, Ligand | under the legend: hide or show the protein or the ligand, and choose the protein's style |
| SAS | P2Rank's solvent accessible surface points, coloured by score from green (0) to red (0.7), larger in the pockets. Greyed out for a protein without them (older results, pockets from a CSV file) |
| Pocket chips | the docked pockets. Click one to show its surface and its pose |
| Mode bar | the poses of a pocket with their Vina scores. The arrow keys step through them |
| Save image | a PNG of the current view, 2 or 4 times larger |
| *Top hits* | the ligands of the protein, highest-affinity first. Click one to select it |
| *Selected pair* | 2D structure, SMILES, InChIKey, Vina score and rank, the lowest Vina score per pocket, contacts |
| *Pockets* | every P2Rank pocket with its score, residues and centre |
| *Download* | the poses as SDF and PDBQT, the protein as PDB and the pocket residues as CSV |

You can change the page with these options of `show()`:

| Option | Meaning |
| --- | --- |
| `max_ligands=100` | ligands per protein, highest-affinity first. `None` shows all |
| `theme="auto"` | colours: `"auto"`, `"light"` or `"dark"` |
| `height=760` | height in a notebook, in pixels |
| `molstar="url"` | load the viewer from the Atlas website. The page is much smaller but needs internet |

## Selecting poses

`select(protein, ligand, pocket=None, mode=1)` picks the poses of one protein and ligand. `show()`, `bust()` and
`export()` then work on these poses.

| `select(protein="1fbl", ligand="molecule1", ...)` | selects | `export()` writes |
| --- | --- | --- |
| (nothing more) | the highest-affinity pose | `PV_1fbl_molecule1_top.sdf` |
| `pocket=2, mode=3` | mode 3 of pocket 2 | `PV_1fbl_molecule1_pk2_m03.sdf` |
| `pocket=2, mode="all"` | every mode of pocket 2 | `PV_1fbl_molecule1_pk2_modes.sdf` |
| `pocket="all"` | the highest-affinity pose in each pocket | `PV_1fbl_molecule1_pocketall_top.sdf` |
| `mode="all"` | every mode of the highest-affinity pocket | `PV_1fbl_molecule1_pk1_modes.sdf` if pocket 1 has the highest-affinity pose |

Pockets are numbered by their P2Rank rank, and modes by their rank within a pocket. Both start at 1: the
best-ranked pocket and the highest-affinity mode. In your own runs, each pocket means each pocket the ligand was
docked into.
`pair.poses` lists the selected poses. A wrong name, pocket or mode gives an error that lists the valid ones.

## Desktop viewers

To open poses in PyMOL, ChimeraX or another viewer, save them as files.

`export()` saves the poses of `pair` as an SDF file with one record per pose, here
`PV_1fbl_molecule1_pk2_modes.sdf`:

```python
pair.export()
```

With a file name and `fmt="pdbqt"`, it saves them as PDBQT:

```python
pair.export("1fbl_molecule1.pdbqt", fmt="pdbqt")
```

With `protein=True`, it also saves the protein, as `PV_1fbl.pdb` next to the poses, and returns both paths:

```python
pair.export(protein=True)
```

`pair.export(ligand=False, protein=True)` saves only the protein. `pair.export_protein()` does the same and still
works.

To export the highest-affinity pose of each protein and ligand of a whole run:

```bash
pocketvina export --results results --what best --format sdf --out best_poses
```

## Single poses with pocketvina.viz

`pocketvina.viz` draws one pose in a simple scene, with Mol\* or py3Dmol. `viz.show` draws the highest-affinity
pose of a protein and ligand:

```python
from pocketvina import viz

viz.show("results", "1fbl", "molecule1")
```

The view of `viz.show("results", "1fbl", "molecule1")` for the quick start run:

<iframe src="_static/examples/mvs_view.html" title="A single pose drawn with Mol*" width="100%" height="520" loading="lazy"
        style="border: 1px solid var(--color-background-border); border-radius: 4px;"></iframe>

With `pocket` and `mode`, it draws another pose, here mode 3 of pocket 2:

```python
viz.show("results", "1fbl", "molecule1", pocket=2, mode=3)
```

`viz.show_modes` draws all modes of one pocket:

```python
viz.show_modes("results", "1fbl", "molecule1", pocket=2)
```

`viz.show` with `backend="py3dmol"` draws with py3Dmol:

```python
viz.show("results", "1fbl", "molecule1", backend="py3dmol")
```

`viz.to_html` saves the view as a web page:

```python
viz.to_html("results", "1fbl", "molecule1", path="pose.html")
```

The same views from the command line. `pocketvina show` saves the view of the highest-affinity pose as a web page:

```bash
pocketvina show --results results --protein 1fbl --ligand molecule1 --out pose.html
```

With `--pocket` and `--mode`, it saves another pose, here mode 3 of pocket 2:

```bash
pocketvina show --results results --protein 1fbl --ligand molecule1 --pocket 2 --mode 3 --out pose2.html
```

Unlike the 3D page, these views load the viewer from the internet. Mol\* needs Python 3.10 or later, so on
Python 3.9 py3Dmol is used.
