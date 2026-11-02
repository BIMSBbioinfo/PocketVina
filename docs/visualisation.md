# 3D views

## The 3D page

`show()` draws the results of a run in 3D:

```python
pocketvina.show()                         # every protein of the run, with a protein menu
pocketvina.show(protein="1fbl")           # one protein
pocketvina.show(path="results.html")      # an HTML file, to open in a browser or to share

pair = pocketvina.select(protein="1fbl", ligand="molecule1", pocket=2, mode="all")
pair.show()                               # the page of 1fbl, with every mode of pocket 2
```

Here `pocketvina` is the run from the [quick start](quickstart.md). For an earlier run, `Results("results")` has the same `show()`
and `select()`.

This is the page that `pocketvina.show()` writes for the quick start run. It is interactive: rotate the protein,
choose a ligand in *Top hits* or switch pockets.

<iframe src="_static/examples/results.html" title="The 3D page of the quick start run" width="100%" height="800" loading="lazy"
        style="border: 1px solid var(--color-background-border); border-radius: 4px;"></iframe>

The page of `pair.show()` from the example above, with every mode of pocket 2:

<iframe src="_static/examples/pair.html" title="The 3D page of one protein and ligand" width="100%" height="800" loading="lazy"
        style="border: 1px solid var(--color-background-border); border-radius: 4px;"></iframe>

The page has the layout of the protein page of the
[PocketVina Atlas](https://bimsbstatic.mdc-berlin.de/akalin/asarigun/pocketvina_atlas/). In Jupyter, Colab or
VS Code it appears below the cell. In a script it is saved as an HTML file, such as `PV_1fbl.html`, and the path is
printed. The page contains all its data and works offline.

| Part | What it shows |
| --- | --- |
| 3D view | the protein, the pocket surfaces, the pose, the residues within 4 Å and their contacts with the ligand |
| Pocket chips | the docked pockets. Click one to show its surface and its pose |
| Mode bar | the poses of a pocket with their Vina scores. The arrow keys step through them |
| Save image | a PNG of the current view, 2 or 4 times larger |
| *Top hits* | the ligands of the protein, best first. Click one to select it |
| *Selected pair* | 2D structure, SMILES, InChIKey, Vina score and rank, best score per pocket, contacts |
| *Pockets* | every P2Rank pocket with its score, residues and centre |
| *Download* | the poses as SDF and PDBQT, the protein as PDB and the pocket residues as CSV |

You can change the page with these options of `show()`:

| Option | Meaning |
| --- | --- |
| `max_ligands=100` | ligands per protein, best first. `None` shows all |
| `theme="auto"` | colours: `"auto"`, `"light"` or `"dark"` |
| `height=760` | height in a notebook, in pixels |
| `molstar="url"` | load the viewer from the Atlas website. The page is much smaller but needs internet |

## Selecting poses

`select(protein, ligand, pocket=None, mode=1)` picks the poses of one protein and ligand. `show()`, `bust()` and
`export()` then work on these poses.

| `select(protein="1fbl", ligand="molecule1", ...)` | selects | `export()` writes |
| --- | --- | --- |
| (nothing more) | the best pose | `PV_1fbl_molecule1_top.sdf` |
| `pocket=2, mode=3` | mode 3 of pocket 2 | `PV_1fbl_molecule1_pk2_m03.sdf` |
| `pocket=2, mode="all"` | every mode of pocket 2 | `PV_1fbl_molecule1_pk2_modes.sdf` |
| `pocket="all"` | the best pose in every docked pocket | `PV_1fbl_molecule1_pocketall_top.sdf` |
| `mode="all"` | every mode of the best pocket | `PV_1fbl_molecule1_pk1_modes.sdf` if pocket 1 is best |

Pockets are numbered by their P2Rank rank, and modes by their rank within a pocket. Both start at 1, the best.
`pair.poses` lists the selected poses. A wrong name, pocket or mode gives an error that lists the valid ones.

## Desktop viewers

To open poses in PyMOL, ChimeraX or another viewer, save them as files:

```python
pair.export()                                   # PV_1fbl_molecule1_pk2_modes.sdf, one record per pose
pair.export("1fbl_molecule1.pdbqt", fmt="pdbqt")
pair.export_protein()                           # PV_1fbl.pdb
```

To export the best poses of a whole run:

```bash
pocketvina export --results results --what best --format sdf --out best_poses
```

## Single poses with pocketvina.viz

`pocketvina.viz` draws one pose in a simple scene, with Mol\* or py3Dmol:

```python
from pocketvina import viz

viz.show("results", "1fbl", "molecule1")                         # the best pose
viz.show("results", "1fbl", "molecule1", pocket=2, mode=3)       # another pose
viz.show_modes("results", "1fbl", "molecule1", pocket=2)         # all modes of one pocket
viz.show("results", "1fbl", "molecule1", backend="py3dmol")      # with py3Dmol
viz.to_html("results", "1fbl", "molecule1", path="pose.html")    # a web page
```

The view of `viz.show("results", "1fbl", "molecule1")` for the quick start run:

<iframe src="_static/examples/mvs_view.html" title="A single pose drawn with Mol*" width="100%" height="520" loading="lazy"
        style="border: 1px solid var(--color-background-border); border-radius: 4px;"></iframe>

The same views from the command line:

```bash
pocketvina show --results results --protein 1fbl --ligand molecule1 --out pose.html
pocketvina show --results results --protein 1fbl --ligand molecule1 --pocket 2 --mode 3 --out pose2.html
```

Unlike the 3D page, these views load the viewer from the internet. Mol\* needs Python 3.10 or later, so on
Python 3.9 py3Dmol is used.
