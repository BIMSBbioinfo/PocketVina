# PocketVina Atlas

<p align="center">
  <a href="https://pocketvina-atlas.mdc-berlin.de/"><img src="https://img.shields.io/badge/PocketVina_Atlas-Database-0F766E?style-for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjIgMC41IDE5LjUgMjIuNSIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNUVFQUQ0IiBzdHJva2Utd2lkdGg9IjIuMSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48Y2lyY2xlIGN4PSIxMSIgY3k9IjEwIiByPSI2LjIiLz48cGF0aCBkPSJNNC44IDEwaDEyLjQiLz48ZWxsaXBzZSBjeD0iMTEiIGN5PSIxMCIgcng9IjIuNSIgcnk9IjYuMiIvPjxwYXRoIGQ9Ik0xNi40IDNBOC45IDguOSAwIDAgMSA1LjcgMTcuMSIvPjxwYXRoIGQ9Ik0xMSAxOC45djIuNk03LjYgMjEuNWg2LjgiLz48L3N2Zz4K"/></a>
  <a href="https://colab.research.google.com/github/BIMSBbioinfo/PocketVina/blob/main/notebooks/atlas_colab.ipynb"><img src="https://img.shields.io/badge/Google_Colab-Atlas-F9AB00?style-for-the-badge&logo=googlecolab"/></a>
</p>

The [PocketVina Atlas](https://pocketvina-atlas.mdc-berlin.de/) is a public database of
docking results computed in advance: a library of drugs docked with PocketVina against the whole human proteome (the
AlphaFold models of human proteins). `pocketvina.Atlas` reads these results from Python. You can search them, rank
them, look at a protein–drug pair in 3D, save its poses and download a part of the Atlas. You run no docking yourself,
so no GPU is needed. The website describes the data and how it was made.

To try it in your browser, open the
[Colab notebook](https://colab.research.google.com/github/BIMSBbioinfo/PocketVina/blob/main/notebooks/atlas_colab.ipynb).

## Step by step

<!-- The outputs on this page are made by running its code against the public Atlas: python docs/atlas_outputs.py -->
Each step below shows its output, from the public Atlas. In a table, long text is cut short.

### Open the Atlas

`Atlas()` reads the public database over the internet. It fetches only what a call needs, and keeps the files it
fetched for the next time (see [Where files are kept](#where-files-are-kept)):

```python
from pocketvina import Atlas

atlas = Atlas()
atlas
```

<!-- output -->

```{code-block} text
:class: pv-output

PocketVina Atlas v1.1: 18,713 proteins, 2,559 drugs, 48,406,051 pairs [https://bimsbstatic.mdc-berlin.de/akalin/asarigun/pocketvina_atlas/data/v1.1/]
```

### Search

`search` finds proteins and drugs by any of their names or identifiers, best match first:

```python
atlas.search("ABL1")
```

<!-- output -->

```{code-block} text
:class: pv-output

      kind      id                         name       matched                        value
0  protein  P00519  ABL1: Tyrosine-protein k...          gene                         ABL1
1  protein  Q8TDN4  CABLES1: CDK5 and ABL1 e...  protein name  CDK5 and ABL1 enzyme sub...
2  protein  Q9BTV7  CABLES2: CDK5 and ABL1 e...  protein name  CDK5 and ABL1 enzyme sub...
```

Part of a name works too:

```python
atlas.search("tinib", top=5)
```

<!-- output -->

```{code-block} text
:class: pv-output

   kind        id                      name matched                     value
0  drug  PVL00043        Masitinib (AB1010)    name        Masitinib (AB1010)
1  drug  PVL00079    Infigratinib Phosphate    name    Infigratinib Phosphate
2  drug  PVL00105  Pemigatinib (INCB054828)    name  Pemigatinib (INCB054828)
3  drug  PVL00119                  Axitinib    name                  Axitinib
4  drug  PVL00121     Selumetinib (AZD6244)    name     Selumetinib (AZD6244)
```

### A protein and a drug

`protein` shows what the Atlas knows about a protein:

```python
atlas.protein("ABL1")
```

<!-- output -->

```{code-block} text
:class: pv-output

ProteinInfo P00519 (ABL1): Tyrosine-protein kinase ABL1
  AlphaFold model AF-P00519-F1-model_v6, mean pLDDT 63.4
  class Enzyme > Kinase
  5 pockets (P2Rank)
  10 associated diseases (Open Targets)
```

`drug` shows the identifiers of a drug:

```python
atlas.drug("imatinib")
```

<!-- output -->

```{code-block} text
:class: pv-output

DrugInfo PVL00129: Imatinib (STI571)
  ChEMBL ID: CHEMBL1642
  catalogue numbers: S1026, S2475
  InChIKey: KTUFNOKKBVMGRW-UHFFFAOYSA-N
  parent SMILES: Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1
  library SMILES: CN1CCN(CC1)CC2=CC=C(C=C2)C(=O)NC3=CC=C(C)C(=C3)NC4=NC=CC(=N4)C5=CC=CN=C5.C[S](O)(=O)=O
  docked SMILES: Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1
  synonyms: Glamox, Imatinib
  ATC: L01EA01
```

In Jupyter, Colab or VS Code both are small cards, with links to UniProt, AlphaFold DB and Open Targets for the
protein, and to ChEMBL for the drug.

### Best scores

The drugs with the lowest Vina score on a protein, highest-affinity first (in kcal/mol: lower is better):

```python
atlas.best_scores(protein="ABL1", top=5)
```

<!-- output -->

```{code-block} text
:class: pv-output

  protein_id  gene ligand_id                  name  vina  best_pocket  best_mode  ligand_efficiency
0     P00519  ABL1  PVL00898             Hypericin -13.4            1          1              0.353
1     P00519  ABL1  PVL01982  Midostaurin (PKC412) -11.4            1          1              0.265
2     P00519  ABL1  PVL00250          Drospirenone -11.0            1          1              0.407
3     P00519  ABL1  PVL02025          Sanguinarine -10.8            1          1              0.432
4     P00519  ABL1  PVL00271         Staurosporine -10.7            1          1              0.306
```

The proteins on which a drug has its lowest Vina score:

```python
atlas.best_scores(ligand="imatinib", top=5)
```

<!-- output -->

```{code-block} text
:class: pv-output

  protein_id    gene ligand_id               name  vina  best_pocket  best_mode  ligand_efficiency
0     Q9H9P8  L2HGDH  PVL00129  Imatinib (STI571) -12.0            1          1              0.324
1     Q9BZH6   WDR11  PVL00129  Imatinib (STI571) -11.9            1          1              0.322
2     Q7KZN9   COX15  PVL00129  Imatinib (STI571) -11.7            1          1              0.316
3     Q8WZ60   KLHL6  PVL00129  Imatinib (STI571) -11.7            1          1              0.316
4     O75908   SOAT2  PVL00129  Imatinib (STI571) -11.6            2          1              0.314
```

Without `top`, you get every row.

### One protein–drug pair

`select` takes the poses of one pair. With `pocket="all"` it takes the highest-affinity pose in each pocket. The
star marks the pocket of the pair's highest-affinity pose:

```python
pair = atlas.select(protein="ABL1", ligand="imatinib", pocket="all")
pair
```

<!-- output -->

```{code-block} text
:class: pv-output

Selection PV_P00519_PVL00129: protein P00519, ligand PVL00129, the highest-affinity pose in each pocket: 5 poses
  SMILES: CN1CCN(CC1)CC2=CC=C(C=C2)C(=O)NC3=CC=C(C)C(=C3)NC4=NC=CC(=N4)C5=CC=CN=C5.C[S](O)(=O)=O
  pocket  lowest Vina score (kcal/mol)  modes
     1 *                          -8.5  1
     2                            -6.9  1
     3                            -7.8  1
     4                            -7.7  1
     5                            -6.8  1
```

`show` draws the pair in 3D. In Jupyter, Colab or VS Code the page appears below the cell. In a script it is saved as
an HTML file, and its path is printed:

```python
pair.show()
```

<!-- page: _static/examples/atlas.html -->
<iframe src="_static/examples/atlas.html" title="ABL1 and imatinib in the PocketVina Atlas" width="100%" height="800"
        loading="lazy" style="border: 1px solid var(--color-background-border); border-radius: 4px;"></iframe>

The page is interactive: rotate the protein, switch pockets, or pick another drug in *Top hits*.

`bust` runs the PoseBusters checks on these poses. It gives one row per pose, with one column per check; these are
the summary columns:

```python
checks = pair.bust()
checks[["pocket", "mode", "vina", "pb_valid", "pb_status", "failed_checks"]]
```

<!-- output -->

```{code-block} text
:class: pv-output

   pocket  mode  vina  pb_valid pb_status failed_checks
0       1     1  -8.5      True     valid          None
1       2     1  -6.9      True     valid          None
2       3     1  -7.8      True     valid          None
3       4     1  -7.7      True     valid          None
4       5     1  -6.8      True     valid          None
```

`export` saves the poses as an SDF file and returns its path:

```python
pair.export()
```

<!-- output -->

```{code-block} text
:class: pv-output

'/path/to/PV_P00519_PVL00129_pocketall_top.sdf'
```

With `protein=True`, it also saves the protein (its AlphaFold model) as a PDB file, and returns both paths:

```python
pair.export(protein=True)
```

<!-- output -->

```{code-block} text
:class: pv-output

['/path/to/PV_P00519_PVL00129_pocketall_top.sdf', '/path/to/PV_P00519.pdb']
```

`select` and the pair's methods are the same as for your own docking results: see
[Selecting poses](visualisation.md#selecting-poses) and [PoseBusters checks](posebusters.md).

## The 3D page

`atlas.show(protein="ABL1")` shows the protein's page, as `show()` does for a run. `pair.show()` shows the same page
with the pair selected. *Top hits* lists the highest-affinity drugs with a PoseBusters-valid pose, as on the website.
*Selected pair* also has the drug's identifiers, and the page has the protein's information. `path="ABL1.html"` saves
the page to that file. [3D views](visualisation.md) describes the parts of the page.

## Identifiers

`protein=` takes any identifier of a protein: its gene (`"ABL1"`), UniProt accession (`"P00519"`), entry name
(`"ABL1_HUMAN"`), Entrez, HGNC or Ensembl gene ID, a gene synonym or the protein name.

`ligand=` takes any identifier of a drug: its name or a synonym (`"imatinib"`), its PVL ID (the Atlas ID of a docked
structure), ChEMBL ID, catalogue number, InChIKey (or its first block) or a SMILES of the same molecule.

Case is ignored. If an identifier belongs to several proteins or drugs, the error lists them; then use the UniProt
accession or the PVL ID. `atlas.search` also finds part of a name.

## Command line

The `pocketvina atlas` commands do the same in a shell. `search` lists the matches:

```bash
pocketvina atlas search imatinib
```

<!-- output -->

```{code-block} text
:class: pv-output

kind       id              name matched             value
drug PVL00129 Imatinib (STI571)    name Imatinib (STI571)
```

`show` writes the 3D page of a protein, here with a drug selected, and prints its path:

```bash
pocketvina atlas show ABL1 --ligand imatinib --out ABL1_imatinib.html
```

<!-- output -->

```{code-block} text
:class: pv-output

/path/to/ABL1_imatinib.html
```

`download` copies proteins for offline use (see below):

```bash
pocketvina atlas download ABL1 EGFR --out my_atlas
```

<!-- output -->

```{code-block} text
:class: pv-output

PocketVina Atlas v1.1: 2,774 files, 295.0 MB, for 2 proteins into /path/to/my_atlas
/path/to/my_atlas
```

See the [command line reference](cli.md) for every option.

## Offline use

`download` copies what the Atlas has on some proteins into a folder. The pose files are large: before it starts, it
says how many files and MB it copies, then it shows a progress bar. It returns the folder:

```python
atlas.download(proteins=["ABL1", "EGFR"], out="my_atlas")
```

<!-- output -->

```{code-block} text
:class: pv-output

PocketVina Atlas v1.1: 2,774 files, 295.0 MB, for 2 proteins into /path/to/my_atlas
'/path/to/my_atlas'
```

`Atlas` then reads that folder, without internet:

```python
offline = Atlas("my_atlas")
offline.best_scores(protein="ABL1", top=5)
```

<!-- output -->

```{code-block} text
:class: pv-output

  protein_id  gene ligand_id                  name  vina  best_pocket  best_mode  ligand_efficiency
0     P00519  ABL1  PVL00898             Hypericin -13.4            1          1              0.353
1     P00519  ABL1  PVL01982  Midostaurin (PKC412) -11.4            1          1              0.265
2     P00519  ABL1  PVL00250          Drospirenone -11.0            1          1              0.407
3     P00519  ABL1  PVL02025          Sanguinarine -10.8            1          1              0.432
4     P00519  ABL1  PVL00271         Staurosporine -10.7            1          1              0.306
```

- A download that stopped goes on where it stopped when you run it again.
- The files are checked against the release's MANIFEST.json (size and sha256; the few files it does not list, by size,
  and P2Rank's SAS points against sas/MANIFEST.json).
- In the folder, `search` and `best_scores(ligand=...)` list only its proteins.

To read the folder by default, set `POCKETVINA_ATLAS` to its full path, for example
`export POCKETVINA_ATLAS=/path/to/my_atlas` in a shell. `Atlas()` and `pocketvina atlas` then read it.

## Where files are kept

Files read over the internet are kept in `~/.cache/pocketvina/atlas/` (or `$XDG_CACHE_HOME/pocketvina/atlas/`), so
the second look is fast. You can delete this folder at any time.

## Citing

If you use the Atlas, cite it as its [citation page](https://pocketvina-atlas.mdc-berlin.de/cite/)
says. For PocketVina, see [Citing and acknowledgements](citing.md).
