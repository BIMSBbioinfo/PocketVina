# PocketVina Atlas: docking results of drugs against the human proteome

The [PocketVina Atlas](https://bimsbstatic.mdc-berlin.de/akalin/asarigun/pocketvina_atlas/) is a public collection of
docking results computed in advance: a library of drugs docked with PocketVina against the whole human proteome (the
AlphaFold models of human proteins). `pocketvina.Atlas` reads these results from Python. You can search them, rank
them, look at a protein–drug pair in 3D, save its poses and download a part of the Atlas. You run no docking yourself,
so no GPU is needed. The website describes the data and how it was made.

To try it in your browser, open the
[Colab notebook](https://colab.research.google.com/github/BIMSBbioinfo/PocketVina/blob/main/notebooks/atlas_colab.ipynb).

## Example

```python
from pocketvina import Atlas

atlas = Atlas()                                 # the public Atlas, read over the internet
atlas.search("imatinib")                        # proteins and drugs that match, best first
atlas.protein("ABL1")                           # the protein: gene, name, AlphaFold model, pockets
atlas.drug("imatinib")                          # the drug's identifiers
atlas.best_scores(protein="ABL1", top=20)       # the 20 drugs with the best Vina score on ABL1
atlas.best_scores(ligand="imatinib", top=20)    # the 20 proteins imatinib scores best on

pair = atlas.select(protein="ABL1", ligand="imatinib", pocket="all")  # the best pose in every pocket
pair.show()                                     # the 3D page with this pair selected
pair.bust()                                     # PoseBusters checks of these poses
pair.export()                                   # the poses as an SDF file
pair.export_protein()                           # the protein as a PDB file
```

`select` and the pair's methods are the same as for your own docking results: see
[Selecting poses](visualisation.md#selecting-poses) and [PoseBusters checks](posebusters.md).

## The 3D page

`atlas.show(protein="ABL1")` shows the protein's page, as `show()` does for a run. *Top hits* lists the drugs that
score best with a PoseBusters-valid pose, as on the website. *Selected pair* also has the drug's identifiers, and the
page has the protein's information. In Jupyter, Colab or VS Code the page appears below the cell. In a script it is
saved as an HTML file, and the path is printed. `path="ABL1.html"` chooses the file. [3D views](visualisation.md)
shows such a page and its parts.

## Identifiers

`protein=` takes any identifier of a protein: its gene (`"ABL1"`), UniProt accession (`"P00519"`), entry name
(`"ABL1_HUMAN"`), Entrez, HGNC or Ensembl gene ID, a gene synonym or the protein name.

`ligand=` takes any identifier of a drug: its name or a synonym (`"imatinib"`), its PVL ID (the Atlas ID of a docked
structure), ChEMBL ID, catalogue number, InChIKey (or its first block) or a SMILES of the same molecule.

Case is ignored. If an identifier belongs to several proteins or drugs, the error lists them; then use the UniProt
accession or the PVL ID. `atlas.search` also finds part of a name.

## Command line

The `pocketvina atlas` commands do the same in a shell:

```bash
pocketvina atlas search imatinib
pocketvina atlas show ABL1 --ligand imatinib --out ABL1_imatinib.html
pocketvina atlas download ABL1 EGFR --out my_atlas
```

See the [command line reference](cli.md) for every option.

## Offline use

`download` copies what the Atlas has on some proteins into a folder. `Atlas` then reads that folder without internet:

```python
atlas.download(proteins=["ABL1", "EGFR"], out="my_atlas")
atlas = Atlas("my_atlas")                       # works offline
atlas.best_scores(protein="ABL1")
```

- The pose files are large. Before it starts, `download` says how many MB it copies, then shows a progress bar.
- A download that stopped goes on where it stopped when you run it again.
- The files are checked against the checksums of the release.
- In the folder, `search` and `best_scores(ligand=...)` list only its proteins.

To read the folder by default, set `POCKETVINA_ATLAS` to its full path, for example
`export POCKETVINA_ATLAS=/path/to/my_atlas` in a shell. `Atlas()` and `pocketvina atlas` then read it.

## Where files are kept

Files read over the internet are kept in `~/.cache/pocketvina/atlas/` (or `$XDG_CACHE_HOME/pocketvina/atlas/`), so
the second look is fast. You can delete this folder at any time.

## Citing

If you use the Atlas, cite it as its [citation page](https://bimsbstatic.mdc-berlin.de/akalin/asarigun/pocketvina_atlas/cite/)
says. For PocketVina, see [Citing and acknowledgements](citing.md).
