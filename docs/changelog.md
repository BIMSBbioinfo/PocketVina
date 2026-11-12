# Changelog

## 0.3.3

### Changed

- The figure on the [first page](index.md) of this documentation and in the README is readable in a dark theme too.
  On the first page its words can be selected.

### Fixed

- In the page of `show()`, after a click into the Download tab's Python lines, the arrow keys scroll them. They
  stepped through the modes and switched to the Selected pair tab.

## 0.3.2

### Changed

- The hint after a download in the page of `show()` says "If nothing was downloaded, downloads are blocked here: use
  the Python lines in the Download tab." Those lines select the pair shown and save it, e.g.
  `pair = pocketvina.select(protein="1fbl", ligand="molecule1")` and `pair.export(protein=True)` (`atlas.select` on
  a page of the PocketVina Atlas). The hint said "Nothing downloaded? Downloads are blocked here: in Python,
  pair.export() ...", also where the download worked (e.g. in a website's frame) and on pages with no `pair` in
  Python; the Download tab said only "In Python: pair.export() and pair.export(protein=True)."
- The hint after a download is up to 90% of the window wide, at most 640 px, and a click or tap hides it. It was at
  most half the window wide, so on a phone it took up to 9 lines. The Download tab's Python lines do not wrap: a line
  too long for its box scrolls sideways. A copy button next to "In Python:" copies both lines.
- The tabs of the page wrap to a second row where the side panel is too narrow for them, e.g. in a window 320 px or
  900 px wide. They made the page wider than the window.

## 0.3.1

### Added

- `pair.export(protein=True)` also writes the protein the poses were docked into, as a PDB file next to the poses,
  and returns both paths. `pair.export(ligand=False, protein=True)` writes only the protein. `pair.export_protein()`
  still works.
- `atlas.download()` also copies the release's `LICENSE` (the licence of the Atlas data) into the folder. A release
  without one is downloaded as before.

### Changed

- With `theme="dark"` the page of `show()` is dark from the start, not white until its script has loaded.
- The hint after a download in the page of `show()` says "Nothing downloaded? Downloads are blocked here: in Python,
  pair.export() writes the poses and pair.export(protein=True) the protein too." It said "This notebook blocks
  downloads: use pair.export() in Python.", also outside notebooks, and named no way to save the protein. The
  Download tab names `pair.export(protein=True)` instead of `pair.export_protein()`.
- `pair.export("out/")` and `pair.export_protein("out/")`, a path that ends in "/", write into the folder `out`,
  made if needed. They wrote a file named `out`.
- A path without an extension is a folder too: `pair.export("poses")` and `pair.export_protein("poses")` write into
  the folder `poses`, made if needed, as `pocketvina atlas export --out poses` does. They wrote a file named `poses`
  (such a file, written before, is now an error that says so, and is kept). A file whose extension is not its
  format's is an error, and nothing is written: `pair.export("abl1.pdb")` (and `--out abl1.pdb`) says "poses are
  written as SDF; use a .sdf name, ..."; it wrote SDF poses into `abl1.pdb`. The protein's file must end in `.pdb`.
- The PocketVina Atlas is read faster over the internet, most of all far from its server (e.g. in Colab). Connections
  to the server stay open and are reused, nearby parts of a file come in one request, and parts read once are kept in
  memory, so a page shown again needs no request. `pair.show()` of the documentation example took 10 s in Colab
  before. With 110 ms added to every round trip to the server, `select()` and `show()` of that example now take
  2 s instead of 9 s.
- `show()` builds its page faster (the page's data is compressed at a lower level; the page is a little larger).
- What a selection shows (`pair` in a notebook or a console) and the page of `show()` use the words of the Atlas
  website: "highest-affinity pose" instead of "best pose". The column "best Vina (kcal/mol)" of a selection is now
  "lowest Vina score (kcal/mol)", the column "Best poses" of the page's Pockets tab is now "Highest-affinity poses",
  and a page that lists only some ligands says "The top 100 of ..." instead of "The best 100 of ...". The label of
  `viz.show_modes` with py3Dmol says "lowest Vina score" instead of "best".
- The [Atlas page](atlas.md) of this documentation goes step by step and shows the real output of each step,
  with the 3D page of ABL1 and imatinib.
- The pages [Getting started](quickstart.md), [3D views](visualisation.md), [PoseBusters checks](posebusters.md),
  [Results format](output-format.md) and [Installation](installation.md) also show their code one step at a time,
  each step with its own sentence.
- Messages, log lines and the docstrings of the [Python API](api.md) say "highest-affinity" or "lowest Vina
  score" where they said "best" for affinity. `pocketvina dock` and `pocketvina info` print
  "highest-affinity poses: ..." instead of "best poses: ...", and `execute_docking()`, when it shows only some
  of the run's pairs, ends with "The top 50 of ..." instead of "Best 50 of ...". `pocketvina info --json` is
  unchanged.

## 0.3.0

### Added

- `pocketvina.Atlas` reads the PocketVina Atlas, a database of docking results of drugs against the human proteome,
  computed in advance. It gives search, protein and drug identifiers, the best scores of a protein or a drug, and `select()` with
  the same `show()`, `bust()`, `export()` and `export_protein()` as for a run. `show()` draws a protein's page, and
  `download()` copies some proteins for offline use ([PocketVina Atlas](atlas.md)).
- The `pocketvina atlas` command with `search`, `show`, `export` and `download` ([Command line reference](cli.md)).
- An example script, `example/atlas.py`, and a Colab notebook for the Atlas. Neither needs a GPU.

### Removed

- The benchmarks page. How to reproduce the benchmarks is in
  [benchmark/README.md](https://github.com/BIMSBbioinfo/PocketVina/blob/main/benchmark/README.md).

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

- PocketVina is on PyPI: `pip install pocketvina` installs everything and compiles nothing. P2Rank is downloaded at the first run, and no
  environment variables need to be set ([Installation](installation.md)).
- The pip package is now `pocketvina`. Uninstall `pocketvina-gpu` first ([Migrating from PocketVina 0.1](migrating.md)).
- Ligands are docked from their largest fragment. `ligand_prep="legacy"` restores the preparation of 0.1.
- The engine compiles its GPU code on first use into `~/.cache/pocketvina` and never writes into the installation.
- Importing `pocketvina` no longer changes environment variables.
