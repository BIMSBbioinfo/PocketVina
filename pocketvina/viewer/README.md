# pocketvina/viewer

The files of the 3D page that `show()` draws in a notebook or writes as an HTML file. The page looks like the
protein page of the [PocketVina Atlas](https://pocketvina-atlas.mdc-berlin.de/).

| File | What it is |
| --- | --- |
| `molstar-pv-5.13.1/` | Mol\* 5.13.1 without its user interface, as bundled for the Atlas website |
| `viewer.js`, `viewer.css` | the 3D viewer of the Atlas website. Each change in `viewer.js` is marked `PocketVina:` |
| `upstream/viewer.js` | the original, to compare: `diff -u upstream/viewer.js viewer.js` |
| `bridge.js`, `workspace.js`, `workspace.css` | the page around the viewer, written for PocketVina |

The Atlas files come from the source of the PocketVina Atlas website, git commit `40e8d59`.

The page (`pocketvina/workspace.py`) keeps itself safe to open:

- Everything it shows is in the page: no request is made, except for the Mol\* bundle with `molstar="url"`.
- Text from the data is escaped before it goes into the page; a ligand's 2D drawing is an image (`data:` URL).
- In a notebook it is an iframe with `sandbox="allow-scripts allow-downloads allow-popups allow-popups-to-escape-sandbox"`:
  its own origin (no `allow-same-origin`), downloads for Save image and the files, and links that open in a new tab.
- Links are only `http(s)` addresses and open in a new tab (`target="_blank" rel="noopener noreferrer"`).

A page of the PocketVina Atlas (`Atlas.show`, `pocketvina/atlas.py`) has optional fields, drawn only when present, so a
page of docking results looks as before:

| Field | What `workspace.js` draws |
| --- | --- |
| `proteins[].info` | title, name, UniProt and AlphaFold links, Annotations, associated diseases with their scores (the site's protein header) |
| `proteins[].pockets[].plddt` | a pLDDT column in Pockets |
| `ligands[].ids` | the identifier rows of Selected pair (label, value, link, tooltip, a collapsed long list), each with a copy button |
| `proteins[].rows[].pb` | the PoseBusters row of the highest-affinity pose (also on a results page, where the run has a result) |
| `proteins[].topNote` | the text above Top hits; the rows are the site's Top hits (drugs with a PoseBusters-valid pose), `rank` and `n` count the drugs (`rank` null: not ranked, e.g. a row docked as a salt counterion) |

On an Atlas page the texts name the protein by its gene (`info.title`) and use the site's words ("rank X of N drugs on
ABL1", "for this compound"). There `rows[].across` can also be `[null, n, top]`: the protein is not among the drug's
`top` best targets, and Selected pair says so.

Licences:

- Mol\* is MIT licensed: `molstar-pv-5.13.1/LICENSE`.
- The other packages in the bundle: `molstar-pv-5.13.1/THIRD-PARTY-LICENSES.txt`.
- How the bundle was built: `molstar-pv-5.13.1/SOURCES.txt`.
- All other files fall under the MIT licence of PocketVina (`LICENSE`).
