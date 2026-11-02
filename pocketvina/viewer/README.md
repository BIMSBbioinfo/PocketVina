# pocketvina/viewer

The files of the 3D page that `show()` draws in a notebook or writes as an HTML file. The page looks like the
protein page of the [PocketVina Atlas](https://bimsbstatic.mdc-berlin.de/akalin/asarigun/pocketvina_atlas/).

| File | What it is |
| --- | --- |
| `molstar-pv-5.13.1/` | Mol\* 5.13.1 without its user interface, as bundled for the Atlas website |
| `viewer.js`, `viewer.css` | the 3D viewer of the Atlas website. Each change in `viewer.js` is marked `PocketVina:` |
| `upstream/viewer.js` | the original, to compare: `diff -u upstream/viewer.js viewer.js` |
| `bridge.js`, `workspace.js`, `workspace.css` | the page around the viewer, written for PocketVina |

The Atlas files come from the source of the PocketVina Atlas website, git commit `40e8d59`.

Licences:

- Mol\* is MIT licensed: `molstar-pv-5.13.1/LICENSE`.
- The other packages in the bundle: `molstar-pv-5.13.1/THIRD-PARTY-LICENSES.txt`.
- How the bundle was built: `molstar-pv-5.13.1/SOURCES.txt`.
- All other files fall under the MIT licence of PocketVina (`LICENSE`).
