<h1 align="center">PocketVina</h1>

<p align="center">
  <a href="https://www.arxiv.org/abs/2506.20043"><img src="https://img.shields.io/badge/arXiv-preprint-B31B1B?style-for-the-badge&logo=arXiv"/></a>
  <a href="https://bioinformatics.mdc-berlin.de/PocketVina/"><img src="https://img.shields.io/badge/Docs-Documentation-0A7BBB?style-for-the-badge&logo=readme"/></a>
  <a href="https://pocketvina-atlas.mdc-berlin.de/"><img src="https://img.shields.io/badge/PocketVina_Atlas-Database-0F766E?style-for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjIgMC41IDE5LjUgMjIuNSIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNUVFQUQ0IiBzdHJva2Utd2lkdGg9IjIuMSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48Y2lyY2xlIGN4PSIxMSIgY3k9IjEwIiByPSI2LjIiLz48cGF0aCBkPSJNNC44IDEwaDEyLjQiLz48ZWxsaXBzZSBjeD0iMTEiIGN5PSIxMCIgcng9IjIuNSIgcnk9IjYuMiIvPjxwYXRoIGQ9Ik0xNi40IDNBOC45IDguOSAwIDAgMSA1LjcgMTcuMSIvPjxwYXRoIGQ9Ik0xMSAxOC45djIuNk03LjYgMjEuNWg2LjgiLz48L3N2Zz4K"/></a>
  <a href="https://pypi.org/project/pocketvina/"><img src="https://img.shields.io/pypi/v/pocketvina?label=PyPI&color=3775A9&logo=pypi&logoColor=white"/></a>
  <a href="https://hub.docker.com/r/asarigun/pocketvina"><img src="https://img.shields.io/badge/Docker-Image-2496ED?style-for-the-badge&logo=docker"/></a>
  <a href="https://www.youtube.com/watch?v=n8-YKOTvHGs"><img src="https://img.shields.io/badge/YouTube-UI Demo-FF0000?style-for-the-badge&logo=youtube"/></a>
  <a href="https://colab.research.google.com/github/BIMSBbioinfo/PocketVina/blob/main/notebooks/quickstart_colab.ipynb"><img src="https://img.shields.io/badge/Google_Colab-Tutorial-F9AB00?style-for-the-badge&logo=googlecolab"/></a>
  <a href="https://colab.research.google.com/github/BIMSBbioinfo/PocketVina/blob/main/notebooks/atlas_colab.ipynb"><img src="https://img.shields.io/badge/Google_Colab-Atlas-F9AB00?style-for-the-badge&logo=googlecolab"/></a>
  <a href="https://zenodo.org/records/15733460"><img src="https://img.shields.io/badge/Zenodo-Dataset & Raw Results-1682D4?style-for-the-badge&logo=zenodo"/></a>
  <a href="https://isocpp.org/"><img src="https://img.shields.io/badge/C++-11.4.0-00599C?style-for-the-badge&logo=cplusplus"/></a>
  <a href="https://gcc.gnu.org/"><img src="https://img.shields.io/badge/C-11.4.0-3C2179?style-for-the-badge&logo=c"/></a>
  <a href="https://adoptium.net/"><img src="https://img.shields.io/badge/Java-17+-007396?style-for-the-badge&logo=openjdk"/></a>
  <a href="https://www.python.org/downloads/release/"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?style-for-the-badge&logo=python"/></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-11.5+-76B900?style-for-the-badge&logo=nvidia"/></a>
  <a href="https://www.khronos.org/opencl/"><img src="https://img.shields.io/badge/OpenCL-3.0-4FC08D?style-for-the-badge&logo=opencl"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style-for-the-badge&logo=legal"/></a>
</p>

<p align="center">
  <a href="https://github.com/BIMSBbioinfo/PocketVina/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/BIMSBbioinfo/PocketVina/ci.yml?branch=main&label=tests&logo=githubactions&logoColor=white"/></a>
  <a href="https://github.com/BIMSBbioinfo/PocketVina/actions/workflows/engine.yml"><img src="https://img.shields.io/github/actions/workflow/status/BIMSBbioinfo/PocketVina/engine.yml?branch=main&label=engine%20build&logo=githubactions&logoColor=white"/></a>
  <a href="https://github.com/BIMSBbioinfo/PocketVina/actions/workflows/notebook.yml"><img src="https://img.shields.io/github/actions/workflow/status/BIMSBbioinfo/PocketVina/notebook.yml?branch=main&label=notebook&logo=githubactions&logoColor=white"/></a>
  <a href="https://github.com/BIMSBbioinfo/PocketVina/actions/workflows/docs.yml"><img src="https://img.shields.io/github/actions/workflow/status/BIMSBbioinfo/PocketVina/docs.yml?branch=main&label=docs&logo=githubactions&logoColor=white"/></a>
</p>

<p align="center">
    GPU-accelerated protein-ligand docking with automated pocket detection, exploring through multi-pocket conditioning.
</p>

<p align="center">
  <img src="doc/assets/pocketvina.gif" width="98%" />
</p>

PocketVina finds the pockets of each protein with P2Rank and docks every ligand into the best pockets on the GPU.

## Installation

```bash
pip install pocketvina
```

PocketVina runs on Linux with an NVIDIA GPU and needs Java 17+ for P2Rank. Other ways to install (conda, Docker) are
in the [documentation](https://bioinformatics.mdc-berlin.de/PocketVina/installation.html).

## Citations

If you use PocketVina, please cite our [preprint](https://arxiv.org/abs/2506.20043):

```bibtex
@misc{sarigun2025pocketvinaenablesscalablehighly,
      title={PocketVina Enables Scalable and Highly Accurate Physically Valid Docking through Multi-Pocket Conditioning},
      author={Ahmet Sarigun and Bora Uyar and Vedran Franke and Altuna Akalin},
      year={2025},
      eprint={2506.20043},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2506.20043},
}
```

PocketVina builds on P2Rank, QuickVina2-GPU, AutoDock Vina, PoseBusters, Mol\* and RDKit. Please cite them too: their
papers are on the [citing page](https://bioinformatics.mdc-berlin.de/PocketVina/citing.html).

## Getting started

- [Quick start](https://bioinformatics.mdc-berlin.de/PocketVina/quickstart.html): dock the example proteins and ligands, then look at the results in 3D
- [Google Colab notebook](https://colab.research.google.com/github/BIMSBbioinfo/PocketVina/blob/main/notebooks/quickstart_colab.ipynb)
- [PocketVina Atlas](https://bioinformatics.mdc-berlin.de/PocketVina/atlas.html): a database of docking results of drugs against the whole human proteome, computed in advance; search, view and download them from Python
- [Command line](https://bioinformatics.mdc-berlin.de/PocketVina/cli.html)
- [Large screens on a cluster (SLURM)](https://bioinformatics.mdc-berlin.de/PocketVina/hpc.html)

## Benchmarks

The benchmark data and results are on [Zenodo](https://zenodo.org/records/15733460). The code to reproduce them is in
[benchmark/](benchmark/README.md).

## Documentation

The [documentation](https://bioinformatics.mdc-berlin.de/PocketVina/) was generated with
[Sphinx](https://www.sphinx-doc.org) from the Markdown pages in [docs/](docs/index.md), using MyST-Parser and the
Furo theme. To build it locally:

```bash
pip install -r docs/requirements.txt
sphinx-build docs docs/_build/html
```

## Contact

For questions and bug reports, open an [issue](https://github.com/BIMSBbioinfo/PocketVina/issues) or start a
[discussion](https://github.com/BIMSBbioinfo/PocketVina/discussions).

## License

PocketVina is released under the MIT License ([LICENSE](LICENSE)). The docking engine is based on AutoDock Vina and
QuickVina2-GPU-2.1 and keeps their Apache License 2.0. The bundled Mol\* viewer is MIT licensed. The data of the
PocketVina Atlas has its own licence: see the
[citing page](https://bioinformatics.mdc-berlin.de/PocketVina/citing.html#licences).
