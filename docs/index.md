# PocketVina

<p align="center">
  <a href="https://www.arxiv.org/abs/2506.20043"><img src="https://img.shields.io/badge/arXiv-preprint-B31B1B?style-for-the-badge&logo=arXiv"/></a>
  <a href="https://github.com/BIMSBbioinfo/PocketVina"><img src="https://img.shields.io/badge/GitHub-Repository-181717?style-for-the-badge&logo=github"/></a>
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
  <a href="https://github.com/BIMSBbioinfo/PocketVina/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style-for-the-badge&logo=legal"/></a>
</p>

PocketVina docks small molecules into proteins on an NVIDIA GPU. It finds the pockets of each protein with
[P2Rank](https://github.com/rdk/p2rank) and docks every ligand into several pockets. You get the highest-affinity pose
of each protein and ligand, a 3D view of the results and optional PoseBusters checks.

```{raw} html
:file: _static/pocketvina_figure.svg
```

To get started, follow the [installation](installation.md) and the [quick start](quickstart.md). Without a local GPU, you can run the quick start
in the
[Colab notebook](https://colab.research.google.com/github/BIMSBbioinfo/PocketVina/blob/main/notebooks/quickstart_colab.ipynb)
on a free GPU. The [PocketVina Atlas](atlas.md) is a database of docking results of drugs against the human
proteome, computed with PocketVina; you can search it from Python, no GPU needed. To cite PocketVina and the tools
it builds on, see [Citing and acknowledgements](citing.md).

```{toctree}
:maxdepth: 2
:caption: User guide

installation
quickstart
visualisation
atlas
posebusters
output-format
gpu-sharing
hpc
migrating
faq
```

```{toctree}
:maxdepth: 2
:caption: Reference

cli
api
changelog
citing
```
