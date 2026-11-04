# PocketVina

PocketVina docks small molecules into proteins on an NVIDIA GPU. It finds the pockets of each protein with
[P2Rank](https://github.com/rdk/p2rank) and docks every ligand into several pockets. You get the best pose of each
protein and ligand, a 3D view of the results and optional PoseBusters checks.

```{image} ../doc/assets/pocketvina_figure.png
:alt: PocketVina: pocket detection with P2Rank, then docking into several pockets on the GPU
:width: 100%
```

To get started, follow the [installation](installation.md) and the [quick start](quickstart.md). Without a local GPU, you can run the quick start
in the
[Colab notebook](https://colab.research.google.com/github/BIMSBbioinfo/PocketVina/blob/main/notebooks/quickstart_colab.ipynb)
on a free GPU. To cite PocketVina and the tools it builds on, see [Citing and acknowledgements](citing.md).

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
