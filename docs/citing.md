# Citing and acknowledgements

## Citing PocketVina

If you use PocketVina in your work, please cite:

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

## Tools PocketVina builds on

Please also cite the tools that PocketVina uses:

### P2Rank

P2Rank finds the pockets of each protein.

```bibtex
@article{krivak2018p2rank,
  title   = {P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure},
  author  = {Krivák, Radoslav and Hoksza, David},
  journal = {Journal of cheminformatics},
  volume  = {10},
  pages   = {1--12},
  year    = {2018},
  publisher = {Springer}
}
```

### Vina family

The docking engine is based on QuickVina2-GPU-2.1, part of Vina-GPU 2.1. Its roots are Vina-GPU 2.0, Vina-GPU 2022,
AutoDock Vina and QuickVina 2.

```bibtex
@article{tang2024vina,
  title   = {Vina-GPU 2.1: towards further optimizing docking speed and precision of AutoDock Vina and its derivatives},
  author  = {Tang, Shidi and Ding, Ji and Zhu, Xiangyu and Wang, Zheng and Zhao, Haitao and Wu, Jiansheng},
  journal = {IEEE/ACM Transactions on Computational Biology and Bioinformatics},
  year    = {2024},
  publisher = {IEEE}
}
```

```bibtex
@article{ding2023vina,
  title   = {Vina-GPU 2.0: further accelerating AutoDock Vina and its derivatives with graphics processing units},
  author  = {Ding, Ji and Tang, Shidi and Mei, Zheming and Wang, Lingyue and Huang, Qinqin and Hu, Haifeng and Ling, Ming and Wu, Jiansheng},
  journal = {Journal of chemical information and modeling},
  volume  = {63},
  number  = {7},
  pages   = {1982--1998},
  year    = {2023},
  publisher = {ACS Publications}
}
```

```bibtex
@article{tang2022accelerating,
  title   = {Accelerating autodock vina with gpus},
  author  = {Tang, Shidi and Chen, Ruiqi and Lin, Mengru and Lin, Qingde and Zhu, Yanxiang and Ding, Ji and Hu, Haifeng and Ling, Ming and Wu, Jiansheng},
  journal = {Molecules},
  volume  = {27},
  number  = {9},
  pages   = {3041},
  year    = {2022},
  publisher = {MDPI}
}
```

```bibtex
@article{trott2010autodock,
  title   = {AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading},
  author  = {Trott, Oleg and Olson, Arthur J},
  journal = {Journal of computational chemistry},
  volume  = {31},
  number  = {2},
  pages   = {455--461},
  year    = {2010},
  publisher = {Wiley Online Library}
}
```

```bibtex
@article{alhossary2015quickvina2,
  title   = {Fast, accurate, and reliable molecular docking with QuickVina 2},
  author  = {Alhossary, Amr and Handoko, Stephanus Daniel and Mu, Yuguang and Kwoh, Chee-Keong},
  journal = {Bioinformatics},
  volume  = {31},
  number  = {13},
  pages   = {2214--2216},
  year    = {2015},
  doi     = {10.1093/bioinformatics/btv082}
}
```

### Mol\*, MolViewSpec and 3Dmol.js

Mol\* draws the 3D page of `show()`. MolViewSpec and 3Dmol.js draw the single-pose views of `pocketvina.viz`.

```bibtex
@article{sehnal2021molstar,
  title   = {Mol* Viewer: modern web app for 3D visualization and analysis of large biomolecular structures},
  author  = {Sehnal, David and Bittrich, Sebastian and Deshpande, Mandar and Svobodov{\'a}, Radka and Berka, Karel and Bazgier, V{\'a}clav and Velankar, Sameer and Burley, Stephen K and Ko{\v{c}}a, Jaroslav and Rose, Alexander S},
  journal = {Nucleic Acids Research},
  volume  = {49},
  number  = {W1},
  pages   = {W431--W437},
  year    = {2021},
  doi     = {10.1093/nar/gkab314}
}
```

```bibtex
@article{bittrich2024molviewspec,
  title   = {Describing and Sharing Molecular Visualizations Using the MolViewSpec Toolkit},
  author  = {Bittrich, Sebastian and Midlik, Adam and Varadi, Mihaly and Velankar, Sameer and Burley, Stephen K and Young, Jasmine Y and Sehnal, David and Vallat, Brinda},
  journal = {Current Protocols},
  volume  = {4},
  pages   = {e1099},
  year    = {2024},
  doi     = {10.1002/cpz1.1099}
}
```

```bibtex
@article{rego2015threedmol,
  title   = {3Dmol.js: molecular visualization with WebGL},
  author  = {Rego, Nicholas and Koes, David},
  journal = {Bioinformatics},
  volume  = {31},
  number  = {8},
  pages   = {1322--1324},
  year    = {2015},
  doi     = {10.1093/bioinformatics/btu829}
}
```

### PoseBusters

PoseBusters checks whether the docked poses are physically valid.

```bibtex
@article{buttenschoen2024posebusters,
  title   = {PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences},
  author  = {Buttenschoen, Martin and Morris, Garrett M and Deane, Charlotte M},
  journal = {Chemical Science},
  volume  = {15},
  number  = {9},
  pages   = {3130--3139},
  year    = {2024},
  doi     = {10.1039/D3SC04185A}
}
```

### RDKit and Open Babel

RDKit builds the 3D ligands from SMILES and writes the poses as SDF. Open Babel converts the ligands for the engine.

```bibtex
@misc{rdkit,
  title        = {RDKit: Open-source cheminformatics},
  howpublished = {\url{https://www.rdkit.org}},
  doi          = {10.5281/zenodo.591637}
}
```

```bibtex
@article{oboyle2011openbabel,
  title   = {Open Babel: An open chemical toolbox},
  author  = {O'Boyle, Noel M and Banck, Michael and James, Craig A and Morley, Chris and Vandermeersch, Tim and Hutchison, Geoffrey R},
  journal = {Journal of Cheminformatics},
  volume  = {3},
  pages   = {33},
  year    = {2011},
  doi     = {10.1186/1758-2946-3-33}
}
```

## Licences

The package and the data of the PocketVina Atlas have separate licences.

### PocketVina (this package)

PocketVina is released under the [MIT licence](https://github.com/BIMSBbioinfo/PocketVina/blob/main/LICENSE). The
docking engine is based on AutoDock Vina and QuickVina2-GPU-2.1 and keeps their Apache License 2.0. The notices are
in the source files. The bundled Mol\* viewer is MIT licensed, with its licence files in
`pocketvina/viewer/molstar-pv-5.13.1/`.

### PocketVina Atlas (the data)

The data that `pocketvina.Atlas` reads from the [PocketVina Atlas](atlas.md) has its own licence:

The atlas's own results (poses, scores, pockets and PoseBusters checks) are under the
[MIT licence](https://opensource.org/license/mit). Data from other resources keeps its licence: AlphaFold DB models
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), UniProt names and annotation
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), ChEMBL data, including the evidence flags derived from it
([CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)), and Open Targets diseases
([CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/)). The compound names and SMILES come from the vendor
library; their redistribution terms are being checked.

Copyright (c) 2026 Akalin Lab, Max Delbrück Center.

The licence is also on the [Atlas website](https://pocketvina-atlas.mdc-berlin.de/downloads/#licence).

## Acknowledgements

We thank these projects for making their code open source:

- [QuickVina2-GPU-2.1](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1/tree/main/QuickVina2-GPU-2.1)
- [P2Rank](https://github.com/rdk/p2rank)
- [Mol\*](https://github.com/molstar/molstar) and [MolViewSpec](https://github.com/molstar/mol-view-spec)
- [3Dmol.js](https://github.com/3dmol/3Dmol.js) and [py3Dmol](https://pypi.org/project/py3Dmol/)
- [PoseBusters](https://github.com/maabuu/posebusters)
- [RDKit](https://github.com/rdkit/rdkit) and [Open Babel](https://github.com/openbabel/openbabel)

## About these pages

These pages are written in Markdown and built with [Sphinx](https://www.sphinx-doc.org),
[MyST-Parser](https://github.com/executablebooks/MyST-Parser), the [Furo](https://github.com/pradyunsg/furo) theme
and [sphinx-click](https://github.com/click-contrib/sphinx-click). They are published with GitHub Pages at
<https://bioinformatics.mdc-berlin.de/PocketVina/>.
