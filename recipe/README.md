# Conda packages

`recipe.yaml` is a [rattler-build](https://rattler.build) recipe. It builds two conda packages from this checkout:

| Package | Platform | Contents |
| --- | --- | --- |
| `pocketvina-engine` | linux-64 | the docking engine, built by `build_engine.sh` with conda-forge's compiler, Boost and OpenCL |
| `pocketvina` | noarch: python | the Python package, the `pocketvina` command, Java for P2Rank and the engine |

## Build

You need linux-64, internet access to conda-forge and rattler-build (`conda install -c conda-forge rattler-build`).
The build takes a few minutes and tests both packages in fresh environments:

```bash
rattler-build build --recipe recipe/ --output-dir conda-bld
```

## Install

```bash
conda create -n pocketvina -c ./conda-bld -c conda-forge pocketvina
conda activate pocketvina
pocketvina setup p2rank      # once: P2Rank 2.5.1 into ~/.cache/pocketvina
```

If you use pixi, also set `export OCL_ICD_VENDORS=/etc/OpenCL/vendors`. Without `conda activate`, for example in a
Jupyter kernel, set `JAVA_HOME=<env>/lib/jvm` so that P2Rank finds Java.

## Test the recipe

```bash
python -m pytest tests/test_recipe.py
```

This needs PyYAML only. With rattler-build on `PATH`, it also renders the recipe. `context.version` must equal
`pocketvina.__version__`. Keep `libboost-devel` at the Boost of conda-forge's current rdkit builds, or environments
get an old rdkit.
