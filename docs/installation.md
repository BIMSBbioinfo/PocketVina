# Installation

## Install

Install PocketVina from PyPI into a new conda environment:

```bash
conda create -n pocketvina -c conda-forge python=3.11 "openjdk>=17" -y
conda activate pocketvina
pip install pocketvina
```

This installs PocketVina with everything it uses, including the docking engine. P2Rank, the pocket finder, is
downloaded into `~/.cache/pocketvina` at the first run. If you already have Python and Java 17 or later,
`pip install pocketvina` alone is enough. Java is needed for P2Rank.

### From source

To use the latest code, or to have the example files of the [quick start](quickstart.md), install from the repository:

```bash
git clone https://github.com/BIMSBbioinfo/PocketVina
cd PocketVina
conda create -n pocketvina -c conda-forge python=3.11 "openjdk>=17" -y
conda activate pocketvina
pip install .
```

If you are upgrading from PocketVina 0.1, run `pip uninstall pocketvina-gpu` first. See [Migrating from PocketVina 0.1](migrating.md).

## Requirements

You need:

- Linux on x86_64 with an NVIDIA GPU and its driver. The CUDA toolkit is not needed.
- The OpenCL loader `libOpenCL.so.1`, from the package `ocl-icd-libopencl1` on Debian and Ubuntu.
- The file `/etc/OpenCL/vendors/nvidia.icd`, which points OpenCL to the NVIDIA driver.
- Python 3.9 or later. The Mol\* views of `pocketvina.viz` need Python 3.10 or later.
- Java 17 or later, for P2Rank.
- For the prebuilt engine, Ubuntu 22.04, Debian 12, RHEL 9 or a newer system. Google Colab works. On older systems,
  use the [conda packages](#conda-packages) or [build the engine](#building-the-engine).

`clinfo -l`, from the package `clinfo`, must list an NVIDIA platform. If it lists none while `nvidia-smi` works,
create the ICD file:

```bash
sudo mkdir -p /etc/OpenCL/vendors
echo libnvidia-opencl.so.1 | sudo tee /etc/OpenCL/vendors/nvidia.icd
```

## Check the installation

These commands show whether PocketVina, P2Rank, Java and the GPU are ready:

```bash
pocketvina --version
pocketvina setup p2rank --check     # the P2Rank and Java that runs use
nvidia-smi                          # the GPUs
clinfo -l                           # OpenCL must list the NVIDIA GPUs too
```

The first run on a GPU compiles the engine's GPU code. This takes a few seconds, once.

## Google Colab

Open the
[Colab notebook](https://colab.research.google.com/github/BIMSBbioinfo/PocketVina/blob/main/notebooks/quickstart_colab.ipynb).
Choose **Runtime → Change runtime type → T4 GPU**, then **Run all**. The first cell installs Java, OpenCL,
PocketVina and P2Rank. After a runtime restart, it skips the steps that are done.

## Docker

The image holds PocketVina, Java, OpenCL, P2Rank and the web UI. The host needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

```bash
cd PocketVina
docker build --network=host -t pocketvina .
docker run --gpus all --rm pocketvina pocketvina --help
# dock the examples of the checkout: the results are written into the current folder
docker run --gpus all --rm -v "$PWD:/work" -w /work pocketvina \
    pocketvina dock --proteins example/proteins --ligands example/ligands.txt --out results --top-pockets 2
# the web UI on port 7860 (app/README.md)
docker run --gpus all -it -p 7860:7860 pocketvina python3 app/app.py
```

## pip without conda

You can also install PocketVina into a plain virtual environment:

```bash
python -m venv .venv && source .venv/bin/activate
pip install pocketvina
pocketvina setup p2rank --check      # the Java and P2Rank that runs use
```

Java 17 or later must then come from your system.

You can add optional extras, for example with `pip install ".[meeko]"`. The extras are `meeko` for ligands prepared
with Meeko, `app` for the web UI in `app/`, `dev` for the tests and `docs` for these pages.

## conda packages

There is no conda channel with PocketVina yet. Build the packages from `recipe/` with
[rattler-build](https://github.com/prefix-dev/rattler-build) and install them from the local folder:

```bash
conda create -n build-tools -c conda-forge rattler-build
conda activate build-tools
cd PocketVina
rattler-build build --recipe recipe/ --output-dir "$HOME/pocketvina-conda"
conda create -n pocketvina -c "$HOME/pocketvina-conda" -c conda-forge pocketvina ocl-icd-system
conda activate pocketvina
```

- The engine is built with conda-forge's libraries, so it also runs on older systems.
- The conda packages need Python 3.10 or later.
- Do not run `pip install .` in this environment. It would replace the conda packages.
- Start Python outside the `PocketVina` folder. Inside it, `import pocketvina` loads the checkout, not the package.
- pixi skips a setup step of `ocl-icd-system`. There, set `OCL_ICD_VENDORS=/etc/OpenCL/vendors`.

`recipe/README.md` has the details.

## Building the engine

`pip install .` ships the prebuilt engine `pocketvina/PocketVina-GPU`. You can build your own, for an older system
or changed code. Run `make` before `pip install .`, here with the compiler, Boost and OpenCL of conda-forge:

```bash
conda create -n pocketvina-src -c conda-forge python=3.11 cxx-compiler make "libboost-devel<1.85" \
    ocl-icd ocl-icd-system opencl-headers "openjdk>=17"
conda activate pocketvina-src
cd PocketVina
make -C pocketvina BOOST_ROOT="$CONDA_PREFIX" OPENCL_ROOT="$CONDA_PREFIX"
pip install .
```

- Boost 1.85 and later lack headers the engine needs, hence `libboost-devel<1.85`.
- The Makefile also reads `PREFIX`, `CXX`, `CXXFLAGS` and `LDFLAGS`. `make -C pocketvina print` shows its settings.
- `make -C pocketvina install PREFIX=<prefix>` installs the engine as `<prefix>/bin/pocketvina-engine`.

PocketVina looks for the engine in this order:

1. `$POCKETVINA_ENGINE`, a path or a command name
2. the binary in the package, `pocketvina/PocketVina-GPU`
3. `pocketvina-engine` on `PATH` or next to the Python interpreter

## Troubleshooting

### The engine finds no GPU

The engine stops with `CL_PLATFORM_NOT_FOUND_KHR` when OpenCL does not see the NVIDIA driver. Check these causes:

- `OCL_ICD_VENDORS` is set. Run `unset OCL_ICD_VENDORS` and remove the line from `~/.bashrc`.
- `clinfo -l` lists no GPU. Create `/etc/OpenCL/vendors/nvidia.icd` as shown under [Requirements](#requirements).
- conda's OpenCL loader `ocl-icd` is installed without `ocl-icd-system`. Install `ocl-icd-system`.
- In Docker, the container was started without `--gpus all` or without the NVIDIA Container Toolkit.

### GLIBC_2.34 not found

The message `version 'GLIBC_2.34' not found`, or `GLIBCXX_3.4.29`, means that your system is older than the
prebuilt engine needs. Use the [conda packages](#conda-packages) or [build the engine](#building-the-engine).

### P2Rank does not start

P2Rank needs Java 17 or later. It runs `$JAVA_HOME/bin/java` if `JAVA_HOME` is set, and `java` on `PATH`
otherwise. Install Java with `conda install -c conda-forge "openjdk>=17"`, or load a Java module.
`pocketvina setup p2rank --check` shows which Java a run uses.

### No internet on the compute nodes

Run `pocketvina setup p2rank` once on a login node. See [HPC and SLURM](hpc.md).

See the [FAQ](faq.md) for more answers.
