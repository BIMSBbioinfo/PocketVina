# Contributing to PocketVina

Everyone who takes part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Bugs and questions

Open an [issue](https://github.com/BIMSBbioinfo/PocketVina/issues). Questions can also go to the
[discussions](https://github.com/BIMSBbioinfo/PocketVina/discussions). For a bug, please include:

- the PocketVina version (`pocketvina --version`) and how you installed it: pip, conda, Docker or from source
- the GPU and the driver version, from `nvidia-smi`
- the command or the Python code you ran
- the full log, from a run with `-v` (e.g. `pocketvina -v dock ...`), which adds debug messages and the traceback
  of an error

Report security problems privately, as [SECURITY.md](SECURITY.md) explains, not in an issue.

## Development setup

Fork the repository on GitHub, clone your fork and start a branch from `dev`:

```bash
git clone https://github.com/<your-name>/PocketVina
cd PocketVina
git remote add upstream https://github.com/BIMSBbioinfo/PocketVina
git fetch upstream
git switch -c my-change upstream/dev
```

Install PocketVina in editable mode with the development tools (pytest, the pinned ruff, the notebook runner) and
Meeko, which some tests use:

```bash
conda create -n pocketvina-dev -c conda-forge python=3.11 "openjdk>=17" -y
conda activate pocketvina-dev
pip install -e ".[dev,meeko]"
```

The code must run on Python 3.9 (CI tests 3.9 to 3.12): no `match` statements, and `Optional[int]` instead of
`int | None`. Please ask in an issue before you add a dependency.

The engine's sources (C++ and OpenCL) are in `pocketvina/main`, `pocketvina/lib` and `pocketvina/OpenCL`.
[Building the engine](docs/installation.md#building-the-engine) shows how to compile it.

## Tests

Most tests need no GPU. They replace the docking engine with `tests/fake_engine.py`, a Python script that takes the
engine's command line and writes output files in the engine's format. CI runs them with:

```bash
python -m pytest tests -m "not gpu"
```

Tests marked `gpu` need an NVIDIA GPU and the real engine. They are skipped unless you ask for them:

```bash
POCKETVINA_GPU_TESTS=1 python -m pytest tests -m gpu
```

Tests marked `network` read the public PocketVina Atlas and run only with `POCKETVINA_NETWORK_TESTS=1`.

## Lint

CI checks the code with ruff 0.17.3, the version the `dev` extra installs (newer versions add rules):

```bash
ruff check pocketvina/io pocketvina/*.py tests
```

## Docs

The docs are the Markdown pages in `docs/`, built with Sphinx. Build them as CI does, with warnings as errors:

```bash
pip install -r docs/requirements.txt
sphinx-build -W -b html docs docs/_build/html
```

Then open `docs/_build/html/index.html`.

## Pull requests

Open pull requests against the `dev` branch, not `main`. Add an entry for your change to
[docs/changelog.md](docs/changelog.md) under `## Not released yet`, in the matching part (Added, Changed, Fixed or
Removed), written for users. If that section is not there yet, add it at the top. If you change what the code does,
add or update a test for it. For every pull request, CI runs ruff and the tests (on Python 3.9 to 3.12). It also
builds the docs when the docs or the code change.

By contributing, you agree that your contribution is released under the same licence as the files it changes (see
the License section of the [README](README.md#license)).
