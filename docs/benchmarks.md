# Benchmark results

All benchmark data and results of the paper, including TargetDock-AI, are on
[Zenodo](https://zenodo.org/records/15733460).

To reproduce a benchmark, go into its folder under `benchmark/` and run `main.py`. The folders are `astex`,
`dockgen`, `pdbbind` and `posebuster`:

```bash
cd benchmark/astex
python main.py
```

The published results were computed with PocketVina 0.1. To use its ligand preparation, set
`ligand_prep="legacy"` in `PocketVinaConfig`.
