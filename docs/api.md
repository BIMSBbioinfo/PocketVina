# Python API

You can import the configuration classes, `PocketVinaGPU`, `P2RankProcessor` and `Results` directly from
`pocketvina`.

## Configuration and runs

```{eval-rst}
.. autoclass:: pocketvina.main.PocketVinaConfig

.. autoclass:: pocketvina.main.P2RankConfig

.. autoclass:: pocketvina.main.PocketConfig

.. autoclass:: pocketvina.main.PocketVinaGPU
   :members: execute_docking, best_scores, show, select, results

.. autoclass:: pocketvina.main.P2RankProcessor
   :members: process_file_list
```

`PocketVinaGPU(config, htvs_mode=True)` docks the 5 best-ranked pockets of each protein if `top_pockets` is not
set. After `execute_docking()`, `failed_proteins` maps each protein that could not be docked to the reason.

```{eval-rst}
.. autofunction:: pocketvina.pipeline.run_atlas

.. autofunction:: pocketvina.pipeline.plan_engine_slots
```

## Results

```{eval-rst}
.. module:: pocketvina.io.results
```

{class}`~pocketvina.io.results.Results` reads a results folder ([Results format](output-format.md)). It sees only the proteins
that are done.

```{eval-rst}
.. autoclass:: pocketvina.io.results.Results
   :members:

.. autofunction:: pocketvina.io.importer.import_legacy
```

## Selections

```{eval-rst}
.. autoclass:: pocketvina.selection.Selection
   :members: show, bust, export, poses, name
```

## PoseBusters

```{eval-rst}
.. automodule:: pocketvina.bust
   :members: bust_results, check_names, require_posebusters, default_workers
```

## Visualisation

```{eval-rst}
.. automodule:: pocketvina.viz
   :members: show, show_modes, to_html, table, MolstarView
```

## GPUs

```{eval-rst}
.. automodule:: pocketvina.gpu
   :members: GPU, list_gpus, select_gpus, plan_slots, fitting_procs
```

## P2Rank and the engine

```{eval-rst}
.. automodule:: pocketvina.p2rank
   :members: p2rank_home, find_p2rank, ensure_p2rank, check_java, P2RankError, JavaError

.. autofunction:: pocketvina.engine.engine_path

.. autofunction:: pocketvina.engine.kernel_cache_dir([visible_devices])

.. autofunction:: pocketvina.engine.resolve_tmp_dir
```
