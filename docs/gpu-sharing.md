# GPUs and GPU sharing

A run uses one GPU, the first visible one. With SLURM, this is the GPU assigned to the job. PocketVina docks as many
pockets at the same time as the free GPU memory holds, about 6.4 GB each: 3 on a 24 GB Tesla P40 and 5 on a 40 GB
A100. The other pockets wait for a free slot.

To use several GPUs, run one job per GPU, for example a SLURM array over parts of the protein list ([HPC and SLURM](hpc.md)).
Alternatively, `--gpus all` uses every visible GPU in one run.

## Options

| `pocketvina dock` | `PocketVinaConfig` | Meaning | Default |
| --- | --- | --- | --- |
| `--gpus 1` | `gpus="1"` | the GPU, as an `nvidia-smi` index or UUID. `all` or `0,2` for several | the first visible GPU |
| `--procs-per-gpu auto` | `procs_per_gpu="auto"` | as many engines per GPU as its free memory holds | `auto` |
| `--procs-per-gpu N` | `procs_per_gpu=N` | N engines per GPU, limited by its memory | |
| `--vram-per-proc MIB` | `vram_per_proc_mib=MIB` | GPU memory one engine needs | `6600` |
| `--ligands-per-call N` | `ligands_per_call=N` | ligands per engine call | `1000` |

`CUDA_VISIBLE_DEVICES` is respected, and `--gpus` selects among the visible GPUs. An engine that runs out of GPU
memory is retried with one engine less on that GPU. Each engine also keeps its ligands in RAM, about 0.8 GB for
1000 ligands. The results do not depend on the number of GPUs or engines, so a stopped run can resume on other GPUs.

## Sharing a GPU

One engine keeps a GPU busy most of the time. A second engine fills the gaps, at the cost of another 6.4 GB of GPU
memory. These numbers were measured on a Tesla P40 with `thread=8000` and the 1fbl example:

| Engines per GPU | Throughput | GPU memory |
| --- | --- | --- |
| 1 | 1.00 × | 6.4 GB |
| 2 | 1.10 × | 12.8 GB |
| 3 | 1.12 × | 19.2 GB |

The gain depends on the GPU, the box, `thread` and the ligands. You can measure it on your own GPU before a large
run:

```bash
pocketvina bench gpu --protein example/proteins/1fbl.pdb --ligands example/ligands.txt \
    --n-ligands 100 --procs 1,2,3 --out bench
```

This runs 1, 2 and 3 engines at the same time on one GPU, each docking the same 100 ligands into pocket 1. The
results are printed and saved in `bench/bench_gpu.csv`. The column `ligands_per_hour` gives the total throughput,
`speedup` the gain over one engine and `peak_vram_mib` the GPU memory of one engine.
