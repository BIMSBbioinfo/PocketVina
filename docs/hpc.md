# HPC and SLURM

PocketVina runs on any cluster with Python, Java and NVIDIA GPUs. The examples use SLURM, but the same steps work
with other schedulers.

## Set up once

### Install PocketVina and P2Rank

Install PocketVina into an environment on a shared filesystem ([Installation](installation.md)). Then download P2Rank on a
login node with internet access:

```bash
pocketvina setup p2rank          # P2Rank 2.5.1 into ~/.cache/pocketvina, and a check of Java 17+
```

The compute nodes then find P2Rank in your home directory. There are two alternatives:

- For a group, install P2Rank into a shared folder with `pocketvina setup p2rank --dir /shared/tools`, and set
  `export P2RANK_HOME=/shared/tools/p2rank-2.5.1` in the job scripts.
- Without internet access, download `p2rank_2.5.1.tar.gz` from the
  [P2Rank releases](https://github.com/rdk/p2rank/releases) on another computer and run
  `pocketvina setup p2rank --from p2rank_2.5.1.tar.gz`.

### Check Java

P2Rank needs Java 17 or later. If the default Java of your cluster is older, load a Java 17 module in the job script.
You can check what a job will use:

```bash
pocketvina setup p2rank --check
```

### GPU code cache

The engine compiles its GPU code once per GPU model and driver into `~/.cache/pocketvina/kernels/`, which nodes
with the same GPUs share. If the compute nodes cannot write to your home directory, set `POCKETVINA_KERNEL_CACHE` to
another folder. Use one such folder per GPU model, and empty it after you update PocketVina.

## Temp files

Temp files go to the first of `--tmp-dir`, `$POCKETVINA_TMPDIR`, `$SLURM_TMPDIR`, `$TMPDIR` and `/tmp`. They are
removed at the end, also when a job is cancelled. Keep them on the node's local disk. The results folder itself can
be on a network filesystem. If your cluster has local scratch elsewhere, point PocketVina to it:

```bash
pocketvina dock ... --tmp-dir /local/scratch/$USER     # or: export POCKETVINA_TMPDIR=/local/scratch/$USER
```

## One job

This job docks all proteins on one GPU:

```bash
#!/bin/bash
#SBATCH --job-name=pocketvina
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source ~/envs/pocketvina/bin/activate      # or: conda activate pocketvina
pocketvina dock --proteins proteins.txt --ligands ligands.csv --out results \
    --top-pockets 5 --engine-timeout 7200
```

The CPUs prepare the ligands and run P2Rank (`--p2rank-threads`, default 4). `--engine-timeout` stops an engine call
that hangs. Its ligands are marked `engine_error`, and the run goes on.

## Array jobs

Several jobs can write different proteins into the same results folder at the same time. This is the simplest way
to dock on many GPUs.

### Split the protein list

Keep the parts in the folder of the list, because relative paths in a list are relative to its folder:

```bash
split -n l/20 -d -a 3 proteins.txt proteins.part.     # proteins.part.000 ... proteins.part.019
```

### Submit the array

Every task docks one part. All tasks must use the same ligands and settings:

```bash
#!/bin/bash
#SBATCH --job-name=pocketvina
#SBATCH --array=0-19%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00

source ~/envs/pocketvina/bin/activate
part=$(printf "proteins.part.%03d" "$SLURM_ARRAY_TASK_ID")
pocketvina dock --proteins "$part" --ligands ligands.csv --out results --top-pockets 5
```

The first task prepares the ligands, and later tasks reuse them. Do not use `--force` in array tasks. It replaces
the run that the other tasks write into.

## Resume and check

To resume, submit the same job or array again. Proteins that are done are skipped. A cancelled job cleans up after
itself, so `#SBATCH --requeue` is safe.

To see how far a run is:

```bash
pocketvina info --results results --proteins proteins.txt
```

This shows the settings, the counts, the proteins with problems and the proteins of the list that are not done yet.
If `pocketvina dock` exits with code 2, some proteins failed, and the log names them. You can read the results while
jobs are still running.

## Size a run

- Docking time grows with proteins × pockets × ligands. `--top-pockets` is the main lever.
- `--ligands-per-call` sets the ligands per engine call, 1000 by default. Larger calls are a bit faster but need
  more RAM, about 0.8 MB per ligand, and lose more work if a call fails.
- A second engine per GPU adds about 10 % on a Tesla P40 ([GPUs and GPU sharing](gpu-sharing.md)).
- Most of the disk space goes to the pose files, which keep every pose. The tables are small.
