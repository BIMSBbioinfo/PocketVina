"""GPUs for the docking engine: the NVIDIA GPUs this process may use (``list_gpus``) and how many engine processes run
on each of them at the same time (``plan_slots``).

Each engine process ("slot") is pinned to one GPU through its own ``CUDA_VISIBLE_DEVICES``, which NVIDIA's OpenCL
honours like CUDA does. One engine at the default ``thread=8000`` holds about 6.4 GB of GPU memory (measured on a
Tesla P40; more threads need more), hence ``DEFAULT_VRAM_PER_PROC_MIB`` = 6600 MiB. Several engines on one GPU share
its compute units: that pays off when one engine leaves the GPU partly idle (e.g. during the CPU-side work of each
call), which depends on the GPU and the workload.

Standard library only (``nvidia-smi`` for the GPU list), so the "files" output format still docks without pyarrow.
"""
import logging
import numbers
import os
import shutil
import subprocess
import threading
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Union

logger = logging.getLogger(__name__)

DEFAULT_VRAM_PER_PROC_MIB = 6600  # 6.4 GB measured at thread=8000 on a Tesla P40, plus headroom
DEFAULT_MARGIN_MIB = 1024         # left free on every GPU by procs_per_gpu="auto"
AUTO = "auto"
ALL_GPUS = "all"
NVIDIA_SMI_QUERY = ("nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free",
                    "--format=csv,noheader,nounits")
NVIDIA_SMI_TIMEOUT_S = 30
NVIDIA_SMI_APPS = ("nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits")
MEMORY_POLL_S = 1.0  # how often MemorySampler measures
# what the engine prints (OpenCL/src/wrapcl.cpp checkErr: "Err<code>:<name>") when a GPU has no memory left for it
OUT_OF_MEMORY_ERRORS = ("CL_MEM_OBJECT_ALLOCATION_FAILURE", "CL_OUT_OF_RESOURCES", "CL_OUT_OF_HOST_MEMORY")
# what it prints when OpenCL cannot set up its GPU at all (lib/main_procedure_cl.cpp does that after parsing the
# ligands, before docking them): no platform or GPU (CUDA_VISIBLE_DEVICES names none OpenCL sees), the GPU not
# available (e.g. busy in an exclusive compute mode), or no context on it (wrapcl.cpp SetupContext prints the code,
# e.g. -5 with no memory left for one, then fails on the NULL context with CL_INVALID_CONTEXT)
DEVICE_ERRORS = ("CL_PLATFORM_NOT_FOUND_KHR", "CL_DEVICE_NOT_FOUND", "CL_DEVICE_NOT_AVAILABLE",
                 "Fail to create context!")


class GPU(NamedTuple):
    """One GPU as ``nvidia-smi`` reports it; the memory fields are None when it reports none (``[N/A]``)."""

    index: int
    uuid: str
    name: str
    memory_total_mib: Optional[int]
    memory_free_mib: Optional[int]


def _mib(text: str) -> Optional[int]:
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_gpu_list(text: str) -> List[GPU]:
    """GPUs from the output of ``NVIDIA_SMI_QUERY`` (lines it cannot read are skipped)."""
    gpus = []
    for line in text.splitlines():
        head = [p.strip() for p in line.split(",", 2)]
        if len(head) != 3:
            continue
        tail = [p.strip() for p in head[2].rsplit(",", 2)]  # a GPU name may hold a comma
        if len(tail) != 3 or not head[0].isdigit():
            continue
        gpus.append(GPU(int(head[0]), head[1], tail[0], _mib(tail[1]), _mib(tail[2])))
    return gpus


def _matches(entry: str, gpu: GPU) -> bool:
    """``entry`` (a ``CUDA_VISIBLE_DEVICES`` item) names ``gpu``: its index, or its UUID or a prefix of it."""
    if entry.isdigit():
        return int(entry) == gpu.index
    return entry.startswith("GPU-") and gpu.uuid.startswith(entry)


def same_numbering(gpus: Sequence[GPU], device_order: Optional[str]) -> bool:
    """CUDA numbers ``gpus`` (all that ``nvidia-smi`` lists) as ``nvidia-smi`` does: with
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, or when they are all of one model (CUDA's default order, ``FASTEST_FIRST``,
    then keeps their PCI bus order, which ``nvidia-smi`` uses)."""
    return (device_order or "").strip().upper() == "PCI_BUS_ID" or len({g.name for g in gpus}) <= 1


def visible_gpus(gpus: Sequence[GPU], visible_devices: Optional[str],
                 device_order: Optional[str] = None) -> List[GPU]:
    """The GPUs of ``gpus`` that ``CUDA_VISIBLE_DEVICES=visible_devices`` leaves visible, in its order.

    None (unset) leaves all of them. Entries are (prefixes of) UUIDs or CUDA device numbers, which are ``nvidia-smi``
    indices when ``same_numbering(gpus, device_order)`` (``device_order``: ``$CUDA_DEVICE_ORDER``); otherwise (GPUs of
    different models) the GPU a number names is not known here, and the list ends before it. As in CUDA, the list
    ends at the first entry that names no GPU (``-1``, ``NoDevFiles``), and an empty value hides every GPU. Inside a
    SLURM job that confines GPUs with cgroups, ``nvidia-smi`` lists only the job's GPUs, numbered from 0 like
    ``CUDA_VISIBLE_DEVICES``.
    """
    if visible_devices is None:
        return list(gpus)
    numbers_ok = same_numbering(gpus, device_order)
    chosen = []
    for entry in visible_devices.split(","):
        entry = entry.strip()
        if entry.isdigit() and not numbers_ok:
            logger.warning("CUDA_VISIBLE_DEVICES=%s: CUDA's GPU %s need not be nvidia-smi's (GPUs of different "
                           "models and CUDA_DEVICE_ORDER is not PCI_BUS_ID), so neither it nor the entries after it "
                           "are matched; with no GPU matched the engines get CUDA_VISIBLE_DEVICES as it is (name GPUs "
                           "by UUID or set CUDA_DEVICE_ORDER=PCI_BUS_ID to dock on several)", visible_devices, entry)
            break
        found = [g for g in gpus if _matches(entry, g)] if entry else []
        if len(found) != 1:
            break
        if found[0] not in chosen:
            chosen.append(found[0])
    return chosen


def list_gpus() -> List[GPU]:
    """The NVIDIA GPUs this process may use: ``nvidia-smi`` filtered by ``$CUDA_VISIBLE_DEVICES`` (see
    ``visible_gpus``; ``$CUDA_DEVICE_ORDER`` tells how it numbers them). [] without ``nvidia-smi`` or when it fails
    (e.g. a container with the driver's OpenCL library but not ``nvidia-smi``: the engine then uses the first GPU of
    NVIDIA's OpenCL platform; it docks on NVIDIA GPUs only).
    """
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        out = subprocess.run(list(NVIDIA_SMI_QUERY), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                             timeout=NVIDIA_SMI_TIMEOUT_S, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("no GPU list from nvidia-smi: %s", e)
        return []
    return visible_gpus(parse_gpu_list(out), visible_devices, os.environ.get("CUDA_DEVICE_ORDER"))


def check_gpu_spec(value) -> Optional[str]:
    """``value`` (None, ``"all"``, an index, a comma-separated string of ``nvidia-smi`` indices/UUIDs, or a list of
    them) as a normalized comma-separated string (or ``"all"``); ``ValueError`` for an empty or malformed one."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == ALL_GPUS:
        return ALL_GPUS
    if isinstance(value, bool):
        raise ValueError(f"gpus must be GPU indices or UUIDs (e.g. \"0,1\") or None, got {value!r}")
    if isinstance(value, numbers.Integral):
        items = [str(int(value))]
    elif isinstance(value, str):
        items = [v.strip() for v in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(int(v)) if isinstance(v, numbers.Integral) and not isinstance(v, bool) else v for v in value]
        if not all(isinstance(v, str) for v in items):
            raise ValueError(f"gpus must be GPU indices or UUIDs (e.g. \"0,1\") or None, got {value!r}")
        items = [v.strip() for v in items]
    else:
        raise ValueError(f"gpus must be GPU indices or UUIDs (e.g. \"0,1\") or None, got {value!r}")
    if not items or any(not (v.isdigit() or (v.startswith("GPU-") and len(v) > 4)) for v in items):
        raise ValueError(f"gpus must be nvidia-smi GPU indices or UUIDs (GPU-...), comma-separated, got {value!r}")
    if len(set(items)) != len(items):
        raise ValueError(f"gpus names a GPU twice: {value!r} (procs_per_gpu sets the processes per GPU)")
    return ",".join(items)


def select_gpus(gpus: Sequence[GPU], spec: Optional[str]) -> List[GPU]:
    """The GPUs that ``spec`` (``check_gpu_spec``) names, in its order: None the first of ``gpus`` (one GPU per run,
    e.g. the one SLURM gives the job), ``"all"`` all of them; ``ValueError`` if it names one that is not in ``gpus``
    (not visible) or the same GPU twice."""
    spec = check_gpu_spec(spec)
    if spec is None:
        return list(gpus[:1])
    if spec == ALL_GPUS:
        return list(gpus)
    chosen = []
    for entry in spec.split(","):
        found = [g for g in gpus if _matches(entry, g)]
        if len(found) != 1:
            visible = ", ".join(f"{g.index} ({g.uuid})" for g in gpus) or "none"
            raise ValueError(f"gpus={spec!r}: {entry!r} is not {'a single' if found else 'a'} visible GPU "
                             f"(visible: {visible})")
        if found[0] in chosen:
            raise ValueError(f"gpus={spec!r} names GPU {found[0].index} twice")
        chosen.append(found[0])
    return chosen


def check_procs_per_gpu(value) -> Union[int, str]:
    """``value`` as an int >= 1 or ``"auto"``; ``ValueError`` otherwise."""
    if isinstance(value, str) and value.strip().lower() == AUTO:
        return AUTO
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f'procs_per_gpu must be an integer >= 1 or "auto", got {value!r}')
    return int(value)


def _check_mib(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum} (MiB), got {value!r}")
    return int(value)


def fitting_procs(gpu: GPU, vram_per_proc_mib: int, margin_mib: int = DEFAULT_MARGIN_MIB) -> Optional[int]:
    """Engine processes of ``vram_per_proc_mib`` that fit in the GPU's free memory less ``margin_mib`` (at least
    1); None when ``nvidia-smi`` reports no free memory figure."""
    if gpu.memory_free_mib is None:
        return None
    return max(1, (gpu.memory_free_mib - margin_mib) // vram_per_proc_mib)


def device_id(gpu: GPU) -> str:
    """The ``CUDA_VISIBLE_DEVICES`` value that selects ``gpu``: its UUID (the same whatever the device order), else
    its index."""
    return gpu.uuid if gpu.uuid.startswith("GPU-") else str(gpu.index)


def plan_slots(gpus: Sequence[GPU], procs_per_gpu: Union[int, str] = 1,
               vram_per_proc_mib: int = DEFAULT_VRAM_PER_PROC_MIB, margin_mib: int = DEFAULT_MARGIN_MIB) -> List[str]:
    """One entry per engine slot, the ``CUDA_VISIBLE_DEVICES`` value of its engine (``device_id``: the GPU's UUID).

    ``procs_per_gpu`` an int: that many slots per GPU, but no more than its free memory holds at
    ``vram_per_proc_mib`` each (keeping ``margin_mib`` free; always at least 1). ``"auto"``: as many as fit
    (``fitting_procs``; 1 when ``nvidia-smi`` reports no free memory). Slots alternate between the GPUs
    (``[g0, g1, g0, g1, ...]``), so that fewer tasks than slots still spread over all GPUs. [] for no GPUs.
    """
    procs = check_procs_per_gpu(procs_per_gpu)
    vram = _check_mib("vram_per_proc_mib", vram_per_proc_mib, 1)
    margin = _check_mib("margin_mib", margin_mib, 0)
    counts: Dict[str, int] = {}
    for g in gpus:
        if device_id(g) in counts:
            continue
        fit = fitting_procs(g, vram, margin)
        n = (fit or 1) if procs == AUTO else min(procs, fit or procs)
        if procs != AUTO and n < procs:
            logger.warning("GPU %d (%s): %s MiB free hold only %d engine processes of %d MiB (procs_per_gpu=%d)",
                           g.index, g.name, g.memory_free_mib, n, vram, procs)
        counts[device_id(g)] = n
    slots = []
    for round_no in range(max(counts.values(), default=0)):
        slots += [uuid for uuid, n in counts.items() if n > round_no]
    return slots


def is_out_of_memory(log_text: str) -> bool:
    """The engine log (tail) reports an OpenCL out-of-memory error (``OUT_OF_MEMORY_ERRORS``)."""
    return any(name in log_text for name in OUT_OF_MEMORY_ERRORS)


def is_device_error(log_text: str) -> bool:
    """The engine log (tail) reports that OpenCL could not set up the GPU (``DEVICE_ERRORS``): a problem of the GPU
    or of the engine's ``CUDA_VISIBLE_DEVICES``, whatever ligands the call docks."""
    return any(name in log_text for name in DEVICE_ERRORS)


def opencl_hint(log_text: str, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """What to try when the engine found no OpenCL platform with ``$OCL_ICD_VENDORS`` set (the v0.1 README set it):
    the CUDA toolkit's ``libOpenCL.so.1`` then misses the GPU that ``/etc/OpenCL/vendors`` names; None otherwise,
    also when ``environ`` is None (this process) and the engine already runs without the variable (its retry
    without it, ``engine.run_engine``, found a platform once)."""
    from . import engine

    if environ is None and engine.ocl_vendors_dropped():
        return None
    vendors = (os.environ if environ is None else environ).get(engine.OCL_VENDORS_ENV)
    if not vendors or engine.NO_PLATFORM_ERROR not in log_text:
        return None
    return (f"OCL_ICD_VENDORS is set ({vendors}): if `clinfo -l` lists the GPU, unset it (`unset OCL_ICD_VENDORS`, "
            "and remove it from ~/.bashrc), see Troubleshooting in docs/installation.md")


def _rss_mib(pid) -> Optional[float]:
    """Resident memory of a process in MiB (Linux ``/proc``), or None."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class MemorySampler:
    """Peak memory while engines run, measured every ``interval`` seconds in a thread: the GPU memory of the engines
    that ``pids()`` names (nvidia-smi: on the GPU that holds the most, per engine, and on all GPUs together) and the
    host memory (RSS) of this process plus those engines (Linux ``/proc``). ``track()`` starts a record of the peaks
    (a dict that every later measurement updates) and ``untrack`` ends it; ``run`` is the record of the sampler's
    whole time. A value stays None when it could not be measured (no nvidia-smi or ``/proc``, or no engine running
    at a measurement)."""

    def __init__(self, pids, interval: float = MEMORY_POLL_S):
        self.pids, self.interval = pids, interval
        self._smi = shutil.which(NVIDIA_SMI_APPS[0]) is not None
        self._records: List[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="pocketvina-memory", daemon=True)
        self.run = self.track()

    def start(self) -> "MemorySampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def track(self) -> dict:
        record = dict(gpu_mib=None, gpu_one_mib=None, gpu_engine_mib=None, gpus=0, host_mib=None, engines=0)
        with self._lock:
            self._records.append(record)
        return record

    def untrack(self, record: dict) -> None:
        with self._lock:  # by identity: records with the same values are still different records
            if record is not self.run:
                self._records = [r for r in self._records if r is not record]

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
            except Exception as e:  # measuring must never stop a run
                logger.debug("memory sample failed: %s", e)

    def _gpu_mib(self, pids) -> Dict[int, tuple]:
        """{pid: (GPU UUID, MiB)} of the engines in ``pids`` that nvidia-smi lists."""
        if not self._smi or not pids:
            return {}
        try:
            out = subprocess.run(list(NVIDIA_SMI_APPS), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                 timeout=NVIDIA_SMI_TIMEOUT_S, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return {}
        used = {}
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[1].isdigit() and int(parts[1]) in pids:
                try:
                    used[int(parts[1])] = (parts[0], float(parts[2]))
                except ValueError:  # "[N/A]"
                    continue
        return used

    def sample(self) -> None:
        """One measurement, applied to every record."""
        pids = set(self.pids())
        gpu_used = self._gpu_mib(pids)
        host = [_rss_mib(pid) for pid in [os.getpid(), *pids]]
        host_mib = sum(h for h in host if h is not None) if host[0] is not None else None

        def peak(old, new):
            return new if old is None else old if new is None else max(old, new)

        per_gpu: Dict[str, float] = {}
        for uuid, mib in gpu_used.values():
            per_gpu[uuid] = per_gpu.get(uuid, 0.0) + mib
        with self._lock:
            for r in self._records:
                if gpu_used:
                    r["gpu_mib"] = peak(r["gpu_mib"], sum(per_gpu.values()))
                    r["gpu_one_mib"] = peak(r["gpu_one_mib"], max(per_gpu.values()))
                    r["gpu_engine_mib"] = peak(r["gpu_engine_mib"], max(mib for _, mib in gpu_used.values()))
                    r["gpus"] = max(r["gpus"], len(per_gpu))
                r["host_mib"] = peak(r["host_mib"], host_mib)
                r["engines"] = max(r["engines"], len(pids))


def format_memory(record: dict) -> str:
    """A ``MemorySampler`` record in words: "GPU 6.3 GiB on one GPU at most (4 GPUs, 4 engines of at most 6.3 GiB;
    25.0 GiB on all GPUs together), CPU 1.1 GiB"."""
    def gib(mib):
        return f"{mib / 1024:.1f} GiB"

    def count(n, word):
        return f"{n} {word}{'s' * (n != 1)}"

    parts = []
    if record.get("gpu_mib") is not None:
        n_gpus, n_engines = record.get("gpus") or 1, record.get("engines") or 1
        detail = f"{count(n_gpus, 'GPU')}, {count(n_engines, 'engine')} of at most {gib(record['gpu_engine_mib'])}"
        if n_gpus > 1:
            detail += f"; {gib(record['gpu_mib'])} on all GPUs together"
        parts.append(f"GPU {gib(record['gpu_one_mib'])} on one GPU at most ({detail})")
    else:
        parts.append("GPU not measured")
    parts.append(f"CPU {gib(record['host_mib'])}" if record.get("host_mib") is not None else "CPU not measured")
    return ", ".join(parts)
