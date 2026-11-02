"""Build hook of the pocketvina wheel (its tag); the project metadata is in pyproject.toml.

``pip install .`` only copies files: nothing is downloaded, compiled or installed besides the Python package. The
docking engine ``pocketvina/PocketVina-GPU`` is a prebuilt binary shipped as package data, with the OpenCL kernel
sources it compiles at first use (into a per-user cache, never the package). Rebuilding the engine is explicit:
``make -C pocketvina`` before installing (see pocketvina/Makefile), or the conda package pocketvina-engine. P2Rank is
downloaded at first use or by ``pocketvina setup p2rank``; Java 17+ comes from the system or conda.

Since the wheel holds that binary it is not tagged ``py3-none-any`` but ``py3-none-linux_<arch>``, with the
architecture read from the binary's ELF header: the code is pure Python, so one wheel serves every Python 3 on that
platform (it is not a manylinux wheel: the engine needs a libOpenCL.so.1 and a glibc/libstdc++ as new as the build
machine's). Built without the binary it is a pure ``py3-none-any`` wheel, for an engine from conda or
``$POCKETVINA_ENGINE``.
"""
import os
import sysconfig

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pocketvina", "PocketVina-GPU")
ELF_MACHINES = {62: "x86_64", 183: "aarch64"}  # ELF e_machine -> wheel platform architecture


def engine_platform(path: str) -> str:
    """Wheel platform tag of the engine binary at ``path``: ``linux_<arch>`` for a known ELF architecture, else the
    platform of this build."""
    with open(path, "rb") as f:
        head = f.read(20)
    if len(head) == 20 and head[:4] == b"\x7fELF":
        arch = ELF_MACHINES.get(int.from_bytes(head[18:20], "little" if head[5] == 1 else "big"))
        if arch:
            return f"linux_{arch}"
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


class BdistWheel(bdist_wheel):
    """Tag ``py3-none-<platform of the engine binary>`` (``--plat-name`` overrides it), or ``py3-none-any`` without
    the binary. The files stay at the wheel's root (installed into purelib): only the tag changes."""

    def get_tag(self):
        if not os.path.isfile(ENGINE):
            return "py3", "none", "any"
        plat = super().get_tag()[2] if self.plat_name_supplied else engine_platform(ENGINE)
        return "py3", "none", plat


setup(cmdclass={"bdist_wheel": BdistWheel})
