import os

from .main import P2RankConfig, P2RankProcessor, Paths, PocketConfig, PocketVinaConfig, PocketVinaGPU

__version__ = "0.3.0"
# not run_atlas or Results (see __getattr__): `from pocketvina import *` would import pyarrow, which the "files"
# format lacks
__all__ = ["PocketVinaGPU", "P2RankProcessor", "PocketVinaConfig", "P2RankConfig", "PocketConfig", "Paths"]

# Paths that importing pocketvina used to put into os.environ (P2RANK_HOME, PATH, LD_LIBRARY_PATH, OPENCL_ROOT),
# kept as attributes for scripts that read them. Importing changes no environment variable any more: P2Rank is found
# by pocketvina.p2rank ($P2RANK_HOME, `prank` on PATH, this p2rank_2.5 folder of older source installs, the user
# cache) and the engine gets its environment from pocketvina.engine.
package_dir = os.path.dirname(os.path.abspath(__file__))
p2rank_home = os.path.join(package_dir, 'p2rank_2.5')
boost_lib_path = os.path.join(package_dir, "boost_1_77_0/stage/lib")
opencl_path = os.path.join(package_dir, "OpenCL")


def __getattr__(name):
    # imported on first use: the atlas pipeline and the results reader load pyarrow
    if name == "run_atlas":
        from .pipeline import run_atlas
        return run_atlas
    if name == "Results":
        from .io.results import Results
        return Results
    if name == "Atlas":
        from .atlas import Atlas
        return Atlas
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
