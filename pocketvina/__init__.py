import os
from .main import PocketVinaGPU, P2RankProcessor, PocketVinaConfig, P2RankConfig, PocketConfig, Paths

__version__ = "0.1.0"
__all__ = ["PocketVinaGPU", "P2RankProcessor", "PocketVinaConfig", "P2RankConfig", "PocketConfig", "Paths"]

# Set P2Rank environment variables
package_dir = os.path.dirname(os.path.abspath(__file__))
p2rank_home = os.path.join(package_dir, 'p2rank_2.5')
os.environ['P2RANK_HOME'] = p2rank_home
os.environ['PATH'] = f"{p2rank_home}:{os.environ.get('PATH', '')}"

# Set Boost and OpenCL environment variables
boost_lib_path = os.path.join(package_dir, "boost_1_77_0/stage/lib")
opencl_path = os.path.join(package_dir, "OpenCL")
os.environ['LD_LIBRARY_PATH'] = f"{boost_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
os.environ['OPENCL_ROOT'] = opencl_path