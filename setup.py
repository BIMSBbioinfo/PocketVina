import os
import platform
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info

def install_java():
    """Install OpenJDK 17 based on the platform"""
    system = platform.system().lower()
    
    try:
        if system == 'linux':
            # Try using conda first (preferred method)
            try:
                subprocess.run(['conda', 'install', '-c', 'conda-forge', 'openjdk=17', '-y'], 
                             check=True)
                return True
            except subprocess.CalledProcessError:
                # If conda fails, try apt-get for Debian/Ubuntu
                try:
                    subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                    subprocess.run(['sudo', 'apt-get', 'install', '-y', 'openjdk-17-jdk'], 
                                 check=True)
                    return True
                except subprocess.CalledProcessError:
                    print("Could not install Java automatically.")
                    print("Please install Java 17 manually:")
                    print("  conda install -c conda-forge openjdk=17")
                    print("  or")
                    print("  sudo apt-get install openjdk-17-jdk")
                    return False
        else:
            print(f"Automatic Java installation not supported on {system}")
            print("Please install Java 17 manually from: https://adoptium.net/")
            return False
            
    except Exception as e:
        print(f"Error installing Java: {e}")
        return False


def check_java_version():
    """Check if compatible Java version is installed"""
    try:
        result = subprocess.run(
            ['java', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        version_output = result.stderr  # Java outputs version to stderr
        version_line = version_output.split('\n')[0]
        if 'version' in version_line:
            version = version_line.split('"')[1].split('.')[0]
            if version.isdigit():
                java_version = int(version)
                if java_version >= 17:
                    print("Java 17+ detected!")
                    return True
                
        print("WARNING: Java 17 or higher is required.")
        if install_java():
            print("Java 17 installed successfully!")
            return True
        return False
    
    except subprocess.CalledProcessError:
        print("Java not found. Attempting to install Java 17...")
        return install_java()
    

def setup_p2rank():
    """Download and setup P2Rank"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    pocketvina_dir = os.path.join(current_dir, "pocketvina")
    p2rank_path = os.path.join(pocketvina_dir, "p2rank_2.5")
    prank_executable = os.path.join(p2rank_path, "prank")
    # Only download and setup if P2Rank doesn't exist
    if not os.path.exists(p2rank_path) or not os.path.exists(prank_executable):
        # Check Java version first
        if not check_java_version():
            print("WARNING: Proceeding with installation, but P2Rank won't work without Java 17+")
            
        # Create p2rank directory if it doesn't exist
        os.makedirs(p2rank_path, exist_ok=True)
            
        # Download P2Rank
        subprocess.run([
                'wget', 
                'https://github.com/rdk/p2rank/releases/download/2.5/p2rank_2.5.tar.gz',
                '-P', pocketvina_dir
            ], check=True)
            
        # Extract P2Rank
        subprocess.run([
                'tar', '-xzf', os.path.join(pocketvina_dir, 'p2rank_2.5.tar.gz'),
                '-C', pocketvina_dir
            ], check=True)
            
        # Make prank executable
        os.chmod(prank_executable, 0o755)
            
        # Clean up tar file
        os.remove(os.path.join(pocketvina_dir, 'p2rank_2.5.tar.gz'))

        print("P2Rank downloaded and extracted successfully!")
    else:
        print("P2Rank already exists, skipping download...")
    
    try:
        # Test P2Rank installation
        try:
            result = subprocess.run(
                [prank_executable, '--version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("P2Rank installation verified!")
        except subprocess.CalledProcessError as e:
            print("WARNING: P2Rank installation might not be complete.")
            print("Please ensure Java 17+ is installed and try running:")
            print("  conda install -c conda-forge openjdk=17")
        
        return True
    except Exception as e:
        print(f"Error setting up P2Rank: {e}")
        return False
    

def compile_cpp():
    """Compile the C++ code only if not already compiled"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pocketvina_dir = os.path.join(current_dir, "pocketvina")
    boost_path = os.path.join(pocketvina_dir, "boost_1_77_0")
    
    # Only download and compile if Boost doesn't exist
    if not os.path.exists(boost_path):
        os.chdir(pocketvina_dir)
        subprocess.run(['wget', 'https://archives.boost.io/release/1.77.0/source/boost_1_77_0.tar.gz'], check=True)
        subprocess.run(['tar', '-xvf', 'boost_1_77_0.tar.gz'], check=True)
        
        os.chdir('boost_1_77_0')
        # Add bootstrap configuration
        subprocess.run(['./bootstrap.sh', '--with-libraries=program_options,system,filesystem,thread'], check=True)
        
        # Build and install Boost
        subprocess.run(['./b2', 
                      f'--prefix={boost_path}',
                      'link=static',
                      'runtime-link=static',
                      'threading=multi',
                      'variant=release',
                      '-j4',
                      'install'], check=True)
        os.chdir(pocketvina_dir)
        
        os.remove('boost_1_77_0.tar.gz')
    
    
    # Set environment variables
    boost_lib_path = os.path.join(boost_path, "lib")
    os.environ['LD_LIBRARY_PATH'] = f"{boost_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    
    # Compile main program
    os.chdir(pocketvina_dir)
    try:
        subprocess.run(['make', 'clean'], check=False)
    except:
        pass
        
    subprocess.run(['chmod', '+r', 'QuickVina2-GPU-2-1/Kernel1_Opt.bin'], check=True)
    subprocess.run(['chmod', '+r', 'QuickVina2-GPU-2-1/Kernel2_Opt.bin'], check=True)
    subprocess.run(['make', 'source'], check=True)

    os.chdir(current_dir)


class CustomInstallCommand(install):
    def run(self):
        setup_p2rank()
        compile_cpp()
        install.run(self)

class CustomDevelopCommand(develop):
    def run(self):
        setup_p2rank()
        compile_cpp()
        develop.run(self)

class CustomEggInfoCommand(egg_info):
    def run(self):
        setup_p2rank()
        compile_cpp()
        egg_info.run(self)

setup(
    name="pocketvina-gpu",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        'pocketvina': [
            'lib/*',
            'main/*',
            'OpenCL/*',
            'boost_1_77_0/stage/lib/*',
            'Makefile',
            'QuickVina2-GPU-2-1/*',
            'PocketVina-GPU',
            'p2rank_2.5/*/*'
        ],
    },
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
        'egg_info': CustomEggInfoCommand,
    },
    install_requires=[
        'numpy',
        'pandas',
        'rdkit',
        'openbabel-wheel',
        'tqdm',
        'wget'
    ],
    include_package_data=True,
    author="Ahmet Sarigun",
    author_email="Ahmet.Sariguen@mdc-berlin.de",
    description=" GPU-accelerated protein-ligand docking with automated pocket detection, exploring through multi-pocket conditioning.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BIMSBbioinfo/PocketVina",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.6",
)