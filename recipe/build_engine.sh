#!/usr/bin/env bash
# Build script of the conda package pocketvina-engine (recipe.yaml): compiles the PocketVina-GPU docking engine with
# conda-forge's C++ compiler, Boost (shared libraries) and OpenCL ICD loader and headers, and installs it with
# pocketvina/Makefile's `make install` as $PREFIX/bin/pocketvina-engine, with the OpenCL kernel sources it compiles
# at its first run on a GPU in $PREFIX/share/pocketvina/OpenCL (pocketvina.engine finds both there).
# CXX, CXXFLAGS and LDFLAGS come from the compiler package's activation (the Makefile takes them from the
# environment); BOOST_ROOT and OPENCL_ROOT are the host prefix, where libboost-devel, ocl-icd and opencl-headers are.
set -euxo pipefail

cd pocketvina
# a source tree may hold a prebuilt engine (git checkout, sdist): never package that one, always this build's
rm -f PocketVina-GPU

# Boost 1.85 removed boost/filesystem/convenience.hpp, whose filesystem::basename (= path::stem) main/main.cpp
# uses: with a newer Boost, compile against a copy of that header's definitions (Boost 1.84's), outside the sources
if grep -q 'boost/filesystem/convenience.hpp' main/main.cpp &&
        [ ! -f "$PREFIX/include/boost/filesystem/convenience.hpp" ]; then
    compat="$SRC_DIR/boost_compat"
    mkdir -p "$compat/boost/filesystem"
    cat > "$compat/boost/filesystem/convenience.hpp" <<'EOF'
// boost/filesystem/convenience.hpp of Boost <= 1.84 (removed in Boost 1.85), for pocketvina-engine's main.cpp
#ifndef BOOST_FILESYSTEM_CONVENIENCE_HPP
#define BOOST_FILESYSTEM_CONVENIENCE_HPP
#include <string>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/system/error_code.hpp>
namespace boost {
namespace filesystem {
inline std::string extension(const path& p) { return p.extension().string(); }
inline std::string basename(const path& p) { return p.stem().string(); }
inline path change_extension(const path& p, const path& new_extension)
{
    path new_p(p);
    new_p.replace_extension(new_extension);
    return new_p;
}
} // namespace filesystem
} // namespace boost
#endif // BOOST_FILESYSTEM_CONVENIENCE_HPP
EOF
    export CXXFLAGS="-I$compat ${CXXFLAGS:-}"
fi

make print BOOST_ROOT="$PREFIX" OPENCL_ROOT="$PREFIX"
make BOOST_ROOT="$PREFIX" OPENCL_ROOT="$PREFIX"
make install PREFIX="$PREFIX"
