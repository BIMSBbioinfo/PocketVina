# PocketVina-GPU docking engine.
#
#   make                build ./PocketVina-GPU, the engine the Python package runs (pip ships it as package data)
#   make install        copy it to $(PREFIX)/bin/pocketvina-engine and the OpenCL kernel sources it compiles at first
#                       use to $(PREFIX)/share/pocketvina/OpenCL ($(DESTDIR) is prepended when set)
#   make source, debug  developer builds that always rebuild the OpenCL kernels
#   make print          show the settings
#
# BOOST_ROOT, OPENCL_ROOT, PREFIX, CXX, CXXFLAGS and LDFLAGS are taken from the environment or the command line, e.g.
#   make BOOST_ROOT=$CONDA_PREFIX OPENCL_ROOT=$CONDA_PREFIX && make install PREFIX=$CONDA_PREFIX
# BOOST_ROOT is a Boost 1.77 source tree with its static libraries built into lib/ (the default ./boost_1_77_0, as
# the README builds it; Boost.Thread's sources are compiled in), or an installed Boost with headers in include/ and
# libraries in lib/ (e.g. conda's libboost-devel), linked dynamically with an rpath to its lib/.

# Need to be modified according to different users (BOOST_LIB_PATH, OPENCL_LIB_PATH: names before v0.2)
BOOST_LIB_PATH ?= $(CURDIR)/boost_1_77_0
BOOST_ROOT ?= $(BOOST_LIB_PATH)
OPENCL_LIB_PATH ?= /usr/local/cuda
OPENCL_ROOT ?= $(OPENCL_LIB_PATH)
PREFIX ?= /usr/local
ifeq ($(origin CXX),default)
CXX = gcc
endif
CXXFLAGS ?=
LDFLAGS ?=
WORK_DIR=$(CURDIR)
OPENCL_VERSION=-DOPENCL_3_0
GPU_PLATFORM=-DNVIDIA_PLATFORM
DOCKING_BOX_SIZE=-DSMALL_BOX

# Should not be modified
ENGINE=PocketVina-GPU
INSTALL_BIN=$(DESTDIR)$(PREFIX)/bin
INSTALL_SHARE=$(DESTDIR)$(PREFIX)/share/pocketvina
# a Boost source tree: Boost.Thread is compiled in; an installed Boost: its library is linked
BOOST_THREAD_SRC:=$(wildcard $(BOOST_ROOT)/libs/thread/src/pthread/thread.cpp $(BOOST_ROOT)/libs/thread/src/pthread/once.cpp)
ifeq ($(BOOST_THREAD_SRC),)
BOOST_THREAD_LIB=-lboost_thread
BOOST_RPATH=-Wl,-rpath,$(BOOST_ROOT)/lib
endif
# Boost.System is header-only since Boost 1.69; its stub library is gone in Boost 1.89
BOOST_SYSTEM_LIB=$(if $(wildcard $(BOOST_ROOT)/lib/libboost_system.*),-lboost_system)
BOOST_INC_PATH=-I$(BOOST_ROOT) -I$(BOOST_ROOT)/boost
VINA_GPU_INC_PATH=-I$(WORK_DIR)/lib -I$(WORK_DIR)/OpenCL/inc
OPENCL_INC_PATH=-I$(OPENCL_ROOT)/include
# an installed Boost's headers; after the others, so that a Boost source tree's own headers come first
INSTALLED_INC_PATH=-I$(BOOST_ROOT)/include
LIB1=-L$(BOOST_ROOT)/lib -lboost_program_options $(BOOST_SYSTEM_LIB) -lboost_filesystem $(BOOST_THREAD_LIB) -lOpenCL
LIB2=-lstdc++ -lstdc++fs
LIB3=-lm -lpthread
LIB_PATH=-L$(BOOST_ROOT)/lib -L$(OPENCL_ROOT)/lib64 -L$(OPENCL_ROOT)/lib $(BOOST_RPATH)
SRC=./lib/*.cpp ./OpenCL/src/wrapcl.cpp $(BOOST_THREAD_SRC)
MACRO=$(OPENCL_VERSION) $(GPU_PLATFORM) $(DOCKING_BOX_SIZE) -DBOOST_TIMER_ENABLE_DEPRECATED #-DDISPLAY_SUCCESS -DDISPLAY_ADDITION_INFO
COMPILE=$(CXX) -o $(ENGINE) $(CXXFLAGS) $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) $(INSTALLED_INC_PATH) ./main/main.cpp
LINK=$(SRC) $(LDFLAGS) $(LIB1) $(LIB2) $(LIB3) $(LIB_PATH) $(MACRO) $(OPTION)
# OpenCL kernels: `make` builds an engine that loads Kernel1_Opt.bin/Kernel2_Opt.bin from opencl_binary_path and,
# if they are missing or rejected by the driver, builds them from $POCKETVINA_KERNEL_SRC/OpenCL (default ".") and
# saves them there (never into the working directory or that source tree). `make source` and `make debug` always
# rebuild the kernels (for kernel development).
.PHONY: all out source debug install clean print
all:out
out:./main/main.cpp
	$(COMPILE) -O3 $(LINK) -DNDEBUG
source:./main/main.cpp
	$(COMPILE) -O3 $(LINK) -DNDEBUG -DBUILD_KERNEL_FROM_SOURCE
debug:./main/main.cpp
	$(COMPILE) -g $(LINK) -DBUILD_KERNEL_FROM_SOURCE
# installs the engine `make` built (or the prebuilt one of a source checkout); pocketvina finds it on PATH and its
# kernel sources in ../share/pocketvina, unless its package holds an engine binary of its own (pip install of a
# checkout), which comes first: then set POCKETVINA_ENGINE=pocketvina-engine
install:
	@test -x $(ENGINE) || { echo "make install: there is no ./$(ENGINE), run make first" >&2; exit 1; }
	install -d $(INSTALL_BIN) $(INSTALL_SHARE)/OpenCL/inc $(INSTALL_SHARE)/OpenCL/src/kernels
	install -m 755 $(ENGINE) $(INSTALL_BIN)/pocketvina-engine
	install -m 644 OpenCL/inc/*.h $(INSTALL_SHARE)/OpenCL/inc
	install -m 644 OpenCL/src/kernels/*.cl $(INSTALL_SHARE)/OpenCL/src/kernels
clean:
	rm -f $(ENGINE)
print:
	@echo "WORK_DIR = $(WORK_DIR)"
	@echo "BOOST_ROOT = $(BOOST_ROOT) ($(if $(BOOST_THREAD_SRC),source tree: Boost.Thread compiled in,installed Boost: -lboost_thread))"
	@echo "BOOST_LIB_PATH = $(BOOST_LIB_PATH)"
	@echo "OPENCL_ROOT = $(OPENCL_ROOT)"
	@echo "PREFIX = $(PREFIX)"
	@echo "CXX = $(CXX)"
	@echo "CXXFLAGS = $(CXXFLAGS)"
	@echo "LDFLAGS = $(LDFLAGS)"
