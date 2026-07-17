# zord C++ graph/supra kernels -> build/<name>
#
# Compiles EVERY .cpp in this directory with the project's standard flags
# (C++17, -O3, -fopenmp) -- exactly how build/graph_algos was built (see the
# `// Build:` header line in each .cpp and scripts/{supra_solve,sota_compare}.py):
#
#   g++ -O3 -std=c++17 -fopenmp src/zord/cpp/<name>.cpp -o build/<name>
#
# Usage (from anywhere; paths are absolute via $(REPO)):
#   make -C src/zord/cpp            # build all kernels
#   make -C src/zord/cpp clean      # remove the built binaries
#
# The binaries land in <repo>/build/, which is where every Python wrapper
# (partition.cpp_kernel, frontend.ingest, profiler.prober, partition.allocate,
# runtime.bufferpool, schedule.dynamic_online) resolves them by default.

CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp

# This Makefile lives at <repo>/src/zord/cpp/Makefile -> repo root is 3 dirs up.
SRCDIR   := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO     := $(abspath $(SRCDIR)/../../..)
BUILDDIR := $(REPO)/build

# Every kernel = every .cpp in this directory.
SRCS     := $(wildcard $(SRCDIR)*.cpp)
BINS     := $(patsubst $(SRCDIR)%.cpp,$(BUILDDIR)/%,$(SRCS))

.PHONY: all clean
all: $(BINS)

$(BUILDDIR)/%: $(SRCDIR)%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -f $(BINS)
