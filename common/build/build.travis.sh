#!/bin/bash
## Build script for TravisCI

# Be verbose and stop on first error
set -ev

# Explicitly download Boost, if a specific version is requested
boost_cmake_params=""
no_boost_libs=false
if [ -n "$ALPS_BOOST_VERSION" ]; then
  download_dir=$HOME/boost
  mkdir -pv $download_dir
  boost_tgz=$download_dir/boost_${ALPS_BOOST_VERSION}.tar.gz
  boost_url=https://sourceforge.net/projects/boost/files/boost/${ALPS_BOOST_VERSION//_/.}/boost_${ALPS_BOOST_VERSION}.tar.gz/download
  wget -S -O $boost_tgz $boost_url
  tar -C $download_dir -xzf $boost_tgz
  boost_cmake_params="-DBoost_NO_SYSTEM_PATHS=true -DBoost_NO_BOOST_CMAKE=true -DBOOST_ROOT=$download_dir/boost_${ALPS_BOOST_VERSION}"
  no_boost_libs=true
fi

# FIXME: A hack to suppress warnings in MPI headers
mpi_warning_hack=""
if [[ "$ENABLE_MPI" == "ON" ]]; then
    mpi_incdirs=" $(mpic++ -showme:incdirs)"
    mpi_warning_hack="${mpi_incdirs// / -isystem }"
fi

# Build ALPSCore
alpscore_src=$PWD
mkdir -pv build
mkdir -pv install
cd build
cmake $alpscore_src                                   \
-DCMAKE_BUILD_TYPE=Debug                              \
-DCMAKE_C_COMPILER=${ALPS_CC:-${CC}}                  \
-DALPS_CXX_STD=$ALPS_CXX_STD                          \
-DCMAKE_CXX_FLAGS="-Wall -Wno-missing-braces -Wmissing-field-initializers -Werror ${mpi_warning_hack}"   \
-DCMAKE_CXX_COMPILER=${ALPS_CXX:-${CXX}}              \
-DCMAKE_INSTALL_PREFIX=$TRAVIS_BUILD_DIR/installed    \
-DALPS_INSTALL_EIGEN=true                             \
-DALPS_BUNDLE_DOWNLOAD_TRIES=3                        \
-DMPIEXEC=mpiexec -DMPIEXEC_NUMPROC_FLAG='-n'         \
-DENABLE_MPI=$ENABLE_MPI                              \
${boost_cmake_params}

# TravisCI provides 2 cores, with possible bursts;
# We use exactly as many cores as available to us.
if which nproc; then
    ncores=$(nproc)
else
    ncores=2 # FIXME: MacOS does not have nproc
fi

time make -j$ncores

# Run MPI tests with a slight oversubscription
# (this might help detect timing-dependent bugs)
time env ALPS_TEST_MPI_NPROC=$[$ncores+1] make test
make install

# Test tutorials build
mkdir -pv tutorials
cd tutorials
ALPSCore_DIR=$TRAVIS_BUILD_DIR/installed cmake $alpscore_src/tutorials -DALPS_TUTORIALS_NO_BOOST_LIBS=${no_boost_libs}
