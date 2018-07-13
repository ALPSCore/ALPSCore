#!/bin/bash

## DEBUG VERSION OF THE SCRIPT --- DO NOT USE IN PRODUCTION!

## Build script for TravisCI

# Be verbose and stop on first error
set -ev

# Explicitly download Boost, if a specific version is requested
boost_cmake_params=""
if [ -n "$ALPS_BOOST_VERSION" ]; then
  download_dir=$HOME/boost
  mkdir -pv $download_dir
  boost_tgz=$download_dir/boost_${ALPS_BOOST_VERSION}.tar.gz
  boost_url=https://sourceforge.net/projects/boost/files/boost/${ALPS_BOOST_VERSION//_/.}/boost_${ALPS_BOOST_VERSION}.tar.gz/download
  wget -S -O $boost_tgz $boost_url
  tar -C $download_dir -xzf $boost_tgz
  boost_cmake_params="-DBoost_NO_SYSTEM_PATHS=true -DBoost_NO_BOOST_CMAKE=true -DBOOST_ROOT=$download_dir/boost_${ALPS_BOOST_VERSION}"
fi

# Build ALPSCore
mkdir -pv build
mkdir -pv install
cd build
cmake ..                                              \
-DCMAKE_BUILD_TYPE=Debug                              \
-DCMAKE_C_COMPILER=${ALPS_CC:-${CC}}                  \
-DALPS_CXX_STD=$ALPS_CXX_STD                          \
-DCMAKE_CXX_COMPILER=${ALPS_CXX:-${CXX}}              \
-DCMAKE_INSTALL_PREFIX=$TRAVIS_BUILD_DIR/installed    \
-DALPS_INSTALL_EIGEN=true                             \
-DALPS_BUNDLE_DOWNLOAD_TRIES=3                        \
-DMPIEXEC=mpiexec -DMPIEXEC_NUMPROC_FLAG='-n'         \
-DENABLE_MPI=$ENABLE_MPI                              \
${boost_cmake_params}

# TravisCI provides 2 cores, with possible bursts
# DEBUG: what does cpuinfo say?
cat /proc/cpuinfo
# DEBUG: test compilation with 3 processes
time make -j3
# DEBUG: test compilation on 2 processes
make clean
time make -j2

time env ALPS_TEST_MPI_NPROC=3 make test
make install
