#!/bin/bash
## Build script for TravisCI

# Be verbose and stop on first error
set -ev

# Explicitly download Boost, if a specific version is requested
boost_no_system_paths=false
if [ -n "$ALPS_BOOST_VERSION" ]; then
  boost_tgz=boost_${ALPS_BOOST_VERSION}.tar.gz
  boost_url=https://sourceforge.net/projects/boost/files/boost/${ALPS_BOOST_VERSION//_/.}/boost_${ALPS_BOOST_VERSION}.tar.gz/download
  wget -S -O $boost_tgz $boost_url
  tar -xzf $boost_tgz
  export BOOST_ROOT=$PWD/boost_${ALPS_BOOST_VERSION}
  boost_no_system_paths=true
fi

# Build ALPSCore
mkdir build
mkdir install
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
-DBoost_NO_SYSTEM_PATHS=${boost_no_system_paths}
make -j3
env ALPS_TEST_MPI_NPROC=3 make test
make install
