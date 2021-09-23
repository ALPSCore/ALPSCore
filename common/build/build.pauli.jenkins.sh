#!/usr/bin/env bash
# Script used by Jenkins to build ALPSCore on Pauli cluster

# This script expects the following environment variables
# COMPILER - the compiler to build with
# MPI_VERSION - MPI library to link with
# BUILD_DIR - (optional) A directory name to build in (default: formed from COMPILER and MPI_VERSION)
# PHASE - 'cmake' | 'make' | 'test' | 'install' (all of the above if empty)
# The scripts creates directories under ./build.tmp/


BASE_DIR=$(/bin/pwd)

# This function sets build environment unless it's already set (as determined by env. var `build_environment_set`)
function setup_environment() {
    [[ $build_environment_set == 1 ]] && return 0

    # Jenkins doesn't seem to load environment by default
    . /etc/profile

    module purge
    module add cmake/3.15.4 eigen3/3.3.7

    # Replace '_' by '/'
    _COMPILER_MODULE="${COMPILER/_//}"

    case $COMPILER in
        gcc_10.3.0)
            module add ${_COMPILER_MODULE} hdf5/${_COMPILER_MODULE}/1.10.7 boost/${_COMPILER_MODULE}/1.76.0
            export CC=$(which gcc)
            export CXX=$(which g++)
            ;;
        aocc_3.1.0)
            module add ${_COMPILER_MODULE} hdf5/${_COMPILER_MODULE}/1.10.7 boost/${_COMPILER_MODULE}/1.76.0
            export CC=$(which clang)
            export CXX=$(which clang++)
            ;;
        intel_2021.3.0)
            module add ${_COMPILER_MODULE} hdf5/${_COMPILER_MODULE}/1.10.7 boost/${_COMPILER_MODULE}/1.75.0
            export CC=$(which icc)
            export CXX=$(which icpc)
            ;;
        *)
            echo "Unsupported compiler passed via COMPILER='$COMPILER'; valid values are:" 2>&1
            echo "gcc_10.3.0 aocc_3.1.0 intel_2021.3.0"
            exit 1
            ;;

    esac

    case $MPI_VERSION in
        MPI_OFF)
            ENABLE_MPI=OFF
            ;;
        OpenMPI)
            ENABLE_MPI=ON
            module add openmpi/${_COMPILER_MODULE}/4.1.1
            ;;
        *)
            echo "Unsupported MPI version passed via MPI_VERSION='$MPI_VERSION'; valid values are:" 2>&1
            echo "MPI_OFF OpenMPI" 2>&1
            exit 1
   	    ;;
    esac

    [[ $BUILD_DIR ]] || BUILD_DIR="build.tmp/${COMPILER}_${MPI_VERSION}"
    mkdir -pv "$BUILD_DIR"
    cd "$BUILD_DIR"

    build_environment_set=1
}

function run_cmake() {
    rm -rf *
    cmake -DCMAKE_INSTALL_PREFIX=$PWD/install \
          -DTesting=ON -DExtensiveTesting=OFF \
          -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_MPI=${ENABLE_MPI} \
          -DTestXMLOutput=TRUE \
          -DDocumentation=OFF \
          "$BASE_DIR"
}

function run_make() {
    make -j8
}

function run_test() {
    make test
}

function run_install() {
    make -j8 install
}


set build_env_set=0

if [[ $PHASE == 'cmake' || $PHASE == '' ]]; then
    setup_environment
    run_cmake || exit $?
fi

if [[ $PHASE == 'make' || $PHASE == '' ]]; then
    setup_environment
    run_make || exit $?
fi

if [[ $PHASE == 'test' || $PHASE == '' ]]; then
    setup_environment
    run_test || exit $?
fi

if [[ $PHASE == 'install' || $PHASE == '' ]]; then
    setup_environment
    run_install || exit $?
fi

