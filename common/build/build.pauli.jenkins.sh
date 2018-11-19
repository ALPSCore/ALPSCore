#!/usr/bin/env bash
# Script used by Jenkins to build ALPSCore on Pauli cluster

# This script expects the following environment variables
# COMPILER - the compiler to build with
# MPI_VERSION - MPI library to link with
# PHASE - 'cmake' | 'make' | 'test' | 'install' (all of the above if empty)

# This function sets build environment unless it's already set (as determined by env. var `build_environment_set`)
function setup_environment() {
    [[ $build_environment_set == 1 ]] && return 0
    
    module purge
    module add cmake/3.11.1
    case $COMPILER in
        gcc_4.8.5) 
            export CC=$(which gcc)
            export CXX=$(which g++)
            ;;
        gcc_5.4.0)
            module add gnu/5.4.0
            export CC=$(which gcc)
            export CXX=$(which g++)
            ;;
        clang_3.4.2) 
            export CC=$(which clang)
            export CXX=$(which clang++)
            ;;
        clang_5.0.1)
            module add llvm5/5.0.1
            export CC=$(which clang)
            export CXX=$(which clang++)
            ;;
        intel_18.0.5)
            . /opt/intel/bin/compilervars.sh intel64
            # we have to load GNU CC before OpenMPI, but will use Intel
            [[ $MPI_VERSION = OpenMPI ]] && module add gnu/5.4.0 openmpi/1.10.7
            export CC=/opt/intel/bin/icc
            export CXX=/opt/intel/bin/icpc
            ;;
        *) 
            echo "Unsupported compiler passed via COMPILER='$COMPILER'; valid values are:" 2>&1
            echo "gcc_4.8.5 gcc_5.4.0 clang_3.4.2 clang_5.0.1 intel_18.0.5"
            exit 1
            ;;
        
    esac

    case $MPI_VERSION in
        MPI_OFF)
            ENABLE_MPI=OFF
            ;;
        OpenMPI)
            ENABLE_MPI=ON
            module add openmpi/1.10.7
            ;;
        *)
            echo "Unsupported MPI version passed via MPI_VERSION='$MPI_VERSION'; valid values are:" 2>&1
            echo "MPI_OFF OpenMPI" 2>&1
            exit 1
   	    ;;
    esac

    export BOOST_ROOT=/opt/ohpc/pub/libs/gnu/openmpi/boost/1.66.0

    mkdir -pv build.tmp
    cd build.tmp

    build_environment_set=1
}

function run_cmake() {
    cmake -DCMAKE_INSTALL_PREFIX=$PWD/install \
          -DTesting=ON -DExtensiveTesting=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DENABLE_MPI=${ENABLE_MPI} \
          -DTestXMLOutput=TRUE \
          -DEIGEN3_INCLUDE_DIR=$HOME/.local/packages/eigen-3.3.4 \
          -DDocumentation=OFF \
          ..
}

function run_make() {
    make -j8 VERBOSE=1
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

