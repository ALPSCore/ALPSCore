#!/usr/bin/env bash
# Script used by the build server the build
# all modules of ALPSCore

# This script expects the following environment variables
# BOOST_ROOT - location for boost distribution
# GTEST_ROOT - location for gtest sources/binaries
# HDF5_ROOT - location for the HDF5 distribution
# FAST_BUILD - don't remove old build files if set and true

# This script accepts optional arguments:
# --make-flags [extra_flags_for_make...]
# --cmake-flags [extra_flags_for_cmake...]

declare -a cmake_flags make_flags
dst=''

while [ "$1" != "" ]; do
    case "$1" in
        --cmake-flags) dst='cmake' ;;
        --make-flags) dst='make' ;;
        *) 
            case "$dst" in
                cmake) cmake_flags+=("$1") ;;
                make)  make_flags+=("$1") ;;
                *)
                    echo "Usage: $0 [--cmake-flags cmake_flag [cmake_flag...]] [--make_flags [make_flag make_flag...]]" >&2
                    exit 1;;
            esac;;
    esac
    shift
done


# Make sure we are in top directory for the repository
SCRIPTDIR=$(dirname $0)
cd $SCRIPTDIR/../..

if [[ -z "$BOOST_ROOT" ]]
then
  BOOST_SYSTEM=NO
else
  BOOST_SYSTEM=YES
fi

echo "Using BOOST at $BOOST_ROOT - no system path $BOOST_SYSTEM"

INSTALLDIR=$PWD/install
BUILDDIR=$PWD/build.tmp
ROOTDIR=$PWD

mkdir $INSTALLDIR 2>/dev/null

if [[ -z "$FAST_BUILD" || "$FAST_BUILD" == 0 || "$FAST_BUILD" == 'false' ]]; then
    rm -rf $BUILDDIR
else
    rm -f $BUILDDIR/CMakeCache.txt
fi
mkdir -p $BUILDDIR
cd $BUILDDIR
rm -rf "$INSTALLDIR"
mkdir -p "$INSTALLDIR"

cmake \
  -DCMAKE_INSTALL_PREFIX="${INSTALLDIR}" \
  -DTesting=ON \
  -DExtensiveTesting=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT="${BOOST_ROOT}" \
  -DGTEST_ROOT="${GTEST_ROOT}" \
  -DBoost_NO_SYSTEM_PATHS="${BOOST_SYSTEM}" \
  -DENABLE_MPI=TRUE \
  -DTestXMLOutput=TRUE \
  -DDocumentation=OFF \
  "${cmake_flags[@]}" \
  ${ROOTDIR}

make "${make_flags[@]}" || exit 1 
make test
make install || exit 1

echo "*** Done ***"
