# Script used by the build server the build
# all modules of ALPSCore

# This script expects the following environment variables
# BOOST_ROOT - location for boost distribution
# GTEST_ROOT - location for gtest sources/binaries
# HDF5_ROOT - location for the HDF5 distribution

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

rm -rf $BUILDDIR
mkdir $BUILDDIR
cd $BUILDDIR

cmake \
  -DCMAKE_INSTALL_PREFIX="${INSTALLDIR}" \
  -DTesting=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBOOST_ROOT="${BOOST_ROOT}" \
  -DGTEST_ROOT="${GTEST_ROOT}" \
  -DBoost_NO_SYSTEM_PATHS="${BOOST_SYSTEM}" \
  -DENABLE_MPI=TRUE \
  -DTestXMLOutput=TRUE \
  ${ROOTDIR}

make || exit 1 
make test
make install || exit 1

echo "*** Done ***"
