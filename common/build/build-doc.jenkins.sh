# Script used by the build server the build
# all modules of ALPSCore

# This script expects the following environment variables
# BOOST_ROOT - location for boost distribution
# GTEST_ROOT - location for gtest sources/binaries
# HDF5_ROOT - location for the HDF5 distribution

# Make sure we are in top directory for the repository
SCRIPTDIR=$(dirname $0)
cd $SCRIPTDIR/../..

INSTALLDIR=$PWD/install
BUILDDIR=$PWD/build-doc.tmp
ROOTDIR=$PWD
DOCDIR=$INSTALLDIR/doc

mkdir $INSTALLDIR 2>/dev/null
mkdir $DOCDIR 2>/dev/null

rm -rf $BUILDDIR
mkdir $BUILDDIR
cd $BUILDDIR

cmake \
  -DCMAKE_INSTALL_PREFIX="${INSTALLDIR}" \
  -DDocumentationOnly=ON \
  -DDOXYFILE_OUTPUT_DIR="${DOCDIR}" \
  ${ROOTDIR}

make doc || exit 1

echo "*** Done ***"
