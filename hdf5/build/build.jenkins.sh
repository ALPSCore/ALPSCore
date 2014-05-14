
#guarantees we are in build directory
BASEDIR=$(dirname $0)
cd $BASEDIR
# now we are in the module directory
cd ..

MODULEDIR=`pwd`
BUILDDIR=${MODULEDIR}/target
TARGET=${MODULEDIR}/../INSTALL_DIR
cd ..
ROOTDIR=`pwd`

# alps-utility
${ROOTDIR}/utility/build/build.jenkins.sh
export gtest_ROOT=${ROOTDIR}/utility/target

mkdir ${BUILDDIR}
cd ${BUILDDIR} 

echo "Module is in ${MODULEDIR}"
echo "Building in ${BUILDDIR}"

cmake \
-DCMAKE_INSTALL_PREFIX="${TARGET}" \
-DTesting=ON \
-DCMAKE_BUILD_TYPE=Release \
${MODULEDIR}

make || exit 1 
make test || exit 1
make install || exit 1

echo "all done."
