#guarantees we are in build directory
BASEDIR=$(dirname $0)
cd $BASEDIR
# now we are in the module directory
cd ..

MODULEDIR=`pwd`
BUILDDIR=${MODULEDIR}/target
TARGET=${MODULEDIR}/../INSTALL_DIR
mkdir ${BUILDDIR}
cd ${BUILDDIR} 

echo "Module is in ${MODULEDIR}"
echo "Building in ${BUILDDIR}"

cmake \
-DCMAKE_INSTALL_PREFIX="${TARGET}" \
-DTesting=ON \
-DCMAKE_BUILD_TYPE=Release \
-DBOOST_ROOT="${BOOST_ROOT}" \
${MODULEDIR}

make || exit 1 
make test || exit 1
make install || exit 1

echo "all done."
