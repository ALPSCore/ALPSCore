#guarantees we are in build directory
BASEDIR=$(dirname $0)
cd $BASEDIR
# now we are in the module directory
cd ..

MODULEDIR=`pwd`
TARGET=${MODULEDIR}/target
mkdir ${TARGET}
cd ${TARGET} 

echo "Module is in ${MODULEDIR}"
echo "Building in ${TARGET}"

cmake \
-DCMAKE_INSTALL_PREFIX="./INSTALL_DIR" \
-DTesting=ON \
-DCMAKE_BUILD_TYPE=Release \
${MODULEDIR}

make || exit 1 
make test || exit 1

echo "all done."
