#!/bin/sh

mydir=$(dirname $0)
mydir=$(cd $mydir && pwd -P)
if [ ".$mydir" = . ]; then
    echo "Cannot determine my directory." >&2
    exit 1
fi

if ! [ -r "$mydir/CMakeLists.txt" ]; then
    echo "Cannot find or read CMakeLists.txt in my directory '$mydir'" >&2
    echo "Installation cannot proceed." >&2
    exit 1
fi

if [ ".$ALPSCore_DIR" = . ]; then
    echo "Please make ALPSCore_DIR environment variable point to the directory you wish to install this library into." >&2
    exit 1
fi

if [ -z "${ALPSCore_DIR##[!/]*}" ]; then
    echo "ALPSCore_DIR environment variable (set to '$ALPSCore_DIR' now) must be an absolute path." >&2
    exit 1
fi

if [ ".$ALPS_USE_MPI" = . ]; then 
    echo "A sequential version will be built"
    defs="-DENABLE_MPI=OFF"
else
    echo "A parallel version will be built"
    defs="-DENABLE_MPI=ON"
fi

if [ ".$HDF5_ROOT" != . ]; then
    echo "HDF5 will be searched in $HDF5_ROOT directory"
fi

if [ ".$BOOST_ROOT" != . ]; then
    echo "Boost will be searched in $BOOST_ROOT directory"
    defs="$defs -DBoost_NO_SYSTEM_PATHS=true -DBoost_NO_BOOST_CMAKE=true"
fi

bdir="000build.tmp"
if  ! mkdir -pv $bdir; then
    echo "Cannot create a build subdirectory '$bdir'" >&2
    exit 1
fi

if ! cd $bdir; then
    echo "Cannot change to the build subdirectory '$bdir'" >&2
    exit 1
fi

echo "*** Generating the build ***" >&2

if ! cmake $mydir -DCMAKE_INSTALL_PREFIX="$ALPSCore_DIR" $defs; then
    echo >&2
    echo "Generating the build failed." >&2
    echo "Correct any reported problems and try again!" >&2
    exit 1
fi

echo "*** Compiling the code ***" >&2

if ! make ; then
    echo >&2
    echo "Compilation failed." >&2
    echo "Correct any reported problems and try again!" >&2
    exit 1
fi

echo "*** Testing the code ***" >&2
if ! make test; then
    echo >&2
    echo "Testing failed." >&2
    echo "If you wish to install nevertheless, change to the directory '$bdir' and run 'make install'" >&2
    exit 1
fi

echo "*** Installing the code ***" >&2
if ! make install; then
    echo >&2
    echo "Installation failed." >&2
    echo "After fixing the problem, change to the directory '$bdir' and run 'make install'" >&2
    exit 1
fi

echo "*** Success ***" >&2
echo "The code is installed in '$ALPSCore_DIR'"
exit 0
