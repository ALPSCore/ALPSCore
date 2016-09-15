The ALPSCore project, based on the ALPS (Algorithms and Libraries for Physics Simulations) project, provides generic algorithms and utilities for physics problems. It strives to increase software reuse in the physics community.

For copyright see COPYRIGHT.txt
For licensing see LICENSE.txt
For acknowledgment in scientific publications see ACKNOWLEDGE.txt

INSTALLATION
============

The instructions below are intended to help to build and install a
working version of ALPSCore library in most common environments. For
complete up-to-date installation instructions, please refer to
https://github.com/ALPSCore/ALPSCore/wiki/Installation .

The instructions assume `bash` user shell.

Prerequisites:
--------------

To install ALPSCore, the following is needed:

* C++ compiler: g++ >= 4.2 OR Intel >= 10.0 OR Clang >= 3.2
* CMake >= 2.8.12
* HDF5 library >=1.8
* Boost >= 1.54.0

Optional requirements:

* An MPI library and headers for compiling multi-cpu versions of the libraries.
* Doxygen for creating low-level documentation.

Generating the build
--------------------

Unpack the archive into the current directory, and change to the
subdirectory containing the source; e.g., for version 0.5.5:

    $ tar -xzf ALPSCore-0.5.5.tar.gz
    $ cd ALPSCore-0.5.5

Set the environment variable `ALPSCore_DIR` to the absolute location of the
directory where you wish to install the library. If Boost and HDF5 are
installed at non-standard locations on your system, point to the
corresponding directories using the environment variables `BOOST_ROOT`
and `HDF5_ROOT`, respectively:

    $ export BOOST_ROOT=/where/boost/installed
    $ export HDF5_ROOT=/where/hdf5/installed

By default, a sequential verison of the library will be installed. To
install a parallel version, set the environment variable
`ALPS_USE_MPI` to a non-empty value:

    $ export ALPS_USE_MPI=1

Then, run the installation script:

    $ ./install_simple.sh

The script will generate the build, compile the library, run the test
suite and install the library to the location specified by the
`ALPSCore_DIR` environment variable.

EXAMPLE: USING THE LIBRARY
==========================

The following instructions demonstrate how to use the installed
library. For complete information, please refer to the site
http://alpscore.org, particularly the tutorial section.

We now assume that the library is installed at location pointed to by
the `ALPSCore_DIR` environment variable, and the current directory is
the library source directory. Then, perform the following sequence of
actions to build the example 2D Ising simulation code:

    $ cd tutorials/mc/ising2_mc
    $ mkdir -p 000build
    $ cd 000build
    $ cmake ..
    $ make ising2_mc

Then, run the generated executable (sequential version of the code):

    $ ./ising2_mc ../ising2_mc.ini --temperature=5 --sweeps=500000

The output should be the following (the exact values of floating-point numbers might differ insignificantly):

     Initializing parameters...
     Creating simulation
     Running simulation
     Checkpointing simulation to ising2_mc.clone.h5
     All measured results:
     AbsMagnetization: Mean +/-error (tau): 0.344702 +/-0.00176017(12.83)
     Energy: Mean +/-error (tau): -0.461347 +/-0.00296763(11.2385)
     Magnetization: Mean +/-error (tau): 0.00405324 +/-0.00511815(36.5184)
     Magnetization^2: Mean +/-error (tau): 0.176925 +/-0.0015774(14.0395)
     Magnetization^4: Mean +/-error (tau): 0.0740855 +/-0.00112082(12.838)

     Simulation ran for 500001 steps.
     Binder cumulant: Mean +/-error (tau): 0.211088 +/-0.00371005(12.838) Relative error: 0.0175759


DIRECTORY STRUCTURE
===================

This section describes the source directory structure of the library.

    root
     |
     +-- common/                            Files that are common for all modules
     |    |
     |    +-- cmake/                        Module files for CMake
     |    |
     |    +-- deps/                         Source of ALPSCore build dependencies
     |    |
     |    +-- scripts/                      Various scripts needed for development
     |    |
     |    +-- build/                        Build scripts for Jenkins
     |
     +-- accumulators/                      Accumulators module
     |    |
     |    +-- CMakeLists.txt                CMakeLists file for the Accumulators module
     |    |
     |    +-- src/                          Sources for the Accumulators module
     |    |
     |    +-- include/alps/                 Headers for the Accumulators module
     |    |            |
     |    |            +-- accumulators.hpp    Main header file for the Accumulators module
     |    |            |
     |    |            +-- accumulators/       Headers for the Accumulators module
     |    |
     |    +-- test/                         Test sources and headers for the Accumulators module
     |
     |
     +-- gf/                      Green's Functions module
     |    |
     |    +-- CMakeLists.txt                CMakeLists file for the Green's Functions module
     |    |
     |    +-- src/                          Sources for the Green's Functions module
     |    |
     |    +-- include/alps/gf               Headers for the Green's Functions module
     |    |
     |    +-- test/                         Test sources and headers for the Green's Functions module
     |
     +-- hdf5/                              HDF5 (Archive) module
     |    |
     |    +-- CMakeLists.txt                CMakeLists file for the HDF5 (Archive) module
     |    |
     |    +-- src/                          Sources for the HDF5 (Archive) module
     |    |
     |    +-- include/alps/                 Headers for the HDF5 (Archive) module
     |    |            |
     |    |            +-- hdf5.hpp         Main header file for the HDF5 (Archive) module
     |    |            |
     |    |            +-- hdf5/            Headers for the HDF5 (Archive) module
     |    |
     |    +-- test/                         Test sources and headers for the HDF5 (Archive) module
     |
     |
     +-- mc/                                Monte Carlo Scheduler module
     |    |
     |    +-- CMakeLists.txt                CMakeLists file for the Monte Carlo Scheduler module
     |    |
     |    +-- src/                          Sources for the Monte Carlo Scheduler module
     |    |
     |    +-- include/alps/mc/              Headers for the Monte Carlo Scheduler module
     |    |
     |    +-- test/                         Test sources and headers for the Monte Carlo Scheduler module
     |
     +-- params/                            Parameters module
     |    |
     |    +-- CMakeLists.txt                CMakeLists file for the Parameters module
     |    |
     |    +-- src/                          Sources for the Parameters module
     |    |
     |    +-- include/alps/                 Headers for the Parameters module
     |    |            |
     |    |            +-- params.hpp       Main header file for the Parameters module
     |    |            |
     |    |            +-- params/          Headers for the Parameters module
     |    |
     |    +-- test/                         Test sources and headers for the Parameters module
     |
     +-- utilities/                         Utilities module (used by other modules)
     |    |
     |    +-- CMakeLists.txt                CMakeLists file for the Utilities module
     |    |
     |    +-- src/                          Sources for the Utilities module
     |    |
     |    +-- include                       Headers for the Utilities module
     |    |   |
     |    |   +-- config.hpp.in             ALPSCore configuration header file template
     |    |   |
     |    |   +-- alps/
     |    |        |
     |    |        +-- numeric/            Headers for operations with numbers and vectors
     |    |        |
     |    |        +-- type_traits/        Headers for operations with C++ types
     |    |        |
     |    |        +-- utilities/          Headers for filename, signals and MPI operations
     |    |
     |    +- test/                         Test sources and headers for the Utilities module
     |
     +-- tutorials/                        Tutorials
          |
          +- accumulators/                 Accumulators usage tutorials
          |
          +- hdf5/                         Archive usage tutorial
          |
          +- linking/                      Tutorial for linking with ALPSCore
          |
          +- mc/                           Monte Carlo scheduler tutorials
          |   |
          |   +-- CMakeLists.txt           CMakeLists file for MC tutorials
          |   |
          |   +-- simple_mc/               Simple MC simulation
          |   |
          |   +-- 01-c++/                  1D Ising simulation without use of mcbase class
          |   |
          |   +-- 02-mcbase/               1D Ising simulation using mcbase class (legacy)
          |   |
          |   +-- ising_mc/                1D Ising simulation using mcbase class (new)
          |   |
          |   +-- ising2_mc/               2D Ising simulation using mcbase class
          |   
          +- params/                       Parameter usage tutorial
