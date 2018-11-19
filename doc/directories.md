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
     +-- accumulators/                      Legacy accumulators module
     |    |
     |    +-- CMakeLists.txt                CMakeLists file for the legacy Accumulators module
     |    |
     |    +-- src/                          Sources for the legacy Accumulators module
     |    |
     |    +-- include/alps/                 Headers for the legacy Accumulators module
     |    |            
     |    +-- test/                         Test sources and headers for the legacy Accumulators module
     |
     +-- alea/
     |    |
     |    +-- alps-alea.pc.in               Package managaer template for ALEA module
     |    |
     |    +-- CMakeLists.txt                CMakeLists file for ALEA module
     |    |
     |    +-- include/alps                  Headers for ALEA module
     |    |
     |    +-- src                           Sources for ALEA module
     |    |
     |    +-- test                          Test sources and headers for the ALEA module
     |    |
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
          +- accumulators/                 legacy Accumulators usage tutorials
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
