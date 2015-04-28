# 
# alps-accumulators cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8.12)

# project name
project (alps-accumulators CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)

# add common definitions
include(ALPSCommonModuleDefinitions)

if (NOT DocumentationOnly)
  include(ALPSEnableMPI)

  # Boost
  if(ENABLE_MPI)
    add_boost(filesystem system serialization mpi)
  else()
    add_boost(filesystem system)
  endif()
  add_hdf5()
  add_alps_package(alps-utilities alps-hdf5)
  add_this_package()
  add_testing()
  gen_pkg_config()
  # gen_find_module(alps/accumulators/accumulator.hpp)
endif(NOT DocumentationOnly)
gen_documentation()
