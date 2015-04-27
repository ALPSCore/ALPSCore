# 
# alps-mc cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8)

# define project name
project (alps-mc CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)

# add common definitions
include(ALPSCommonModuleDefinitions)

if (NOT DocumentationOnly)
  include(ALPSEnableMPI)

  # Boost
  #set(Boost_USE_STATIC_LIBS        ON)
  #set(Boost_USE_STATIC_RUNTIME    OFF)
  set(BOOST_LIBRARY_LIST chrono program_options filesystem system)
  if(ENABLE_MPI)
    list(APPEND BOOST_LIBRARY_LIST mpi)
  endif(ENABLE_MPI)
  add_boost(${BOOST_LIBRARY_LIST})

  add_hdf5()
  add_alps_package(alps-utilities alps-hdf5 alps-params alps-accumulators)

  add_this_package()

  add_testing()

  if (ALPS_BUILD_TUTORIALS)
      # add tutorial
      add_subdirectory(tutorials)
  endif(ALPS_BUILD_TUTORIALS)

  gen_pkg_config()
  # gen_find_module(alps/mc/mcbase.hpp)

endif (NOT DocumentationOnly)
gen_documentation()
