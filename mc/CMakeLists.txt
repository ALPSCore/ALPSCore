# 
# alps-mc cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8.12)

# define project name
project (alps-mc CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)
# add common definitions
include(ALPSCommonModuleDefinitions)

gen_documentation()
if (DocumentationOnly)
  return()
endif ()

add_this_package(mcbase api parseargs stop_callback)

include(ALPSEnableMPI)

set(boost_libs_ chrono program_options filesystem system serialization)
if(ENABLE_MPI)
  list(APPEND boost_libs_ mpi)
endif(ENABLE_MPI)
add_boost(${boost_libs_})

add_hdf5()
add_alps_package(alps-utilities alps-hdf5 alps-params alps-accumulators)

add_testing()

gen_cfg_module()
gen_pkg_config()
