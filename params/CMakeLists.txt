# 
# alps-params cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8.12)

# define project name
project (alps-params CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)

# add common definitions
include(ALPSCommonModuleDefinitions)

gen_documentation()

if (DocumentationOnly)
  return()
endif()

add_this_package(params)

include(ALPSEnableMPI)

# Boost
if(ALPS_HAVE_MPI)
  add_boost(filesystem system program_options serialization mpi)
else()
  add_boost(filesystem system program_options serialization)
endif()
add_hdf5()

add_alps_package(alps-utilities alps-hdf5)

add_testing()

gen_pkg_config()
gen_cfg_module()


