# 
# alps-mc cmake configuration script.
# 

cmake_minimum_required (VERSION 3.1)

# define project name
project (alps-mc CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)
# add common definitions, if being configured independently
if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  include(ALPSCommonModuleDefinitions)
  include(ALPSEnableMPI)
endif()


gen_documentation()
if (DocumentationOnly)
  return()
endif ()

add_this_package(mcbase api stop_callback)

add_boost()

add_hdf5()
add_alps_package(alps-utilities alps-hdf5 alps-params alps-accumulators)

add_testing()

gen_cfg_module()
gen_pkg_config()
