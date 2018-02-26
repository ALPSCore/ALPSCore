### gftools cmake configuration

cmake_minimum_required (VERSION 3.1)

# gftools use only CXX compiler 
project (alps-gf CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# find additional cmake module in cmake dir
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../common/cmake)
# add common definitions, if being configured independently
if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  include(ALPSCommonModuleDefinitions)
  include(ALPSEnableMPI)
  include(ALPSEnableEigen)
endif()


gen_documentation()
if (DocumentationOnly)
  return()
endif()

# A gentle reminder for a developer that uses -DCMAKE_BUILD_TYPE=Debug
# (does not really try to analyze compilation flags)
string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)
if (CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  message("Warning: Debug build of GF module may produce slow code. Do not use in production.")
endif()

add_this_package(mesh)

add_boost()
add_hdf5()
add_eigen()
add_alps_package(alps-utilities alps-hdf5)
add_testing()

gen_cfg_module()
gen_pkg_config()
    
