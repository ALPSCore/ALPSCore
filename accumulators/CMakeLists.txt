#
# alps-accumulators cmake configuration script.
#

cmake_minimum_required (VERSION 3.1)

# project name
project (alps-accumulators CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)

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

add_this_package(namedaccumulators
                 accumulator_wrapper
                 result_wrapper
                 wrapper_set
                 wrapper_set_hdf5
                 mpi
                 feature/count
                 feature/mean
                 feature/error
                 feature/binning_analysis
                 feature/max_num_binning)

add_boost()
add_hdf5()
add_eigen()
add_alps_package(alps-utilities alps-hdf5)
add_testing()
gen_pkg_config()
gen_cfg_module()
