#
# alps-hdf5 cmake configuration script.
#

cmake_minimum_required (VERSION 3.1)

# define project name
project (alps-hdf5 CXX)
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
endif()

add_this_package(archive
                 archivecontext
                 archive_free
                 archive_write
                 archive_read
                 archive_read_scalar_helpers
                 archive_read_vector_data_helper
                 archive_read_vector_attribute_helper)

add_boost()
add_hdf5()

add_alps_package(alps-utilities)

add_testing()
gen_pkg_config()
gen_cfg_module()
