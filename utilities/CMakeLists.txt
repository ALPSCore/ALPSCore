# 
# alps-utilities cmake configuration script.
# 

cmake_minimum_required (VERSION 3.1)

# define project name
project (alps-utilities CXX)
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

include(CheckIncludeFile)

add_this_package(
    # copyright //FIXME
    unique_file
    filename_operations
    stacktrace
    signal
    gtest_par_xml_output
)

add_boost()
add_eigen()

add_testing()
CHECK_INCLUDE_FILE(unistd.h ALPS_HAVE_UNISTD_H)
gen_pkg_config()
gen_cfg_module()
