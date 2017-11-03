# 
# alps-params cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8.12)

# define project name
project (alps-params CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)

# add common definitions, if being configured independently
if (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  include(ALPSCommonModuleDefinitions)
endif()

gen_documentation()

if (DocumentationOnly)
  return()
endif()

add_this_package(params dict_value dictionary iniparser_interface)

include(ALPSEnableMPI)

add_boost()
add_hdf5()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/deps/iniparser" "./iniparser")
if (ALPS_BUILD_SHARED OR ALPS_BUILD_PIC)
  # FIXME: is it necessary or CMake can figure it out?
  set_target_properties(libiniparser PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE libiniparser)

add_alps_package(alps-utilities alps-hdf5)

add_testing()

gen_pkg_config()
gen_cfg_module()


