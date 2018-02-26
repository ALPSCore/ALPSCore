# 
# alps-params cmake configuration script.
# 

cmake_minimum_required (VERSION 3.1)

# define project name
project (alps-params CXX)
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

# The `EXTRA` is a workaround against a misfeature in CMake 3.1
#  not allowing $<TARGET_OBJECTS:tgt> in `target_sources()`
add_this_package(params dict_value dictionary iniparser_interface EXTRA $<TARGET_OBJECTS:libiniparser>)

add_boost()
add_hdf5()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../common/deps/iniparser" "./iniparser")
if (ALPS_BUILD_SHARED OR ALPS_BUILD_PIC)
  # FIXME: is it necessary or CMake can figure it out?
  set_target_properties(libiniparser PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
target_include_directories(${PROJECT_NAME} PRIVATE $<TARGET_PROPERTY:libiniparser,INTERFACE_INCLUDE_DIRECTORIES>)
# This does not work in CMake 3.1:
# target_sources(${PROJECT_NAME} PRIVATE $<TARGET_OBJECTS:libiniparser>)

add_alps_package(alps-utilities alps-hdf5)

add_testing()

gen_pkg_config()
gen_cfg_module()


