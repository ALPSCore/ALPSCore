#
# global cmake configuration file for ALPSCore
#

cmake_minimum_required (VERSION 2.8.12)

# add includes and libs for each module
macro(alps_add_module module module_path)
    set(${module}_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/${module_path}/include ${CMAKE_BINARY_DIR}/${module_path}/include)
    set(${module}_LIBRARIES ${module})
endmacro(alps_add_module)

# include cmake scripts from common/cmake directory
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/./common/cmake)

# check that compiler version is OK.
include(ALPSCompilerVersionCheck)

# take care of compiler-specific tweaks
include(ALPSCompilerTweaks)

# ALPS_GLOBAL_BUILD means building project all at once
set(ALPS_GLOBAL_BUILD true)

# disable in-source builds
if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory")
endif()

# Documentation
option(DocumentationOnly "Build only documentation, no code or tests" OFF)
mark_as_advanced(DocumentationOnly)

option(Documentation "Build documentation" ON)
if (DocumentationOnly)
  set(Documentation ON)
endif (DocumentationOnly)

set(_am_building_documentation ${Documentation})
if (Documentation)
    message(STATUS "Enable build of documentation")
    set(Documentation OFF) # not to spawn make doc jobs for each module
endif(Documentation)

# enable Testing
option(Testing "Enable testing" ON)
option(ExtensiveTesting "Enable extensive testing, including time consuming tests" OFF)
mark_as_advanced(ExtensiveTesting)
if (Testing AND NOT DocumentationOnly)
    enable_testing()
    include(ALPSEnableTests) #defined in common/cmake
endif(Testing AND NOT DocumentationOnly)

# FIXME: remove the following
# # enable tutorials
# option(BuildTutorials "Build tutorials" OFF)
# if (BuildTutorials)
#     set(ALPS_BUILD_TUTORIALS ON)
# endif()

# each module is defined as a cmake project in a subdirectory
add_subdirectory(utilities)
alps_add_module(alps-utilities utilities)
add_subdirectory(hdf5)
alps_add_module(alps-hdf5 hdf5)
add_subdirectory(accumulators)
alps_add_module(alps-accumulators accumulators)
add_subdirectory(params)
alps_add_module(alps-params params)
add_subdirectory(mc)
alps_add_module(alps-mc mc)

# Python is no more built as part of the main ALPSCore project.
# if (BuildPython)
#     set(ALPS_HAVE_PYTHON ON)
#     # python libraries
#     add_subdirectory(python)
#     alps_add_module(alps-python python)
# endif()

if(_am_building_documentation)
    set(DOXYFILE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/common/doc")
    message(STATUS "Building documentation for sources in ${DOXYFILE_EXTRA_SOURCES}")  
    set(DOXYFILE_IN "${CMAKE_SOURCE_DIR}/common/doc/Doxyfile.in") 
    set(CMAKE_PROJECT_NAME "ALPSCore reference") # to provide name for the documentation 
    include(UseDoxygen)
    unset(CMAKE_PROJECT_NAME)
endif(_am_building_documentation)
# unset(_am_building_documentation CACHE)

# miscellaneous operations
# create module file for lmod manager 
configure_file("${CMAKE_SOURCE_DIR}/common/misc/alpscore.lmod.in" "${CMAKE_BINARY_DIR}/alpscore.lmod")
