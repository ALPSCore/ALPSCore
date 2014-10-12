#
# global cmake configuration file for ALPSCore
#

cmake_minimum_required (VERSION 2.8)

# add includes and libs for each module
macro(alps_add_module module module_path)
    set(${module}_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/${module_path}/include ${CMAKE_BINARY_DIR}/${module_path}/include)
    set(${module}_LIBRARIES ${module})
endmacro(alps_add_module)

# include cmake scripts from common/cmake directory
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/./common/cmake)

# check that compiler version is OK.
include(ALPSCompilerVersionCheck)

# ALPS_GLOBAL_BUILD means building project all at once
set(ALPS_GLOBAL_BUILD true)

if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory")
endif()

# enable Testing
option(Testing "Enable testing" ON)
option(ExtensiveTesting "Enable extensive testing, including time consuming tests" OFF)
mark_as_advanced(ExtensiveTesting)
if (Testing)
    enable_testing()
    include(ALPSEnableTests) #defined in common/cmake
endif(Testing)

# Documentation
option(Documentation ON)
set(_am_building_documentation ${Documentation})
if (Documentation)
    message(STATUS "Enable build of documentation")
    set(Documentation OFF) # not to spawn make doc jobs for each module
endif(Documentation)

# each module is defined as a cmake project in a subdirectory
add_subdirectory(utilities)
alps_add_module(alps-utilities utilities)
add_subdirectory(hdf5)
alps_add_module(alps-hdf5 hdf5)
add_subdirectory(accumulator)
alps_add_module(alps-accumulator accumulator)
add_subdirectory(params)
alps_add_module(alps-params params)
add_subdirectory(mc)
alps_add_module(alps-mc mc)

if(_am_building_documentation)
    set(DOXYFILE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/common/doc")
    message(STATUS "Building documentation for sources in ${DOXYFILE_EXTRA_SOURCES}")  
    set(DOXYFILE_IN "${CMAKE_SOURCE_DIR}/common/doc/Doxyfile.in") 
    include(UseDoxygen)
endif(_am_building_documentation)
