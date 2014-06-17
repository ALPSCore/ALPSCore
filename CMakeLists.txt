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
# ALPS_GLOBAL_BUILD means building project all at once
set(ALPS_GLOBAL_BUILD true)

if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory")
endif()

# enable Testing
option(Testing "Enable testing" ON)
if (Testing)
    enable_testing()
    include(ALPSEnableTests) #defined in common/cmake
endif(Testing)

# each module is defined as a cmake project in a subdirectory
add_subdirectory(utility)
alps_add_module(alps-utility utility)
add_subdirectory(hdf5)
alps_add_module(alps-hdf5 hdf5)
add_subdirectory(accumulator)
alps_add_module(alps-accumulator accumulator)
add_subdirectory(params)
alps_add_module(alps-params params)
add_subdirectory(mc)
alps_add_module(alps-mc mc)

