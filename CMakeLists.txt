#
# global cmake configuration file for ALPSCore
#

cmake_minimum_required (VERSION 3.0)
project(ALPSCore C CXX)

# include cmake scripts from common/cmake directory
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/./common/cmake)

include(ALPSCommonModuleDefinitions)

# check that compiler version is OK.
include(ALPSCompilerVersionCheck)

# take care of compiler-specific tweaks
include(ALPSCompilerTweaks)

include(ALPSEnableMPI)
include(ALPSEnableEigen)

# ALPS_GLOBAL_BUILD means building project all at once
set(ALPS_GLOBAL_BUILD true)

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
add_subdirectory(gf)
alps_add_module(alps-gf gf)
add_subdirectory(alea)
alps_add_module(alps-alea alea)

#Doxygen building is a function to prevent namespace damage
function(build_documentation_)
    set(DOXYFILE_SOURCE_DIR "${CMAKE_SOURCE_DIR}/common/doc")
    message(STATUS "Building documentation for sources in ${DOXYFILE_EXTRA_SOURCES}")  
    set(DOXYFILE_IN "${CMAKE_SOURCE_DIR}/common/doc/Doxyfile.in") 
    set(CMAKE_PROJECT_NAME "ALPSCore reference") # to provide name for the documentation 
    include(UseDoxygen)
    unset(CMAKE_PROJECT_NAME)
endfunction()
if(_am_building_documentation)
  build_documentation_()
endif(_am_building_documentation)

# unset(_am_building_documentation CACHE)

# miscellaneous operations
# Create the main config file
gen_cfg_main()
# Create the main *.hpp config file
gen_main_hpp_config()
# create module file for lmod manager 
configure_file("${CMAKE_SOURCE_DIR}/common/misc/alpscore.lmod.in" "${CMAKE_BINARY_DIR}/alpscore.lmod")

