#
# global cmake configuration file for ALPSCore
#

cmake_minimum_required (VERSION 3.1)
project(ALPSCore C CXX)

# # TODO: should probably a bit more fine-grained
# # Is expected to be taken care of by ALPS_CXX_STD, which works transitively
# set(CMAKE_CXX_STANDARD 11)

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

# Normalize the list of disabled modules (aka packages)
# FIXME: this inconsistent dual-naming "${module}" vs "alps-${module}" should go away!
foreach(module_ ${ALPS_MODULES_DISABLE})
  list(APPEND ALPS_MODULES_DISABLE "alps-${module_}")
endforeach()

# each module is defined as a cmake project in a subdirectory
# also add to utilities/include/config.hpp.in under "Available ALPSCore components"
set(ALPS_KNOWN_COMPONENTS)
foreach(module_ utilities hdf5 accumulators params mc gf alea)
  list(FIND ALPS_MODULES_DISABLE ${module_} disabled_index_)
  if (disabled_index_ EQUAL -1)
    add_subdirectory(${module_})
    alps_add_module("alps-${module_}" ${module_})
    list(APPEND ALPS_KNOWN_COMPONENTS ${module_})
  else()
    message("\nNOTE: module ${module_} is disabled")
  endif()
endforeach()

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

