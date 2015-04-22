#  Try to find ALPSCore. Depends on  Once done this will define
#  ALPSCore_FOUND        - System has ALPSCore
#  ALPSCore_INCLUDE_DIRS - The ALPSCore include directories
#  ALPSCore_LIBRARIES    - The ALPSCore libraries
#  ALPSCore_DEFINITIONS  - Compiler switches required for using ALPSCore

# The script relies on installed Find Modules for each component

# list(APPEND CMAKE_MODULE_PATH ${ALPS_ROOT}/share/cmake/Modules)

# This function adds the given directory and 3 levels of parent directories to the list 
function(alps_add_parent_dirs list_ dir_)
  message(STATUS "Inside: list_=${list_} contains ${${list_}}")
  list(APPEND ${list_} ${dir_} ${dir_}/.. ${dir_}/../.. ${dir_}/../../..)
  message(STATUS "Inside: list_=${list_} contains ${${list_}}")
  set(${list_} ${${list_}} PARENT_SCOPE)
endfunction()

# Create a list of known components
list(APPEND known_components hdf5 accumulators params mc python)

# Create a list of directories to search
list(APPEND alpscore_dirs ${ALPS_ROOT} ${ALPSCore_DIR})
message(STATUS "alpscore_dirs=${alpscore_dirs}")
alps_add_parent_dirs(alpscore_dirs  ${CMAKE_CURRENT_LIST_DIR})
message(STATUS "alpscore_dirs=${alpscore_dirs}")

# Start searching from utility
find_package(alps-utilities HINTS ${alpscore_dirs})
set(ALPSCore_INCLUDES ${alps-utilities_INCLUDE_DIRS}) 
set(ALPSCore_LIBRARIES ${alps-utilities_LIBRARIES})
list(APPEND alpscore_dirs ${alps-utilities_DIR})

# if no components required - search for everything
list(LENGTH ALPSCore_FIND_COMPONENTS comp_len)
if (${comp_len} EQUAL 0) 
    set(ALPSCore_FIND_COMPONENTS ${known_components})
endif()

# check for each required component
foreach(component ${ALPSCore_FIND_COMPONENTS})
    list(FIND known_components ${component} has_module)
    if (${has_module} EQUAL -1) 
        message(FATAL_ERROR "ALPSCore : Unknown component ${component}")
    else()
        set(alps_module "alps-${component}")
        find_package(${alps_module} REQUIRED HINTS ${alpscore_dirs})
        list(APPEND ${alpscore_dirs} ${${alps_module}_DIR})

        # Add includes (do not add more than once)
        list(FIND ALPSCore_INCLUDES ${${alps_module}_INCLUDE_DIRS} is_already_there)
        if (${is_already_there} EQUAL -1)
            list(APPEND ALPSCore_INCLUDES ${${alps_module}_INCLUDE_DIRS}) 
        endif(${is_already_there} EQUAL -1)

        if (NOT ${${alps_module}_LIBRARIES} STREQUAL "")
            # Add libraries (do not add more than once)
            list(FIND ALPSCore_LIBRARIES ${${alps_module}_LIBRARIES} is_already_there)
            if (${is_already_there} EQUAL -1)
                list(APPEND ALPSCore_LIBRARIES ${${alps_module}_LIBRARIES})
            endif(${is_already_there} EQUAL -1)
        endif (NOT ${${alps_module}_LIBRARIES} STREQUAL "")
    endif()
endforeach()

# Reverse the list of libraries to put utility in the end
# This is vital for old linkers 
list(REVERSE ALPSCore_LIBRARIES)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ALPSCore DEFAULT_MSG ALPSCore_LIBRARIES ALPSCore_INCLUDES)

# Definitions for compatibility
set(ALPSCore_INCLUDE_DIR  ${ALPSCore_INCLUDES} )
set(ALPSCore_INCLUDE_DIRS ${ALPSCore_INCLUDES} )
set(ALPSCore_LIBRARY      ${ALPSCore_LIBRARIES})

mark_as_advanced(
    ALPSCore_INCLUDES
    ALPSCore_INCLUDE_DIR
    ALPSCore_INCLUDE_DIRS
    ALPSCore_LIBRARIES
    ALPSCore_LIBRARY
    )
