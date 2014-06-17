#
# Provide common definitions for building alps modules 
#

# Disable in-source builds
if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory")
endif()

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# RPATH fix
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
 set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
else()
 set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

# Build static XOR shared 
# Defines ALPS_BUILD_TYPE=STATIC|DYNAMIC .
option(BuildStatic "Build static libraries" ON)
option(BuildShared "Build shared libraries" OFF)
if (BuildStatic AND NOT BuildShared) 
    message(STATUS "Building static libraries")
    set(ALPS_BUILD_TYPE STATIC)
elseif(BuildShared AND NOT BuildStatic)
    message(STATUS "Building shared libraries")
    set(ALPS_BUILD_TYPE SHARED)
else()
    message(FATAL_ERROR "Please choose BuildStatic XOR BuildShared type of building libraries.")
endif()

# Define ALPS_ROOT and add it to cmake module path 
if (NOT DEFINED ALPS_ROOT)
    set(ALPS_ROOT ${CMAKE_INSTALL_PREFIX})
endif ()

if (NOT IS_ABSOLUTE ${ALPS_ROOT})
    set (ALPS_ROOT ${CMAKE_BINARY_DIR}/${ALPS_ROOT}) # FIXME: unstable
endif()

message (STATUS "ALPS_ROOT: ${ALPS_ROOT}")
list(APPEND CMAKE_MODULE_PATH ${ALPS_ROOT}/share/cmake/Modules)


