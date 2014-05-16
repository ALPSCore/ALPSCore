#
# Provide common definitions for building alps modules 
#

# Disable in-source builds
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
