### gftools cmake configuration

cmake_minimum_required (VERSION 2.8.12)

# gftools use only CXX compiler 
project (ALPSCoreGF CXX)

# Disable build in source
if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory.")
endif()

#set(CMAKE_DISABLE_SOURCE_CHANGES ON)
#set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# find additional cmake module in cmake dir
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


### Dependencies
# Boost
#find_package (Boost) 
find_package(ALPSCore REQUIRED hdf5)

unset(LINK_ALL)
list(APPEND LINK_ALL ${ALPSCore_LIBRARIES})

# Build and enable tests
option(Testing "Enable testing" ON)
if (Testing)
    message(STATUS "Building tests")
    include(EnableGtests)
    add_subdirectory(test)
    enable_testing()
endif (Testing)

