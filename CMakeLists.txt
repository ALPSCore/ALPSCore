### gftools cmake configuration

cmake_minimum_required (VERSION 2.8.12)

# gftools use only CXX compiler 
project (ALPSCoreGF CXX)

# find additional cmake module in cmake dir
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


### Dependencies
find_package(ALPSCore REQUIRED hdf5)
include_directories("gf/include")

unset(LINK_ALL)
list(APPEND LINK_ALL ${ALPSCore_LIBRARIES})
    

add_library(gf gf/src/mesh.cpp gf/src/gf.cpp)
target_link_libraries(gf ${ALPSCore_LIBRARIES})
list(APPEND LINK_ALL gf)

# Build and enable tests
option(Testing "Enable testing" ON)
if (Testing)
    include(EnableGtests)
    enable_testing()
    add_subdirectory(test)
endif (Testing)

