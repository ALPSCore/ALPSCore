### gftools cmake configuration

cmake_minimum_required (VERSION 2.8.12)

# gftools use only CXX compiler 
project (ALPSCoreGF CXX)

# find additional cmake module in cmake dir
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Where to install?
message(STATUS "Will be installed in ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Redifine CMAKE_INSTALL_PREFIX if you wish to change install location")

### Dependencies
find_package(ALPSCore REQUIRED hdf5)
find_package(MPI REQUIRED)
include_directories("gf/include")
include_directories(${MPI_CXX_INCLUDE_PATH})

unset(LINK_ALL)
list(APPEND LINK_ALL ${ALPSCore_LIBRARIES})
    
add_library(gf gf/src/mesh.cpp gf/src/gf.cpp)
target_link_libraries(gf ${ALPSCore_LIBRARIES} ${MPI_CXX_LIBRARIES})
list(APPEND LINK_ALL gf)

# Build and enable tests
option(Testing "Enable testing" ON)
if (Testing)
    include(EnableGtests)
    enable_testing()
    add_subdirectory(test)
endif (Testing)

### Configuration file
configure_file("./cmake/GFToolsConfig.cmake.in" "./GFToolsConfig.cmake" @ONLY)

### Installation
install(TARGETS gf 
        LIBRARY DESTINATION "lib/"
        ARCHIVE DESTINATION "lib/")
install(DIRECTORY "gf/include" DESTINATION "." FILES_MATCHING PATTERN "*.hpp")
install(FILES "${CMAKE_BINARY_DIR}/GFToolsConfig.cmake" DESTINATION "./share")
