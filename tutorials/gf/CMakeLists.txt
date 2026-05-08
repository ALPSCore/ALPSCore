# This is a minimal CMake file that illustrates how to use GF component
# in project that uses ALPSCore

# we need at least cmake version 3.10
cmake_minimum_required(VERSION 3.10)

# the project is called 'gf', and it is a C++ project
project(gf CXX)

set(CMAKE_CXX_STANDARD 11)

# the project relies on the ALPSCore package. If ALPSCore is not found
# automatically, specify its location using:
# export ALPSCore_DIR=/location/to/ALPSCORE/
find_package(ALPSCore 2.2 REQUIRED gf)

# we only have one cpp file: it's called gf.cpp
add_executable(${PROJECT_NAME} gf.cpp)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES})
