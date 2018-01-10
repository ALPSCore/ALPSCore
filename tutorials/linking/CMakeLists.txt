# This is a minimal CMake file that illustrates how to link your
# project to ALPSCore
# Several additional options are possible (or even recommended), 
# this tutorial only illustrates the minimal options

# we need at least cmake version 3.1
cmake_minimum_required(VERSION 3.1)

# the project is called 'linking', and it is a C++ project
project(linking CXX)

# the project relies on the ALPSCore package. If ALPSCore is not found
# automatically, specify its location using:
# export ALPSCore_DIR=/location/to/ALPSCORE/
find_package(ALPSCore 2.0  REQUIRED)

# we only have one cpp file: it's called linking.cpp
add_executable(${PROJECT_NAME} linking.cpp)


# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES})
