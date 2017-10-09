# This is a minimal CMake file that illustrates how to use Eigen
# in project that uses ALPSCore

# we need at least cmake version 2.8.12
cmake_minimum_required(VERSION 2.8.12)

# the project is called 'use_eigen', and it is a C++ project
project(use_eigen CXX)

# the project relies on the ALPSCore package. If ALPSCore is not found
# automatically, specify its location using:
# export ALPSCore_DIR=/location/to/ALPSCORE/
find_package(ALPSCore 1.0  REQUIRED)

# check if Eigen is present:
if (ALPSCore_HAS_EIGEN_VERSION)
  message(STATUS "ALPSCore provides Eigen, version ${ALPSCore_HAS_EIGEN_VERSION}")
else()
  message("WARNING: ALPSCore does not provide Eigen, the program won't compile!")
endif()

# we only have one cpp file: it's called use_eigen.cpp
add_executable(${PROJECT_NAME} use_eigen.cpp)


# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES})
