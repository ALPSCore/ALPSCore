cmake_minimum_required(VERSION 3.1)
project(mpi_simple CXX)

# Not strictly necessary, but will create executables that are
# aware of ALPSCore location
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Request the ALPSCore package
# The ALPSCore package must be installed in some standard place (like /usr/local),
# or somewhere in your PATH. 
# Otherwise, point ALPSCore_DIR environment variable to the ALPScore installation tree.
find_package(ALPSCore 2.0 REQUIRED)

# Check that ALPSCore is built with MPI support
if (NOT ALPSCore_HAS_MPI)
  message(FATAL_ERROR "Your version of ALPSCore is built without MPI support; please rebuild it.")
endif()

add_executable(${PROJECT_NAME} main.cpp)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore
# Note that we do not request MPI library: ALPSCore provides it as a transitive dependency
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES})
