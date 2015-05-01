cmake_minimum_required(VERSION 2.8.12)
project(ini_params CXX)

# Not strictly necessary, but will create executables that are
# aware of ALPSCore location
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(${PROJECT_NAME} main.cpp)

# Request the ALPSCore package with `parms` components.
# The ALPSCore package must be installed in some standard place (like /usr/local),
# or somewhere in your PATH. 
# Otherwise, point ALPSCore_DIR environment variable to the ALPScore installation tree.
find_package(ALPSCore REQUIRED params)

# FIXME: this must not be necessary --- make dependencies transitive! (NOTE: library dependencies are already transitive)
if (ALPSCore_HAS_MPI)
  # If ALPSCore uses MPI, add MPI headers and libraries to the project
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH})
endif()

# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES})
