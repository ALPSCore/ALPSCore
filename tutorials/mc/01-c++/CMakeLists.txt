# generate ising program
cmake_minimum_required(VERSION 2.8.12)
project(tutorial_01_ising CXX)

# Not strictly necessary, but will create executables that are
# aware of ALPSCore location
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(${PROJECT_NAME} ising.cpp main.cpp)

# Request the ALPSCore package with `mc` components.
# Note that the components that `mc` is dependent on are requested automatically.
# The ALPSCore package must be installed in some standard place (like /usr/local),
# or somewhere in your PATH. 
# Otherwise, point ALPSCore_DIR environment variable to the ALPScore installation tree.
find_package(ALPSCore REQUIRED mc)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES})
