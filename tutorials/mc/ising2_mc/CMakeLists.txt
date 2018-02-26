cmake_minimum_required(VERSION 3.1)
project(ising2_mc CXX)

add_executable(${PROJECT_NAME} main.cpp ising.cpp)
add_executable(${PROJECT_NAME}_mpi main_mpi.cpp ising.cpp)

# Request the ALPSCore package (with all components)
# The ALPSCore package must be installed in some
# standard place (like /usr/local),
# or somewhere in your PATH. 
# Otherwise, point ALPSCore_DIR environment variable
# to the ALPScore installation tree.
find_package(ALPSCore 2.0 REQUIRED)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_mpi ${ALPSCore_LIBRARIES})
