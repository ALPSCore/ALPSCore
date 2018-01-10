cmake_minimum_required(VERSION 3.1)
project(ini_params CXX)

# Not strictly necessary, but will create executables that are
# aware of ALPSCore location
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(${PROJECT_NAME} main.cpp)

# Request the ALPSCore package with `params` components.
# The ALPSCore package must be installed in some standard place (like /usr/local),
# or somewhere in your PATH. 
# Otherwise, point ALPSCore_DIR environment variable to the ALPScore installation tree.
find_package(ALPSCore 2.0 REQUIRED params)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES})
