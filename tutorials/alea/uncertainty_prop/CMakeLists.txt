cmake_minimum_required(VERSION 3.10)
project(uncertainty_prop CXX)

# Not strictly necessary, but will create executables that are
# aware of ALPSCore location
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(${PROJECT_NAME} main.cpp)

# Request the ALPSCore package with `alea` component.  It must be installed
#   - either in a standard system path `/usr`
#   - somewhere in your $PATH
#   - in the directory corresponding to $ALPSCore_DIR
find_package(ALPSCore REQUIRED alea)

# Use ALPSCore_LIBRARIES variable to link to ALPSCore
target_link_libraries(${PROJECT_NAME} ${ALPSCore_LIBRARIES})
