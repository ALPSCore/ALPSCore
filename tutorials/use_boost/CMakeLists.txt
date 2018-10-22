# This is a minimal CMake file that illustrates how to use Boost
# in project that uses ALPSCore

# we need at least cmake version 3.1
cmake_minimum_required(VERSION 3.1)

# the project is called 'use_boost', and it is a C++ project
project(use_boost CXX)

# the project relies on the ALPSCore package. If ALPSCore is not found
# automatically, specify its location using:
# export ALPSCore_DIR=/location/to/ALPSCORE/
find_package(ALPSCore 2.0  REQUIRED)

# if our code only uses header-only Boost, nothing special to be done:
# we only have one cpp file: it's called use_boost_header.cpp
# and it generates a program called "use_boost_header"
add_executable(use_boost_header use_boost_header.cpp)
# Use ALPSCore_LIBRARIES variable to link to ALPSCore 
target_link_libraries(use_boost_header ${ALPSCore_LIBRARIES})

# If our code uses some Boost library, we will need to ask CMake to find it.
# We should ask CMake to find the same Boost version
# that ALPSCore used.
# Let's ask for "filesystem" package (which itself depends on "system")

message("ALPSCore used Boost version ${ALPSCore_BOOST_VERSION}")
find_package(Boost ${ALPSCore_BOOST_VERSION} EXACT REQUIRED filesystem system)

message("Found Boost libraries: ${Boost_LIBRARIES}")

# Our executable "use_boost_fs" is made from a single source file "use_boost_fs.cpp"
add_executable(use_boost_fs use_boost_fs.cpp)

# ...and it depends on Boost libraries and ALPSCore
target_link_libraries(use_boost_fs ${ALPSCore_LIBRARIES} ${Boost_LIBRARIES})
