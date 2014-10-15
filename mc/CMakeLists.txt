# 
# alps-mc cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8)

# define project name
project (alps-mc CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)

# add common definitions
include(ALPSCommonModuleDefinitions)
include(ALPSEnableMPI)

# Boost
#set(Boost_USE_STATIC_LIBS        ON)
#set(Boost_USE_STATIC_RUNTIME    OFF)
if(ALPS_HAVE_MPI)
  add_boost(chrono program_options filesystem system serialization mpi)
else()
  add_boost(chrono program_options filesystem system)
endif()
add_hdf5()
add_alps_package(alps-utilities alps-hdf5 alps-params alps-accumulator)

add_this_package()

add_testing()

# add tutorial
add_subdirectory(tutorials)

gen_pkg_config()
gen_find_module(alps/mc/mcbase.hpp)
gen_documentation()
