# 
# alps-hdf5 cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8.12)

# define project name
project (alps-hdf5 CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)

include(ALPSCommonModuleDefinitions)

if (NOT DocumentationOnly)
  include(ALPSEnableMPI)

  # Boost
  #set(Boost_USE_STATIC_LIBS        ON)
  #set(Boost_USE_STATIC_RUNTIME    OFF)
  add_boost(filesystem system program_options)
  add_hdf5()

  add_alps_package(alps-utilities)

  add_this_package()
  add_testing()

  gen_pkg_config()
  # gen_find_module(alps/hdf5.hpp)
endif (NOT DocumentationOnly)
gen_documentation()
