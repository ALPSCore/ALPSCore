# 
# alps-python cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8)

# define project name
project (alps-python CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../common/cmake)

# add common definitions
include(ALPSCommonModuleDefinitions)

gen_documentation()

if (DocumentationOnly)
  return()
endif (DocumentationOnly)

# enable mpi
include(ALPSEnableMPI)
# search for python
find_package(PythonLibs)
if (NOT PYTHONLIBS_FOUND)
  message(FATAL_ERROR " Python libraries not found")
else(NOT PYTHONLIBS_FOUND)
  set(ALPS_HAVE_PYTHON True)
endif (NOT PYTHONLIBS_FOUND)
# find numpy
find_package(NumPy REQUIRED)
include_directories(${NUMPY_INCLUDE_DIRS})
# Boost
#set(Boost_USE_STATIC_LIBS        ON)
#set(Boost_USE_STATIC_RUNTIME    OFF)
set(BOOST_LIBRARY_LIST chrono program_options filesystem system)
if(ENABLE_MPI)
  list(APPEND BOOST_LIBRARY_LIST mpi)
endif(ENABLE_MPI)
if(ALPS_HAVE_PYTHON)
  list(APPEND BOOST_LIBRARY_LIST python)
endif(ALPS_HAVE_PYTHON)
add_boost(${BOOST_LIBRARY_LIST})

add_hdf5()
add_alps_package(alps-utilities alps-hdf5 alps-params alps-accumulators alps-mc)

add_this_package()

add_testing()

gen_pkg_config()
gen_header_only_find_module(alps/utilities/boost_python.hpp) 

