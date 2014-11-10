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

# configurable option
option(ENABLE_PYTHON "Enable Python libraries" ON)

if (NOT DocumentationOnly AND ENABLE_PYTHON)
  include(ALPSEnableMPI)

  find_package(PythonLibs)
  if (NOT PYTHONLIBS_FOUND)
    message(FATAL_ERROR " Python libraries not found")
  endif (NOT PYTHONLIBS_FOUND)

  # Boost
  #set(Boost_USE_STATIC_LIBS        ON)
  #set(Boost_USE_STATIC_RUNTIME    OFF)
  set(BOOST_LIBRARY_LIST chrono program_options filesystem system)
  if(ENABLE_MPI)
    list(APPEND BOOST_LIBRARY_LIST mpi)
  endif(ENABLE_MPI)
  if(ENABLE_PYTHON)
    list(APPEND BOOST_LIBRARY_LIST python)
  endif(ENABLE_PYTHON)
  add_boost(${BOOST_LIBRARY_LIST})

  add_hdf5()
  add_alps_package(alps-utilities alps-hdf5 alps-params alps-accumulators alps-mc)

  add_this_package()

  add_testing()

  gen_pkg_config()

endif (NOT DocumentationOnly AND ENABLE_PYTHON)
gen_documentation()
