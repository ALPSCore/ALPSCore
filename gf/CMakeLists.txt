### gftools cmake configuration

cmake_minimum_required (VERSION 2.8.12)

# gftools use only CXX compiler 
project (alps-gf CXX)
message ("\nALPSCore module : ${PROJECT_NAME}\n")

# find additional cmake module in cmake dir
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../common/cmake)
include(ALPSCommonModuleDefinitions)

# GF uses boost::multi_array, to supress extra checks we need to define extra flags,
# otherwise codes will slow down to a crawl.
if (NOT ALPS_DEBUG)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DBOOST_DISABLE_ASSERTS")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DBOOST_DISABLE_ASSERTS")
endif()

gen_documentation()
if (DocumentationOnly)
  return()
endif()

add_this_package(mesh gf)
include(ALPSEnableMPI)

add_boost()
add_hdf5()
add_alps_package(alps-utilities alps-hdf5)
add_testing()

gen_cfg_module()
gen_pkg_config()
    
