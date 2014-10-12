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
  find_package (Boost COMPONENTS chrono program_options filesystem system serialization mpi REQUIRED)
else()
  find_package (Boost COMPONENTS chrono program_options filesystem system REQUIRED)
endif()
    message(STATUS "Boost includes: " ${Boost_INCLUDE_DIRS} )
    message(STATUS "Boost libs: " ${Boost_LIBRARIES} )
find_package (HDF5 REQUIRED)
    message(STATUS "HDF5 includes: " ${HDF5_INCLUDE_DIRS} )
    message(STATUS "HDF5 libs: " ${HDF5_LIBRARIES} )

if (NOT DEFINED ALPS_GLOBAL_BUILD)
find_package (alps-utilities REQUIRED)
    message(STATUS "alps-utilities includes: " ${alps-utilities_INCLUDE_DIRS} )
    message(STATUS "alps-utilities libs: " ${alps-utilities_LIBRARIES} )
find_package (alps-hdf5 REQUIRED)
    message(STATUS "alps-hdf5 includes: " ${alps-hdf5_INCLUDE_DIRS} )
    message(STATUS "alps-hdf5 libs: " ${alps-hdf5_LIBRARIES} )
find_package (alps-params REQUIRED)
    message(STATUS "alps-params includes: " ${alps-params_INCLUDE_DIRS} )
    message(STATUS "alps-params libs: " ${alps-params_LIBRARIES} )
find_package (alps-accumulator REQUIRED)
    message(STATUS "alps-accumulator includes: " ${alps-accumulator_INCLUDE_DIRS} )
    message(STATUS "alps-accumulator libs: " ${alps-accumulator_LIBRARIES} )
endif(NOT DEFINED ALPS_GLOBAL_BUILD)


# LINK_ALL - link for libraries
set (LINK_ALL ${alps-accumulator_LIBRARIES} ${alps-params_LIBRARIES} ${alps-hdf5_LIBRARIES} ${alps-utilities_LIBRARIES} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} )

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${alps-utilities_INCLUDE_DIRS}
    ${alps-hdf5_INCLUDE_DIRS}
    ${alps-params_INCLUDE_DIRS}
    ${alps-accumulator_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)

#configure_file("${PROJECT_SOURCE_DIR}/include/config.h.in" "${PROJECT_BINARY_DIR}/include/alps/config.h")
add_subdirectory(src)

# add tests - take two parameters Testing and TestXMLOutput
option(Testing "Enable testing" ON)
if (Testing)
    enable_testing()
    add_subdirectory(test)
endif (Testing)

# add tutorial
add_subdirectory(tutorials)

# installation
install(DIRECTORY include DESTINATION .
          FILES_MATCHING PATTERN "*.hpp" PATTERN "*.hxx"
        )

# Documentation
set(DOXYFILE_EXTRA_SOURCES "${DOXYFILE_EXTRA_SOURCES} ${PROJECT_SOURCE_DIR}/include" PARENT_SCOPE)
option(Documentation OFF)
if (Documentation)
    set(DOXYFILE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/include")
    set(DOXYFILE_IN "${PROJECT_SOURCE_DIR}/../common/doc/Doxyfile.in") 
    include(UseDoxygen)
endif(Documentation)

# Generate pkg-config file
configure_file("${PROJECT_SOURCE_DIR}/${PROJECT_NAME}.pc.in" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.pc")
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.pc" DESTINATION "lib/pkgconfig")

# Generate cmake Find module
set(PROJECT_SEARCH_FILE alps/mc/mcbase.hpp)
configure_file("${PROJECT_SOURCE_DIR}/../common/cmake/FindALPSModule.cmake.in" "${PROJECT_BINARY_DIR}/Find${PROJECT_NAME}.cmake" @ONLY)
install(FILES "${PROJECT_BINARY_DIR}/Find${PROJECT_NAME}.cmake" DESTINATION "share/cmake/Modules")

