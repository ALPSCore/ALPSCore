# 
# alps-mc cmake configuration script.
# 

cmake_minimum_required (VERSION 2.8)

# add common modules from ../common/cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/../common/cmake)

message (STATUS ${CMAKE_MODULE_PATH}) 
include(ALPSCommonModuleDefinitions)
message (STATUS ${CMAKE_MODULE_PATH}) 

# define project name
project (alps-mc CXX)

# Boost
#set(Boost_USE_STATIC_LIBS        ON)
#set(Boost_USE_STATIC_RUNTIME    OFF)
find_package (Boost COMPONENTS chrono program_options filesystem system REQUIRED)
    message(STATUS "Boost includes: " ${Boost_INCLUDE_DIRS} )
    message(STATUS "Boost libs: " ${Boost_LIBRARIES} )
find_package (HDF5 REQUIRED)
    message(STATUS "HDF5 includes: " ${HDF5_INCLUDE_DIRS} )
    message(STATUS "HDF5 libs: " ${HDF5_LIBRARIES} )

find_package (alps-utility REQUIRED)
    message(STATUS "alps-utility includes: " ${alps-utility_INCLUDE_DIRS} )
    message(STATUS "alps-utility libs: " ${alps-utility_LIBRARIES} )
find_package (alps-hdf5 REQUIRED)
    message(STATUS "alps-hdf5 includes: " ${alps-hdf5_INCLUDE_DIRS} )
    message(STATUS "alps-hdf5 libs: " ${alps-hdf5_LIBRARIES} )
find_package (alps-params REQUIRED)
    message(STATUS "alps-params includes: " ${alps-params_INCLUDE_DIRS} )
    message(STATUS "alps-params libs: " ${alps-params_LIBRARIES} )
find_package (alps-accumulator REQUIRED)
    message(STATUS "alps-accumulator includes: " ${alps-accumulator_INCLUDE_DIRS} )
    message(STATUS "alps-accumulator libs: " ${alps-accumulator_LIBRARIES} )


# LINK_ALL - link for libraries
set (LINK_ALL ${alps-accumulator_LIBRARIES} ${alps-params_LIBRARIES} ${alps-hdf5_LIBRARIES} ${alps-utility_LIBRARIES} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} )
# LINK_TEST - link for tests
set (LINK_TEST )

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${alps-utility_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
)

#configure_file("${PROJECT_SOURCE_DIR}/include/config.h.in" "${CMAKE_BINARY_DIR}/include/alps/config.h")
add_subdirectory(src)

# add tests - take two parameters Testing and TestXMLOutput
option(Testing "Enable testing" ON)
if (Testing)
    enable_testing()
    add_subdirectory(test)
endif (Testing)

# installation
install(DIRECTORY include DESTINATION .
          FILES_MATCHING PATTERN "*.hpp" PATTERN "*.hxx"
        )

# Documentation
set(DOXYFILE_SOURCE_DIR "./include")
include(UseDoxygen)

# Generate pkg-config file
configure_file("${PROJECT_SOURCE_DIR}/${PROJECT_NAME}.pc.in" "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc")
install(FILES "${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc" DESTINATION "lib/pkgconfig")

# Generate cmake Find module
set(PROJECT_SEARCH_FILE alps/mcbase.hpp)
configure_file("${CMAKE_SOURCE_DIR}/../common/cmake/FindALPSModule.cmake.in" "${CMAKE_BINARY_DIR}/Find${PROJECT_NAME}.cmake" @ONLY)
install(FILES "${CMAKE_BINARY_DIR}/Find${PROJECT_NAME}.cmake" DESTINATION "share/cmake/Modules")

