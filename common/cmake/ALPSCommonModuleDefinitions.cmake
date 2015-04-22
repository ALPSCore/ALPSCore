#
# Provide common definitions for building alps modules 
#

# Disable in-source builds
if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory")
endif()

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# RPATH fix
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
 set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")
else()
 set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

#policy update CMP0042
if(APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif()

# Build static XOR shared 
# Defines ALPS_BUILD_TYPE=STATIC|DYNAMIC .
option(BuildStatic "Build static libraries" OFF)
option(BuildShared "Build shared libraries" ON)
option(BuildPython "Build Python interface" ON)
if (BuildStatic AND NOT BuildShared) 
    message(STATUS "Building static libraries")
    set(ALPS_BUILD_TYPE STATIC)
    set(BUILD_SHARED_LIBS OFF)
elseif(BuildShared AND NOT BuildStatic)
    message(STATUS "Building shared libraries")
    set(ALPS_BUILD_TYPE SHARED)
    set(BUILD_SHARED_LIBS ON)
else()
    message(FATAL_ERROR "Please choose EITHER BuildStatic OR BuildShared type of building libraries, NOT both")
endif()
if (BuildPython AND NOT BuildShared)
    message(FATAL_ERROR "Python interface requires a shared (BuildShared=ON) build")
endif()


# Define ALPS_ROOT and add it to cmake module path 
if (NOT DEFINED ALPS_ROOT)
    set(ALPS_ROOT ${CMAKE_INSTALL_PREFIX})
endif ()

if (NOT IS_ABSOLUTE ${ALPS_ROOT})
    set (ALPS_ROOT ${CMAKE_BINARY_DIR}/${ALPS_ROOT}) # FIXME: unstable
endif()

message (STATUS "ALPS_ROOT: ${ALPS_ROOT}")
list(APPEND CMAKE_MODULE_PATH ${ALPS_ROOT}/share/cmake/Modules)


## Some macros

macro(add_boost) # usage: add_boost(component1 component2...)
  find_package (Boost COMPONENTS ${ARGV} REQUIRED)
  message(STATUS "Boost includes: ${Boost_INCLUDE_DIRS}" )
  message(STATUS "Boost libs: ${Boost_LIBRARIES}" )
  include_directories(${Boost_INCLUDE_DIRS})
  list(APPEND LINK_ALL ${Boost_LIBRARIES})
endmacro(add_boost)

macro(add_hdf5) 
  find_package (HDF5 REQUIRED)
  message(STATUS "HDF5 includes: ${HDF5_INCLUDE_DIRS}" )
  message(STATUS "HDF5 libs: ${HDF5_LIBRARIES}" )
  include_directories(${HDF5_INCLUDE_DIRS})
  list(APPEND LINK_ALL ${HDF5_LIBRARIES})
endmacro(add_hdf5)

macro(add_alps_package) # usage add_alps_package(pkgname1 pkgname2...)
  foreach(pkg_ ${ARGV})
    if (NOT DEFINED ALPS_GLOBAL_BUILD)
      find_package(${pkg_} REQUIRED)
      message(STATUS "${pkg_} includes: ${${pkg_}_INCLUDE_DIRS}" )
      message(STATUS "${pkg_} libs: ${${pkg_}_LIBRARIES}" )
    endif (NOT DEFINED ALPS_GLOBAL_BUILD)
    include_directories(${${pkg_}_INCLUDE_DIRS})
    list(APPEND LINK_ALL ${${pkg_}_LIBRARIES})
  endforeach(pkg_)
  gen_cfg_module(${ARGV})
endmacro(add_alps_package) 


macro(add_this_package)
  include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
  )
  add_subdirectory(src)
  install(DIRECTORY include DESTINATION .
          FILES_MATCHING PATTERN "*.hpp" PATTERN "*.hxx"
         )
endmacro(add_this_package)

# Parameters: list of source files
macro(add_source_files)
  add_library(${PROJECT_NAME} ${ALPS_BUILD_TYPE} ${ARGV})
  set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(${PROJECT_NAME} ${LINK_ALL})
  install(TARGETS ${PROJECT_NAME} 
          EXPORT ${PROJECT_NAME} 
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          INCLUDES DESTINATION include)
  install(EXPORT ${PROJECT_NAME} NAMESPACE alps:: DESTINATION share/${PROJECT_NAME})
endmacro(add_source_files)  


macro(add_testing)
  option(Testing "Enable testing" ON)
  if (Testing)
    enable_testing()
    add_subdirectory(test)
  endif (Testing)
endmacro(add_testing)

macro(gen_documentation)
  set(DOXYFILE_EXTRA_SOURCES "${DOXYFILE_EXTRA_SOURCES} ${PROJECT_SOURCE_DIR}/include" PARENT_SCOPE)
  option(Documentation "Build documentation" OFF)
  if (Documentation)
    set(DOXYFILE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/include")
    set(DOXYFILE_IN "${PROJECT_SOURCE_DIR}/../common/doc/Doxyfile.in") 
    include(UseDoxygen)
  endif(Documentation)
endmacro(gen_documentation)

macro(gen_hpp_config)
  configure_file("${PROJECT_SOURCE_DIR}/include/config.hpp.in" "${PROJECT_BINARY_DIR}/include/alps/config.hpp")
  install(FILES "${PROJECT_BINARY_DIR}/include/alps/config.hpp" DESTINATION include/alps) 
endmacro(gen_hpp_config)

macro(gen_pkg_config)
  # Generate pkg-config file
  configure_file("${PROJECT_SOURCE_DIR}/${PROJECT_NAME}.pc.in" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.pc")
  install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.pc" DESTINATION "lib/pkgconfig")
endmacro(gen_pkg_config)


# Function: generates package-specific CMake configs
# Arguments: list of dependencies
function(gen_cfg_module)
  set(DEPENDS ${ARGV})
  configure_file("${PROJECT_SOURCE_DIR}/../common/cmake/ALPSModuleConfig.cmake.in" 
                 "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY)
  install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" DESTINATION "share/${PROJECT_NAME}/")
endfunction()

# # Requred parameters:
# #  project_search_file_ : filename helping to identify the location of the project 
# # Optional parameters:
# #  HEADER_ONLY : the package does not contain libraries
# #
# function(gen_find_module project_search_file_)
#   set(PROJECT_SEARCH_FILE ${project_search_file_})
#   set (NOT_HEADER_ONLY true)
#   foreach(arg ${ARGV})
#     if (arg STREQUAL "HEADER_ONLY")
#       set(NOT_HEADER_ONLY false)
#     endif()
#   endforeach()
#   configure_file("${PROJECT_SOURCE_DIR}/../common/cmake/ALPSModuleConfig.cmake.in" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY)
#   # configure_file("${PROJECT_SOURCE_DIR}/../common/cmake/FindALPSModule.cmake.in" "${PROJECT_BINARY_DIR}/Find${PROJECT_NAME}.cmake" @ONLY)
#   # install(FILES "${PROJECT_BINARY_DIR}/Find${PROJECT_NAME}.cmake" DESTINATION "share/cmake/Modules/")
#   install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" DESTINATION "share/${PROJECT_NAME}/")
#   install(FILES "${PROJECT_SOURCE_DIR}/../common/cmake/ALPSCoreConfig.cmake" DESTINATION "share/ALPSCore/")
#   install(FILES "${PROJECT_SOURCE_DIR}/../common/cmake/FindALPSCore.cmake" DESTINATION "share/cmake/Modules/")
# endfunction(gen_find_module)
