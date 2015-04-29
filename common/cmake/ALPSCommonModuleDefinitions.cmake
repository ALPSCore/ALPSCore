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

# Set ALPS_ROOT as a hint for standalone component builds
if (DEFINED ENV{ALPS_ROOT})
  set(ALPS_ROOT "$ENV{ALPS_ROOT}" CACHE PATH "Path to ALPSCore installation (for standalone component builds)")
  mark_as_advanced(ALPS_ROOT)
endif()


## Some macros

macro(add_boost) # usage: add_boost(component1 component2...)
  find_package (Boost 1.54.0 COMPONENTS ${ARGV} REQUIRED)
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

# Usage: add_alps_package(pkgname1 pkgname2...)
# Sets variable ${PROJECT_NAME}_DEPENDS
# Adds to variable LINK_ALL
macro(add_alps_package)
    set(${PROJECT_NAME}_DEPENDS "${ARGV}")
    foreach(pkg_ ${ARGV})
        if (DEFINED ALPS_GLOBAL_BUILD)
            include_directories(${${pkg_}_INCLUDE_DIRS})
            message(STATUS "${pkg_} includes: ${${pkg_}_INCLUDE_DIRS}" )
        else(DEFINED ALPS_GLOBAL_BUILD)
            string(REGEX REPLACE "^alps-" "" pkgcomp_ ${pkg_})
            find_package(ALPSCore QUIET COMPONENTS ${pkgcomp_} HINTS ${ALPS_ROOT})
            if (ALPSCore_${pkgcomp_}_FOUND) 
              # message(STATUS "DEBUG: found as an ALPSCore component")
              set(${pkg_}_LIBRARIES ${ALPSCore_${pkgcomp_}_LIBRARIES})
            else()
              # message(STATUS "DEBUG: could not find ALPSCore, searching for the component directly")
              find_package(${pkg_} REQUIRED HINTS ${ALPS_ROOT})
            endif()
            # Imported targets returned by find_package() contain info about include dirs, no need to assign them
        endif (DEFINED ALPS_GLOBAL_BUILD)
        list(APPEND LINK_ALL ${${pkg_}_LIBRARIES})
        message(STATUS "${pkg_} libs: ${${pkg_}_LIBRARIES}")
    endforeach(pkg_)
endmacro(add_alps_package) 

# Usage: add_this_package([exported_target1 exported_target2...])
# (by default, alps::${PROJECT_NAME} is the only exported target)
# Affected by variable ${PROJECT_NAME}_DEPENDS
function(add_this_package)
  include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
  )
  add_subdirectory(src)
  install(DIRECTORY include DESTINATION .
          FILES_MATCHING PATTERN "*.hpp" PATTERN "*.hxx"
         )
  # FIXME: exported targets are explicitly listed for Python only -- this logic should be separated.
  set(tgt_list_ "")
  foreach(tgt_ ${ARGV})
      list(APPEND tgt_list_ "alps::${tgt_}")
  endforeach()
  if (tgt_list_)
      gen_cfg_module(DEPENDS ${${PROJECT_NAME}_DEPENDS} EXPORTS ${tgt_list_})
  else()
      gen_cfg_module(DEPENDS ${${PROJECT_NAME}_DEPENDS})
  endif()
endfunction(add_this_package)

# Parameters: list of source files
macro(add_source_files)
  add_library(${PROJECT_NAME} ${ALPS_BUILD_TYPE} ${ARGV})
  set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(${PROJECT_NAME} ${LINK_ALL})
  # (FIXME) The following requires a newer (2.8.12 ?) CMake:
  # install(TARGETS ${PROJECT_NAME} 
  #         EXPORT ${PROJECT_NAME} 
  #         LIBRARY DESTINATION lib
  #         ARCHIVE DESTINATION lib
  #         INCLUDES DESTINATION include)
  # so it is replaced: BEGIN...
  install(TARGETS ${PROJECT_NAME} 
          EXPORT ${PROJECT_NAME} 
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib)
  set_target_properties(${PROJECT_NAME} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_INSTALL_PREFIX}/include)
  # ... END
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
# Arguments: [DEPENDS <list-of-dependencies>] [EXPORTS <list-of-exported-targets>]
# If no exported targets are present, alps::${PROJECT_NAME} is assumed.
function(gen_cfg_module)
    include(CMakeParseArguments) # arg parsing helper
    cmake_parse_arguments(gen_cfg_module "" "" "DEPENDS;EXPORTS" ${ARGV})
    if (gen_cfg_module_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Incorrect call of gen_cfg_module(DEPENDS ... [EXPORTS ...]): ARGV=${ARGV}")
    endif()
    set(DEPENDS ${gen_cfg_module_DEPENDS})
    if (gen_cfg_module_EXPORTS)
        set(EXPORTS ${gen_cfg_module_EXPORTS})
    else()
        set(EXPORTS alps::${PROJECT_NAME})
    endif()
    configure_file("${PROJECT_SOURCE_DIR}/../common/cmake/ALPSModuleConfig.cmake.in" 
                   "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" @ONLY)
    configure_file("${PROJECT_SOURCE_DIR}/../common/cmake/ALPSCoreConfig.cmake.in" 
                   "${PROJECT_BINARY_DIR}/ALPSCoreConfig.cmake" @ONLY)
    install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake" DESTINATION "share/${PROJECT_NAME}/")
    install(FILES "${PROJECT_BINARY_DIR}/ALPSCoreConfig.cmake" DESTINATION "share/ALPSCore/")
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
