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
option(ALPS_BUILD_STATIC "Do static build" OFF)
option(ALPS_BUILD_SHARED "Do shared build" ON)
if (ALPS_BUILD_STATIC)
  if (ALPS_BUILD_SHARED)
    message(FATAL_ERROR "Please choose EITHER ALPS_BUILD_STATIC OR ALPS_BUILD_SHARED type of build, NOT both")
  endif()
  message(STATUS "Doing static build")
  # set(BUILD_SHARED_LIBS OFF CACHE BOOL "Generate shared libraries")
  option(BUILD_SHARED_LIBS "Generate shared libraries" OFF)
elseif(ALPS_BUILD_SHARED)
  if (ALPS_BUILD_STATIC)
    message(FATAL_ERROR "Please choose EITHER ALPS_BUILD_STATIC OR ALPS_BUILD_SHARED type of build, NOT both")
  endif()
  message(STATUS "Building shared libraries")
  # set(BUILD_SHARED_LIBS OFF CACHE BOOL "Generate shared libraries")
  option(BUILD_SHARED_LIBS "Generate shared libraries" ON)
else()
  option(BUILD_SHARED_LIBS "Generate shared libraries" ON) # << user will likely override it
  message("NOTE: Will generate libraries depending on BUILD_SHARED_LIBS option, which is set to ${BUILD_SHARED_LIBS}")
endif()
option(ALPS_BUILD_PIC "Generate position-independent code (PIC)" OFF)

# Set ALPS_ROOT as a hint for standalone component builds
if (DEFINED ENV{ALPS_ROOT})
  set(ALPS_ROOT "$ENV{ALPS_ROOT}" CACHE PATH "Path to ALPSCore installation (for standalone component builds)")
  mark_as_advanced(ALPS_ROOT)
endif()


## Some macros

# add includes and libs for each module
macro(alps_add_module module module_path)
    set(${module}_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/${module_path}/include ${CMAKE_BINARY_DIR}/${module_path}/include)
    set(${module}_LIBRARIES ${module})
endmacro(alps_add_module)

macro(add_boost) # usage: add_boost(component1 component2...)
  if (ALPS_BUILD_STATIC)
    set(Boost_USE_STATIC_LIBS        ON)
    #set(Boost_USE_STATIC_RUNTIME    OFF)
  endif()
  find_package (Boost 1.54.0 COMPONENTS ${ARGV} REQUIRED)
  message(STATUS "Boost includes: ${Boost_INCLUDE_DIRS}" )
  message(STATUS "Boost libs: ${Boost_LIBRARIES}" )
  target_include_directories(${PROJECT_NAME} PUBLIC ${Boost_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${Boost_LIBRARIES})
endmacro(add_boost)

macro(add_hdf5) 
  if (ALPS_BUILD_STATIC)
    set(HDF5_USE_STATIC_LIBRARIES ON)
  endif()
  find_package (HDF5 REQUIRED)
  message(STATUS "HDF5 includes: ${HDF5_INCLUDE_DIRS}" )
  message(STATUS "HDF5 libs: ${HDF5_LIBRARIES}" )
  target_include_directories(${PROJECT_NAME} PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${HDF5_LIBRARIES})
endmacro(add_hdf5)

# Usage: add_alps_package(pkgname1 pkgname2...)
# Sets variable ${PROJECT_NAME}_DEPENDS
macro(add_alps_package)
    list(APPEND ${PROJECT_NAME}_DEPENDS ${ARGV})
    foreach(pkg_ ${ARGV})
        if (DEFINED ALPS_GLOBAL_BUILD)
            include_directories(${${pkg_}_INCLUDE_DIRS}) # this is needed to compile tests
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
        target_link_libraries(${PROJECT_NAME} PUBLIC ${${pkg_}_LIBRARIES})
        message(STATUS "${pkg_} libs: ${${pkg_}_LIBRARIES}")
    endforeach(pkg_)
endmacro(add_alps_package) 

# Usage: add_this_package(srcs...)
# The `srcs` are source file names in directory "src/"
# Defines ${PROJECT_NAME} target
# Exports alps::${PROJECT_NAME} target
function(add_this_package)
   # This is needed to compile tests:
   include_directories(
     ${PROJECT_SOURCE_DIR}/include
     ${PROJECT_BINARY_DIR}/include
   )
  
  set(src_list_ "")
  foreach(src_ ${ARGV})
    list(APPEND src_list_ "src/${src_}.cpp")
  endforeach()
  add_library(${PROJECT_NAME} ${src_list_})
  if (ALPS_BUILD_PIC) 
    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()

  install(TARGETS ${PROJECT_NAME} 
          EXPORT ${PROJECT_NAME} 
          LIBRARY DESTINATION lib
          ARCHIVE DESTINATION lib
          INCLUDES DESTINATION include)
  install(EXPORT ${PROJECT_NAME} NAMESPACE alps:: DESTINATION share/${PROJECT_NAME})
  target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_BINARY_DIR}/include)

  install(DIRECTORY include DESTINATION .
          FILES_MATCHING PATTERN "*.hpp" PATTERN "*.hxx"
         )
endfunction(add_this_package)

macro(add_testing)
  option(Testing "Enable testing" ON)
  if (Testing)
    enable_testing()
    add_subdirectory(test)
  endif (Testing)
endmacro(add_testing)

macro(gen_documentation)
  set(DOXYFILE_EXTRA_SOURCES "${DOXYFILE_EXTRA_SOURCES} ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src" PARENT_SCOPE)
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
# If no <list-of-dependencies> are present, the contents of ${PROJECT_NAME}_DEPENDS is used
# If no exported targets are present, alps::${PROJECT_NAME} is assumed.
function(gen_cfg_module)
    include(CMakeParseArguments) # arg parsing helper
    cmake_parse_arguments(gen_cfg_module "" "" "DEPENDS;EXPORTS" ${ARGV})
    if (gen_cfg_module_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Incorrect call of gen_cfg_module([DEPENDS ...] [EXPORTS ...]): ARGV=${ARGV}")
    endif()
    if (gen_cfg_module_DEPENDS)
        set(DEPENDS ${gen_cfg_module_DEPENDS})
    else()
        set(DEPENDS ${${PROJECT_NAME}_DEPENDS})
    endif()
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
