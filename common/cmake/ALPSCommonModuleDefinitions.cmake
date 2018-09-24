#
# Provide common definitions for building alps modules 
#

# Do not forget to adjust as needed!
set(ALPSCORE_VERSION "2.2.0")

# Disable in-source builds
if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    message(FATAL_ERROR "In source builds are disabled. Please use a separate build directory")
endif()

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# Ignore special meaning of PackageName_ROOT variables (since CMake 3.12)
if (NOT CMAKE_VERSION VERSION_LESS 3.12)
  cmake_policy(SET CMP0074 OLD)
endif()

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

#Do Release-with-debug build by default
#If it is not set, remove it from the cache
if (NOT CMAKE_BUILD_TYPE)
  unset(CMAKE_BUILD_TYPE CACHE)
endif()
set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type, such as `Debug` or `Release`")
mark_as_advanced(CMAKE_BUILD_TYPE)

# This option is checked, e.g., when adding -DBOOST_DISABLE_ASSERTS.
option(ALPS_DEBUG "Set to TRUE to supress auto-adjusting your compilation flags" false)
mark_as_advanced(ALPS_DEBUG)

# GF uses boost::multi_array, to supress extra checks we need to define extra flags,
# otherwise codes will slow down to a crawl.
if (NOT ALPS_DEBUG)
  if (NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "(^| )-DBOOST_DISABLE_ASSERTS( |$)")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DBOOST_DISABLE_ASSERTS" CACHE STRING "ALPSCore Release compilation flags" FORCE)
  endif()
  if (NOT CMAKE_CXX_FLAGS_RELWITHDEBINFO MATCHES "(^| )-DBOOST_DISABLE_ASSERTS( |$)")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DBOOST_DISABLE_ASSERTS" CACHE STRING "ALPSCore Release-with-Debug-Info compilation flags" FORCE)
  endif()
  if (NOT CMAKE_CXX_FLAGS_MINSIZEREL MATCHES "(^| )-DBOOST_DISABLE_ASSERTS( |$)")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -DBOOST_DISABLE_ASSERTS" CACHE STRING "ALPSCore Release-minimum-size-executables flags" FORCE)
  endif()
  if (NOT CMAKE_CXX_FLAGS_DEBUG MATCHES "(^| )-DALPS_GF_DEBUG( |$)")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DALPS_GF_DEBUG" CACHE STRING "ALPSCore Debug compilation flags" FORCE)
  endif()
endif()

# Build static XOR shared 
# Defines ALPS_BUILD_TYPE=STATIC|DYNAMIC .
set(ALPS_BUILD_TYPE "dynamic" CACHE STRING "Build type: `static`, `dynamic` or `unspecified`")
set_property(CACHE ALPS_BUILD_TYPE PROPERTY STRINGS static dynamic unspecified)
string(TOLOWER ${ALPS_BUILD_TYPE}  ALPS_BUILD_TYPE)

# We do not want those variables in cache:
unset(ALPS_BUILD_SHARED)
unset(ALPS_BUILD_STATIC)
if (DEFINED ALPS_BUILD_STATIC OR DEFINED ALPS_BUILD_SHARED)
  message(WARNING "Setting ALPS_BUILD_SHARED, ALPS_BUILD_STATIC in cache does not have any effect.")
endif()
unset(ALPS_BUILD_SHARED CACHE)
unset(ALPS_BUILD_STATIC CACHE)

if (ALPS_BUILD_TYPE STREQUAL dynamic)
  set(ALPS_BUILD_SHARED true)
  set(ALPS_BUILD_STATIC false)
  message(STATUS "Building shared libraries")
  unset(BUILD_SHARED_LIBS CACHE)
  option(BUILD_SHARED_LIBS "Generate shared libraries" ON)
elseif (ALPS_BUILD_TYPE STREQUAL static)
  set(ALPS_BUILD_SHARED false)
  set(ALPS_BUILD_STATIC true)
  message(STATUS "Doing static build")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")
  unset(BUILD_SHARED_LIBS CACHE)
  option(BUILD_SHARED_LIBS "Generate shared libraries" OFF)
elseif (ALPS_BUILD_TYPE STREQUAL unspecified)
  # Special case: just go after BUILD_SHARED_LIBS, everything else is default.
  set(ALPS_BUILD_SHARED false)
  set(ALPS_BUILD_STATIC false)
  option(BUILD_SHARED_LIBS "Generate shared libraries" ON) # << user will likely override it
  message(WARNING "NOTE: Will generate libraries depending on BUILD_SHARED_LIBS option, which is set to ${BUILD_SHARED_LIBS}")
  message(WARNING "Be sure you know what you are doing!")
else()
  message(FATAL_ERROR "ALPS_BUILD_TYPE should be set to either 'static' or 'dynamic' (or 'unspecified' only if you know what your are doing)")
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
  if (ALPS_BUILD_SHARED)
    set(Boost_USE_STATIC_LIBS        OFF)
  endif()
  find_package (Boost 1.56.0 COMPONENTS ${ARGV} REQUIRED)
  # Remember Boost version
  set(ALPSCore_BOOST_VERSION ${Boost_MAJOR_VERSION})
  if (DEFINED Boost_MAJOR_VERSION AND DEFINED Boost_MINOR_VERSION)
    if (DEFINED Boost_SUBMINOR_VERSION)
      set(ALPSCore_BOOST_VERSION ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION})
    else()
      set(ALPSCore_BOOST_VERSION ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION})
    endif()
  endif()
  set(ALPSCore_BOOST_VERSION ${ALPSCore_BOOST_VERSION}  CACHE INTERNAL "Version of Boost used by ALPSCore")
  # Remember Boost location hints
  foreach(varname_ BOOST_ROOT BOOSTROOT BOOST_INCLUDEDIR BOOST_LIBRARYDIR)
    set(env_  $ENV{${varname_}})
    set(var_ ${${varname_}})
    if (NOT var_ AND env_)
      set(var_ ${env_})
    endif()
    set(ALPSCore_${varname_} "${var_}")
  endforeach()
  if (ALPSCore_BOOSTROOT AND NOT ALPSCore_BOOST_ROOT)
    set(ALPSCore_BOOST_ROOT "${ALPSCore_BOOSTROOT}")
  endif()
  foreach(varname_ BOOST_ROOT BOOST_INCLUDEDIR BOOST_LIBRARYDIR)
    set(ALPSCore_${varname_} ${ALPSCore_${varname_}} CACHE INTERNAL "${varname_} as seen by ALPSCore")
  endforeach()

  message(STATUS "Boost includes: ${Boost_INCLUDE_DIRS}" )
  message(STATUS "Boost libs: ${Boost_LIBRARIES}" )
  target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${Boost_LIBRARIES})
endmacro(add_boost)

macro(add_hdf5) 
  if (ALPS_BUILD_STATIC)
    set(HDF5_USE_STATIC_LIBRARIES ON)
  endif()
  if (ALPS_BUILD_SHARED)
    set(HDF5_USE_STATIC_LIBRARIES OFF)
  endif()
  set(HDF5_NO_FIND_PACKAGE_CONFIG_FILE TRUE)
  find_package (HDF5 REQUIRED)
  message(STATUS "HDF5 includes: ${HDF5_INCLUDE_DIRS}" )
  message(STATUS "HDF5 libs: ${HDF5_LIBRARIES}" )
  target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${HDF5_LIBRARIES})
endmacro(add_hdf5)

# Usage: add_alps_package(pkgname1 pkgname2...)
# Sets variable ${PROJECT_NAME}_DEPENDS
macro(add_alps_package)
    foreach(pkg_ ${ARGV})
      list(FIND ALPS_MODULES_DISABLE ${pkg_} disabled_index_)
      if (NOT disabled_index_ EQUAL -1)
        message(FATAL_ERROR "Module ${PROJECT_NAME} depends on ${pkg_} which is disabled.")
      endif()
    endforeach()
    list(APPEND ${PROJECT_NAME}_DEPENDS ${ARGV})
    foreach(pkg_ ${ARGV})
        if (DEFINED ALPS_GLOBAL_BUILD)
            include_directories(BEFORE ${${pkg_}_INCLUDE_DIRS}) # this is needed to compile tests (FIXME: why?)
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

# Usage: add_this_package(srcs... EXTRA extra_srcs...)
# The `srcs` are source file names in directory "src/"
# After `EXTRA`, `extra_srcs` are added verbatim
# Defines ${PROJECT_NAME} target
# Exports alps::${PROJECT_NAME} target
# Defines internal cache variable ALPS_HAVE_ALPS_${MODULE} (where MODULE=upcase(PROJECT_NAME))
function(add_this_package)
  include(CMakeParseArguments)
  cmake_parse_arguments(THIS_PACKAGE "" "" "EXTRA" ${ARGV})
   # This is needed to compile tests:
   include_directories(
     ${PROJECT_SOURCE_DIR}/include
     ${PROJECT_BINARY_DIR}/include
   )
  
  set(src_list_ "")
  foreach(src_ ${THIS_PACKAGE_UNPARSED_ARGUMENTS})
    list(APPEND src_list_ "src/${src_}.cpp")
  endforeach()
  add_library(${PROJECT_NAME} ${src_list_} ${THIS_PACKAGE_EXTRA})
  if (ALPS_CXX_FEATURES)
    target_compile_features(${PROJECT_NAME} PUBLIC ${ALPS_CXX_FEATURES})
  endif()
  if (ALPS_CXX_FLAGS)
    target_compile_options(${PROJECT_NAME} PUBLIC ${ALPS_CXX_FLAGS})
  endif()
  if (ALPS_BUILD_PIC) 
    set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()

  if (ALPS_HAVE_MPI)
    if (MPI_CXX_INCLUDE_PATH)
      target_include_directories(${PROJECT_NAME} PUBLIC ${MPI_CXX_INCLUDE_PATH})
      message(STATUS "MPI C++ include path: ${MPI_CXX_INCLUDE_PATH}")
    endif()
    if (MPI_C_INCLUDE_PATH)
      target_include_directories(${PROJECT_NAME} PUBLIC ${MPI_C_INCLUDE_PATH})
      message(STATUS "MPI C include path: ${MPI_C_INCLUDE_PATH}")
    endif()
    if (MPI_CXX_LIBRARIES)
      target_link_libraries(${PROJECT_NAME} PUBLIC ${MPI_CXX_LIBRARIES})
      message(STATUS "MPI libraries: ${MPI_CXX_LIBRARIES}")
    endif()
  endif(ALPS_HAVE_MPI)

  string(REPLACE "alps-" "" upcase_project_name_ ${PROJECT_NAME})
  string(TOUPPER ${upcase_project_name_} upcase_project_name_)
  set(ALPS_HAVE_ALPS_${upcase_project_name_} 1 CACHE INTERNAL "")
  
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

function(gen_main_hpp_config)
  configure_file("${CMAKE_SOURCE_DIR}/utilities/include/config.hpp.in" "${CMAKE_BINARY_DIR}/utilities/include/alps/config.hpp")
  install(FILES "${CMAKE_BINARY_DIR}/utilities/include/alps/config.hpp" DESTINATION include/alps) 
endfunction(gen_main_hpp_config)

macro(gen_pkg_config)
  # Generate pkg-config file
  configure_file("${PROJECT_SOURCE_DIR}/${PROJECT_NAME}.pc.in" "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.pc")
  install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}.pc" DESTINATION "lib/pkgconfig")
endmacro(gen_pkg_config)

# Function: generates main ALPSCore config
function(gen_cfg_main)
  configure_file("${PROJECT_SOURCE_DIR}/common/cmake/ALPSCoreConfig.cmake.in" 
                 "${PROJECT_BINARY_DIR}/stage/ALPSCoreConfig.cmake" @ONLY)
  configure_file("${PROJECT_SOURCE_DIR}/common/cmake/ALPSCoreConfigVersion.cmake.in" 
                 "${PROJECT_BINARY_DIR}/stage/ALPSCoreConfigVersion.cmake" @ONLY)
  install(FILES "${PROJECT_BINARY_DIR}/stage/ALPSCoreConfig.cmake" DESTINATION "share/ALPSCore/")
  install(FILES "${PROJECT_BINARY_DIR}/stage/ALPSCoreConfigVersion.cmake" DESTINATION "share/ALPSCore/")
endfunction()


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
                   "${PROJECT_BINARY_DIR}/stage/${PROJECT_NAME}Config.cmake" @ONLY)
    install(FILES "${PROJECT_BINARY_DIR}/stage/${PROJECT_NAME}Config.cmake" DESTINATION "share/${PROJECT_NAME}/")
endfunction()
