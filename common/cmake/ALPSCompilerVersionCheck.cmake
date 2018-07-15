#
# Check for minimum compiler version and throw error if we know the compiler doesn't work
# Warn if we can't test for it.
#
if (DEFINED CMAKE_SKIP_COMPILER_VERSION_TEST)
# for some experimental compilers the version parser fails and we need to skip the version test altogether
  message(STATUS "Unable to determine C++ compiler version. Version check manually disabled (-DCMAKE_SKIP_COMPILER_VERSION_TEST)")
else()
  if ((DEFINED CMAKE_CXX_COMPILER_ID) AND (DEFINED CMAKE_CXX_COMPILER_VERSION))
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.2") #we can probably support older versions but nobody has any left to test
        message(FATAL_ERROR "Insufficient gcc version")
      endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
      if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0")
        message(FATAL_ERROR "Insufficient Intel compiler version")
      endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.2")
        message(FATAL_ERROR "Insufficient Clang compiler version")
      endif()
    endif()
  else()
    message(WARNING "Unknown C++ compiler version: might not be supported")
  endif()
endif()

unset(ALPS_CXX_STD) # ensure the var is read from cache, if any
if (DEFINED ALPS_CXX_STD)
  if (ALPS_CXX_STD MATCHES "^[cC][+][+](03|98)$")
    unset(ALPS_CXX_STD CACHE)
    message(FATAL_ERROR "ALPSCore cannot be compiled with C++98/C++03; at least C++11 is required")
  endif()
  if (NOT ALPS_CXX_STD MATCHES "^[cC][+][+](11|14|17)|custom$")
    message(FATAL_ERROR "Invalid value of ALPS_CXX_STD='${ALPS_CXX_STD}'. Only 'c++11', 'c++14', 'c++17' and 'custom' are supported.")
  endif()
  string(TOLOWER ${ALPS_CXX_STD} ALPS_CXX_STD)
else()
  set(ALPS_CXX_STD "c++11")
endif()
set(ALPS_CXX_STD ${ALPS_CXX_STD} CACHE STRING "C++ standard used to compile ALPSCore" FORCE)
set_property(CACHE ALPS_CXX_STD PROPERTY STRINGS "c++11" "c++14" "c++17" "custom")
mark_as_advanced(ALPS_CXX_STD)

set(CMAKE_CXX_EXTENSIONS OFF)

# FIXME: In future we can try to set this ourselves if CMake does not recognize the compiler
set(ALPS_CXX_FLAGS "" CACHE INTERNAL "C++ compilation flags to be set as interface")
set(ALPS_CMAKE_MINIMUM_VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

if (ALPS_CXX_STD STREQUAL "c++11")
  if (CMAKE_VERSION VERSION_LESS 3.8)
    set(ALPS_CXX_FEATURES "cxx_auto_type;cxx_constexpr" CACHE INTERNAL "List of C++ features required by ALPSCore")
  else()
    set(ALPS_CMAKE_MINIMUM_VERSION 3.8)
    set(ALPS_CXX_FEATURES "cxx_std_11" CACHE INTERNAL "List of C++ features required by ALPSCore")
  endif()
  message(STATUS "ALPSCore will use C++11")
endif()

if (ALPS_CXX_STD STREQUAL "c++14")
  if (CMAKE_VERSION VERSION_LESS 3.8)
    set(ALPS_CXX_FEATURES "cxx_auto_type;cxx_constexpr;cxx_decltype_auto" CACHE INTERNAL "List of C++ features required by ALPSCore")
  else()
    set(ALPS_CMAKE_MINIMUM_VERSION 3.8)
    set(ALPS_CXX_FEATURES "cxx_std_14" CACHE INTERNAL "List of C++ features required by ALPSCore")
  endif()
  message(STATUS "ALPSCore will use C++14")
endif()

if (ALPS_CXX_STD STREQUAL "c++17")
  if (CMAKE_VERSION VERSION_LESS 3.8)
    message(FATAL_ERROR "To use ALPS_CXX_STD=c++17 you need at least CMake version 3.8; "
      "this CMake version is ${CMAKE_VERSION}. "
      "Please set ALPS_CXX_STD=custom and pass the proper compilation flags via CXX_COMPILE_FLAGS.")
  else()
    set(ALPS_CMAKE_MINIMUM_VERSION 3.8)
    set(ALPS_CXX_FEATURES "cxx_std_17" CACHE INTERNAL "List of C++ features required by ALPSCore")
  endif()
  message(STATUS "ALPSCore will use C++17")
endif()

if (ALPS_CXX_STD STREQUAL "custom")
  set(ALPS_CXX_FEATURES "" CACHE INTERNAL "List of C++ features required by ALPSCore")
  message("CAUTION: ALPSCore C++ standard will be set by compiler flags;"
    " client code will not have any way to inquire the C++ standard used by ALPSCore!")
endif()

set(ALPS_CMAKE_MINIMUM_VERSION ${ALPS_CMAKE_MINIMUM_VERSION} CACHE INTERNAL "Minimum CMake version required to use ALPSCore")
