# Try to guess the flags needed for the current compiler to activate a given C++ standard
function(alps_guess_cxx_std_flag std okvar flagvar)
  include(CheckCXXCompilerFlag)
  if (CMAKE_CXX_COMPILER_ID MATCHES Intel)
    set(flag "-std=${std}")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES PGI)
    set(flag "--${std}")
  else()
    set(${okvar} false PARENT_SCOPE)
    set(${flagvar} "" PARENT_SCOPE)
    return()
  endif()
  CHECK_CXX_COMPILER_FLAG(${flag} has_flag)
  if (has_flag)
    set(${okvar} true PARENT_SCOPE)
    set(${flagvar} ${flag} PARENT_SCOPE)
  else()
    set(${okvar} false PARENT_SCOPE)
    set(${flagvar} "" PARENT_SCOPE)
  endif()
endfunction()


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

set(ALPS_CXX_FLAGS "")
set(ALPS_CXX_FEATURES "")
set(ALPS_CMAKE_MINIMUM_VERSION ${CMAKE_MINIMUM_REQUIRED_VERSION})

if (ALPS_CXX_STD STREQUAL "custom")

  message("CAUTION: ALPSCore C++ standard will be set by compiler flags;"
    " client code will not have any way to inquire the C++ standard used by ALPSCore!")

else()

  # Does CMake know how to use C++XX for this compiler?
  if (CMAKE_CXX_COMPILE_FEATURES)
    if (ALPS_CXX_STD STREQUAL "c++11")
      if (CMAKE_VERSION VERSION_LESS 3.8)
        set(ALPS_CXX_FEATURES "cxx_auto_type;cxx_constexpr")
      else()
        set(ALPS_CMAKE_MINIMUM_VERSION 3.8)
        set(ALPS_CXX_FEATURES "cxx_std_11")
      endif()
      message(STATUS "ALPSCore will use C++11")
    endif()

    if (ALPS_CXX_STD STREQUAL "c++14")
      if (CMAKE_VERSION VERSION_LESS 3.8)
        set(ALPS_CXX_FEATURES "cxx_auto_type;cxx_constexpr;cxx_decltype_auto")
      else()
        set(ALPS_CMAKE_MINIMUM_VERSION 3.8)
        set(ALPS_CXX_FEATURES "cxx_std_14")
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
        set(ALPS_CXX_FEATURES "cxx_std_17")
      endif()
      message(STATUS "ALPSCore will use C++17")
    endif()
    
  else()
    
    message(WARNING "This version of CMake does not know how to activate ${ALPS_CXX_STD} features "
      "for your compiler ${CMAKE_CXX_COMPILER} (id: ${CMAKE_CXX_COMPILER_ID} version: ${CMAKE_CXX_COMPILER_VERSION}). "
      "We will try to guess.")
    alps_guess_cxx_std_flag(${ALPS_CXX_STD} ok flags)
    if (ok)
      message(STATUS "Guessed C++ flags: ${flags}")
      set(ALPS_CXX_FLAGS ${flags})
    else()
      message(FATAL_ERROR "Could not guess the flags. "
        "You may try to use a newer version of CMake, or set ALPS_CXX_STD=custom and pass the required "
        "compiler flags via CMAKE_CXX_FLAGS.")
    endif()

  endif()

endif()

set(ALPS_CXX_FEATURES ${ALPS_CXX_FEATURES} CACHE INTERNAL "List of C++ features required by ALPSCore")
set(ALPS_CXX_FLAGS ${ALPS_CXX_FLAGS} CACHE INTERNAL "C++ compilation flags to be set as interface")
set(ALPS_CMAKE_MINIMUM_VERSION ${ALPS_CMAKE_MINIMUM_VERSION} CACHE INTERNAL "Minimum CMake version required to use ALPSCore")
