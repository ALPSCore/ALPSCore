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

# if both options are unset, make sure default settings are acquired
if (NOT ALPS_USE_CXX03 AND NOT ALPS_USE_CXX11)
  unset(ALPS_USE_CXX03)
  unset(ALPS_USE_CXX11)
  if (NOT ALPS_USE_CXX03 AND NOT ALPS_USE_CXX11)
    unset(ALPS_USE_CXX03 CACHE)
    unset(ALPS_USE_CXX11 CACHE)
  endif()
endif()

option(ALPS_USE_CXX03 "Use C++03 to compile ALPSCore" ON)
option(ALPS_USE_CXX11 "Use C++11 to compile ALPSCore" OFF)
mark_as_advanced(ALPS_USE_CXX03)
mark_as_advanced(ALPS_USE_CXX11)

if (ALPS_USE_CXX03 AND ALPS_USE_CXX11)
  message(FATAL_ERROR "Both ALPS_USE_CXX03 AND ALPS_USE_CXX11 options are set, but they are mutually exclusive")
endif()

function(alps_get_cxx03_flag flagvar)
  include(CheckCXXCompilerFlag)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(try_options "-std=c++03" "-std=c++98")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(try_options "-std=c++98")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
    set(try_options "--c++03" "-std=c++03")
  else()
    message(AUTHOR_WARNING "Do not know how to make compiler ID='${CMAKE_CXX_COMPILER_ID}' set C++98/C++03 standard")
    set(try_options "-std=c++03" "-std=c++98")
  endif()

  set(retval)
  foreach (flag ${try_options})
    check_cxx_compiler_flag("-std=c++03" supported)
    if (supported)
      set(retval ${flag})
    endif()
  endforeach()

  if (NOT retval)
    message(AUTHOR_WARNING "Cannot determine a vald option for compiler ID='${CXX_COMPILER_ID}'
 to set C++98/C++03 standard, trying empty option")
  endif()

  set(cxxfile "${CMAKE_BINARY_DIR}/get_cxx_version.cxx")
  file(WRITE  ${cxxfile}
"#include <cstdio>
#define STRINGIFY_HELPER(s) #s
#define STRINGIFY(s) STRINGIFY_HELPER(s)
int main() { puts(STRINGIFY(__cplusplus)); return 0; }
")
  
  try_run(run_result compile_result
    ${CMAKE_BINARY_DIR} ${cxxfile}
    COMPILE_DEFINITIONS ${retval}
    RUN_OUTPUT_VARIABLE run_output)

  if (NOT compile_result)
    message(FATAL_ERROR "Unable to compile test executable")
  endif()
  if (run_result STREQUAL "FAILED_TO_RUN")
    message(FATAL_ERROR "Unable to run test executable")
  endif()
  
  if (NOT run_output MATCHES "199711L")
    message(AUTHOR_WARNING "Setting C++ standard to C++03 has apparently failed:
Compiler ID='${CXX_COMPILER_ID}' with option='${retval}' using standard '${run_output}'
which is not C++03 standard. Proceeding anyway.")
  endif()

  set(${flagvar} ${retval} PARENT_SCOPE)
endfunction()

set(CMAKE_CXX_EXTENSIONS OFF)

if (ALPS_USE_CXX03)
  # FIXME: we have to "downgrade" the standard if compiler's default is C++1x
  alps_get_cxx03_flag(cxx03_flag_)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxx03_flag_}")
  set(ALPS_CXX_FEATURES "" CACHE INTERNAL "List of C++ features required by ALPSCore")
  set(ALPS_CXX_FLAGS ${cxx03_flag_} CACHE INTERNAL "C++ compilation flags to be set as interface")
endif()

if (ALPS_USE_CXX11)
  set(ALPS_CXX_FEATURES "cxx_auto_type;cxx_constexpr" CACHE INTERNAL "List of C++ features required by ALPSCore")
  set(ALPS_CXX_FLAGS "" CACHE INTERNAL "C++ compilation flags to be set as interface")  
endif()
