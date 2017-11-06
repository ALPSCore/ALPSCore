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
