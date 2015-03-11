#
# Check compiler version and add specific options, if needed
#
include(CheckCXXCompilerFlag)
if (NOT DEFINED CMAKE_CXX_COMPILER_ID OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  CHECK_CXX_COMPILER_FLAG("-ftemplate-depth=1024" has_template_depth)
  if (has_template_depth)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=1024")
  endif()
endif()
