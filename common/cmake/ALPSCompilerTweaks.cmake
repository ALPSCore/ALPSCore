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

#this is needed because we use boost multiarray. Otherwise codes will slow down to a crawl
if (NOT DEFINED BOOST_DO_NOT_DISABLE_ASSERTS)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_DISABLE_ASSERTS")
endif()
