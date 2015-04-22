# FIXME: FindALPSCore.cmake
include(FindPackageHandleStandardArgs)

message(STATUS "Going to search in: ${ALPS_ROOT} $ENV{ALPS_ROOT} ${CMAKE_CURRENT_LIST_DIR}/../../..")
find_package(ALPSCore CONFIG QUIET HINTS ${ALPS_ROOT} $ENV{ALPS_ROOT} ${CMAKE_CURRENT_LIST_DIR}/../../..)
find_package_handle_standard_args(ALPSCore CONFIG_MODE)
