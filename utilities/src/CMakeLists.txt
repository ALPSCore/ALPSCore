set (alps_utils_src
    # copyright.cpp //FIXME
    os.cpp
    temporary_filename.cpp
    stacktrace.cpp
    signal.cpp
    parseargs.cpp
    )

#include_directories(
#    ${Boost_INCLUDE_DIRS}
#    ${PROJECT_SOURCE_DIR}/include
#    ${PROJECT_BINARY_DIR}/include
#)

add_library(${PROJECT_NAME} ${ALPS_BUILD_TYPE} ${alps_utils_src})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${PROJECT_NAME} ${LINK_ALL})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)


