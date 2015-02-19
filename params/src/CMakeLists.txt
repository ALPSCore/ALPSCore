set (alps-params_src
    params
    paramproxy
    paramvalue
    )

add_library(${PROJECT_NAME} ${ALPS_BUILD_TYPE} ${alps-params_src})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${PROJECT_NAME} ${LINK_ALL})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

