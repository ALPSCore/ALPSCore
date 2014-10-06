set (alps-mc_src
    mcbase
    api
    parseargs
    stop_callback
    )

add_library(${PROJECT_NAME} ${ALPS_BUILD_TYPE} ${alps-mc_src})
target_link_libraries(${PROJECT_NAME} ${LINK_ALL})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

