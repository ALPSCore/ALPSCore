set (alps-accumulators_src
    accumulator.cpp
    )

add_library(${PROJECT_NAME} STATIC ${alps-accumulators_src})
target_link_libraries(${PROJECT_NAME} ${LINK_ALL})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

