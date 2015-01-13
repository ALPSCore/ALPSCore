set (alps-accumulators_src
    accumulator.cpp
    accumulators.cpp
    wrapper_set.cpp
    )

add_library(${PROJECT_NAME} ${ALPS_BUILD_TYPE} ${alps-accumulators_src})
target_link_libraries(${PROJECT_NAME} ${LINK_ALL})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

