set (alps-accumulator_src
    accumulator.cpp
    )

add_library(${PROJECT_NAME} STATIC ${alps-accumulator_src})
target_link_libraries(${PROJECT_NAME} ${LINK_ALL})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

