set (alps-hdf5_src
    archive.cpp
    )

add_library(${PROJECT_NAME} STATIC ${alps-hdf5_src})
target_link_libraries(${PROJECT_NAME} ${LINK_ALL})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

