# generate ising program
add_executable(tutorial_01_ising ising.cpp main.cpp)
target_link_libraries(tutorial_01_ising ${PROJECT_NAME} ${LINK_ALL})
