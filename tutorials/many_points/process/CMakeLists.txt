project(Tutorial_manypoints C CXX)
cmake_minimum_required(VERSION 3.0)

find_package(ALPSCore 2.0 REQUIRED)

add_executable(binder_cumulant "main.cpp")
target_link_libraries(binder_cumulant ${ALPSCore_LIBRARIES})

