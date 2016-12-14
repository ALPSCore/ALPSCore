project(Tutorial_manypoints C CXX)
cmake_minimum_required(VERSION 2.8.12)

find_package(ALPSCore REQUIRED)

add_executable(binder_cumulant "main.cpp")
target_link_libraries(binder_cumulant ${ALPSCore_LIBRARIES})

