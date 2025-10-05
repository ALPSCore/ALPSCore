cmake_minimum_required(VERSION 3.1)
project(iniparser C)

add_library(libiniparser OBJECT src/dictionary.c src/iniparser.c)
target_include_directories(libiniparser PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
