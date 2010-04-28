#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

FOREACH (name alea alea2 io)
  add_executable(${name} ${name}.C)
  add_dependencies(${name} alps)
  target_link_libraries(${name} alps)
  add_alps_test(${name})
ENDFOREACH(name)

#  h5multiarray h5groups h5alea h5enum h5overwrite
FOREACH (name h5vector h5ptr h5valarray h5ublasvector h5scalar h5attribute h5discontinous h5class h5pair)
    add_executable(${name} ${name}.cpp)
    add_dependencies(${name} boost)
    target_link_libraries(${name} boost)
    if(MSVC)
        get_target_property(EXE_NAME ${name} LOCATION)
        add_custom_command(TARGET ${name} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${EXE_NAME} ${PROJECT_BINARY_DIR}/bin)
    endif(MSVC)
    ADD_TEST(
        ${name} 
        ${CMAKE_COMMAND}
            -Dcmd=${name}
            -Dsourcedir=${CMAKE_CURRENT_SOURCE_DIR}
            -Dbinarydir=${CMAKE_CURRENT_BINARY_DIR}
            -Ddllexedir=${PROJECT_BINARY_DIR}/bin
            -Doutput=${name}.output
            -Dtimeout=600
            -P ${PROJECT_SOURCE_DIR}/config/run_test.cmake
    )
ENDFOREACH(name)
