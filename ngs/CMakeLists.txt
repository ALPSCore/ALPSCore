#  Copyright Matthias Troyer 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)


include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

FOREACH (name mchdf5)
    add_executable(${name} ${name}.cpp)
    add_dependencies(${name} alps)
    target_link_libraries(${name} alps)
    add_alps_test(${name})
ENDFOREACH(name)

if (MPI_FOUND)
#    add_executable(test_ngs main.cpp)
#    add_dependencies(test_ngs alps)
#    target_link_libraries(test_ngs alps)
endif(MPI_FOUND)

add_executable(ngs_sim1 sim1.cpp)
add_dependencies(ngs_sim1 alps)
target_link_libraries(ngs_sim1 alps)

add_test(ngs_sim1
    ${CMAKE_COMMAND}
        -Dsourcedir=${PROJECT_SOURCE_DIR}
        -Dbinarydir=${PROJECT_BINARY_DIR}
        -Dcmddir=${CMAKE_CURRENT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_ngs_sim1_test.cmake
)
