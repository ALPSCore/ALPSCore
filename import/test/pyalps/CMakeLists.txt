#  Copyright Matthias Troyer, Synge Todo and Lukas Gamper 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

#add_executable(loadobs loadobs.cpp)
#add_dependencies(loadobs alps)
#target_link_libraries(loadobs alps)
#add_alps_test(loadobs)

enable_testing()
if (ALPS_BUILD_PYTHON AND BUILD_SHARED_LIBS)
#    FOREACH (name pyioarchive pyhdf5io numpylarge pyparams hlist_test mcdata pyhdf5 mcanalyze)
    FOREACH (name pyioarchive pyhdf5io pyparams hlist_test mcdata pyhdf5 mcanalyze)
        add_test(python_${name}
            ${CMAKE_COMMAND}
                -Dpython_interpreter=${PYTHON_INTERPRETER}
                -Dcmd=${name}.py
                -Dinput=${name}
                -Doutput=${name}
                -Dpythonpath=${PROJECT_BINARY_DIR}/lib/pyalps:${PROJECT_SOURCE_DIR}/lib
                -Dsourcedir=${CMAKE_CURRENT_SOURCE_DIR}
                -Dbinarydir=${CMAKE_CURRENT_BINARY_DIR}
                -Dcmddir=${CMAKE_CURRENT_SOURCE_DIR}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/run_python_test.cmake
        )
        set_property(TEST python_${name} PROPERTY LABELS pyalps)
    ENDFOREACH(name)
ENDIF(ALPS_BUILD_PYTHON AND BUILD_SHARED_LIBS)
