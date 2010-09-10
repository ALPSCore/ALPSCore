#  Copyright Matthias Troyer, Synge Todo and Lukas Gamper 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

if (ALPS_BUILD_PYTHON)
    FOREACH (name value_with_error mcdata)
        add_test(python_${name}
            ${CMAKE_COMMAND}
                -Dcmd=${name}.py
                -Dsourcedir=${PROJECT_SOURCE_DIR}
                -Dbinarydir=${PROJECT_BINARY_DIR}
                -Dcmddir=${CMAKE_CURRENT_SOURCE_DIR}
                -P ${PROJECT_SOURCE_DIR}/config/run_python_test.cmake
        )
  ENDFOREACH(name)
ENDIF(ALPS_BUILD_PYTHON)
