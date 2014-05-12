#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

IF(ALPS_NGS_USE_NEW_ALEA)
	FOREACH (name count mean)
		add_executable(accumulator_${name} ${name}.cpp)
		add_dependencies(accumulator_${name} alps)
		target_link_libraries(accumulator_${name} alps)
		add_alps_test(accumulator_${name})
	ENDFOREACH(name)
ENDIF(ALPS_NGS_USE_NEW_ALEA)