#  Copyright Matthias Troyer 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/tool)
include_directories(${Boost_ROOT_DIR})

#define tests here. Each test neet ${name}.C ${name}.input ${name}.output
FOREACH (name detailedbinning histogram signed simpleobseval testobservableset observableset_xml vectorobseval obsbinneddata value_with_error.test)

	add_executable(${name} ${name}.C)
	add_dependencies(${name} alps)
	target_link_libraries(${name} alps)
	# install(TARGETS ${name} RUNTIME DESTINATION bin)
        if(MSVC)
          get_target_property(EXE_NAME ${name} LOCATION)
          add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${EXE_NAME} ${PROJECT_BINARY_DIR}/bin)
        endif(MSVC)
	
	ADD_TEST(
		${name} 
		${CMAKE_COMMAND}
			-Dcmd=${name}
                        -Dsourcedir=${CMAKE_CURRENT_SOURCE_DIR}
                        -Dbinarydir=${CMAKE_CURRENT_BINARY_DIR}
                        -Ddllexedir=${PROJECT_BINARY_DIR}/bin
			-Dinput=${name}.input
			-Doutput=${name}.output
			-Dtimeout=600
			-P ${PROJECT_SOURCE_DIR}/config/CMake/run_test.cmake
	)
ENDFOREACH(name)
