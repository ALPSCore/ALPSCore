include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/tool)
include_directories(${Boost_ROOT_DIR})

#define tests here. Each test neet ${name}.C ${name}.input ${name}.output
FOREACH (name xmlhandler xmlparser xmlstream)

	add_executable(${name} ${name}.C)
	add_dependencies(${name} alps)
	target_link_libraries(${name} alps)
	install(TARGETS ${name} RUNTIME DESTINATION bin)
	
	ADD_TEST(
		${name} 
		${CMAKE_COMMAND}
			-Dcmd=${name}
                        -Dsourcedir=${CMAKE_CURRENT_SOURCE_DIR}
                        -Dbinarydir=${CMAKE_CURRENT_BINARY_DIR}
			-Dinput=${name}.input
			-Doutput=${name}.output
			-Dtimeout=600
			-P ${PROJECT_SOURCE_DIR}/config/CMake/run_test.cmake
	)
ENDFOREACH(name)
