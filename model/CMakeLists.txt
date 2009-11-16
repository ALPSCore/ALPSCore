include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/tool)
include_directories(${Boost_ROOT_DIR})

#define tests here. Each test neet ${name}.C ${name}.input ${name}.output
FOREACH (name example1 example2 example3 example4 example5 example6 example7 example8 example9 example10 example11 example12 example13 example14 example15 example16 example17 example18)

	add_executable(model_${name} ${name}.C)
	add_dependencies(model_${name} alps)
	target_link_libraries(model_${name} alps)
	# install(TARGETS model_${name} RUNTIME DESTINATION bin)
	
	ADD_TEST(
		model_${name} 
		${CMAKE_COMMAND}
			-Dcmd=model_${name}
                        -Dsourcedir=${CMAKE_CURRENT_SOURCE_DIR}
                        -Dbinarydir=${CMAKE_CURRENT_BINARY_DIR}
			-Dinput=${name}.input
			-Doutput=${name}.output
			-Dtimeout=600
			-P ${PROJECT_SOURCE_DIR}/config/CMake/run_test.cmake
	)
ENDFOREACH(name)
