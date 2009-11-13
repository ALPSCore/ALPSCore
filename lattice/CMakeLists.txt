include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/tool)
include_directories(${Boost_ROOT_DIR})

#define tests here. Each test neet ${name}.C ${name}.input ${name}.output
FOREACH (name example1 example2 example3 example4 example5 example6 example7 example8 example9 example10 example11)

	add_executable(lattice_${name} ${name}.C)
	add_dependencies(lattice_${name} alps)
	target_link_libraries(lattice_${name} alps)
	install(TARGETS lattice_${name} RUNTIME DESTINATION bin)
	
	ADD_TEST(
		lattice_${name} 
		${CMAKE_COMMAND}
			-Dcmd=lattice_${name}
			-Dinput=${CMAKE_CURRENT_SOURCE_DIR}/${name}.input
			-Doutput=${CMAKE_CURRENT_SOURCE_DIR}/${name}.output
			-Dtimeout=600
			-P ${PROJECT_SOURCE_DIR}/config/CMake/run_test.cmake
	)
ENDFOREACH(name)
