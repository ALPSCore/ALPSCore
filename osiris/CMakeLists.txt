include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/tool)
include_directories(${Boost_ROOT_DIR})

configure_file(xdrdump2.input.cmake ${CMAKE_CURRENT_BINARY_DIR}/boostdump2.input)
configure_file(xdrdump2.input.cmake ${CMAKE_CURRENT_BINARY_DIR}/xdrdump2.input)

#define tests here. Each test neet ${name}.C ${name}.input ${name}.output
FOREACH (name os sizeof xdrdump xdrdump2 boostdump boostdump2 boostdump3 boostdump4)
# FOREACH (name os sizeof xdrdump xdrdump2)

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
