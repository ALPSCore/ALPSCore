include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/tool)
include_directories(${Boost_ROOT_DIR})

#define tests here. Each test needs ${name}.C ${name}.input ${name}.output
FOREACH (name example1 example2 example3 example4 example5 example6 example7 example8 example9 example10 example11)

	add_executable(lattice_${name} ${name}.C)
	add_dependencies(lattice_${name} alps)
	target_link_libraries(lattice_${name} alps)
	# install(TARGETS lattice_${name} RUNTIME DESTINATION bin)
        if(MSVC)
          get_target_property(EXE_NAME lattice_${name} LOCATION)
          add_custom_command(TARGET lattice_${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy ${EXE_NAME} ${PROJECT_BINARY_DIR}/bin)
        endif(MSVC)
	
	ADD_TEST(
		lattice_${name} 
		${CMAKE_COMMAND}
			-Dcmd=lattice_${name}
                        -Dsourcedir=${CMAKE_CURRENT_SOURCE_DIR}
                        -Dbinarydir=${CMAKE_CURRENT_BINARY_DIR}
                        -Ddllexedir=${PROJECT_BINARY_DIR}/bin
			-Dinput=${name}.input
			-Doutput=${name}.output
			-Dtimeout=600
			-P ${PROJECT_SOURCE_DIR}/config/CMake/run_test.cmake
	)
ENDFOREACH(name)

FOREACH (name coloring label parity)

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
