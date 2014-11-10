
foreach (name
	pyrandom01 
	pyhdf5
	api
	pymcbase
)
    add_library(${name}_c MODULE ${name}.cpp import_numpy.cpp)

    set_target_properties(${name}_c PROPERTIES COMPILE_DEFINITIONS "${ALPS_SHARED_CPPFLAGS}")
    set_target_properties(${name}_c PROPERTIES PREFIX "")
	target_link_libraries(${name}_c ${PYTHON_LIBRARIES} ${LINK_ALL})
    include_directories(${PYTHON_INCLUDE_DIRS})

	install(TARGETS ${name}_c DESTINATION python)
endforeach(name)
