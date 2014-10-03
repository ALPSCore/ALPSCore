
# rule for generating single core ising program
add_executable(tutorial_02_single ising.cpp single.cpp)
target_link_libraries(tutorial_02_single ${PROJECT_NAME} ${LINK_ALL})

if (ALPS_HAVE_MPI)

	add_executable(tutorial_02_mpi mpi.cpp ising.cpp)
	target_link_libraries(tutorial_02_mpi ${PROJECT_NAME} ${LINK_ALL})

#	add_executable(tutorial_02_pscan pscan.cpp single.cpp)
#	target_link_libraries(tutorial_02_pscan ${PROJECT_NAME} ${LINK_ALL})

endif (ALPS_HAVE_MPI)

if (ALPS_HAVE_PYTHON)

	# rule for generating python export
	set_property(GLOBAL APPEND PROPERTY PY_MODULES_LIST ising_c)
	if(BUILD_SHARED_LIBS)
	    add_library(ising_c MODULE ising.cpp export.cpp)
		set_target_properties(ising_c PROPERTIES COMPILE_DEFINITIONS "${ALPS_SHARED_CPPFLAGS}")
	else(BUILD_SHARED_LIBS)
		set_property(GLOBAL APPEND PROPERTY PY_STATIC_MODULES_LIST ising_c)
		add_library(ising_c STATIC ising.cpp export.cpp)
	endif (BUILD_SHARED_LIBS)
	set_target_properties(ising_c PROPERTIES PREFIX "")
	target_link_libraries(ising_c ${PROJECT_NAME} ${LINK_ALL})

endif (ALPS_HAVE_PYTHON)
