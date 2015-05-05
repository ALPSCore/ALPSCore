
# rule for generating single core ising program
add_executable(tutorial_02_single ising.cpp single.cpp)
target_link_libraries(tutorial_02_single ${PROJECT_NAME})

if (ENABLE_MPI)

	add_executable(tutorial_02_mpi ising.cpp mpi.cpp)
	target_link_libraries(tutorial_02_mpi ${PROJECT_NAME})

        #FIXME: removed for now!
	#add_executable(tutorial_02_pscan ising.cpp mpi_pscan.cpp)
	#target_link_libraries(tutorial_02_pscan ${PROJECT_NAME})

endif(ENABLE_MPI)
