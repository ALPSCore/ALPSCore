include(ALPSEnableTests)

set (test_src
    test_utils_test
    unique_file
    filename_operations
    temporary_filename
    gtest_par_xml_output
    inf
    type_traits_test
    vector_functions
    rectangularize
    tensor_test
    )

set (test_src_mpi
    mpi_utils_env
    mpi_utils_comm
    mpi_utils_bcast
    mpi_utils_bcast_optional
    mpi_utils_reduce
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

if(ALPS_HAVE_MPI)
  foreach(test ${test_src_mpi})
    alps_add_gtest(${test} PARTEST NOMAIN)
  endforeach(test)
  alps_add_gtest(mpi_utils_env_exceptions1 PARTEST SRCS mpi_test_support)
  alps_add_gtest(mpi_utils_env_exceptions2 PARTEST SRCS mpi_test_support)
endif()
