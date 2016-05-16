include(ALPSEnableTests)

set (test_src
    temporary_filename
    gtest_par_xml_output
    inf
    type_traits
    vector_functions
    rectangularize
    )

set (test_src_mpi
    mpi_utils
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

if(ALPS_HAVE_MPI)
  foreach(test ${test_src_mpi})
    alps_add_gtest(${test} NOMAIN PARTEST)
  endforeach(test)
endif()
