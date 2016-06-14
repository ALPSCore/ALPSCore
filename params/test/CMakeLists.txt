include(ALPSEnableTests)

set (test_src
     file_read
     assign
     compare
     defaults
     not_found
     predicates_and_actions
     short_options
     options_help
     missing_iterator
     serialize
     param_ostream
     common_param_tests
    )


foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

if(ALPS_HAVE_MPI)
  alps_add_gtest(mpi_bcast NOMAIN PARTEST)
  alps_add_gtest(mpi_common_param_tests NOMAIN PARTEST)
endif()
