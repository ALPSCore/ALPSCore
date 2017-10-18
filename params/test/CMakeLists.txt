include(ALPSEnableTests)

# set (test_src
#      file_read
#      repeated_options
#      assign
#      compare
#      defaults
#      not_found
#      predicates_and_actions
#      short_options
#      options_help
#      convenience_params
#      missing_iterator
#      serialize
#      param_ostream
#      common_param_tests
#      disappearing_ini
#     )

set (test_src
  dictionary
  params
  params_formats  
  )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

# if(ALPS_HAVE_MPI)
#   alps_add_gtest(mpi_bcast NOMAIN PARTEST)
#   alps_add_gtest(mpi_common_param_tests NOMAIN PARTEST)
# endif()
