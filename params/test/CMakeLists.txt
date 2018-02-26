include(ALPSEnableTests)

set (test_src
  serialize_variant
  dictionary
  params
  dots_in_keys
  params_formats
  params_cmdline
  params_defaulted
  params_has_unused  
  params_origins
  convenience_params
  dictval_eq  
  dictionary_eq
  params_eq
  dictionary_hdf5
  params_hdf5
  )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

if(ALPS_HAVE_MPI)
   alps_add_gtest(serialize_variant_mpi NOMAIN PARTEST)
   alps_add_gtest(dictionary_bcast_mpi NOMAIN PARTEST)
   alps_add_gtest(params_bcast_mpi NOMAIN PARTEST)
endif()
