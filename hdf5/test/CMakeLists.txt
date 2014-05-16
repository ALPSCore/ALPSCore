include(ALPSEnableTests)

set (test_src
    hdf5_complex
    hdf5_bool
    hdf5_copy
    hdf5_exceptions
    hdf5_family
    hdf5_memory
    hdf5_misc
    hdf5_multiarchive
    hdf5_pair
    hdf5_real_complex_vec
    hdf5_replace
    hdf5_vecvecdbl
    hdf5_vecveccplx
    )
#hdf5_complex hdf5_copy hdf5_real_complex_vec hdf5_real_complex_matrix hdf5_bool hdf5_replace hdf5_pair hdf5_parms 
#            hdf5_ising hdf5_family hdf5_valgrind hdf5_observableset hdf5_multi_array hdf5_memory hdf5_misc hdf5_vecveccplx 
#            hdf5_exceptions hdf5_multiarchive hdf5_fortran_string hdf5_vecvecdbl)


foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
