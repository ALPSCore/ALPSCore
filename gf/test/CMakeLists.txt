include(ALPSEnableTests)

set (test_srcs
  gf_check_eigen
  gf_new_test
  gf_new_tail_test
  index_test
  mesh_test
  one_index_gf_test
  gf1_print_test
  two_index_gf_test
  three_index_gf_test
  four_index_gf_test 
  four_index_gf_pn_test 
  seven_index_gf_test
  itime_gf_test
  fourier_test
  grid_test
  piecewise_polynomial_test
    )


foreach(test ${test_srcs})
    alps_add_gtest(${test} SRCS gf_test)
endforeach(test)

set (mpi_test_srcs
    one_index_gf_test_mpi
    four_index_gf_test_mpi
    four_index_gf_test_mismatched_mpi
    four_index_gf_test_mismatched-tail_mpi
    multiarray_bcast_mpi 
    mesh_test_mpi
    gf_new_test_mpi
    gf_new_tail_test_mpi)

if (ALPS_HAVE_MPI) 
    foreach(test ${mpi_test_srcs})
        alps_add_gtest(${test} NOMAIN PARTEST SRCS gf_test mpi_guard)
    endforeach(test)
endif()

