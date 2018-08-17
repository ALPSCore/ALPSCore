include(ALPSEnableTests)

if (HDF5_VERSION VERSION_LESS 1.10)
  add_definitions("-DALPS_HDF5_1_8")
endif()

set (test_src
    testharness
    acc_check_eigen
    assign_acc
    get_results
    add_accumulators
    count
    divide_accumulators
    unary_ops
    mean
    merge
    mult_by_constant
    weighted
    nonexistent
    acc_count_test
    mean_err_count
    save_load
    save_load2
    vec_const_binop_simple
    binop_with_constant
    binop_with_scalar
    binop_mixed
    binop_mixed_faildemo
    single_accumulator
    autocorrelation
    concurrent_access
    print
    scalar_result_type
    negative_error # FIXME!! Incorporate in the corresponding test
    )

#add tests for MPI
if(ALPS_HAVE_MPI)
  set (test_src_mpi
    mpi_merge
    mpi_merge_uneven
    repeated_merge
    zero_vector_mpi
    )
endif()

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

foreach(test ${test_src_mpi})
    alps_add_gtest(${test} NOMAIN PARTEST)
endforeach(test)
