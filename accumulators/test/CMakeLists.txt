include(ALPSEnableTests)

set (test_src
    testharness
    get_results
    add_accumulators
    count
    divide_accumulators
    unary_ops
    mean
    mean_wrapped
    merge
    mult_by_constant
    weighted
    nonexistent
    acc_count_test
    mean_err_count
    save_load
    vec_const_binop_simple
    binop_with_constant
    binop_with_scalar
    single_accumulator
    autocorrelation
    print
    )

#add tests for MPI
if(ALPS_HAVE_MPI)
  set (test_src_mpi
    mpi_merge
    )
endif()

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

foreach(test ${test_src_mpi})
    alps_add_gtest(${test} NOMAIN PARTEST)
endforeach(test)
