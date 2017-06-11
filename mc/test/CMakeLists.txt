include(ALPSEnableTests)

set (test_src
    sum_single
    timer_in_sim
    timer
    check_schedule
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

#add tests for MPI
if(ALPS_HAVE_MPI)

set (test_src_mpi
    sum_mpi
    reduce
    signed_obs
    custom_scheduler
    reduce_unavailable_results
    )
foreach(test ${test_src_mpi})
    alps_add_gtest(${test} NOMAIN PARTEST)
endforeach(test)

endif(ALPS_HAVE_MPI)
