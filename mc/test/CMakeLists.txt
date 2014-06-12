include(ALPSEnableTests)

set (test_src
    sum_single
    sum_mpi
    )


foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
