include(ALPSEnableTests)

set (test_src
    sum_single
    )


foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
