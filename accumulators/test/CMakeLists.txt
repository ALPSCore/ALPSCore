include(ALPSEnableTests)

set (test_src
    add_accumulators
    count
    mean
    merge
    mult_by_constant
    weighted
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
