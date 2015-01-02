include(ALPSEnableTests)

set (test_src
    add_accumulators
    count
    divide_accumulators
    invert_accumulators
    mean
    mean_wrapped
    merge
    mult_by_constant
    weighted
    nonexistent
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
