include(ALPSEnableTests)

set (test_src
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
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
