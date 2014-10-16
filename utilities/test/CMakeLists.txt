include(ALPSEnableTests)

set (test_src
    bitops
    temporary_filename
    vmusage
    vec_reduce
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

