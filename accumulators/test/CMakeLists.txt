include(ALPSEnableTests)

set (test_src
    count
    mean
    weighted
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
