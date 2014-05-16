include(ALPSEnableTests)

set (test_src
    bitops
    temporary_filename
    vmusage
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

