include(ALPSEnableTests)

set (test_src
    temporary_filename
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

