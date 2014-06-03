include(ALPSEnableTests)

set (test_src
    count
    mean 
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
