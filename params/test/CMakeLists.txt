include(ALPSEnableTests)

set (test_src
     assign
     default
     not_found
     ordering
     stream
    )


foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
