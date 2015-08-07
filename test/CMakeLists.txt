set (test_src
     usage
    )


foreach(test ${test_src})
    add_gtest(${test})
endforeach(test)
