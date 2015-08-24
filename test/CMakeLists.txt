set (test_src
     usage
     matsubara_gf
    )


foreach(test ${test_src})
    add_gtest(${test})
endforeach(test)
