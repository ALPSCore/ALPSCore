include(ALPSEnableTests)

set (test_src
    bitops
    temporary_filename
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

# vmusage depends on command line args, so no gtest xml  
alps_add_gtest(vmusage false) 
