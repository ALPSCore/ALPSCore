include(ALPSEnableTests)

set (test_src
     file_read
#     param
     assign
#     default
     not_found
     short_options
     subclass
#     redefinition
#     ordering
#     stream
    )


foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
