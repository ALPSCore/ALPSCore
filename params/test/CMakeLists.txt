include(ALPSEnableTests)

set (test_src
     file_read
#     param
     assign
     defaults
     not_found
     short_options
     subclass
     serialize
     ostream
#     redefinition
#     ordering
#     stream
    )


foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
