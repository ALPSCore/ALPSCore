include(ALPSEnableTests)

set (test_src
     file_read
     assign
     compare
     defaults
     not_found
     predicates_and_actions
     short_options
     subclass
     serialize
     ostream
    )


foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
