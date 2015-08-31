include(ALPSEnableTests)

set (test_src
     file_read
     assign
     defaults
     not_found
     predicates_and_actions
     short_options
     subclass
     serialize
     ostream
     valid_param_tests
     undef_param_tests
     common_param_tests_v2
    )


foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)
