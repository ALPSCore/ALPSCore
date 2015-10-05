include(ALPSEnableTests)

set (test_src
    temporary_filename
    gtest_par_xml_output
    inf
    type_traits
    )

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

