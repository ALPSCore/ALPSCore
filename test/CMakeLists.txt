set (test_src
  four_index_gf_pn_test 
  four_index_gf_test 
  index_test 
  itime_gf_test mesh_test
  three_index_gf_test
    )


foreach(test ${test_src})
    add_gtest(${test})
endforeach(test)
