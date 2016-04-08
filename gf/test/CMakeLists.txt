message(STATUS "Building tests")

add_library(gf_test gf_test.cpp)
target_link_libraries(gf_test ${LINK_ALL})
list(APPEND LINK_ALL gf_test)

set (test_src
  index_test 
  mesh_test
  one_index_gf_test
  two_index_gf_test
  three_index_gf_test
  four_index_gf_test 
  four_index_gf_pn_test 
  itime_gf_test
  fourier_test
    )


foreach(test ${test_src})
    add_gtest(${test})
endforeach(test)
# Tests that need additional source files
add_gtest(four_index_gf_test_mismatched-mpi mpi_guard.cpp)
add_gtest(four_index_gf_test_mismatched-tail-mpi mpi_guard.cpp)
add_gtest(multiarray_bcast_mpi mpi_guard.cpp)
add_gtest(mesh_test_mpi mpi_guard.cpp)
