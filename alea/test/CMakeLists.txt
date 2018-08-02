include(ALPSEnableTests)

set (test_src
     empty
     twogauss
     galois
     result
     transform
     stream_serializer
    )

#add tests for MPI
if(ALPS_HAVE_MPI)
  set (test_src_mpi
       mpi_twogauss
       )
endif()

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

foreach(test ${test_src_mpi})
    alps_add_gtest(${test} NOMAIN PARTEST)
endforeach(test)
