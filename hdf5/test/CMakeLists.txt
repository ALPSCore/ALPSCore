include(ALPSEnableTests)

set (test_src
    hdf5_open_strmode
    hdf5_open_intmode
    hdf5_complex
    hdf5_bool
    hdf5_copy
    hdf5_exceptions
    hdf5_memory
    hdf5_misc
    hdf5_native_type    
    hdf5_multiarchive
    hdf5_pair
    hdf5_real_complex_vec
    hdf5_replace
    hdf5_vecvecdbl
    hdf5_vecveccplx
    hdf5_boost_optional
    hdf5_io_types
    hdf5_attributes
    hdf5_omp #this one was commented out. Any idea why?
    hdf5_tensor
    )

if (ExtensiveTesting)
   set(test_src  ${test_src} hdf5_large_array)
endif()

foreach(test ${test_src})
    alps_add_gtest(${test})
endforeach(test)

if(ExtensiveTesting)
  SET_TARGET_PROPERTIES(hdf5_io_types PROPERTIES COMPILE_FLAGS "-DExtensiveTesting")
endif (ExtensiveTesting)

    # TODO: fix szip and attribute dependence ...
#    if (SZIP_FOUND)
#        configure_file(hdf5_io.cpp.in ${PROJECT_BINARY_DIR}/hdf5/test/hdf5_dz${test_name}.cpp)
#        alps_add_gtest(hdf5_d${test_name} ${PROJECT_BINARY_DIR}/hdf5/test)
#        SET_TARGET_PROPERTIES(hdf5_dz${test_name} PROPERTIES COMPILE_FLAGS "-DSZIP_COMPRESS=true")
#    endif (SZIP_FOUND)

    #configure_file(hdf5_io.cpp.in ${PROJECT_BINARY_DIR}/hdf5/test/hdf5_a${test_name}.cpp)
    #alps_add_gtest(hdf5_a${test_name} ${PROJECT_BINARY_DIR}/hdf5/test)
    #SET_TARGET_PROPERTIES(hdf5_a${test_name} PROPERTIES COMPILE_FLAGS "-DIS_ATTRIBUTE=true")
