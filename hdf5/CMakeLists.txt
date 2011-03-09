#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

option(ALPS_BUILD_HDF5_TESTS "Build the hdf5 tests" OFF)
mark_as_advanced(ALPS_BUILD_HDF5_TESTS)

IF (ALPS_BUILD_HDF5_TESTS)

    find_package(HDF5)

    # tests using ALPS
    FOREACH (name hdf5_alea)
        add_executable(${name} ${name}.cpp)
        add_dependencies(${name} alps)
        target_link_libraries(${name} alps)
        add_alps_test(${name})
    ENDFOREACH(name)

    # tests not using ALPS
    FOREACH (name hdf5_exceptions hdf5_multiarchive)
        add_executable(${name} ${name}.cpp)
        if (Boost_FOUND)
            target_link_libraries(${name} ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 
        else(Boost_FOUND) 
            add_dependencies(${name} boost) 
            target_link_libraries(${name} alps ${ALPS_BOOST_LIBRARY_NAME} ${HDF5_LIBRARIES})
        endif(Boost_FOUND) 
        add_alps_test(${name})
    ENDFOREACH(name)

    # texting type serialization
    SET(ALPS_HDF5_DATA_TYPES
        bool int short long float double
        std::size_t std::string std::complex<float> std::complex<double> "std::complex<long double>"
        boost::int8_t boost::uint8_t boost::int16_t boost::uint16_t boost::int32_t boost::uint32_t boost::int64_t boost::uint64_t
        std::vector<std::size_t> std::vector<short> std::vector<int> std::vector<long> std::vector<float> std::vector<double> "std::vector<std::complex<double> >" std::vector<std::string>
        std::valarray<int> std::valarray<double> "std::valarray<std::complex<double> >" 
        boost::numeric::ublas::vector<int> boost::numeric::ublas::vector<double> "boost::numeric::ublas::vector<std::complex<double> >"
        "std::pair<int *, std::vector<std::size_t> >" "std::pair<double *, std::vector<std::size_t> >" "std::pair<std::complex<double> *,std::vector<std::size_t> >" "std::pair<std::string *,std::vector<std::size_t> >"
        "std::vector<std::vector<int> >" "std::vector<std::vector<double> >" "std::vector<std::vector<std::complex<double> > >" "std::vector<std::vector<std::string> >"
        "std::vector<std::vector<std::vector<int> > >" "std::vector<std::vector<std::vector<double> > >" "std::vector<std::vector<std::vector<std::complex<double> > > >" "std::vector<std::vector<std::vector<std::string> > >"
        "std::vector<std::valarray<int> >" "std::valarray<std::vector<double> >" "std::vector<boost::numeric::ublas::vector<std::complex<double> > >"
        "std::pair<std::vector<int> *, std::vector<std::size_t> >" "std::pair<std::vector<double> *, std::vector<std::size_t> >" "std::pair<std::vector<std::complex<double> > *, std::vector<std::size_t> >" "std::pair<std::vector<std::string> *, std::vector<std::size_t> >"
        enum_type std::vector<enum_type> "std::vector<std::vector<enum_type> >" "std::pair<enum_type *, std::vector<std::size_t> >" "std::vector<std::valarray<enum_type> >"
        "std::pair<std::vector<enum_type> *, std::vector<std::size_t> >" "std::pair<std::vector<std::vector<enum_type> > *, std::vector<std::size_t> >"
        userdefined_class<std::size_t> userdefined_class<short> userdefined_class<int> userdefined_class<long> userdefined_class<float> userdefined_class<double> "userdefined_class<std::complex<double> >" userdefined_class<std::string>
        "std::vector<userdefined_class<double> >" "std::vector<std::vector<userdefined_class<double> > >" "std::pair<userdefined_class<double> *, std::vector<std::size_t> >" 
        "cast_type<int, long>" "cast_type<int, double>" "cast_type<double, std::string>" "cast_type<int, std::string>" "cast_type<float, double>" "cast_type<short, float>"
        "std::vector<cast_type<int, double> >" "std::vector<std::vector<cast_type<int, double> > >" "std::pair<cast_type<int, double> *, std::vector<std::size_t> >" "std::vector<std::valarray<cast_type<int, double> > >"
        "std::vector<cast_type<double, std::string> >" "std::vector<std::vector<cast_type<double, std::string> > >" "std::pair<cast_type<double, std::string> *, std::vector<std::size_t> >" 
#        "boost::multi_array<double, 1>" "boost::multi_array<int, 1>" "boost::multi_array<std::complex<double>, 1>" "boost::multi_array<std::string, 1>"
#        "boost::multi_array<double, 2>" "boost::multi_array<int, 2>" "boost::multi_array<std::complex<double>, 2>" "boost::multi_array<std::string, 2>"
#        "boost::multi_array<double, 3>" "boost::multi_array<int, 3>" "boost::multi_array<std::complex<double>, 3>" "boost::multi_array<std::string, 3>"
        "boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>" "boost::numeric::ublas::matrix<std::complex<double>, boost::numeric::ublas::column_major>"
        "int *" "short *" "long *" "float *" "double *"
        "std::size_t *" "std::string *" "std::complex<double> *"
        "enum_type *" "userdefined_class<double> *" "cast_type<int, double> *" "cast_type<int, std::string> *"
        "boost::shared_array<int>" "boost::shared_array<short>" "boost::shared_array<long>" "boost::shared_array<float>" "boost::shared_array<double>"
        "boost::shared_array<std::size_t>" "boost::shared_array<std::string>" "boost::shared_array<std::complex<double> >"
        "boost::shared_array<enum_type>" "boost::shared_array<userdefined_class<double> >" "boost::shared_array<cast_type<int, double> >" "boost::shared_array<cast_type<int, std::string> >"
        "cast_type<std::vector<int>, std::valarray<int> >" "cast_type<std::vector<int>, boost::numeric::ublas::vector<int> >"
        "cast_type<std::valarray<int>, std::vector<int> >" "cast_type<std::valarray<int>, boost::numeric::ublas::vector<int> >"
        "cast_type<boost::numeric::ublas::vector<int>, std::vector<int> >" "cast_type<boost::numeric::ublas::vector<int>, std::valarray<int> >"
        "cast_type<std::vector<int>, std::valarray<double> >" "cast_type<std::vector<int>, boost::numeric::ublas::vector<double> >"
        "cast_type<std::pair<int *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<int> > > >" "cast_type<std::pair<int *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<double> > > >"
        "std::pair<cast_type<std::vector<int>, std::valarray<long> > *, std::vector<std::size_t> >" "std::vector<cast_type<std::vector<int>, boost::numeric::ublas::vector<double> > >"
        "std::pair<double, int>" "std::pair<double, std::complex<double> >" "std::pair<cast_type<int, std::string>, enum_type>" "std::pair<enum_type, cast_type<int, double> >" 
        "std::pair<std::vector<cast_type<int, std::string> >, std::pair<double, int> >" "std::pair<std::pair<std::vector<enum_type> *, std::vector<std::size_t> >, enum_type>"
    )
    FOREACH (type ${ALPS_HDF5_DATA_TYPES})

        STRING(REGEX REPLACE "\\*" "p" test_name ${type})
        STRING(REGEX REPLACE "(::)|<|>| |\\," "_" test_name ${test_name})
        STRING(REGEX REPLACE "__+" "_" test_name ${test_name})
        
        set(HDF5_TEST_TYPE ${type})
        include_directories(.)
        
        if (ALPS_BUILD_NGS)

            configure_file(type_check.cpp.in ${PROJECT_BINARY_DIR}/test/hdf5/ngs_hdf5_data_${test_name}.cpp)
            add_executable(ngs_hdf5_data_${test_name} ${PROJECT_BINARY_DIR}/test/hdf5/ngs_hdf5_data_${test_name}.cpp)
            SET_TARGET_PROPERTIES(ngs_hdf5_data_${test_name} PROPERTIES COMPILE_FLAGS "-DUSE_NG_ARCHIVE")
	    if (SZIP_FOUND)
              configure_file(type_check.cpp.in ${PROJECT_BINARY_DIR}/test/hdf5/ngs_hdf5_data_szip_${test_name}.cpp)
              add_executable(ngs_hdf5_data_szip_${test_name} ${PROJECT_BINARY_DIR}/test/hdf5/ngs_hdf5_data_szip_${test_name}.cpp)
              SET_TARGET_PROPERTIES(ngs_hdf5_data_szip_${test_name} PROPERTIES COMPILE_FLAGS "-DSZIP_COMPRESS=true -DUSE_NG_ARCHIVE")
            endif (SZIP_FOUND)

            configure_file(type_check.cpp.in ${PROJECT_BINARY_DIR}/test/hdf5/ngs_hdf5_attr_${test_name}.cpp)
            add_executable(ngs_hdf5_attr_${test_name} ${PROJECT_BINARY_DIR}/test/hdf5/ngs_hdf5_attr_${test_name}.cpp)
            SET_TARGET_PROPERTIES(ngs_hdf5_attr_${test_name} PROPERTIES COMPILE_FLAGS "-DIS_ATTRIBUTE=true -DUSE_NG_ARCHIVE")

            add_dependencies(ngs_hdf5_data_${test_name} alps)
            target_link_libraries(ngs_hdf5_data_${test_name} alps)

            if (SZIP_FOUND)
              add_dependencies(ngs_hdf5_data_szip_${test_name} alps)
              target_link_libraries(ngs_hdf5_data_szip_${test_name} alps)
            endif (SZIP_FOUND)

            add_dependencies(ngs_hdf5_attr_${test_name} alps)
            target_link_libraries(ngs_hdf5_attr_${test_name} alps)

            add_alps_test(ngs_hdf5_data_${test_name})

            if (SZIP_FOUND)
              add_alps_test(ngs_hdf5_data_szip_${test_name})
            endif (SZIP_FOUND)

           add_alps_test(ngs_hdf5_attr_${test_name})

        else (ALPS_BUILD_NGS)

	        configure_file(type_check.cpp.in ${PROJECT_BINARY_DIR}/test/hdf5/hdf5_data_${test_name}.cpp)
	        add_executable(hdf5_data_${test_name} ${PROJECT_BINARY_DIR}/test/hdf5/hdf5_data_${test_name}.cpp)

                if (SZIP_FOUND)
		  configure_file(type_check.cpp.in ${PROJECT_BINARY_DIR}/test/hdf5/hdf5_data_szip_${test_name}.cpp)
	          add_executable(hdf5_data_szip_${test_name} ${PROJECT_BINARY_DIR}/test/hdf5/hdf5_data_szip_${test_name}.cpp)
	          SET_TARGET_PROPERTIES(hdf5_data_szip_${test_name} PROPERTIES COMPILE_FLAGS "-DSZIP_COMPRESS=true")
                endif (SZIP_FOUND)

	        configure_file(type_check.cpp.in ${PROJECT_BINARY_DIR}/test/hdf5/hdf5_data_pycplx_${test_name}.cpp)
	        add_executable(hdf5_data_pycplx_${test_name} ${PROJECT_BINARY_DIR}/test/hdf5/hdf5_data_pycplx_${test_name}.cpp)
	        SET_TARGET_PROPERTIES(hdf5_data_pycplx_${test_name} PROPERTIES COMPILE_FLAGS "-DALPS_HDF5_WRITE_PYTHON_COMPATIBLE_COMPLEX")

	        configure_file(type_check.cpp.in ${PROJECT_BINARY_DIR}/test/hdf5/hdf5_attr_${test_name}.cpp)
	        add_executable(hdf5_attr_${test_name} ${PROJECT_BINARY_DIR}/test/hdf5/hdf5_attr_${test_name}.cpp)
	        SET_TARGET_PROPERTIES(hdf5_attr_${test_name} PROPERTIES COMPILE_FLAGS "-DIS_ATTRIBUTE=true")

	        configure_file(type_check.cpp.in ${PROJECT_BINARY_DIR}/test/hdf5/hdf5_attr_pycplx_${test_name}.cpp)
	        add_executable(hdf5_attr_pycplx_${test_name} ${PROJECT_BINARY_DIR}/test/hdf5/hdf5_attr_pycplx_${test_name}.cpp)
	        SET_TARGET_PROPERTIES(hdf5_attr_pycplx_${test_name} PROPERTIES COMPILE_FLAGS "-DIS_ATTRIBUTE=true -DALPS_HDF5_WRITE_PYTHON_COMPATIBLE_COMPLEX")

	        if (Boost_FOUND)

	            target_link_libraries(hdf5_data_${test_name} ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 

		    if (SZIP_FOUND)
	              target_link_libraries(hdf5_data_szip_${test_name} ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 
		    endif (SZIP_FOUND)

	            target_link_libraries(hdf5_data_pycplx_${test_name} ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 

	            target_link_libraries(hdf5_attr_${test_name} ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 

	            target_link_libraries(hdf5_attr_pycplx_${test_name} ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 

	        else(Boost_FOUND) 

	            add_dependencies(hdf5_data_${test_name} alps boost)
	            target_link_libraries(hdf5_data_${test_name} alps ${ALPS_BOOST_LIBRARY_NAME} ${HDF5_LIBRARIES})

		    if (SZIP_FOUND)
	              add_dependencies(hdf5_data_szip_${test_name} alps boost) 
	              target_link_libraries(hdf5_data_szip_${test_name} alps ${ALPS_BOOST_LIBRARY_NAME} ${HDF5_LIBRARIES})
                    endif (SZIP_FOUND)

	            add_dependencies(hdf5_data_pycplx_${test_name} alps boost) 
	            target_link_libraries(hdf5_data_pycplx_${test_name} alps ${ALPS_BOOST_LIBRARY_NAME} ${HDF5_LIBRARIES})

	            add_dependencies(hdf5_attr_${test_name} alps boost) 
	            target_link_libraries(hdf5_attr_${test_name} alps ${ALPS_BOOST_LIBRARY_NAME} ${HDF5_LIBRARIES})

	            add_dependencies(hdf5_attr_pycplx_${test_name} alps boost) 
	            target_link_libraries(hdf5_attr_pycplx_${test_name} alps ${ALPS_BOOST_LIBRARY_NAME} ${HDF5_LIBRARIES})

	        endif(Boost_FOUND) 

	        add_alps_test(hdf5_data_${test_name})

                if (SZIP_FOUND)
	          add_alps_test(hdf5_data_szip_${test_name})
                endif (SZIP_FOUND)

	        add_alps_test(hdf5_data_pycplx_${test_name})

	        add_alps_test(hdf5_attr_${test_name})

	        add_alps_test(hdf5_attr_pycplx_${test_name})

        endif(ALPS_BUILD_NGS) 

    ENDFOREACH(type)

ENDIF (ALPS_BUILD_HDF5_TESTS)
