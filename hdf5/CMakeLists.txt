#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

option(ALPS_BUILD_HDF5_TESTS "Build the hdf5 tests" OFF)

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
            target_link_libraries(${name} boost ${HDF5_LIBRARIES})
        endif(Boost_FOUND) 
        add_alps_test(${name})
    ENDFOREACH(name)

    # texting type serialization
    SET(ALPS_HDF5_DATA_TYPES
        bool int short long float double
        std::size_t std::string std::complex<int> "std::complex<long long>" std::complex<double> "std::complex<long double>"
        boost::int8_t boost::uint8_t boost::int16_t boost::uint16_t boost::int32_t boost::uint32_t boost::int64_t boost::uint64_t
        std::vector<std::size_t> std::vector<short> std::vector<int> std::vector<long> std::vector<float> std::vector<double> "std::vector<std::complex<int> >" "std::vector<std::complex<double> >" std::vector<std::string>
        std::deque<std::size_t> std::deque<short> std::deque<int> std::deque<long> std::deque<float> std::deque<double> "std::deque<std::complex<int> >" "std::deque<std::complex<double> >" std::deque<std::string>
        std::valarray<int> std::valarray<double> "std::valarray<std::complex<int> >" "std::valarray<std::complex<double> >" std::valarray<std::string>
        boost::numeric::ublas::vector<int> boost::numeric::ublas::vector<double> "boost::numeric::ublas::vector<std::complex<int> >" "boost::numeric::ublas::vector<std::complex<double> >"
        "std::pair<int *, std::vector<std::size_t> >" "std::pair<double *, std::vector<std::size_t> >" "std::pair<std::complex<int> *,std::vector<std::size_t> >" "std::pair<std::complex<double> *,std::vector<std::size_t> >" "std::pair<std::string *,std::vector<std::size_t> >"
        "std::vector<std::vector<int> >" "std::vector<std::vector<double> >" "std::vector<std::vector<std::complex<double> > >" "std::vector<std::vector<std::string> >"
        "std::vector<std::vector<std::vector<int> > >" "std::vector<std::vector<std::vector<double> > >" "std::vector<std::vector<std::vector<std::complex<double> > > >" "std::vector<std::vector<std::vector<std::string> > >"
        "std::vector<std::valarray<int> >" "std::valarray<std::vector<double> >" "std::vector<boost::numeric::ublas::vector<std::complex<double> > >" "boost::numeric::ublas::vector<std::valarray<std::string> >"
        "std::pair<std::vector<int> *, std::vector<std::size_t> >" "std::pair<std::vector<double> *, std::vector<std::size_t> >" "std::pair<std::vector<std::complex<double> > *, std::vector<std::size_t> >" "std::pair<std::vector<std::string> *, std::vector<std::size_t> >"
        enum_type std::vector<enum_type> "std::vector<std::vector<enum_type> >" "std::pair<enum_type *, std::vector<std::size_t> >" "std::vector<std::valarray<enum_type> >"
        "std::pair<std::vector<enum_type> *, std::vector<std::size_t> >" "std::pair<std::vector<std::vector<enum_type> > *, std::vector<std::size_t> >"
        userdefined_class<std::size_t> userdefined_class<short> userdefined_class<int> userdefined_class<long> userdefined_class<float> userdefined_class<double> "userdefined_class<std::complex<int> >" "userdefined_class<std::complex<double> >" userdefined_class<std::string>
        "std::vector<userdefined_class<double> >" "std::vector<std::vector<userdefined_class<double> > >" "std::pair<userdefined_class<double> *, std::vector<std::size_t> >" "std::vector<std::valarray<userdefined_class<double> > >"
        "cast_type<int, long>" "cast_type<int, double>" "cast_type<double, std::string>" "cast_type<int, std::string>" "cast_type<float, double>" "cast_type<short, float>"
        "std::vector<cast_type<int, double> >" "std::vector<std::vector<cast_type<int, double> > >" "std::pair<cast_type<int, double> *, std::vector<std::size_t> >" "std::vector<std::valarray<cast_type<int, double> > >"
        "std::vector<cast_type<double, std::string> >" "std::vector<std::vector<cast_type<double, std::string> > >" "std::pair<cast_type<double, std::string> *, std::vector<std::size_t> >" "std::vector<std::valarray<cast_type<double, std::string> > >"
        "boost::multi_array<double, 1>" "boost::multi_array<int, 1>" "boost::multi_array<std::complex<double>, 1>" "boost::multi_array<std::string, 1>"
        "boost::multi_array<double, 2>" "boost::multi_array<int, 2>" "boost::multi_array<std::complex<double>, 2>" "boost::multi_array<std::string, 2>"
        "boost::multi_array<double, 3>" "boost::multi_array<int, 3>" "boost::multi_array<std::complex<double>, 3>" "boost::multi_array<std::string, 3>"
        "boost::multi_array<boost::multi_array<double, 2>, 1>" "boost::multi_array<std::vector<double>, 1>" "std::vector<boost::multi_array<std::complex<double>, 1> >" "std::pair<boost::multi_array<std::string, 1> *, std::vector<std::size_t> >"
        "boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>" "boost::numeric::ublas::matrix<std::complex<double>, boost::numeric::ublas::column_major>"
        "int *" "short *" "long *" "float *" "double *"
        "std::size_t *" "std::string *" "std::complex<int> *" "std::complex<double> *"
        "enum_type *" "userdefined_class<double> *" "cast_type<int, double> *" "cast_type<int, std::string> *"
        "boost::shared_array<int>" "boost::shared_array<short>" "boost::shared_array<long>" "boost::shared_array<float>" "boost::shared_array<double>"
        "boost::shared_array<std::size_t>" "boost::shared_array<std::string>" "boost::shared_array<std::complex<int> >" "boost::shared_array<std::complex<double> >"
        "boost::shared_array<enum_type>" "boost::shared_array<userdefined_class<double> >" "boost::shared_array<cast_type<int, double> >" "boost::shared_array<cast_type<int, std::string> >"
        "cast_type<std::vector<int>, std::valarray<int> >" "cast_type<std::vector<int>, boost::numeric::ublas::vector<int> >"
        "cast_type<std::valarray<int>, std::vector<int> >" "cast_type<std::valarray<int>, boost::numeric::ublas::vector<int> >"
        "cast_type<boost::numeric::ublas::vector<int>, std::vector<int> >" "cast_type<boost::numeric::ublas::vector<int>, std::valarray<int> >"
        "cast_type<std::vector<int>, std::valarray<double> >" "cast_type<std::vector<int>, boost::numeric::ublas::vector<double> >"
        "cast_type<std::pair<int *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<int> > > >" "cast_type<std::pair<int *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<double> > > >"
        "std::pair<cast_type<std::vector<int>, std::valarray<long> > *, std::vector<std::size_t> >" "std::vector<cast_type<std::vector<int>, boost::numeric::ublas::vector<double> > >"
    )
    FOREACH (type ${ALPS_HDF5_DATA_TYPES})

        STRING(REGEX REPLACE "\\*" "p" test_name ${type})
        STRING(REGEX REPLACE "(::)|<|>| |\\," "_" test_name ${test_name})
        STRING(REGEX REPLACE "__+" "_" test_name ${test_name})

        add_executable(hdf5_data_${test_name} type_check.cpp)
        SET_TARGET_PROPERTIES(hdf5_data_${test_name} PROPERTIES COMPILE_FLAGS "-DTYPE='${type}' -DVECTOR_SIZE=25 -DMATRIX_SIZE=5 -DSEED=42")
        
        add_executable(hdf5_data_szip_${test_name} type_check.cpp)
        SET_TARGET_PROPERTIES(hdf5_data_szip_${test_name} PROPERTIES COMPILE_FLAGS "-DTYPE='${type}' -DVECTOR_SIZE=25 -DMATRIX_SIZE=5 -DSEED=42 -DSZIP_COMPRESS=true")
        
        add_executable(hdf5_data_pycplx_${test_name} type_check.cpp)
        SET_TARGET_PROPERTIES(hdf5_data_pycplx_${test_name} PROPERTIES COMPILE_FLAGS "-DTYPE='${type}' -DVECTOR_SIZE=25 -DMATRIX_SIZE=5 -DSEED=42 -DALPS_HDF5_WRITE_PYTHON_COMPATIBLE_COMPLEX")

        add_executable(hdf5_attr_${test_name} type_check.cpp)
        SET_TARGET_PROPERTIES(hdf5_attr_${test_name} PROPERTIES COMPILE_FLAGS "-DTYPE='${type}' -DVECTOR_SIZE=25 -DMATRIX_SIZE=5 -DIS_ATTRIBUTE=true -DSEED=42")

        add_executable(hdf5_attr_pycplx_${test_name} type_check.cpp)
        SET_TARGET_PROPERTIES(hdf5_attr_pycplx_${test_name} PROPERTIES COMPILE_FLAGS "-DTYPE='${type}' -DVECTOR_SIZE=25 -DMATRIX_SIZE=5 -DSEED=42 -DIS_ATTRIBUTE=true -DALPS_HDF5_WRITE_PYTHON_COMPATIBLE_COMPLEX")

        if (Boost_FOUND)

            target_link_libraries(hdf5_data_${test_name} ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 

            target_link_libraries(hdf5_data_szip_${test_name} ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 

            target_link_libraries(hdf5_data_pycplx_${test_name} ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 

            target_link_libraries(hdf5_attr_${test_name} ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 

            target_link_libraries(hdf5_attr_pycplx_${test_name} ${Boost_LIBRARIES} ${HDF5_LIBRARIES}) 

        else(Boost_FOUND) 

            add_dependencies(hdf5_data_${test_name} boost)
            target_link_libraries(hdf5_data_${test_name} boost ${HDF5_LIBRARIES})

            add_dependencies(hdf5_data_szip_${test_name} boost) 
            target_link_libraries(hdf5_data_szip_${test_name} boost ${HDF5_LIBRARIES})

            add_dependencies(hdf5_data_pycplx_${test_name} boost) 
            target_link_libraries(hdf5_data_pycplx_${test_name} boost ${HDF5_LIBRARIES})

            add_dependencies(hdf5_attr_${test_name} boost) 
            target_link_libraries(hdf5_attr_${test_name} boost ${HDF5_LIBRARIES})

            add_dependencies(hdf5_attr_pycplx_${test_name} boost) 
            target_link_libraries(hdf5_attr_pycplx_${test_name} boost ${HDF5_LIBRARIES})

        endif(Boost_FOUND) 

        add_alps_test(hdf5_data_${test_name})

        add_alps_test(hdf5_data_szip_${test_name})

        add_alps_test(hdf5_data_pycplx_${test_name})

        add_alps_test(hdf5_attr_${test_name})

        add_alps_test(hdf5_attr_pycplx_${test_name})

    ENDFOREACH(type)

ENDIF (ALPS_BUILD_HDF5_TESTS)
