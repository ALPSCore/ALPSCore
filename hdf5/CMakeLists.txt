#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

option(ALPS_BUILD_HDF5_TESTS "Build the hdf5 tests" OFF)
mark_as_advanced(ALPS_BUILD_HDF5_TESTS)

find_package(HDF5)

# hdf5_large - produces to large files ...
FOREACH (name hdf5_complex hdf5_copy hdf5_real_complex_vec hdf5_real_complex_matrix hdf5_bool hdf5_replace hdf5_pair hdf5_parms 
            hdf5_ising hdf5_family hdf5_valgrind hdf5_observableset hdf5_multi_array hdf5_memory hdf5_misc hdf5_vecveccplx 
            hdf5_exceptions hdf5_multiarchive hdf5_fortran_string hdf5_vecvecdbl)
    add_executable(${name} ${name}.cpp)
    add_dependencies(${name} alps)
    target_link_libraries(${name} alps)
    set_property(TARGET ${name} PROPERTY LABELS hdf5)
    add_alps_test(${name})
    set_property(TEST ${name} PROPERTY LABELS hdf5)
ENDFOREACH(name)

IF (ALPS_ENABLE_OPENMP AND OPENMP_FOUND)
    add_executable(hdf5_omp hdf5_omp.cpp)
    add_dependencies(hdf5_omp alps)
    target_link_libraries(hdf5_omp alps)
    set_property(TARGET hdf5_omp PROPERTY LABELS hdf5)
    add_alps_test(hdf5_omp)
    set_property(TEST hdf5_omp PROPERTY LABELS hdf5)
ENDIF (ALPS_ENABLE_OPENMP AND OPENMP_FOUND)

IF (ALPS_BUILD_HDF5_TESTS)

    # texting type serialization
    SET(ALPS_HDF5_DATA_TYPES
        bool int short long float double
        std::size_t std::string std::complex<float> std::complex<double> "std::complex<long double>"
        boost::int8_t boost::uint8_t boost::int16_t boost::uint16_t boost::int32_t boost::uint32_t boost::int64_t boost::uint64_t
        std::vector<bool> 
# TODO: fix it
#        "std::vector<std::vector<bool> >" "std::pair<std::vector<bool> *, std::vector<std::size_t> >" "boost::numeric::ublas::vector<bool>"
        std::vector<std::size_t> std::vector<short> std::vector<int> std::vector<long> std::vector<float> std::vector<double> "std::vector<std::complex<double> >" std::vector<std::string>
        std::valarray<int> std::valarray<double> "std::valarray<std::complex<double> >" 
        boost::numeric::ublas::vector<int> boost::numeric::ublas::vector<double> "boost::numeric::ublas::vector<std::complex<double> >"
        "std::pair<int *, std::vector<std::size_t> >" "std::pair<double *, std::vector<std::size_t> >" "std::pair<std::complex<double> *,std::vector<std::size_t> >" "std::pair<std::string *,std::vector<std::size_t> >"
        "std::vector<std::vector<int> >" "std::vector<std::vector<double> >" "std::vector<std::vector<std::complex<double> > >" "std::vector<std::vector<std::string> >"
        "std::vector<std::vector<std::vector<int> > >" "std::vector<std::vector<std::vector<double> > >" "std::vector<std::vector<std::vector<std::complex<double> > > >" "std::vector<std::vector<std::vector<std::string> > >"
        "std::vector<std::valarray<int> >" "std::valarray<std::vector<double> >" "std::vector<boost::numeric::ublas::vector<std::complex<double> > >"
        "std::pair<std::vector<int> *, std::vector<std::size_t> >" "std::pair<std::vector<double> *, std::vector<std::size_t> >" "std::pair<std::vector<std::complex<double> > *, std::vector<std::size_t> >" 
        "std::pair<std::vector<std::string> *, std::vector<std::size_t> >"
        enum_type std::vector<enum_type> "std::vector<std::vector<enum_type> >" "std::pair<enum_type *, std::vector<std::size_t> >" "std::vector<std::valarray<enum_type> >"
        "std::pair<std::vector<enum_type> *, std::vector<std::size_t> >" "std::pair<std::vector<std::vector<enum_type> > *, std::vector<std::size_t> >"
        enum_vec_type std::vector<enum_vec_type> "std::vector<std::vector<enum_vec_type> >" "std::pair<enum_vec_type *, std::vector<std::size_t> >" "std::vector<std::valarray<enum_vec_type> >"
        "std::pair<std::vector<enum_vec_type> *, std::vector<std::size_t> >" "std::pair<std::vector<std::vector<enum_vec_type> > *, std::vector<std::size_t> >"
        userdefined_class<std::size_t> userdefined_class<short> userdefined_class<int> userdefined_class<long> userdefined_class<float> userdefined_class<double> "userdefined_class<std::complex<double> >" userdefined_class<std::string>
        "std::vector<userdefined_class<double> >" "std::vector<std::vector<userdefined_class<double> > >" "std::pair<userdefined_class<double> *, std::vector<std::size_t> >" 
        "cast_type<int, long>" "cast_type<int, double>" "cast_type<double, std::string>" "cast_type<int, std::string>" "cast_type<float, double>" "cast_type<short, float>"
        "std::vector<cast_type<int, double> >" "std::vector<std::vector<cast_type<int, double> > >" "std::pair<cast_type<int, double> *, std::vector<std::size_t> >" "std::vector<std::valarray<cast_type<int, double> > >"
        "std::vector<cast_type<double, std::string> >" "std::vector<std::vector<cast_type<double, std::string> > >" "std::pair<cast_type<double, std::string> *, std::vector<std::size_t> >" 
        "boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>" "boost::numeric::ublas::matrix<std::complex<double>, boost::numeric::ublas::column_major>"
        "int *" "short *" "long *" "float *" "double *"
        "std::size_t *" "std::string *" "std::complex<double> *"
        "enum_type *" "enum_vec_type *" "userdefined_class<double> *" "cast_type<int, double> *" "cast_type<int, std::string> *"
        "boost::shared_array<int>" "boost::shared_array<short>" "boost::shared_array<long>" "boost::shared_array<float>" "boost::shared_array<double>"
        "boost::shared_array<std::size_t>" "boost::shared_array<std::string>" "boost::shared_array<std::complex<double> >" "boost::shared_array<enum_type>"
        "boost::shared_array<enum_vec_type>" "boost::shared_array<userdefined_class<double> >" "boost::shared_array<cast_type<int, double> >" "boost::shared_array<cast_type<int, std::string> >"
        "cast_type<std::vector<int>, std::valarray<int> >" "cast_type<std::vector<int>, boost::numeric::ublas::vector<int> >"
        "cast_type<std::valarray<int>, std::vector<int> >" "cast_type<std::valarray<int>, boost::numeric::ublas::vector<int> >"
        "cast_type<boost::numeric::ublas::vector<int>, std::vector<int> >" "cast_type<boost::numeric::ublas::vector<int>, std::valarray<int> >"
        "cast_type<std::vector<int>, std::valarray<double> >" "cast_type<std::vector<int>, boost::numeric::ublas::vector<double> >"
        "cast_type<std::pair<int *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<int> > > >" "cast_type<std::pair<int *, std::vector<std::size_t> >, std::vector<std::vector<std::vector<double> > > >"
        "std::pair<cast_type<std::vector<int>, std::valarray<long> > *, std::vector<std::size_t> >" "std::vector<cast_type<std::vector<int>, boost::numeric::ublas::vector<double> > >"
        "std::pair<double, int>" "std::pair<double, std::complex<double> >" "std::pair<cast_type<int, std::string>, enum_type>" "std::pair<enum_type, cast_type<int, double> >" 
        "std::pair<std::vector<cast_type<int, std::string> >, std::pair<double, int> >" "std::pair<std::pair<std::vector<enum_type> *, std::vector<std::size_t> >, enum_type>"
        "alps::numeric::matrix<unsigned int>" alps::numeric::matrix<float> alps::numeric::matrix<double> "alps::numeric::matrix<std::complex<float> >" "alps::numeric::matrix<std::complex<double> >"
        "std::vector<alps::numeric::matrix<unsigned int> >" "std::vector<alps::numeric::matrix<float> >" "std::vector<alps::numeric::matrix<double> >" 
        "std::vector<alps::numeric::matrix<std::complex<float> > >" "std::vector<alps::numeric::matrix<std::complex<double> > >"
        "alps::numeric::matrix<std::vector<double> >" "alps::numeric::matrix<std::vector<std::complex<float> > >" "alps::numeric::matrix<alps::numeric::matrix<int> >"
        "alps::numeric::matrix<alps::numeric::matrix<double> >" "alps::numeric::matrix<alps::numeric::matrix<std::complex<double> > >"
        "std::vector<std::size_t, std::allocator<std::size_t> >" "std::vector<short, std::allocator<short> >" "std::vector<int, std::allocator<int> >" "std::vector<long, std::allocator<long> >" 
        "std::vector<float, std::allocator<float> >" "std::vector<double, std::allocator<double> >" "std::vector<std::complex<double>, std::allocator<std::complex<double> > >" "std::vector<std::string, std::allocator<std::string> >"
        "std::vector<std::vector<int, std::allocator<int> > >" "std::vector<std::vector<double>, std::allocator<std::vector<double> > >"
        "std::vector<std::vector<std::complex<double>, std::allocator<std::complex<double> > >, std::allocator<std::vector<std::complex<double>, std::allocator<std::complex<double> > > > >" 
        "std::vector<std::vector<std::string, std::allocator<std::string> >, std::allocator<std::vector<std::string, std::allocator<std::string> > > >"
        "boost::multi_array<double, 1>" "boost::multi_array<int, 1>" "boost::multi_array<std::complex<double>, 1>" "boost::multi_array<std::string, 1>"
        "boost::multi_array<double, 2>" "boost::multi_array<int, 2>" "boost::multi_array<std::complex<double>, 2>" "boost::multi_array<std::string, 2>"
        "boost::multi_array<double, 3>" "boost::multi_array<int, 3>" "boost::multi_array<std::complex<double>, 3>" "boost::multi_array<std::string, 3>"
        "std::vector<boost::multi_array<double, 2> >" "std::vector<boost::multi_array<double, 3> >" "std::vector<boost::multi_array<double, 4> >" 
        "std::pair<boost::multi_array<std::complex<double>, 3> *, std::vector<std::size_t> >" "boost::multi_array<std::complex<double>, 1> *"
        "alps::multi_array<double, 1>" "alps::multi_array<int, 1>" "alps::multi_array<std::complex<double>, 1>" "alps::multi_array<std::string, 1>"
        "alps::multi_array<double, 2>" "alps::multi_array<int, 2>" "alps::multi_array<std::complex<double>, 2>" "alps::multi_array<std::string, 2>"
        "alps::multi_array<double, 3>" "alps::multi_array<int, 3>" "alps::multi_array<std::complex<double>, 3>" "alps::multi_array<std::string, 3>"
        "std::vector<alps::multi_array<double, 2> >" "std::vector<alps::multi_array<double, 3> >" "std::vector<alps::multi_array<double, 4> >" 
        "std::pair<alps::multi_array<std::complex<double>, 3> *, std::vector<std::size_t> >" "alps::multi_array<std::complex<double>, 1> *"
        "boost::array<int, 20>" "boost::array<long double, 20>" "boost::array<float, 20>" "boost::array<unsigned long long, 20>" 
        "boost::array<boost::array<std::complex<double>, 20>, 20>"
        "std::vector<boost::array<int, 4> >" "boost::array<std::vector<int>, 4>" "std::vector<boost::array<std::vector<int>, 4> >"
        "boost::tuple<int, double, float, std::complex<double> >" "std::vector<boost::tuple<char, bool, long long> >"
    )
    
    FOREACH (type ${ALPS_HDF5_DATA_TYPES})

        STRING(REGEX REPLACE "\\*" "p" test_name "_${type}")
        STRING(REGEX REPLACE "(::)|<|>| |\\," "_" test_name ${test_name})
        STRING(REGEX REPLACE "__+" "_" test_name ${test_name})

        STRING(REGEX REPLACE "_std_" "_" test_name ${test_name})
        STRING(REGEX REPLACE "_vector_" "_vec_" test_name ${test_name})
        STRING(REGEX REPLACE "_vector_" "_vec_" test_name ${test_name})
        STRING(REGEX REPLACE "_matrix_" "_mtx_" test_name ${test_name})
        STRING(REGEX REPLACE "_size_t_" "_szt_" test_name ${test_name})
        STRING(REGEX REPLACE "_enum_type_" "_enm_" test_name ${test_name})
        STRING(REGEX REPLACE "_enum_vec_type_" "_enmvec_" test_name ${test_name})
        STRING(REGEX REPLACE "_boost_numeric_ublas_" "_nub_" test_name ${test_name})
        STRING(REGEX REPLACE "_column_major_" "_clm_" test_name ${test_name})
        STRING(REGEX REPLACE "_complex_" "_cpx_" test_name ${test_name})
        STRING(REGEX REPLACE "_cast_type_" "_cst_" test_name ${test_name})
        STRING(REGEX REPLACE "_valarray_" "_valarr_" test_name ${test_name})
        STRING(REGEX REPLACE "_double_" "_dbl_" test_name ${test_name})
        STRING(REGEX REPLACE "_float_" "_flt_" test_name ${test_name})
        STRING(REGEX REPLACE "_long_" "_lng_" test_name ${test_name})
        STRING(REGEX REPLACE "_string_" "_str_" test_name ${test_name})
        STRING(REGEX REPLACE "_boost_shared_array_" "_shrdarr_" test_name ${test_name})
        STRING(REGEX REPLACE "_boost_int" "_bint" test_name ${test_name})
        STRING(REGEX REPLACE "_boost_uint" "_buint" test_name ${test_name})
        STRING(REGEX REPLACE "_boost_array" "_barr" test_name ${test_name})
        STRING(REGEX REPLACE "_userdefined_class_" "_usrcls_" test_name ${test_name})
        STRING(REGEX REPLACE "_pair_" "_pr_" test_name ${test_name})
        STRING(REGEX REPLACE "_multi_array_" "_mularr_" test_name ${test_name})
        STRING(REGEX REPLACE "_aosot_" "_bst_" test_name ${test_name})

        set(HDF5_TEST_TYPE ${type})
        include_directories(.)

        configure_file(type_check.cpp.in ${PROJECT_BINARY_DIR}/test/hdf5/h5d${test_name}.cpp)
        add_executable(h5d${test_name} ${PROJECT_BINARY_DIR}/test/hdf5/h5d${test_name}.cpp)
        
        if (SZIP_FOUND)
            configure_file(type_check.cpp.in ${PROJECT_BINARY_DIR}/test/hdf5/h5dz${test_name}.cpp)
            add_executable(h5dz${test_name} ${PROJECT_BINARY_DIR}/test/hdf5/h5dz${test_name}.cpp)
            SET_TARGET_PROPERTIES(h5dz${test_name} PROPERTIES COMPILE_FLAGS "-DSZIP_COMPRESS=true")
        endif (SZIP_FOUND)

        configure_file(type_check.cpp.in ${PROJECT_BINARY_DIR}/test/hdf5/h5a${test_name}.cpp)
        add_executable(h5a${test_name} ${PROJECT_BINARY_DIR}/test/hdf5/h5a${test_name}.cpp)
        SET_TARGET_PROPERTIES(h5a${test_name} PROPERTIES COMPILE_FLAGS "-DIS_ATTRIBUTE=true")

        add_dependencies(h5d${test_name} alps)
        target_link_libraries(h5d${test_name} alps)

        if (SZIP_FOUND)
            add_dependencies(h5dz${test_name} alps)
            target_link_libraries(h5dz${test_name} alps)
        endif (SZIP_FOUND)

        add_dependencies(h5a${test_name} alps)
        target_link_libraries(h5a${test_name} alps)

        add_alps_test(h5d${test_name})

        if (SZIP_FOUND)
            add_alps_test(h5dz${test_name})
        endif (SZIP_FOUND)

        add_alps_test(h5a${test_name})

    ENDFOREACH(type)

ENDIF (ALPS_BUILD_HDF5_TESTS)
