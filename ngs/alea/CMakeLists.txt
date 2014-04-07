#  Copyright Matthias Troyer 2009 - 2010
#            Lukas Gamper 2012
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

IF(ALPS_NGS_USE_NEW_ALEA)
    FOREACH (name ngs_alea_next ngs_alea_compare)
        add_executable(${name} ${name}.cpp)
        add_dependencies(${name} alps)
        target_link_libraries(${name} alps)
        add_alps_test(${name})
    ENDFOREACH(name)
ENDIF(ALPS_NGS_USE_NEW_ALEA)

IF(FALSE AND ALPS_NGS_USE_NEW_ALEA)
    FOREACH (name   ngs_alea_count_test_compile
                    ngs_alea_count_test_runtime
                    ngs_alea_ctor_test_compile
                    ngs_alea_ctor_test_runtime
                    ngs_alea_error_test_compile
                    ngs_alea_error_test_runtime
                    ngs_alea_fix_size_test_compile
                    ngs_alea_fix_size_test_runtime
                    ngs_alea_max_num_test_compile
                    ngs_alea_max_num_test_runtime
                    ngs_alea_log_test_compile
                    ngs_alea_log_test_runtime
                    ngs_alea_mean_test_runtime
                    ngs_alea_mean_test_compile
                    ngs_alea_stream_test_compile
                    ngs_alea_stream_test_runtime
                    ngs_alea_value_type_test
                    ngs_alea_weight_type_test
                    ngs_alea_wrapper_test_compile
                    ngs_alea_wrapper_test_runtime
    )
        add_executable(${name} ${name}.cpp)
        add_dependencies(${name} alps)
        target_link_libraries(${name} alps)
        add_alps_test(${name})
    ENDFOREACH(name)
ENDIF(FALSE AND ALPS_NGS_USE_NEW_ALEA)
