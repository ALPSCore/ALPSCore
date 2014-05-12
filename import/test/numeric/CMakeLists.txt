#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

if(LAPACK_FOUND)
  include_directories(${PROJECT_BINARY_DIR}/src)
  include_directories(${PROJECT_SOURCE_DIR}/src)
  include_directories(${Boost_ROOT_DIR})
  add_definitions(-DALPS_DISABLE_MATRIX_ELEMENT_ITERATOR_WARNING)
  FOREACH (name matrix_unit_tests matrix_column_view matrix_transpose_view matrix_hdf5 matrix_deprecated_hdf5_format_test matrix_ublas_sparse_functions accumulate_if vector_functions vector_valarray_conversion matrix_algorithms real_tests)
    add_executable(${name} ${name}.C)
    add_dependencies(${name} alps)
    target_link_libraries(${name} alps ${BLAS_LIBRARY} ${LAPACK_LIBRARY})
    set_property(TARGET ${name} PROPERTY LABELS numeric)
    add_alps_test(${name})
    set_property(TEST ${name} PROPERTY LABELS numeric)
  ENDFOREACH(name)
endif(LAPACK_FOUND)

