#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

option(ALPS_BUILD_GRAPH_TESTS "Build the graph tests" OFF)
mark_as_advanced(ALPS_BUILD_GRAPH_TESTS)

IF (ALPS_BUILD_GRAPH_TESTS)

    FOREACH (name
            iso_simple
            canonical_label_test
            canonical_label_with_color_symmetries_test
            is_embeddable_with_color_symmetries_test
            subgraph_generator_test
            subgraph_generator_test_colored_edges
            subgraph_generator_test_colored_edges2
            subgraph_generator_test_colored_edges_with_sym
            lattice_constant_test
            lattice_constant_matrix
            colored_lattice_constant_test
            colored_lattice_constant_test2
    )
      add_executable(${name} ${name}.cpp)
      add_dependencies(${name} alps)
      target_link_libraries(${name} alps)
      add_alps_test(${name})
      set_property(TEST ${name} PROPERTY LABELS graph)
    ENDFOREACH(name)
ENDIF(ALPS_BUILD_GRAPH_TESTS)
