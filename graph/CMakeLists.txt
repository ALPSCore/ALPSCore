#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

# FOREACH (name non_iso_graph_generation_colored_edges embedding_test canonical_label_test)
FOREACH (name iso_simple canonical_label_test)
  add_executable(${name} ${name}.cpp)
  add_dependencies(${name} alps)
  target_link_libraries(${name} alps)
  add_alps_test(${name})
ENDFOREACH(name)
