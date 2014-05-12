#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

configure_file(xdrdump2.input.cmake ${CMAKE_CURRENT_BINARY_DIR}/boostdump2.input)
configure_file(xdrdump2.input.cmake ${CMAKE_CURRENT_BINARY_DIR}/xdrdump2.input)

FOREACH (name os sizeof xdrdump xdrdump2 boostdump boostdump2 boostdump3 boostdump4)
  add_executable(${name} ${name}.C)
  add_dependencies(${name} alps)
  target_link_libraries(${name} alps)
  set_property(TARGET ${name} PROPERTY LABELS osiris)
  add_alps_test(${name})
  set_property(TEST ${name} PROPERTY LABELS osiris)
ENDFOREACH(name)
