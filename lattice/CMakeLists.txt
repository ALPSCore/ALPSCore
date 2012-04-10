#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

IF(NOT ALPS_LLVM_WORKAROUND)
  FOREACH (name example1 example2 example3 example4 example5 example6 example7 example8 example9 example10 example11 label)
    add_executable(lattice_${name} ${name}.C)
    add_dependencies(lattice_${name} alps)
    target_link_libraries(lattice_${name} alps)
    set_property(TARGET lattice_${name} PROPERTY LABELS lattice)
    add_alps_test(lattice_${name} ${name})	
    # set_property(TEST lattice_${name} PROPERTY LABELS lattice)
    set_property(TEST ${name} PROPERTY LABELS lattice)
  ENDFOREACH(name)
  FOREACH (name coloring parity)
    add_executable(${name} ${name}.C)
    add_dependencies(${name} alps)
    target_link_libraries(${name} alps)
    set_property(TARGET ${name} PROPERTY LABELS lattice)
    add_alps_test(${name})
    set_property(TEST ${name} PROPERTY LABELS lattice)
  ENDFOREACH(name)
ENDIF(NOT ALPS_LLVM_WORKAROUND)
