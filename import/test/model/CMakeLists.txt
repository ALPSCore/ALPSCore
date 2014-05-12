#  Copyright Matthias Troyer 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

IF(NOT ALPS_LLVM_WORKAROUND)
  FOREACH (name example1 example2 example3 example4 example5 example6 example7 example8 example9 example10 example11 example12 example13 example14 example15 example16 example17 example18)
    add_executable(model_${name} ${name}.C)
    add_dependencies(model_${name} alps)
    target_link_libraries(model_${name} alps)
    set_property(TARGET model_${name} PROPERTY LABELS model)
    add_alps_test(model_${name} ${name})
    set_property(TEST ${name} PROPERTY LABELS model)
  ENDFOREACH(name)
ENDIF(NOT ALPS_LLVM_WORKAROUND)
