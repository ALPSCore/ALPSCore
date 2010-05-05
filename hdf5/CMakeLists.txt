#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

FOREACH (name alea alea2 io)
  add_executable(${name} ${name}.C)
  add_dependencies(${name} alps)
  target_link_libraries(${name} alps)
  add_alps_test(${name})
ENDFOREACH(name)

#  h5multiarray h5groups h5alea h5overwrite h5boostarray h5carray
FOREACH (name h5vector h5ptr h5valarray h5ublasvector h5scalar h5attribute h5discontinous h5class h5pair h5enum)
    add_executable(${name} ${name}.cpp)
    if (Boost_FOUND)
      target_link_libraries(${name} ${Boost_LIBRARIES})
    else(Boost_FOUND)
      add_dependencies(${name} boost)
      target_link_libraries(${name} boost)
    endif(Boost_FOUND)
    add_alps_test(${name})
ENDFOREACH(name)
