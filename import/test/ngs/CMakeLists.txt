#  Copyright Matthias Troyer 2009 - 2010
#            Lukas Gamper 2012
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)


include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

FOREACH (name ngs_hdf5 ngs_hash)
    add_executable(${name} ${name}.cpp)
    add_dependencies(${name} alps)
    target_link_libraries(${name} alps)
    add_alps_test(${name})
ENDFOREACH(name)

#TODO: does not compile in clang
add_subdirectory(alea)
add_subdirectory(params)
add_subdirectory(scheduler)
