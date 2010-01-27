#  Copyright Matthias Troyer 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/tool)
include_directories(${Boost_ROOT_DIR})

add_subdirectory(alea)
add_subdirectory(fixed_capacity)
add_subdirectory(lattice)
add_subdirectory(hdf5)
add_subdirectory(model)
add_subdirectory(osiris)
add_subdirectory(parameter)
add_subdirectory(parapack)
add_subdirectory(parser)
