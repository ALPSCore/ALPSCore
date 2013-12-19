#  Copyright Matthias Troyer 2009 - 2010
#            Lukas Gamper 2012
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Boost_ROOT_DIR})

if (MPI_FOUND)

    add_executable(scheduler_sum_mpi sum_mpi.cpp)
    add_dependencies(scheduler_sum_mpi alps)
    target_link_libraries(scheduler_sum_mpi alps)

endif (MPI_FOUND)

add_executable(scheduler_sum_single sum_single.cpp)
add_dependencies(scheduler_sum_single alps)
target_link_libraries(scheduler_sum_single alps)
