#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

include_directories(${PROJECT_BINARY_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${Boost_ROOT_DIR})

IF(NOT ALPS_LLVM_WORKAROUND)
  FOREACH (name clone_info clone_phase clone_timer exmc_optimize exp_number footprint info_test integer_range linear_regression merge percentage temperature_scan time version wl_weight)
    add_executable(${name} ${name}.C)
    add_dependencies(${name} alps)
    target_link_libraries(${name} alps)
    add_alps_test(${name})
    set_property(TEST ${name} PROPERTY LABELS parapack)
  ENDFOREACH(name)

  IF(ALPS_HAVE_MPI)
    FOREACH (name collect_mpi comm_mpi filelock_mpi halt_mpi info_test_mpi process_mpi)
      add_executable(${name} ${name}.C)
      add_dependencies(${name} alps)
      target_link_libraries(${name} alps)
      # add_alps_test(${name})
      # set_property(TEST ${name} PROPERTY LABELS parapack)
    ENDFOREACH(name)
  ENDIF(ALPS_HAVE_MPI)
ENDIF(NOT ALPS_LLVM_WORKAROUND)
