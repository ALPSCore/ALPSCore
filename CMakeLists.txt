#  Copyright Matthias Troyer and Synge Todo 2009 - 2010.
#  Distributed under the Boost Software License, Version 1.0.
#      (See accompanying file LICENSE_1_0.txt or copy at
#          http://www.boost.org/LICENSE_1_0.txt)

add_subdirectory(alea)
add_subdirectory(fixed_capacity)
add_subdirectory(lattice)
add_subdirectory(hdf5)
add_subdirectory(model)
if (ALPS_BUILD_NGS)
    add_subdirectory(ngs)
endif(ALPS_BUILD_NGS)
add_subdirectory(osiris)
add_subdirectory(parameter)
add_subdirectory(parapack)
add_subdirectory(parser)
add_subdirectory(numeric)
add_subdirectory(pyalps)
