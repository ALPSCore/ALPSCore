/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file mpi_vector.hpp
    
    @brief Header for object-oriented interface to MPI for std::pair
*/

#ifndef ALPS_UTILITIES_MPI_PAIR_HPP_3acdc0894fa949d59f697673f81e1995
#define ALPS_UTILITIES_MPI_PAIR_HPP_3acdc0894fa949d59f697673f81e1995

#include <utility> // for std::pair
#include <alps/utilities/mpi.hpp>
#include <string>

#include <boost/foreach.hpp>
#include <boost/container/vector.hpp>

namespace alps {
    namespace mpi {

        /// MPI_BCast of a pair
        // FIXME!: make a test
        // FIXME: what is exception safety status?
        template <typename T>
        inline void broadcast(const alps::mpi::communicator& comm, std::pair<std::string, T>& val, int root) {
            using alps::mpi::broadcast;
            broadcast(comm, val.first, root);
            broadcast(comm, val.second, root);
        }

    } // mpi::
} // alps::


#endif /* ALPS_UTILITIES_MPI_PAIR_HPP_3acdc0894fa949d59f697673f81e1995 */
