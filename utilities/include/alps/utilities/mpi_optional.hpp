/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file mpi_optional.hpp
    
    @brief Header for object-oriented interface to MPI for boost::optional
*/

#ifndef ALPS_UTILITIES_MPI_OPTIONAL_HPP_INCLUDED_765ecb98a17c4ace9754b3d3b1c1f567
#define ALPS_UTILITIES_MPI_OPTIONAL_HPP_INCLUDED_765ecb98a17c4ace9754b3d3b1c1f567

#include <alps/utilities/mpi.hpp>
#include <boost/optional.hpp>

namespace alps {
    namespace mpi {

        /// MPI_BCast of a boost::optional
        // FIXME: what is exception safety status?
        template <typename T>
        inline void broadcast(const communicator& comm, boost::optional<T>& val, int root)
        {
            bool is_valid=!!val;
            bool is_root=(comm.rank()==root);
            broadcast(comm, is_valid, root);
            if (!is_valid) {
                if (!is_root) {
                    val=boost::none;
                }
                return;
            }

            if (!is_root) {
                val=T();
            }
            broadcast(comm, *val, root);
        }

    } // mpi::
} // alps::
#endif /* ALPS_UTILITIES_MPI_OPTIONAL_HPP_INCLUDED_765ecb98a17c4ace9754b3d3b1c1f567 */
