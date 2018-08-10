/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file mpi_vector.hpp
    
    @brief Header for object-oriented interface to MPI for vector data structures
*/

#ifndef ALPS_UTILITIES_MPI_VECTOR_HPP_INCLUDED_ba202cf9fb0040e493596702a707ca9b
#define ALPS_UTILITIES_MPI_VECTOR_HPP_INCLUDED_ba202cf9fb0040e493596702a707ca9b

#include <vector>
#include <alps/utilities/mpi.hpp>
#include <string>

#include <boost/container/vector.hpp>

namespace alps {
    namespace mpi {

        /// MPI_BCast of a vector of strings
        // FIXME!: make a test
        // FIXME: what is exception safety status?
        // FIXME: implement generically for non-contiguous types
        // FIXME: inline to have it header-only. A tad too complex to be inlined?
        inline void broadcast(const communicator& comm, std::vector<std::string>& vec, int root)
        {
            // FIXME? it might be better to trade traffic to memory and first combine elements in a vector
            using alps::mpi::broadcast;
            std::size_t root_sz=vec.size();
            broadcast(comm, root_sz, root);
            if (comm.rank() != root) {
                vec.resize(root_sz);
            }
            for(std::string& elem: vec) {
                broadcast(comm, elem, root);
            }
        }


        /// MPI_BCast of a vector of (primitive) type T
        // FIXME!: make a test
        // FIXME: what is exception safety status?
        // FIXME: verify that it is a "primitive" (or, at least, contiguous) type!
        template <typename T>
        inline void broadcast(const communicator& comm, std::vector<T>& vec, int root)
        {
            std::size_t root_sz=vec.size();
            broadcast(comm, root_sz, root);
            if (comm.rank() != root) {
                vec.resize(root_sz);
            }
            broadcast(comm, &vec[0], vec.size(), root);
        }
      
        /// MPI_BCast of a vector of bool
        // FIXME!: make a test
        // FIXME: what is exception safety status?
        // FIXME: implement generically for non-contiguous types
        inline void broadcast(const communicator& comm, std::vector<bool>& vec, int root)
        {
            // FIXME? it might be better to trade traffic to memory and first combine elements in a vector
            using alps::mpi::broadcast;
            typedef std::vector<bool> vector_type;
            typedef vector_type::iterator iter_type;
            
            std::size_t root_sz=vec.size();
            broadcast(comm, root_sz, root);
            if (comm.rank() != root) {
                vec.resize(root_sz);
            }

            for (iter_type it=vec.begin(); it!=vec.end(); ++it) {
                bool elem=*it;
                broadcast(comm, elem, root);
                *it=elem;
            }
        }
      

    } // mpi::
} // alps::


#endif /* ALPS_UTILITIES_MPI_VECTOR_HPP_INCLUDED_ba202cf9fb0040e493596702a707ca9b */
