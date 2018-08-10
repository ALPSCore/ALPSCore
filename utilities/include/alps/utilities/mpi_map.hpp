/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file mpi_map.hpp
    
    @brief Header for object-oriented interface to MPI for std::map
*/

#ifndef ALPS_UTILITIES_MPI_MAP_HPP_INCLUDED_b6555a13ab3b46c2a2b4a434a3484099
#define ALPS_UTILITIES_MPI_MAP_HPP_INCLUDED_b6555a13ab3b46c2a2b4a434a3484099

#include <alps/utilities/mpi.hpp>
#include <map>

namespace alps {
    namespace mpi {

        /// MPI_BCast of an std::map
        /** @todo FIXME: does a series of broadcasts */
        // FIXME: what is exception safety status?
        // FIXME!: make a test
        template <typename K, typename V>
        inline void broadcast(const communicator& comm, std::map<K,V>& a_map, int root)
        {
            typedef std::map<K,V> map_type;
            typedef typename map_type::value_type value_type;

            std::size_t root_sz=a_map.size();
            broadcast(comm, root_sz, root);
              
            if (comm.rank()==root) {
                for(value_type& pair: a_map) {
                    broadcast(comm, const_cast<K&>(pair.first), root);
                    broadcast(comm, pair.second, root);
                }
            } else {
                map_type new_map;
                while (root_sz--) {
                    std::pair<K,V> pair; // FIXME! this requires default ctor
                    broadcast(comm, pair.first, root);
                    broadcast(comm, pair.second, root);
                    new_map.insert(pair);
                }
                using std::swap;
                swap(a_map, new_map);
            }
        }

    } // mpi::
} // alps::

#endif /* ALPS_UTILITIES_MPI_MAP_HPP_INCLUDED_b6555a13ab3b46c2a2b4a434a3484099 */
