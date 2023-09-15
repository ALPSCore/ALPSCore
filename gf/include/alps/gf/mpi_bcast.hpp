/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file mpi_bcast.hpp
    @brief Functions to MPI-broadcast internal data structures
 */

#ifndef ALPS_GF_MPI_BCAST_HPP_c030bec39d4b43b9a24a16b5805f542d
#define ALPS_GF_MPI_BCAST_HPP_c030bec39d4b43b9a24a16b5805f542d

#include <alps/utilities/mpi.hpp>
#include <iostream>
namespace alps {
    namespace gf {
        namespace detail {
            /// Broadcast a vector
            /** @note Non-default allocator is silently unsupported. */ 
            template <typename T>
            void broadcast(const alps::mpi::communicator& comm, std::vector<T>& data, int root) {
                typedef std::vector<T> data_type;
                typedef typename data_type::size_type size_type;
                size_type root_sz=data.size();
                alps::mpi::broadcast(comm, root_sz, root);
                data.resize(root_sz);
                alps::mpi::broadcast(comm, &data[0], root_sz, root);
            }
            
            /// Broadcast a multi-array.
            /**
               @note Non-default allocator is *silently* unsupported!

               @note Only a particular (namely, C) storage order is supported.

               @note Any detected mismatch results in MPI_Abort()
             */
            template <typename T, size_t N>
            void broadcast(const alps::mpi::communicator& comm, alps::numerics::tensor<T,N>& data, int root)
            {
                typedef alps::numerics::tensor<T,N> data_type;
//                typedef typename data_type::index index_type;
                
                int rank=comm.rank();
                const bool is_root=(rank==root);

                try {
                    // Compare dimensions with root. Normally should not be needed,
                    // and incurs extra communication cost ==> enabled only in debug mode.
#ifndef         BOOST_DISABLE_ASSERTS
                    {
                        size_t ndim=N;
                        alps::mpi::broadcast(comm, ndim, root);
                        if (ndim!=N) {
                            throw std::logic_error("Different multi_array dimensions in broadcast:\n"
                                                   "root (rank #"+
                                                   std::to_string(root)+
                                                   ") expects N="+
                                                   std::to_string(ndim)+
                                                   ", rank #"+
                                                   std::to_string(rank)+
                                                   " has N="+std::to_string(N));
                        }
                    }
#endif
                } catch (const std::exception& exc) {
                    // Abort here and now -- otherwise we end up with unmatched broadcast.
                    std::cerr << "Exception in broadcast of boost::multi_array data on rank " << rank << std::endl
                              << exc.what() 
                              << "\nAborting." << std::endl;
                    MPI_Abort(MPI_COMM_WORLD,1);

                    // FIXME: In debug mode we could use MPI_Allreduce() and notify other processes
                    //        of the error. Has to be done carefully to avoid unmatched broadcasts.
                }
                // Broadcast the array shape
                std::array<size_t,N> shape;
                if (is_root) {
                  std::array<size_t, N> root_shape=data.shape();
                  std::copy(root_shape.begin(), root_shape.end(), shape.begin()); // FIXME: this copy is not needed if done carefully
                }
                alps::mpi::broadcast(comm, shape.data(), N, root);

                if (! is_root) {
                    data.reshape(shape);
                }
                
                size_t nelements=data.size();

                // This is an additional broadcast, but we need to ensure MPI broadcast correctness,
                // otherwise it would be a hell to debug down the road figuring out what exactly went wrong.
                unsigned long nelements_root=nelements;
                alps::mpi::broadcast(comm, nelements_root, root);
                if (nelements_root!=nelements) {
                    // Should never happen, but if it has, 
                    // our best course is to abort here and now.
                    std::cerr << "Broadcast of incompatible boost::multi_array data detected on rank " << rank
                              << ".\nRoot sends " << nelements_root << " elements,"
                              << " this process expects " << nelements << " elements."
                              << "\nAborting..."
                              << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                alps::mpi::broadcast(comm, data.data(), nelements, root);
            }

        } // detail::
    } // gf::
} // alps::

#endif /* ALPS_GF_MPI_BCAST_HPP_c030bec39d4b43b9a24a16b5805f542d */
