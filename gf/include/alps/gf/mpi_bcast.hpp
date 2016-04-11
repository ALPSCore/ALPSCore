/** @file mpi_bcast.hpp
    @brief Functions to MPI-broadcast internal data structures
 */

#ifndef ALPS_GF_MPI_BCAST_HPP_c030bec39d4b43b9a24a16b5805f542d
#define ALPS_GF_MPI_BCAST_HPP_c030bec39d4b43b9a24a16b5805f542d

// FIXME: will eventually go to ALPSCore
#include "mpi_wrappers.hpp"
#include <iostream>
namespace alps {
    namespace gf {
        namespace detail {
            /// Broadcast a vector
            /** @note Non-default allocator is silently unsupported. */ 
            template <typename T>
            void bcast(std::vector<T>& data, int root, MPI_Comm comm) {
                typedef std::vector<T> data_type;
                typedef typename data_type::size_type size_type;
                size_type root_sz=data.size();
                alps::mpi::bcast(root_sz, root, comm);
                data.resize(root_sz);
                alps::mpi::bcast(&data[0], root_sz, root, comm);
            }
            
            /// Broadcast a multi-array.
            /**
               @note Non-default allocator is *silently* unsupported!

               @note Only a particular (namely, C) storage order is supported.

               @note Any detected mismatch results in MPI_Abort()
             */
            template <typename T, size_t N>
            void bcast(boost::multi_array<T,N>& data, int root, MPI_Comm comm)
            {
                typedef boost::multi_array<T,N> data_type;
                typedef typename data_type::index index_type;
                typedef typename data_type::size_type size_type;
                
                int rank;
                MPI_Comm_rank(comm,&rank);
                const bool is_root=(rank==root);

                try {
                    // NOTE: questionable; boost::multi_array does not document comparison of storage orders
                    if (! (data.storage_order()==boost::c_storage_order()) )
                        throw std::logic_error("Unsupported storage order in multi_array broadcast at rank #"+
                                               boost::lexical_cast<std::string>(rank));
                
                    // Compare dimensions with root. Normally should not be needed,
                    // and incurs extra communication cost ==> enabled only in debug mode.
#ifndef         BOOST_DISABLE_ASSERTS
                    {
                        size_type ndim=N;
                        MPI_Bcast(&ndim, 1, alps::mpi::mpi_type<size_type>(), root, comm);
                        if (ndim!=N) {
                            throw std::logic_error("Different multi_array dimensions in broadcast:\n"
                                                   "root (rank #"+
                                                   boost::lexical_cast<std::string>(root)+
                                                   ") expects N="+
                                                   boost::lexical_cast<std::string>(ndim)+
                                                   ", rank #"+
                                                   boost::lexical_cast<std::string>(rank)+
                                                   " has N="+boost::lexical_cast<std::string>(N));
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

                // Broadcast the bases
                boost::array<index_type,N> bases;
                const index_type* root_bases=data.index_bases();
                if (is_root) {
                    std::copy(root_bases, root_bases+N, bases.begin()); // FIXME: this copy is not needed if done carefully
                }
                MPI_Bcast(bases.data(), N, alps::mpi::mpi_type<index_type>(), root, comm);

                // Broadcast the array shape
                boost::array<size_type,N> shape;
                if (is_root) {
                    const size_type* root_shape=data.shape();
                    std::copy(root_shape, root_shape+N, shape.begin()); // FIXME: this copy is not needed if done carefully
                }
                MPI_Bcast(shape.data(), N, alps::mpi::mpi_type<size_type>(), root, comm);

                if (! is_root) {
                    data.resize(shape);
                    data.reindex(bases);
                }
                
                size_t nbytes=data.num_elements()*sizeof(T);

                // This is an additional broadcast, but we need to ensure MPI broadcast correctness,
                // otherwise it would be a hell to debug down the road figuring out what exactly went wrong.
                unsigned long nbytes_root=nbytes;
                MPI_Bcast(&nbytes_root, 1, MPI_UNSIGNED_LONG, root, comm);
                if (nbytes_root!=nbytes) {
                    // Should never happen, but if it has, 
                    // our best course is to abort here and now.
                    std::cerr << "Broadcast of incompatible boost::multi_array data detected on rank " << rank
                              << ".\nRoot sends " << nbytes_root << " bytes,"
                              << " this process expects " << nbytes << " bytes."
                              << "\nAborting..."
                              << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                MPI_Bcast(&data(bases), nbytes, MPI_BYTE, root, comm);
            }

        } // detail::
    } // gf::
} // alps::

#endif /* ALPS_GF_MPI_BCAST_HPP_c030bec39d4b43b9a24a16b5805f542d */
