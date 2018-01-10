/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file mpi_variant.hpp
    
    @brief Header for object-oriented interface to MPI for boost::variant

    @todo FIXME: Move to utilities
*/

#ifndef ALPS_PARAMS_MPI_VARIANT_HPP_69d56023ac6048ca987f4be0f67122af
#define ALPS_PARAMS_MPI_VARIANT_HPP_69d56023ac6048ca987f4be0f67122af

#include <alps/utilities/mpi_vector.hpp>
#include <alps/utilities/mpi_pair.hpp>

#include <alps/params/serialize_variant.hpp>

#include <cassert>

namespace alps {
    namespace mpi {

        inline void broadcast(const alps::mpi::communicator&, alps::params_ns::detail::None&, int) {
            //std::cout << "DEBUG: Broadcasting None is no-op" << std::endl;
        }
        
        namespace detail {
            /// Consumer class to send-broadcast an object via MPI
            struct broadcast_sender {
                const alps::mpi::communicator& comm_;
                int root_;

                broadcast_sender(const alps::mpi::communicator& comm, int root) : comm_(comm), root_(root) {}

                template <typename T>
                void operator()(const T& val) {
                    using alps::mpi::broadcast;
                    assert(comm_.rank()==root_ && "Should be only called by broadcast root");
                    broadcast(comm_, const_cast<T&>(val), root_);
                }
            };

            /// Producer class to receive an object broadcast via MPI
            struct broadcast_receiver {
                int target_which;
                int which_count;
                const alps::mpi::communicator& comm_;
                int root_;

                broadcast_receiver(int which, const alps::mpi::communicator& comm, int root)
                    : target_which(which), which_count(0), comm_(comm), root_(root)
                {}

                template <typename T>
                boost::optional<T> operator()(const T*)
                {
                    using alps::mpi::broadcast;
                    assert(comm_.rank()!=root_ && "Should NOT be called by broadcast root");
                    boost::optional<T> ret;
                    if (target_which==which_count) {
                        T val;
                        broadcast(comm_, val, root_);
                        ret=val;
                    }
                    ++which_count;
                    return ret;
                }
            };

            typedef alps::detail::variant_serializer<alps::params_ns::detail::dict_all_types,
                                                     broadcast_sender, broadcast_receiver> var_serializer;
            typedef var_serializer::variant_type variant_type;
            
        } // detail::
        
        /// MPI_BCast of an boost::variant over MPL type sequence MPLSEQ
        template <typename MPLSEQ>
        inline void broadcast(const communicator& comm, typename boost::make_variant_over<MPLSEQ>::type& var, int root)
        {
            using alps::mpi::broadcast;

            int which=var.which();
            broadcast(comm, which, root);
            
            if (comm.rank()==root) {
                detail::broadcast_sender consumer(comm, root);
                detail::var_serializer::consume(consumer, var);
            } else {
                detail::broadcast_receiver producer(which, comm, root);
                var=detail::var_serializer::produce(producer);
            }
        }

    } // mpi::
} // alps::

#endif /* ALPS_PARAMS_MPI_VARIANT_HPP_69d56023ac6048ca987f4be0f67122af */
