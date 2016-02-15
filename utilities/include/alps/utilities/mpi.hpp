/** @file mpi.hpp
    
    @brief Header for object-oriented interface to MPI (similar to boost::mpi)
    
    @details
    The interface provided by this file is intended to be (almost)
    drop-in replacemnt for the subset of boost::mpi functionality used
    by ALPSCore.
*/

#ifndef ALPS_UTILITIES_MPI_HPP_INCLUDED_90206380262d48f0bcbe98fd16edd65d
#define ALPS_UTILITIES_MPI_HPP_INCLUDED_90206380262d48f0bcbe98fd16edd65d

#include <mpi.h>

#include <vector>

// DEBUG:
#include <stdexcept>
// DEBUG:
#include <typeinfo>


namespace alps {
    namespace mpi {

        class communicator {
            MPI_Comm comm_;
            
            public:

            communicator() : comm_(MPI_COMM_WORLD) {}

            /// Returns process rank in this communicator
            int rank() const {
                throw std::logic_error("rank() not implemented");
            }

            /// Returns the number of processes in this communicator
            int size() const {
                throw std::logic_error("size() not implemented");
            }

            /// Barrier on this communicator
            void barrier() const {
                MPI_Barrier(comm_);
            }

            /// Converts this communicator object to MPI communicator
            operator MPI_Comm() const {
                return comm_;
            }
        };

        class environment {
            public:
            environment(int& argc, char**& argv, bool =false) {
                // FIXME: verify MPI initializaton
                MPI_Init(&argc, &argv);
            }

            ~environment(){
                // FIXME: verify MPI finalization
                // FIXME: attempt to see if we are inside an exception to call MPI_Abort()?
                MPI_Finalize();
            }
                
        };

        /// Class-holder for reduction operations for type T
        template <typename T>
        class maximum {
            public:
            maximum() { }
        };

        
        /// Broadcasts value `val` on communicator `comm` with root `root`
        template <typename T>
        void broadcast(const communicator& comm, T& val, int root) {
            throw std::logic_error(std::string("broadcast() is not implemented, called for type T=")
                                   +typeid(T).name());
        }

        /// Returns MPI datatype for the value of type `T`
        template <typename T>
        MPI_Datatype get_mpi_datatype(const T& val) {
            throw std::logic_error(std::string("get_mpi_datatype() is not implemented, called for type T=")
                                   +typeid(T).name());
        }

        /// performs MPI_Allreduce() for type T using operation of type OP
        template <typename T, typename OP>
        T all_reduce(const communicator& comm, const T& val, const OP& op) {
            throw std::logic_error(std::string("T all_reduce(const T&, OP) is not implemented, called for type T=")
                                   +typeid(T).name() + "and OP="+typeid(OP).name() );
        }

        /// performs MPI_Allgather() for type T
        template <typename T>
        void all_gather(const communicator& comm, const T& in_val, std::vector<T>& out_vals) {
            throw std::logic_error(std::string("all_gather() is not implemented, called for type T=")
                                   +typeid(T).name());
        }

        /// Trait for MPI reduction operations
        template <typename OP, typename T>
        class is_mpi_op {
            public:
            static MPI_Op op() {
                throw std::logic_error(std::string("is_mpi_op() is not implemented, called for types OP=")
                                       +typeid(OP).name() + " and T="+typeid(T).name());
            }
        };
        
        
    } // mpi::
} // alps::

#endif /* ALPS_UTILITIES_MPI_HPP_INCLUDED_90206380262d48f0bcbe98fd16edd65d */
