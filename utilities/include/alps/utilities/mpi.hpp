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

        namespace detail {
        /// Translate C++ primitive type into corresponding MPI type
        template <typename T> class mpi_type {};

#define ALPS_MPI_DETAIL_MAKETYPE(_mpitype_, _cxxtype_)          \
        template <>                                             \
        class mpi_type<_cxxtype_> {                             \
          public:                                               \
            operator MPI_Datatype() { return _mpitype_; }       \
        }

        ALPS_MPI_DETAIL_MAKETYPE(MPI_CHAR,char);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_SHORT,signed short int);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_INT,signed int);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_LONG,signed long int);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_LONG_LONG_INT,signed long long int);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_LONG_LONG,signed long long int);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_SIGNED_CHAR,signed char);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_UNSIGNED_CHAR,unsigned char);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_UNSIGNED_SHORT,unsigned short int);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_UNSIGNED,unsigned int);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_UNSIGNED_LONG,unsigned long int);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_UNSIGNED_LONG_LONG,unsigned long long int);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_FLOAT,float);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_DOUBLE,double);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_LONG_DOUBLE,long double);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_WCHAR,wchar_t);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_C_BOOL,_Bool);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_INT8_T,int8_t);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_INT16_T,int16_t);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_INT32_T,int32_t);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_INT64_T,int64_t);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_UINT8_T,uint8_t);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_UINT16_T,uint16_t);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_UINT32_T,uint32_t);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_UINT64_T,uint64_t);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_C_COMPLEX,float _Complex);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_C_FLOAT_COMPLEX,float _Complex);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_C_DOUBLE_COMPLEX,double _Complex);
        // ALPS_MPI_DETAIL_MAKETYPE(MPI_C_LONG_DOUBLE_COMPLEX,long double _Complex);
#undef ALPS_MPI_DETAIL_MAKETYPE
        } // detail::




        class communicator {
            MPI_Comm comm_;
            
            public:

            communicator() : comm_(MPI_COMM_WORLD) {}

            /// Returns process rank in this communicator
            int rank() const {
                int myrank;
                MPI_Comm_rank(comm_,&myrank);
                return myrank;
            }

            /// Returns the number of processes in this communicator
            int size() const {
                int sz;
                MPI_Comm_size(comm_,&sz);
                return sz;
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
            return alps::mpi::detail::mpi_type<T>();
            // throw std::logic_error(std::string("get_mpi_datatype() is not implemented, called for type T=")
            //                        +typeid(T).name());
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
        // FIXME: remove T? or just ensure it's a basic MPI type?
        template <typename OP, typename T>
        class is_mpi_op {
            public:
            static MPI_Op op() {
                throw std::logic_error(std::string("is_mpi_op() is not implemented, called for types OP=")
                                       +typeid(OP).name() + " and T="+typeid(T).name());
            }
        };

        /// Trait for MPI reduction operations: specialization for addition
        // FIXME: remove T? restrict T?
        template <typename T>
        class is_mpi_op<std::plus<T>, T> {
            public:
            static MPI_Op op() {
                return MPI_SUM;
            }
        };
        
        
    } // mpi::
} // alps::

#endif /* ALPS_UTILITIES_MPI_HPP_INCLUDED_90206380262d48f0bcbe98fd16edd65d */
