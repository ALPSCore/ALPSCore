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

#include <boost/scoped_array.hpp> /* for std::string broadcast */

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

        
        /// Broadcasts array `vals` of a primitive type `T`, length `count` on communicator `comm` with root `root`
        template <typename T>
        void broadcast(const communicator& comm, T* vals, std::size_t count, int root) {
            MPI_Bcast(vals, count, detail::mpi_type<T>(), root, comm);
        }

        /// MPI_BCast of an array: overload for bool
        inline void broadcast(const communicator& comm, bool* vals, std::size_t count, int root) {
            // sizeof() returns size in chars (FIXME? should it be bytes?)
            MPI_Bcast(vals, count*sizeof(bool), MPI_CHAR, root, comm); 
        }

        /// Broadcasts value `val` of a primitive type `T` on communicator `comm` with root `root`
        template <typename T>
        void broadcast(const communicator& comm, T& val, int root) {
            broadcast(comm, &val, 1, root);
        }

        /// MPI_BCast of a single value: overload for std::string
        // FIXME: what is exception safety status?
        // FIXME: inline to have it header-only. A tad too complex to be inlined?
        inline void broadcast(const communicator& comm, std::string& val, int root) {
            std::size_t root_sz=val.size();
            broadcast(comm, root_sz, root);
            if (comm.rank()==root) {
                // NOTE: at root rank the value being broadcast does not change, so const cast is safe
                broadcast(comm, const_cast<char*>(val.data()), root_sz, root);
            } else {
                // FIXME: not very efficient --- any better way without heap alloc?
                //        Note, there is no guarantee in C++03 that modifying *(&val[0]+i) is safe!
                boost::scoped_array<char> buf(new char[root_sz]);
                broadcast(comm, buf.get(), root_sz, root);
                val.assign(buf.get(), root_sz);
            }
        }

        
        /// Returns MPI datatype for the value of type `T`
        template <typename T>
        MPI_Datatype get_mpi_datatype(const T& val) {
            return detail::mpi_type<T>();
            // throw std::logic_error(std::string("get_mpi_datatype() is not implemented, called for type T=")
            //                        +typeid(T).name());
        }

        /// performs MPI_Allgather() for primitive type T
        /** @NOTE Vector `out_vals` is resized */
        template <typename T>
        void all_gather(const communicator& comm, const T& in_val, std::vector<T>& out_vals) {
            out_vals.resize(comm.size());
            MPI_Allgather((void*)&in_val, 1, detail::mpi_type<T>(),
                          &out_vals.front(), 1, detail::mpi_type<T>(),
                          comm);
            // throw std::logic_error(std::string("all_gather() is not implemented, called for type T=")
            //                        +typeid(T).name());
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
        
        /// Trait for MPI reduction operations: specialization for maximum
        // FIXME: remove T? restrict T?
        template <typename T>
        class is_mpi_op<maximum<T>, T> {
            public:
            static MPI_Op op() {
                return MPI_MAX;
            }
        };
        
        
    } // mpi::
} // alps::

#endif /* ALPS_UTILITIES_MPI_HPP_INCLUDED_90206380262d48f0bcbe98fd16edd65d */
