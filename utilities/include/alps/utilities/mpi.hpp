/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file utilities/mpi.hpp

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
#include <complex>
#include <exception> /* for std::uncaught_exception() */
#include <functional> /* for std::plus */
#include <algorithm> /* for std::max */

#include <memory> /* for proper copy/assign of managed communicators */

#include <stdexcept>
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
            typedef _cxxtype_ value_type;                       \
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
#ifdef ALPS_MPI_HAS_MPI_CXX_BOOL
       ALPS_MPI_DETAIL_MAKETYPE(MPI_CXX_BOOL,bool);
#endif

#if defined(ALPS_MPI_HAS_MPI_CXX_DOUBLE_COMPLEX) && defined(ALPS_MPI_HAS_MPI_CXX_FLOAT_COMPLEX)
        ALPS_MPI_DETAIL_MAKETYPE(MPI_CXX_DOUBLE_COMPLEX,std::complex<double>);
        ALPS_MPI_DETAIL_MAKETYPE(MPI_CXX_FLOAT_COMPLEX,std::complex<float>);
#endif

#undef ALPS_MPI_DETAIL_MAKETYPE
        } // detail::


        /// Possible ways to make a C++ object from MPI communicator
        enum comm_create_kind {
            comm_attach, ///< do not destroy when going out of scope
            comm_duplicate, ///< duplicate and destroy when going out of scope
            take_ownership ///< do not duplicate, but *destroy* when going out of scope
        };

        /// Encapsulation of an MPI communicator and some communicator-related operations
        /**
           A communicator object can be created from an existing MPI
           communicator, either by simple wrapping, by duplicating, or
           by taking complete ownership (see `communicator(const MPI_Comm&, comm_create_kind)`).

           Some MPI operations are implemented as mthods of this
           class.  Many ALPSCore classes have `%broadcast()` methods
           that take the communicator object as an argument.  The
           communicator object is implicitly convertible to the
           wrapped MPI communicator and therefore can be used as an
           argument to MPI calls directly.
         */
        class communicator {
            std::shared_ptr<MPI_Comm> comm_ptr_;

            // Internal functor class to destroy communicator when needed
            struct comm_deleter {
                void operator()(MPI_Comm* comm_ptr) {
                    int finalized;
                    MPI_Finalized(&finalized);
                    if (!finalized) MPI_Comm_free(comm_ptr);
                    delete comm_ptr;
                }
            };

            public:

            /// Creates an `MPI_COMM_WORLD` communicator object
            communicator() : comm_ptr_(new MPI_Comm(MPI_COMM_WORLD)) {} // FIXME? Shall we deprecate it?

            // FIXME: introduce error checking!!

            /// Creates a communicator object from an MPI communicator
            /**
               @param comm MPI communicator
               @param kind How to manage the communicator (see alps::mpi::comm_create_kind)
            */
            communicator(const MPI_Comm& comm, comm_create_kind kind) {
                switch (kind) {
                  default:
                      throw std::logic_error("alps::mpi::communicator(): unsupported `kind` argument.");
                      break;
                  case comm_attach:
                      comm_ptr_.reset(new MPI_Comm(comm));
                      break;
                  case take_ownership:
                      comm_ptr_.reset(new MPI_Comm(comm), comm_deleter());
                      break;
                  case comm_duplicate:
                      MPI_Comm* newcomm_ptr=new MPI_Comm();
                      MPI_Comm_dup(comm, newcomm_ptr);
                      comm_ptr_.reset(newcomm_ptr, comm_deleter());
                      break;
                }
            }

            /// Returns process rank in this communicator
            int rank() const {
                int myrank;
                MPI_Comm_rank(*comm_ptr_,&myrank);
                return myrank;
            }

            /// Returns the number of processes in this communicator
            int size() const {
                int sz;
                MPI_Comm_size(*comm_ptr_,&sz);
                return sz;
            }

            /// Barrier on this communicator
            void barrier() const {
                MPI_Barrier(*comm_ptr_);
            }

            /// Converts this communicator object to MPI communicator
            operator MPI_Comm() const {
                return *comm_ptr_;
            }
        };


        /// MPI environment RAII class
        class environment {
            bool initialized_;
            bool abort_on_exception_;
            public:

            /// Call `MPI_Abort()`
            static void abort(int rc=0)
            {
                MPI_Abort(MPI_COMM_WORLD,rc);
            }

            /// Returns initialized status of MPI
            static bool initialized()
            {
                int ini;
                MPI_Initialized(&ini);
                return ini;
            }

            /// Returns finalized status of MPI
            static bool finalized()
            {
                int fin;
                MPI_Finalized(&fin);
                return fin;
            }

            /// Initializes MPI environment unless it's already active
            /**
               @param argc `argc` argument from `main(argc, argv)`
               @param argv `argv` argument from `main(argc, argv)`
               @param abort_on_exception If true and MPI environment object is getting destroyed
                                         while an exception is being handled, call `MPI_Abort()`

            */
            environment(int& argc, char**& argv, bool abort_on_exception=true)
                : initialized_(false), abort_on_exception_(abort_on_exception)
            {
                if (!initialized()) {
                    MPI_Init(&argc, &argv);
                    initialized_=true;
                }
            }

            /// Initializes MPI environment unless it's already active
            /** This ctor does not pass `argc` and `argv`.
               @param abort_on_exception If true and MPI environment object is getting destroyed
                                         while an exception is being handled, call `MPI_Abort()`
            */
            environment(bool abort_on_exception=true)
                : initialized_(false), abort_on_exception_(abort_on_exception)
            {
                if (!initialized()) {
                    MPI_Init(NULL,NULL);
                    initialized_=true;
                }
            }

            /// Finalizes MPI unless already finalized or was already initilaized when ctor was called
            /**
               If called during an exception unwinding, may call `MPI_Abort()` (see the corresponding ctors)
            */
            ~environment()
            {
                if (!initialized_) return; // we are not in control, don't mess up other's logic.
                if (finalized()) return; // MPI is finalized --- don't touch it.
                if (abort_on_exception_ && std::uncaught_exception()) {
                    this->abort(255); // FIXME: make the return code configurable?
                }
                MPI_Finalize();
            }

        };

        /// Class-holder for reduction operations (and a functor) for type T.
        template <typename T>
        class maximum {
            public:
            maximum() { }
            T operator()(const T& a, const T& b) const {
                using std::max;
                return max(a,b);
            }
        };


        /// Broadcasts array `vals` of a primitive type `T`, length `count` on communicator `comm` with root `root`
        template <typename T>
        void broadcast(const communicator& comm, T* vals, std::size_t count, int root) {
            MPI_Bcast(vals, count, detail::mpi_type<T>(), root, comm);
        }

#ifndef ALPS_MPI_HAS_MPI_CXX_BOOL
        /// MPI_BCast of an array: overload for bool
        inline void broadcast(const communicator& comm, bool* vals, std::size_t count, int root) {
            // sizeof() returns size in chars (FIXME? should it be bytes?)
            MPI_Bcast(vals, count*sizeof(bool), MPI_CHAR, root, comm);
        }
#endif /* ALPS_MPI_HAS_MPI_BOOL */


#if !defined(ALPS_MPI_HAS_MPI_CXX_DOUBLE_COMPLEX) || !defined(ALPS_MPI_HAS_MPI_CXX_FLOAT_COMPLEX)
        /// MPI_BCast of an array: overload for std::complex
        template <typename T>
        inline void broadcast(const communicator& comm, std::complex<T>* vals, std::size_t count, int root) {
            // sizeof() returns size in chars (FIXME? should it be bytes?)
            MPI_Bcast(vals, count*sizeof(std::complex<T>), MPI_CHAR, root, comm);
        }
#endif

        /// Broadcasts value `val` of a primitive type `T` on communicator `comm` with root `root`
        template <typename T>
        void broadcast(const communicator& comm, T& val, int root) {
            broadcast(comm, &val, 1, root);
        }

        /// MPI_BCast of a single value: overload for `std::string`
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
                std::unique_ptr<char[]> buf(new char[root_sz]);
                broadcast(comm, buf.get(), root_sz, root);
                val.assign(buf.get(), root_sz);
            }
        }


        /// Returns MPI datatype for the value of type `T`
        template <typename T>
        MPI_Datatype get_mpi_datatype(const T&) {
            return detail::mpi_type<T>();
        }

        /// performs `MPI_Allgather()` for primitive type T
        /** @note Vector `out_vals` is resized */
        template <typename T>
        void all_gather(const communicator& comm, const T& in_val, std::vector<T>& out_vals) {
            out_vals.resize(comm.size());
            MPI_Allgather((void*)&in_val, 1, detail::mpi_type<T>(),
                          &out_vals.front(), 1, detail::mpi_type<T>(),
                          comm);
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

        /// Performs `MPI_Allreduce` for array of a primitive type, T[n]
        template <typename T, typename OP>
        void all_reduce(const alps::mpi::communicator& comm, const T* val, int n,
                        T* out_val, const OP& /*op*/)
        {
            if (n<=0) {
                throw std::invalid_argument("Non-positive array size in mpi::all_reduce()");
            }
            // @todo FIXME: implement in-place operations
            if (val==out_val) {
                throw std::invalid_argument("Implicit in-place mpi::all_reduce() is not implemented");
            }
            MPI_Allreduce(const_cast<T*>(val), out_val, n, detail::mpi_type<T>(),
                          is_mpi_op<OP,T>::op(), comm);
        }

        /// Performs `MPI_Allreduce` for a primitive type T
        template <typename T, typename OP>
        void all_reduce(const alps::mpi::communicator& comm, const T& val,
                        T& out_val, const OP& op)
        {
            all_reduce(comm, &val, 1, &out_val, op);
        }

        /// Performs `MPI_Allreduce` for a primitive type T
        template <typename T, typename OP>
        T all_reduce(const alps::mpi::communicator& comm, const T& val, const OP& op)
        {
            T out_val;
            all_reduce(comm, val, out_val, op);
            return out_val;
        }

    } // mpi::
} // alps::


#endif /* ALPS_UTILITIES_MPI_HPP_INCLUDED_90206380262d48f0bcbe98fd16edd65d */
