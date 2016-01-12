/** @file mpi_wrappers.hpp
    @brief Partial replacement for boost::mpi.
*/

#ifndef ALPS_MPI_MPI_WRAPPERS_85342464d9af4d07ad532b78166841cf
#define ALPS_MPI_MPI_WRAPPERS_85342464d9af4d07ad532b78166841cf

#include <mpi.h>
#include <string>
#include <boost/scoped_array.hpp> /* for broadcasting strings */

namespace alps {
    namespace mpi {

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

        /// Convenience function to get rank from the given communicator
        inline int rank(MPI_Comm comm) {
            int myrank;
            MPI_Comm_rank(comm, &myrank);
            return myrank;
        }

        /// Generic MPI_BCast of an array (guesses type)
        template <typename T>
        int bcast(T* pval, int count, int root, MPI_Comm comm) {
            return MPI_Bcast(pval, count, mpi_type<T>(), root, comm);
        }

        /// MPI_BCast of an array: overload for bool
        inline int bcast(bool* val, int count, int root, MPI_Comm comm) {
            // sizeof() returns size in chars (FIXME? should it be bytes?)
            return MPI_Bcast(val, count*sizeof(bool), MPI_CHAR, root, comm); 
        }

        /// Generic MPI_BCast of a single value (guesses type)
        template <typename T>
        int bcast(T& val, int root, MPI_Comm comm) {
            return bcast(&val, 1, root, comm);
        }

        /// MPI_BCast of a single value: overload for std::string
        // FIXME: what is exception safety status?
        // FIXME: inline to have it header-only. A tad too complex to be inlined?
        inline int bcast(std::string& val, int root, MPI_Comm comm) {
            std::size_t root_sz=val.size();
            int rc=bcast(root_sz, root, comm);
            if (rc!=0) return rc;
            int myrank=rank(comm);
            if (myrank==root) {
                // NOTE: at root rank the value being broadcast does not change, so const cast is safe
                rc=bcast(const_cast<char*>(val.data()), root_sz, root, comm);
            } else {
                // FIXME: not very efficient --- any better way without heap alloc?
                //        Note, there is no guarantee in C++03 that modifying *(&val[0]+i) is safe!
                boost::scoped_array<char> buf(new char[root_sz]);
                rc=bcast(buf.get(), root_sz, root, comm);
                if (rc==0) val.assign(buf.get(), root_sz);
            }
            return rc;
        }

        
        
    } // mpi::
} // alps::
#endif /* ALPS_MPI_MPI_WRAPPERS_85342464d9af4d07ad532b78166841cf */
