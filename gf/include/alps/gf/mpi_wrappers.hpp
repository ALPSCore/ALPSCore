/** @file mpi_wrappers.hpp
    @brief Partial replacement for boost::mpi.
*/

#ifndef ALPS_MPI_MPI_WRAPPERS_85342464d9af4d07ad532b78166841cf
#define ALPS_MPI_MPI_WRAPPERS_85342464d9af4d07ad532b78166841cf

#include <mpi.h>

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

    } // mpi::
} // alps::
#endif /* ALPS_MPI_MPI_WRAPPERS_85342464d9af4d07ad532b78166841cf */
