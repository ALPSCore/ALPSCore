/*
 * 
 * Copyright (c) Toon Knapen, Kresimir Fresl and Matthias Troyer 2003
 *
 * Permission to copy, modify, use and distribute this software 
 * for any non-commercial or commercial purpose is granted provided 
 * that this license appear on all copies of the software source code.
 *
 * Authors assume no responsibility whatsoever for its use and makes 
 * no guarantees about its quality, correctness or reliability.
 *
 * KF acknowledges the support of the Faculty of Civil Engineering, 
 * University of Zagreb, Croatia.
 *
 */

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_SYEV_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_SYEV_HPP

#include <boost/numeric/bindings/traits/type_traits.hpp>
#include <boost/numeric/bindings/traits/traits.hpp>
#include <boost/numeric/bindings/lapack/lapack.h>
#include <boost/numeric/bindings/lapack/ilaenv.hpp>
#include <boost/numeric/bindings/traits/detail/array.hpp>
#include <boost/numeric/bindings/traits/detail/utils.hpp>

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK 
#  include <boost/static_assert.hpp>
#  include <boost/type_traits.hpp>
#endif 

#include <cassert>

#include <alps/bindings/lapack.h>

namespace boost { namespace numeric { namespace bindings { 

  namespace lapack {

    /////////////////////////////////////////////////////////////////////
    //
    // eigenvalue solver A * X = lambda * X with A symmetric matrix
    //
    /////////////////////////////////////////////////////////////////////

    /*
     * syev() computes the solution to a system of linear eigenvalue equations 
     * A * X = lambda * X, where A is an N-by-N symmetric matrix, 
     * lambda is an eigenvalue and X and eigenvector
     */

    namespace detail {

      inline 
      void syev (char const jobz, char const uplo, int const n,
                 float* a, int const lda, float* w,
                 float* work, int* lwork, int* info) 
      {
        LAPACK_SSYEV (&jobz, &uplo, &n, a, &lda, w, work, lwork, info);
      }

      inline 
      void syev (char const jobz, char const uplo, int const n,
                 double* a, int const lda, double* w,
                 double* work, int* lwork, int* info) 
      {
        LAPACK_DSYEV (&jobz, &uplo, &n, a, &lda, w, work, lwork, info);
      }

      template <typename SymmA, typename Vector, typename Work>
      inline
      int syev (char const jobz, char const ul, SymmA& a, Vector& w, 
                Work& work, int& lwork) {

        int const n = traits::matrix_size1 (a);
        assert (n == traits::matrix_size2 (a)); 
        assert (n == traits::vector_size (w)); 

        int info; 
        syev (jobz, ul, n, 
              traits::matrix_storage (a), 
              traits::leading_dimension (a),
              traits::vector_storage (w),  
              traits::vector_storage (work),  
              &lwork,&info);
        return info; 
      }

    }

      template <typename SymmA, typename Vector, typename Work>
      inline
      int syev (char const jobz, char const ul, SymmA& a, Vector& w, 
                Work& work) {

      assert (jobz == 'V' || jobz == 'N'); 
      assert (ul == 'U' || ul == 'L'); 

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK
      BOOST_STATIC_ASSERT((boost::is_same<
        typename traits::matrix_traits<SymmA>::matrix_structure, 
        traits::general_t
      >::value));
#endif

      int const n = traits::matrix_size1 (a);
      int const lw = traits::vector_size (work); 
      assert (lw >= std::max(1,3*n-1)); 
      return detail::syev (jobz,ul,a,w,work,lw); 
    }

    template <typename SymmA, typename Vector, typename Work>
    inline
    int syev (char const jobz, SymmA& a, Vector& w, Work& work) {

      assert ((jobz == 'V' || jobz == 'N')); 

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK
      BOOST_STATIC_ASSERT((boost::is_same<
        typename traits::matrix_traits<SymmA>::matrix_structure, 
        traits::symmetric_t
        >::value));
#endif

      int const lw = traits::vector_size (work); 
      assert (lw >= std::max(1,3*n-1)); 
      char uplo = traits::matrix_uplo_tag (a);
      return detail::syev (jobz, uplo, a, w, work, lw); 
    }

    template <typename SymmA, typename Vector>
    inline
    int syev (char const jobz, char const ul, SymmA& a, Vector& w) {
      // with 'internal' work vectors 

      assert (ul == 'U' || ul == 'L'); 
      assert (jobz == 'V' || jobz == 'N'); 

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK
      BOOST_STATIC_ASSERT((boost::is_same<
        typename traits::matrix_traits<SymmA>::matrix_structure, 
        traits::general_t
      >::value));
#endif

#ifndef BOOST_NUMERIC_BINDINGS_POOR_MANS_TRAITS
      typedef typename traits::matrix_traits<SymmA>::value_type val_t; 
#else 
      typedef typename SymmA::value_type val_t; 
#endif 
      char opts[2];
      opts[0]=jobz;
      opts[1]=ul;
      // assuming same block size for double and single precision
      int info = ilaenv(1,"DSYEV",opts,traits::matrix_size1 (a));
      if (info>=0) {
        int nb = std::max(1,info);
        info = -102; 
        int lw = std::max(1,(nb+2)*traits::matrix_size1 (a)); 
        traits::detail::array<val_t> work (lw); 
        if (work.valid())
          info =  detail::syev (jobz,ul, a, w, work, lw); 
      }
      return info; 
    }

    template <typename SymmA, typename Vector>
    inline
    int syev (char jobz, SymmA& a, Vector& w) {
      // with 'internal' work vectors 

      assert (jobz == 'V' || jobz == 'N'); 

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK
      BOOST_STATIC_ASSERT((boost::is_same<
        typename traits::matrix_traits<SymmA>::matrix_structure, 
        traits::symmetric_t
      >::value));
#endif

      char uplo = traits::matrix_uplo_tag (a);

#ifndef BOOST_NUMERIC_BINDINGS_POOR_MANS_TRAITS
      typedef typename traits::matrix_traits<SymmA>::value_type val_t; 
#else 
      typedef typename SymmA::value_type val_t; 
#endif 
      char opts[2];
      opts[0]=jobz;
      opts[1]=uplo;
      // assuming same block size for double and single precision
      info = ilaenv(1,"DSYEV",opts,traits::matrix_size1 (a));
      if (info>=0) {
        int nb = std::max(1,info);
        int lw = std::max(1,(nb+2)*traits::matrix_size1 (a)); 
        info = -102; 
        traits::detail::array<val_t> work (lw); 
        if (work.valid()) 
          info =  detail::syev (jobz,uplo, a, w, work, lw);
      }
      return info; 
    }
  }
}}}

#endif 
