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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_HEEV_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_HEEV_HPP

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
    // eigenvalue solver A * X = lambda * X with A hermitian matrix
    //
    /////////////////////////////////////////////////////////////////////

    /*
     * heev() computes the solution to a system of linear eigenvalue equations 
     * A * X = lambda * X, where A is an N-by-N hermitian matrix, 
     * lambda is an eigenvalue and X and eigenvector
     */

    namespace detail {

      inline 
      void heev (char const jobz, char const uplo, int const n,
                 traits::complex_f* a, int const lda, float* w,
                 traits::complex_f* work, int* lwork, float* rwork, int* info) 
      {
        LAPACK_CHEEV (&jobz, &uplo, &n, 
                      traits::complex_ptr(a), &lda, w, 
                      traits::complex_ptr(work), lwork, 
                      rwork, info);
      }

      inline 
      void heev (char const jobz, char const uplo, int const n,
                 traits::complex_d* a, int const lda, double* w,
                 traits::complex_d* work, int* lwork, double* rwork, int* info) 
      {
        LAPACK_ZHEEV (&jobz, &uplo, &n, 
                      traits::complex_ptr(a), &lda, w, 
                      traits::complex_ptr(work), 
                      lwork,rwork,info);
      }

      template <typename HermA, typename Vector, typename Work, typename RWork>
      inline
      int heev (char const jobz, char const ul, HermA& a, Vector& w, 
                Work& work, int& lwork, RWork& rwork) {

        int const n = traits::matrix_size1 (a);
        assert (n == traits::matrix_size2 (a)); 
        assert (n == traits::vector_size (w)); 

        int info; 
        heev (jobz, ul, n, 
              traits::matrix_storage (a), 
              traits::leading_dimension (a),
              traits::vector_storage (w),  
              traits::vector_storage (work),  
              &lwork,
              traits::vector_storage (rwork),
              &info);
        return info; 
      }

    }

      template <typename HermA, typename Vector, typename Work, typename RWork>
      inline
      int heev (char const jobz, char const ul, HermA& a, Vector& w, 
                Work& work, RWork& rwork) {

      assert (jobz == 'V' || jobz == 'N'); 
      assert (ul == 'U' || ul == 'L'); 

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK
      BOOST_STATIC_ASSERT((boost::is_same<
        typename traits::matrix_traits<HermA>::matrix_structure, 
        traits::general_t
      >::value));
#endif

      int const n = traits::matrix_size1 (a);
      int const lw = traits::vector_size (work); 
      assert (lw >= std::max(1,2*n-1)); 
      assert (traits::vector_size (rwork)>=std::max(1,3*n-2));
      return detail::heev (jobz,ul,a,w,work,lw,rwork); 
    }

    template <typename HermA, typename Vector, typename Work, typename RWork>
    inline
    int heev (char const jobz, HermA& a, Vector& w, Work& work, RWork& rwork) {

      assert ((jobz == 'V' || jobz == 'N')); 

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK
      BOOST_STATIC_ASSERT((boost::is_same<
        typename traits::matrix_traits<HermA>::matrix_structure, 
        traits::hermitian_t
        >::value));
#endif

      int const n = traits::matrix_size1 (a);
      int const lw = traits::vector_size (work); 
      char uplo = traits::matrix_uplo_tag (a);
      assert (lw >= std::max(1,2*n-1)); 
      assert (traits::vector_size (rwork)>=std::max(1,3*n-2));
      return detail::heev (jobz,uplo,a,w,work,lw,rwork); 
    }

    template <typename HermA, typename Vector>
    inline
    int heev (char const jobz, char const ul, HermA& a, Vector& w) {
      // with 'internal' work vectors 

      assert (ul == 'U' || ul == 'L'); 
      assert (jobz == 'V' || jobz == 'N'); 

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK
      BOOST_STATIC_ASSERT((boost::is_same<
        typename traits::matrix_traits<HermA>::matrix_structure, 
        traits::general_t
      >::value));
#endif

#ifndef BOOST_NUMERIC_BINDINGS_POOR_MANS_TRAITS
      typedef typename traits::matrix_traits<HermA>::value_type val_t; 
      typedef typename traits::vector_traits<Vector>::value_type real_t; 
#else 
      typedef typename HermA::value_type val_t; 
      typedef typename Vector::value_type real_t; 
#endif 
      char opts[2];
      opts[0]=jobz;
      opts[1]=ul;
      // assuming same block size for double and single precision
      int info = ilaenv(1,"ZHEEV",opts,traits::matrix_size1 (a));
      if (info>=0) {
        int const nb = std::max(1,info);
        int const n = traits::matrix_size1 (a);
        info = -102; 
        int lw = std::max(1,(nb+1)*traits::matrix_size1 (a)); 
        traits::detail::array<val_t> work (lw); 
        traits::detail::array<real_t> rwork (std::max(1,3*n-2)); 
        if (work.valid() && rwork.valid())
          info =  detail::heev (jobz, ul, a, w, work, lw, rwork); 
      }
      return info; 
    }

    template <typename HermA, typename Vector>
    inline
    int heev (char jobz, HermA& a, Vector& w) {
      // with 'internal' work vectors 

      assert (jobz == 'V' || jobz == 'N'); 

#ifndef BOOST_NUMERIC_BINDINGS_NO_STRUCTURE_CHECK
      BOOST_STATIC_ASSERT((boost::is_same<
        typename traits::matrix_traits<HermA>::matrix_structure, 
        traits::hermitian_t
      >::value));
#endif

      char uplo = traits::matrix_uplo_tag (a);

#ifndef BOOST_NUMERIC_BINDINGS_POOR_MANS_TRAITS
      typedef typename traits::matrix_traits<HermA>::value_type val_t; 
      typedef typename traits::vector_traits<Vector>::value_type real_t; 
#else 
      typedef typename HermA::value_type val_t; 
      typedef typename Vector::value_type real_t; 
#endif 
      char opts[2];
      opts[0]=jobz;
      opts[1]=uplo;
      // assuming same block size for double and single precision
      int info = ilaenv(1,"ZHEEV",opts,traits::matrix_size1 (a));
      if (info>=0) {
        int nb = std::max(1,info);
        info = -102; 
        int lw = std::max(1,(nb+1)*traits::matrix_size1 (a)); 
        int n = traits::matrix_size1 (a);
        traits::detail::array<val_t> work (lw); 
        traits::detail::array<real_t> rwork (std::max(1,3*n-2)); 
        if (work.valid() && rwork.valid())
          info =  detail::heev (jobz, uplo, a, w, work, lw, rwork); 
      }
      return info; 
    }
  }
}}}

#endif 
