/*
 * 
 * Copyright (c) Toon Knapen, Kresimir Fresl and Synge Todo 2004
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

#ifndef BOOST_NUMERIC_BINDINGS_LAPACK_GELS_HPP
#define BOOST_NUMERIC_BINDINGS_LAPACK_GELS_HPP

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
    // Simple Driver Routines for Solving Least Squares Problems
    //
    /////////////////////////////////////////////////////////////////////

    /*
     * gels() computes the least squares solution to an over-determined system
     * of linear equations, A X=B or A**H X=B, or the minimum norm solution of
     * an under-determined system, where A is a general rectangular matrix of
     * full rank, using a QR or LQ factorization of A.
     */

    namespace detail {

      inline 
      void gels (char const trans, int const m, int const n, int const nrhs,
		 float* a, int const lda, float* b, int const ldb,
		 float* work, int const lwork, int* info)
      {
	LAPACK_SGELS(&trans, &m ,&n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
      }

      inline 
      void gels (char const trans, int const m, int const n, int const nrhs,
		 double* a, int const lda, double* b, int const ldb,
		 double* work, int const lwork, int* info)
      {
	LAPACK_DGELS(&trans, &m ,&n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
      }

    }

    template<class T>
    int gels (char const trans, int const m, int const n, int const nrhs,
	      T* a, int const lda, T* b, int const ldb) {
      int info;
      int lwork = std::min(m,n) + std::max(std::max(1,m), std::max(n,nrhs)) * 64;
      T* work = new T[lwork];
      detail::gels(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, &info);
      delete [] work;
      return info;
    }
  }
}}}

#endif 
