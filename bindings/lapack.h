/*
 * 
 * Copyright (c) Toon Knapen, Kresimir Fresl, Matthias Troyer, & Synge Todo 2003,2004
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

#ifndef ALPS_NUMERIC_BINDINGS_LAPACK_LAPACK_H
#define ALPS_NUMERIC_BINDINGS_LAPACK_LAPACK_H

#include <boost/numeric/bindings/traits/type.h>
#include <alps/bindings/lapack_names.h>

extern "C" {

  /**********************************************************************/
  /* eigenproblems */
  /**********************************************************************/

  /* symmetric/Hermitian indefinite and complex symmetric */

  void LAPACK_SSYEV (const char* jobz, const char* uplo, const int* n, float* a, 
                     const int * lda, float* w, float* work, const int * lwork,
                     int* info);

  void LAPACK_DSYEV (const char* jobz, const char* uplo, const int* n, double* a, 
                     const int * lda, double* w, double* work, const int * lwork,
                     int* info);

  void LAPACK_CHEEV (const char* jobz, const char* uplo, const int* n, fcomplex_t* a, 
                     const int * lda, float* w, fcomplex_t* work, const int * lwork,
                     float* rwork, int* info);

  void LAPACK_ZHEEV (const char* jobz, const char* uplo, const int* n, dcomplex_t* a, 
                     const int * lda, double * w, dcomplex_t * work, const int * lwork,
                     double * rwork, int* info);

  /**********************************************************************/
  /* linear least squares problems */
  /**********************************************************************/

  void LAPACK_SGELS (const char* trans, const int* m, const int* n, const int* nrhs,
		     float * a, const int* lda, float * b, const int* ldb,
		     float * work, const int* lwork, int* info);

  void LAPACK_DGELS (const char* trans, const int* m, const int* n, const int* nrhs,
		     double * a, const int* lda, double * b, const int* ldb,
		     double * work, const int* lwork, int* info);

  void LAPACK_CGELS (const char* trans, const int* m, const int* n, const int* nrhs,
		     fcomplex_t * a, const int* lda, fcomplex_t * b, const int& ldb,
		     fcomplex_t * work, const int* lwork, int* info);

  void LAPACK_ZGELS (const char* trans, const int* m, const int* n, const int* nrhs,
		     dcomplex_t * a, const int* lda, dcomplex_t * b, const int& ldb,
		     dcomplex_t * work, const int* lwork, int* info);

}

#endif 
