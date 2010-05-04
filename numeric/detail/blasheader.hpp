/*****************************************************************************
 *
 * ALPS DMFT Project - BLAS Compatibility headers
 *  BLAS headers for accessing BLAS from C++.
 *
 * Copyright (C) 2005 - 2009 by 
 *                              Emanuel Gull <gull@phys.columbia.edu>,
 *
 *
* This software is part of the ALPS Applications, published under the ALPS
* Application License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Application License along with
* the ALPS Applications; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/


#include<complex>
//function declaration for BLAS functions
namespace blas{
//matrix vector multiplication
extern "C" void dgemv_(const void *trans, const int *size1, const int *size2, const double *alpha, const double *values_, const int *memory_size,const double *v1, const int *inc, const double *beta, double *v2, const int *incb);
extern "C" void zgemv_(const void *trans, const int *size1, const int *size2, const std::complex<double> *alpha, const std::complex<double>*values_, const int *memory_size,const std::complex<double>*v1, const int *inc, const std::complex<double>*beta, std::complex<double> *v2, const int *incb);
extern "C" void sgemv_(const void *trans, const int *size1, const int *size2, const float *alpha, const float *values_, const int *memory_size,const float *v1, const int *inc, const float *beta, float*v2, const int *incb);
//outer product
//outer product
extern "C" void dger_(const int *m, const int * n, const double* alpha,const double*  x, const int* incx, const double*y, const int*  incy, const double* a,const int* lda) ;
extern "C" void sger_(const int *m, const int * n, const float* alpha,const float*  x, const int* incx, const float*y, const int*  incy, const float* a,const int* lda) ;
//matrix matrix product
extern "C" void sgemm_(const char *tA, const char *tB, const int *M,
             const int *N, const int *K, const float *alpha,
             const float *A, const int *lda, const float *B,
             const int *ldb, const float *beta, float *C,
             const int *ldc);
extern "C" void dgemm_(const char *tA, const char *tB, const int *M,
             const int *N, const int *K, const double *alpha,
             const double *A, const int *lda, const double *B,
             const int *ldb, const double *beta, double *C,
             const int *ldc);
//matrix product for complex matrices
extern "C" void zgemm_(const char *tA, const char *tB, const int *M,
             const int *N, const int *K, const void *alpha,
             const void *A, const int *lda, const void *B,
             const int *ldb, const void *beta, void *C,
             const int *ldc);
//vector maximum
extern "C" int idamax_(const int *n,const double*  x,const int* incx);
extern "C" int isamax_(const int *n,const float*  x,const int* incx);
extern "C" int izamax_(const int *n,const void  *  x,const int* incx);
//vector norm
extern "C" int dnrm2_ (const int *n,const double*  x,const int* incx);
//vector copy
extern "C" int scopy_ (const int *n,const float*  x,const int* incx, float*y, const int *incy);
extern "C" int dcopy_ (const int *n,const double*  x,const int* incx, double *y, const int *incy);
//vector swap 
extern "C" int sswap_ (const int *n,float*  x,const int* incx, float*y, const int *incy);
extern "C" int dswap_ (const int *n,double*  x,const int* incx, double *y, const int *incy);
//vector dot product
extern "C" double ddot_(const int *size,const double * v1,const int *inc,const double *v2,const int *inc2);
extern "C" float sdot_(const int *size,const float * v1,const int *inc,const float *v2,const int *inc2);
//vector scale.
extern "C" void sscal_(const int *size, const float *alpha, float *v, const int *inc); //double 
extern "C" void saxpy_(const int *size, const float *alpha, const float *x, const int *inc, float *y, const int *incy); //double 
extern "C" void dscal_(const int *size, const double *alpha, double *v, const int *inc); //double 
extern "C" void daxpy_(const int *size, const double *alpha, const double *x, const int *inc, double *y, const int *incy); //double 
extern "C" void zaxpy_(const int *size, const std::complex<double> *alpha, const std::complex<double> *x, const int *inc, std::complex<double> *y, const int *incy); //double 
extern "C" void zscal_(const int *size, const void *alpha, void *v, const int *inc);     //complex double
}

namespace lapack{
extern "C" void ssyev_(const char *jobs, const char *uplo, const int *size, float *matrix, const int *lda, float *eigenvalues, float *work, const int *lwork, int *info);
extern "C" void dsyev_(const char *jobs, const char *uplo, const int *size, double *matrix, const int *lda, double *eigenvalues, double *work, const int *lwork, int *info);
extern "C" void sgesv_(const int *N, const int *NRHS, float *A, const int *LDA, int *IPIV, float *B, const int *LDB, int *INFO);
extern "C" void dgesv_(const int *N, const int *NRHS, double *A, const int *LDA, int *IPIV, double *B, const int *LDB, int *INFO);
extern "C" void zgesv_(const int *N, const int *NRHS, std::complex<double> *A, const int *LDA, int *IPIV, std::complex<double> *B, const int *LDB, int *INFO);
extern "C" void dgetrs_(const char *trans, const int *N, const int *NRHS, double *A, const int *LDA, int *IPIV, double *B, const int *LDB, int *INFO);
extern "C" void dtrtri_(const char *uplo, const char *diag, const int *N, double *A, const int *LDA, int *INFO);
}
namespace vecLib{
extern "C" void vvexp(double * /* y */,const double * /* x */,const int * /* n */);
}
namespace acml{
extern "C" void vrda_exp(const int, double *, const double *);
}
namespace mkl{
extern "C" void vdExp(const int, const double *, double *);
extern "C" void vfExp(const int, const float *, float *);
extern "C" void vzmul_(const int*, const std::complex<double>*, const std::complex<double> *, std::complex<double> *);
}
