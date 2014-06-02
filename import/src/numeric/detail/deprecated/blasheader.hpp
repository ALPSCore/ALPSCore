/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef BOOST_NUMERIC_DETAIL_BLASHEADER_HPP
#define BOOST_NUMERIC_DETAIL_BLASHEADER_HPP

#include <boost/numeric/bindings/blas.hpp>
#include <boost/numeric/bindings/lapack.hpp>

#include<complex>

namespace lapack{
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

#endif
