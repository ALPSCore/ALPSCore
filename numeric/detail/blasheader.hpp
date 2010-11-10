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
