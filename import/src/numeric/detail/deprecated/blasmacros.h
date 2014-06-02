/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <boost/numeric/bindings/blas.hpp>
#include<vector>

// provide overloads for types where blas can be used        

namespace blas{

#define IMPLEMENT_FOR_REAL_BLAS_TYPES(F) F(float) F(double)

#define IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) \
F(std::complex<float>) \
F(std::complex<double>)

#define IMPLEMENT_FOR_ALL_BLAS_TYPES(F) \
IMPLEMENT_FOR_REAL_BLAS_TYPES(F) \
IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) 
} // namespace
