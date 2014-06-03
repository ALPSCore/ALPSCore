/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_MATRIX_BLASMACROS_HPP
#define ALPS_MATRIX_BLASMACROS_HPP

// provide overloads for types where blas can be used        

namespace alps {
    namespace numeric {

    #define ALPS_IMPLEMENT_FOR_REAL_BLAS_TYPES(F) F(float) F(double)

    #define ALPS_IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F) \
    F(std::complex<float>) \
    F(std::complex<double>)

    #define ALPS_IMPLEMENT_FOR_ALL_BLAS_TYPES(F) \
    ALPS_IMPLEMENT_FOR_REAL_BLAS_TYPES(F) \
    ALPS_IMPLEMENT_FOR_COMPLEX_BLAS_TYPES(F)
    } // namespave numeric
} // namespace alps

#endif // ALPS_MATRIX_BLASMACROS_HPP
