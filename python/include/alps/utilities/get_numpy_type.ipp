/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
 
#include <alps/utilities/get_numpy_type.hpp>

namespace alps {
    namespace detail {

        int get_numpy_type(bool) { return PyArray_BOOL; }
        int get_numpy_type(char) { return PyArray_CHAR; }
        int get_numpy_type(unsigned char) { return PyArray_UBYTE; }
        int get_numpy_type(signed char) { return PyArray_BYTE; }
        int get_numpy_type(short) { return PyArray_SHORT; }
        int get_numpy_type(unsigned short) { return PyArray_USHORT; }
        int get_numpy_type(int) { return PyArray_INT; }
        int get_numpy_type(unsigned int) { return PyArray_UINT; }
        int get_numpy_type(long) { return PyArray_LONG; }
        int get_numpy_type(unsigned long) { return PyArray_ULONG; }
        int get_numpy_type(long long) { return PyArray_LONGLONG; }
        int get_numpy_type(unsigned long long) { return PyArray_ULONGLONG; }
        int get_numpy_type(float) { return PyArray_FLOAT; }
        int get_numpy_type(double) { return PyArray_DOUBLE; }
        int get_numpy_type(long double) { return PyArray_LONGDOUBLE; }
        int get_numpy_type(std::complex<float>) { return PyArray_CFLOAT; }
        int get_numpy_type(std::complex<double>) { return PyArray_CDOUBLE; }
        int get_numpy_type(std::complex<long double>) { return PyArray_CLONGDOUBLE; }
    }
}
