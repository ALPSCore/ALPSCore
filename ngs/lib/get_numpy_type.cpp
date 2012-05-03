/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <alps/ngs/detail/get_numpy_type.hpp>

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
