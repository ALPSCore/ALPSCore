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

#ifndef ALPS_NGS_DETAIL_GET_NUMPY_TYPE_HPP
#define ALPS_NGS_DETAIL_GET_NUMPY_TYPE_HPP

#include <alps/ngs/config.hpp>

#if !defined(ALPS_HAVE_PYTHON)
    #error numpy is only available if python is enabled
#endif

#include <alps/ngs/boost_python.hpp>
#include <alps/ngs/detail/numpy_import.hpp>

#include <complex>

#define ALPS_NGS_FOREACH_NATIVE_NUMPY_TYPE(CALLBACK)                                                                                            \
    CALLBACK(bool)                                                                                                                              \
    CALLBACK(char)                                                                                                                              \
    CALLBACK(signed char)                                                                                                                       \
    CALLBACK(unsigned char)                                                                                                                     \
    CALLBACK(short)                                                                                                                             \
    CALLBACK(unsigned short)                                                                                                                    \
    CALLBACK(int)                                                                                                                               \
    CALLBACK(unsigned)                                                                                                                          \
    CALLBACK(long)                                                                                                                              \
    CALLBACK(unsigned long)                                                                                                                     \
    CALLBACK(long long)                                                                                                                         \
    CALLBACK(unsigned long long)                                                                                                                \
    CALLBACK(float)                                                                                                                             \
    CALLBACK(double)                                                                                                                            \
    CALLBACK(long double)                                                                                                                       \
    CALLBACK(std::complex<float>)                                                                                                               \
    CALLBACK(std::complex<double>)                                                                                                              \
    CALLBACK(std::complex<long double>)

namespace alps {
    namespace detail {
        #define ALPS_NGS_DECL_NUMPY_TYPE(T)                                                                                                     \
            ALPS_DECL int get_numpy_type(T);
        ALPS_NGS_FOREACH_NATIVE_NUMPY_TYPE(ALPS_NGS_DECL_NUMPY_TYPE)
        #undef ALPS_NGS_DECL_NUMPY_TYPE
    }
}

#endif
