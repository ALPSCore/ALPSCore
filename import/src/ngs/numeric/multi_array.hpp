/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2014 by Jan Gukelberger <gukelberger@phys.ethz.ch>                *
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

#ifndef ALPS_NGS_NUMERIC_MULTIARRAY_HEADER
#define ALPS_NGS_NUMERIC_MULTIARRAY_HEADER

#include <alps/multi_array/functions.hpp>

// Import multi_array functions into ngs::numeric namespace.
namespace alps {
    namespace ngs {
        namespace numeric {
            
            using alps::sin;
            using alps::cos;
            using alps::tan;
            using alps::sinh;
            using alps::cosh;
            using alps::tanh;
            using alps::asin;
            using alps::acos;
            using alps::atan;
            using alps::abs;
            using alps::sqrt;
            using alps::exp;
            using alps::log;
            using alps::fabs;

            using alps::sq;
            using alps::cb;
            using alps::cbrt;
            
            using alps::pow;
            using alps::sum;
        }
    }
}

#endif
