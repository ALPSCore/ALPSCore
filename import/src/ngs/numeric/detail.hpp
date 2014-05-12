/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2012 by Mario Koenz <mkoenz@ethz.ch>                       *
 * Copyright (C) 2012 - 2014 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_NUMERIC_DETAIL_HEADER
#define ALPS_NGS_NUMERIC_DETAIL_HEADER

#include <alps/ngs/stacktrace.hpp>
#include <alps/utility/resize.hpp>

#include <alps/multi_array.hpp>

#include <boost/array.hpp>

#include <vector>
#include <stdexcept>

namespace alps {
    namespace ngs { //merged with alps/numerics/vector_function.hpp
        namespace numeric {
            namespace detail {

                template<typename T, typename U>
                inline void check_size(T & a, U const & b) {}

                template<typename T, typename U>
                inline void check_size(std::vector<T> & a, std::vector<U> const & b) {
                    if(a.size() == 0)
                        alps::resize_same_as(a, b);
                    else if(a.size() != b.size())
                        boost::throw_exception(std::runtime_error("vectors must have the same size!" + ALPS_STACKTRACE));
                }

                template<typename T, typename U, std::size_t N, std::size_t M>
                inline void check_size(boost::array<T, N> & a, boost::array<U, M> const & b) {
                    boost::throw_exception(std::runtime_error("boost::array s must have the same size!" + ALPS_STACKTRACE));
                }

                template<typename T, typename U, std::size_t N>
                inline void check_size(boost::array<T, N> & a, boost::array<U, N> const & b) {}

                template<typename T, typename U, std::size_t D>
                inline void check_size(alps::multi_array<T, D> & a, alps::multi_array<U, D> const & b) {}
                
            }
        }
    }
}

#endif
