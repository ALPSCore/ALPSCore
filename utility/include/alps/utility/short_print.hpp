/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_SHORT_PRINT_HPP
#define ALPS_NGS_SHORT_PRINT_HPP

#include <vector>
#include <boost/array.hpp>
#include <alps/multi_array.hpp>
#include <ostream>

namespace alps {
    namespace detail {
        template<typename T> struct short_print_proxy {
            public:
                explicit short_print_proxy(T const & v, std::size_t p): value(v), precision(p) {};
                short_print_proxy(short_print_proxy<T> const & rhs): value(rhs.value) {};
                T const & value;
                std::size_t precision;
        };

        template <typename T> std::ostream & operator<<(std::ostream & os, short_print_proxy<T> const & v) {
            return os << v.value;
        }
    }

    template<typename T> detail::short_print_proxy<T const> short_print(T const & v, std::size_t p = 6) {
        return detail::short_print_proxy<T const>(v, p);
    }
    
    namespace detail {
        std::ostream & operator<<(std::ostream & os, short_print_proxy<float> const & v);
        std::ostream & operator<<(std::ostream & os, short_print_proxy<double> const & v);
        std::ostream & operator<<(std::ostream & os, short_print_proxy<long double> const & v);

        template<typename T>
        std::ostream & print_for_sequence(std::ostream & os, T const & value)
        {
            switch (value.size()) {\
                case 0: \
                    return os << "[]";\
                case 1: \
                    return os << "[" << short_print(value.front()) << "]";\
                case 2: \
                    return os << "[" << short_print(value.front()) << "," << short_print(value.back()) << "]";\
                default: \
                    return os << "[" << short_print(value.front()) << ",.." << short_print(value.size()) << "..," << short_print(value.back()) << "]";\
            }
        }
        
        template <typename T> 
        std::ostream & operator<<(std::ostream & os, short_print_proxy<std::vector<T> const> const & v) 
        {
            return print_for_sequence(os, v.value);
        }
        
        template <typename T, std::size_t N> 
        std::ostream & operator<<(std::ostream & os, short_print_proxy<boost::array<T, N> const> const & v)
        {
            return print_for_sequence(os, v.value);
        }
        
        template <typename T, std::size_t N> std::ostream & operator<<(std::ostream & os, short_print_proxy<alps::multi_array<T, N> const> const & v) {
            switch (v.value.num_elements()) {
                case 0: 
                    return os << "[]";
                case 1: 
                    return os << "[" << short_print(*(v.value.data())) << "]";
                case 2: 
                    return os << "[" << short_print(*(v.value.data())) << "," << short_print(*(v.value.data()+v.value.num_elements()-1)) << "]";
                default: 
                    return os << "[" << short_print(*(v.value.data())) << ",.." << short_print(v.value.num_elements()) << "..," << short_print(*(v.value.data()+v.value.num_elements()-1)) << "]";
            }
        }
    }
}

#endif
