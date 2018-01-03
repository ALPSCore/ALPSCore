/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <vector>
#include <boost/array.hpp>
//#include <alps/multi_array.hpp> // FIXME
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
//DEPENDENCE ON MULTIARRAY TO BE REMOVED 
/*        template <typename T, std::size_t N> std::ostream & operator<<(std::ostream & os, short_print_proxy<alps::multi_array<T, N> const> const & v) {
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
         // FIXME -> move to multi_array
        }*/
    }
}

