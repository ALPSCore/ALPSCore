/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NUMERIC_CHECK_SIZE_HEADER
#define ALPS_NUMERIC_CHECK_SIZE_HEADER

#include <alps/type_traits/is_sequence.hpp>
#include <alps/utilities/stacktrace.hpp>

#include <boost/array.hpp>

#include <type_traits>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace alps {
    namespace numeric {

        namespace detail {

            template <class X, class Y>
            inline typename std::enable_if<!(is_sequence<X>::value || is_sequence<Y>::value),void>::type
                resize_same_as(X&, const Y&) {}

            template <class X, class Y>
            inline typename std::enable_if<is_sequence<X>::value && is_sequence<Y>::value,void>::type
            resize_same_as(X& a, const Y& y)  {
                a.resize(y.size());
            }

            template<typename T, typename U, std::size_t N>
            inline void resize_same_as(boost::array<T, N> & a, boost::array<U, N> const & y) {}
        }

        template<typename T, typename U>
        inline void check_size(T & /*a*/, U const & /*b*/) {}

        template<typename T, typename U>
        inline void check_size(std::vector<T> & a, std::vector<U> const & b) {
            if(a.size() == 0)
                detail::resize_same_as(a, b);
            else if(a.size() != b.size())
                boost::throw_exception(std::runtime_error("vectors must have the same size!" + ALPS_STACKTRACE));
        }

        template<typename T, typename U, std::size_t N, std::size_t M>
        inline void check_size(boost::array<T, N> & a, boost::array<U, M> const & b) {
            boost::throw_exception(std::runtime_error("boost::array s must have the same size!" + ALPS_STACKTRACE));
        }

        template<typename T, typename U, std::size_t N>
        inline void check_size(boost::array<T, N> & a, boost::array<U, N> const & b) {}

    }
}

#endif
