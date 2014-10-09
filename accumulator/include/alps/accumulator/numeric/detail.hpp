/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_ACCUMULATOR_NUMERIC_DETAIL_HEADER
#define ALPS_ACCUMULATOR_NUMERIC_DETAIL_HEADER

#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/resize.hpp>

// #include <alps/multi_array.hpp>

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

                // template<typename T, typename U, std::size_t D>
                // inline void check_size(alps::multi_array<T, D> & a, alps::multi_array<U, D> const & b) {}
                
            }
        }
    }
}

#endif
