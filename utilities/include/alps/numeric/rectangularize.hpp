/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file rectangularize.hpp
    @brief Defines a function to rectangularize a vector-of-vectors
*/

#ifndef ALPS_NUMERIC_RECTANGULARIZE_HPP_08679765945f4171af5c396beff05503
#define ALPS_NUMERIC_RECTANGULARIZE_HPP_08679765945f4171af5c396beff05503

#include <vector>
#include <boost/foreach.hpp>

namespace alps {
    namespace numeric {

        /// Make sure that vector-of-vectors is a rectangular matrix (generic dummy template)
        template <typename T>
        void rectangularize(const T&) {}

        /// Make sure that vector-of-vectors is a rectangular matrix
        /**  Converts an uneven vector-of-vectors to a rectangular matrix,
             resizing rows as needed (that is, right-padding the rows with 0)

             @note Does not work with 3-tensors.
         */
        template <typename T>
        void rectangularize(std::vector< std::vector<T> >& vec)
        {
            std::size_t mx_size=0;
            BOOST_FOREACH(std::vector<T>& val, vec) {
                // FIXME: // would be needed for 3-tensors, but not quite working:
                // FIXME: rectangularize(val);
                if (mx_size<val.size()) mx_size=val.size();
            }
            BOOST_FOREACH(std::vector<T>& val, vec) {
                val.resize(mx_size);
            }
        }

    }
}

#endif /* ALPS_NUMERIC_RECTANGULARIZE_HPP_08679765945f4171af5c396beff05503 */
