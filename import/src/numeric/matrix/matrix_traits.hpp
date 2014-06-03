/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#ifndef ALPS_MATRIX_TRAITS_HPP
#define ALPS_MATRIX_TRAITS_HPP

namespace alps {
namespace numeric {

    template <typename Matrix>
    struct associated_diagonal_matrix
    {
    };

    template <typename Matrix>
    struct associated_real_diagonal_matrix
    {
    };

    template <typename Matrix>
    struct associated_vector
    {
    };

    template <typename Matrix>
    struct associated_real_vector
    {
    };

} // end namespace numeric
} // end namespace alps
#endif //ALPS_MATRIX_TRAITS_HPP
