/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_NUMERIC_MATRIX_DETAIL_PRINT_MATRIX_HPP
#define ALPS_NUMERIC_MATRIX_DETAIL_PRINT_MATRIX_HPP

#include <ostream>


namespace alps {
namespace numeric {
namespace detail {

template <typename Matrix>
void print_matrix(std::ostream& os, Matrix const& m)
{
    os << "[";
    for(typename Matrix::size_type i=0; i < num_rows(m); ++i)
    {
        os << "[ ";
        if(num_cols(m) > 0)
        {
            for(typename Matrix::size_type j=0; j < num_cols(m)-1; ++j)
                os << m(i,j) << ", ";
            os << m(i,num_cols(m)-1);
        }
        os << "]";
        if(i+1 < num_rows(m))
            os << "," << std::endl;
    }
    os << "]" << std::endl;
}

} // end namespace detail
} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_MATRIX_DETAIL_PRINT_MATRIX_HPP
