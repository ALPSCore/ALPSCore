/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_NUMERIC_MATRIX_DETAIL_PRINT_VECTOR_HPP
#define ALPS_NUMERIC_MATRIX_DETAIL_PRINT_VECTOR_HPP

#include <ostream>

namespace alps {
namespace numeric {
namespace detail {

template <typename Vector>
void print_vector(std::ostream& os, Vector const& v)
{
    os<<"[";
    if(v.size() > 0)
    {
        for(unsigned int i=0;i<v.size()-1;++i)
          os<<v(i)<<", ";
        os<< v(v.size()-1);
    }
    os << "]"<<std::endl;
}

} // end namespace detail
} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_MATRIX_DETAIL_PRINT_VECTOR_HPP
