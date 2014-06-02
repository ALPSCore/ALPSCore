/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_MATRIX_CONJ_HPP
#define ALPS_MATRIX_CONJ_HPP

#include <alps/numeric/conj.hpp>

namespace alps {
namespace numeric {

/**
  * Does an conj_inplace on all elements of the matrix
  */
template <typename Matrix>
void conj_inplace(Matrix& m, tag::matrix)
{
    // TODO discuss conj() for matrix may be misleading:
    //      elementwise conj() <-> adjoint()
    //
    typedef typename Matrix::col_element_iterator col_element_iterator;
    for(typename Matrix::size_type j=0; j < num_cols(m); ++j)
       for(std::pair<col_element_iterator,col_element_iterator> range = col(m,j); range.first != range.second; ++range.first)
           conj_inplace(*range.first);
}

template <typename Vector>
void conj_inplace(Vector& t, tag::vector)
{
    typename Vector::iterator const end = t.end();
    for(typename Vector::iterator it=t.begin(); it != end; ++it)
        conj_inplace(*it);
}

/**
  * Returns a matrix or a vector of type T containing a the complex conjugates of the original object.
  * It does an element-wise conjugation.
  */
template <typename T>
typename boost::enable_if<
      boost::mpl::or_<
            boost::is_same<typename get_entity<T>::type, tag::vector>
          , boost::is_same<typename get_entity<T>::type, tag::matrix>
       >
    , T
>::type conj(T const& t)
{
    T r(t);
    conj_inplace(r);
    return r;
}

} // end namespace numeric
} // end namespace alps 

#endif //ALPS_MATRIX_CONJ_HPP
