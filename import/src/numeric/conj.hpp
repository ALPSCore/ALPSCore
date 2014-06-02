/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_CONJ_HPP
#define ALPS_NUMERIC_CONJ_HPP

#include <alps/numeric/matrix/entity.hpp>
#include <complex>
#include <boost/type_traits/is_fundamental.hpp>
#include <boost/static_assert.hpp>

namespace alps { namespace numeric {


template <class T>
typename boost::enable_if<boost::is_fundamental<T>,T>::type conj (T x)
{ 
  return x;
}

// if std::complex<T> is used std::conj will be called by argument dependent look-up

template <class T>
typename boost::enable_if<boost::is_fundamental<T>,void>::type conj_inplace(T& t, tag::scalar)
{
}

template <class T>
void conj_inplace(std::complex<T>& x, tag::scalar)
{
  BOOST_STATIC_ASSERT((boost::is_fundamental<T>::value));
  using std::conj;
  x = conj(x);
}

template <typename T>
void conj_inplace(T& t)
{
    conj_inplace(t, typename get_entity<T>::type());
}

} }  // end namespace alps::numeric

#endif // ALPS_NUMERIC_CONJ_HPP
