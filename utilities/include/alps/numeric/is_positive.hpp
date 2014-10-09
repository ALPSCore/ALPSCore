/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_IS_POSITIVE_HPP
#define ALPS_NUMERIC_IS_POSITIVE_HPP

#include <alps/numeric/is_zero.hpp>

namespace alps { namespace numeric {

//
// is_positive
//

template<unsigned int N, class T>
inline bool is_positive(T x,
  typename boost::enable_if<boost::is_float<T> >::type* = 0)
{ return is_nonzero<N>(x) && x > T(0); }
template<unsigned int N, class T>
inline bool is_positive(T x,
  typename boost::enable_if<boost::is_integral<T> >::type* = 0)
{ return x > T(0); }
template<unsigned int N, class T>
inline bool is_positive(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_nonzero<N>(x) && x > T(0); }

template<class T>
inline bool is_positive(T x,
  typename boost::enable_if<boost::is_float<T> >::type* = 0)
{ return is_nonzero(x) && x > T(0); }
template<class T>
inline bool is_positive(T x,
  typename boost::enable_if<boost::is_integral<T> >::type* = 0)
{ return x > T(0); }
template<class T>
inline bool is_positive(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_nonzero(x) && x > T(0); }


} } // end namespace alps::numeric

#endif // ALPS_NUMERIC_IS_POSITIVE_HPP
