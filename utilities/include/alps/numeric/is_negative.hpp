/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_IS_NEGATIVE_HPP
#define ALPS_NUMERIC_IS_NEGATIVE_HPP

#include <alps/numeric/is_nonzero.hpp>

namespace alps { namespace numeric {

//
// is_negative
//

template<unsigned int N, class T>
inline bool is_negative(T x,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_nonzero<N>(x) && x < T(0); }
template<unsigned int N, class T>
inline bool is_negative(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_nonzero<N>(x) && x < T(0); }

template<class T>
inline bool is_negative(T x,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_nonzero(x) && x < T(0); }
template<class T>
inline bool is_negative(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_nonzero(x) && x < T(0); }

inline bool is_negative(unsigned char) { return false; }
inline bool is_negative(unsigned short) { return false; }
inline bool is_negative(unsigned int) { return false; }
inline bool is_negative(unsigned long) { return false; }

} } // end namespace

#endif // ALPS_NUMERIC_IS_NEGATIVE_HPP
