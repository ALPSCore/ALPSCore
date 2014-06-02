/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_IS_NONZERO_HPP
#define ALPS_NUMERIC_IS_NONZERO_HPP

#include <alps/numeric/is_zero.hpp>

namespace alps { namespace numeric {

//
// is_nonzero
//

/// \brief checks if a number is not zero
/// in case of a floating point number, absolute values less than
/// epsilon (1e-50 by default) count as zero
/// \return returns true if the value is not zero

template<unsigned int N, class T>
inline bool is_nonzero(T x)
{ 
  return !is_zero<N>(x); 
}


template<class T>
inline bool is_nonzero(T x)
{ 
  return !is_zero(x); 
}


} } // end namespace alps::alea

#endif // ALPS_NUMERIC_IS_NONZERO_HPP
