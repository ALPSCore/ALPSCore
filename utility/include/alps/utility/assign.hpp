/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_UTILITY_ASSIGN_HPP
#define ALPS_UTILITY_ASSIGN_HPP

#include <valarray>

namespace alps {

template <class X, class Y> 
inline void assign(X& x,const Y& y) 
{
  x=y;
}

template <class X, class Y> 
inline void assign(std::valarray<X>& x, std::valarray<Y> const& y) 
{
  x.resize(y.size()); 
  for (std::size_t i=0;i<y.size();++i) 
    x[i]=y[i];
}

template <class X> 
inline void assign(std::valarray<X>& x, std::valarray<X> const& y) 
{
  x.resize(y.size()); 
  x=y;
}

} // end namespace alps

#endif // ALPS_UTILITY_ASSIGN_HPP
