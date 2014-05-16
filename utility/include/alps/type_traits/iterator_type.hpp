/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_TYPE_TRAITS_ITERATOR_TYPE_HPP
#define ALPS_TYPE_TRAITS_ITERATOR_TYPE_HPP

#include <valarray>

namespace alps {

template <class Collection> 
struct iterator_type
{
  typedef typename Collection::iterator type;
};

template <class Collection> 
struct const_iterator_type
{
  typedef typename Collection::iterator type;
};

template <class T> 
struct iterator_type<std::valarray<T> >
{
  typedef T* type;
};

template <class T> 
struct const_iterator_type<std::valarray<T> >
{
  typedef T const * type;
};

} // namespace alps

#endif // ALPS_TYPE_TRAITS_ITERATOR_TYPE_HPP
