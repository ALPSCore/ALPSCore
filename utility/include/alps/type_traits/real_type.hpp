/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_REAL_TYPE_H
#define ALPS_TYPE_TRAITS_REAL_TYPE_H

#include <boost/mpl/bool.hpp>
#include <complex>
#include <valarray>
#include <vector>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <class T>
struct real_type 
{
  typedef T type;
};

template <class T>
struct real_type<std::complex<T> > : public real_type<T> {};


template <class T>
struct real_type<std::valarray<T> > {
  typedef std::valarray<typename real_type<T>::type> type;
};

template <class T, class A>
struct real_type<std::vector<T,A> > {
  typedef std::vector<typename real_type<T>::type,A> type;
};



} // end namespace alps

#endif // ALPS_TYPE_TRAITS_REAL_TYPE_H
