/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


/* $Id: nobinning.h 3520 2009-12-11 16:49:53Z gamperl $ */

#ifndef ALPS_VECTOR_VALARRAY_CONVERSION
#define ALPS_VECTOR_VALARRAY_CONVERSION

#include <alps/utilities/data.hpp>
#include <vector>
#include <valarray>
#include <algorithm>


namespace alps {

  template <class T>
  struct vector2valarray_type {
    typedef T type;
  };

  template <class T, class A>
  struct vector2valarray_type<std::vector<T, A> > {
    typedef std::valarray<T> type;
  };


  template <class T>
  struct valarray2vector_type {
    typedef T type;
  };

  template <class T>
  struct valarray2vector_type<std::valarray<T> > {
    typedef std::vector<T> type;
  };


  namespace numeric {

    template <class T>
    T valarray2vector (const T& value) {return value;}

    template<class T>
    std::vector<T> valarray2vector(std::valarray<T> const & from)
    {
      std::vector<T> to;
      to.reserve(from.size());
      std::copy(data(from),data(from)+from.size(),std::back_inserter(to));
      return to;
    }



    template <class T>
    T vector2valarray (const T& value) {return value;}

    template<class T>
    std::valarray<T> vector2valarray(std::vector<T> const & from)
    {
      std::valarray<T> to(from.size());
      std::copy(from.begin(),from.end(),data(to));
      return to;
    }

    template<class T1, class T2>
    std::valarray<T2> vector2valarray(std::vector<T1> const & from)
    {
      std::valarray<T2> to(from.size());
      std::copy(from.begin(),from.end(),data(to));
      return to;
    }

  }
}

#endif
