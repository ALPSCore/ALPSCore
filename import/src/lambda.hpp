/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_LAMBDA_HPP
#define ALPS_LAMBDA_HPP

#include <boost/lambda/lambda.hpp>
#include <valarray>

namespace boost { 
namespace lambda {
  
template<class Act, class T> 
struct plain_return_type_2<arithmetic_action<Act>, std::valarray<T>, std::valarray<T> > {
  typedef std::valarray<T> type;
};

template<class Act, class T, class U> 
struct plain_return_type_2<arithmetic_action<Act>, std::valarray<T>, U> {
  typedef std::valarray<T> type;
};

template<class Act, class T, class U> 
struct plain_return_type_2<arithmetic_action<Act>, U, std::valarray<T> > {
  typedef std::valarray<T> type;
};

}
}

#endif // ALPS_LAMBDA_HPP
