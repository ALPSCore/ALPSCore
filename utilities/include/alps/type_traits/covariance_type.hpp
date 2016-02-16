/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_COVARIANCE_TYPE_H
#define ALPS_TYPE_TRAITS_COVARIANCE_TYPE_H

#include <alps/type_traits/average_type.hpp>
#include <alps/type_traits/element_type.hpp>
#include <alps/type_traits/is_sequence.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/mpl/if.hpp>


namespace alps {

template <class T>
struct covariance_type
{
 typedef typename boost::mpl::if_<
     is_sequence<T>,
     typename boost::numeric::ublas::matrix<
       typename average_type<typename element_type<T>::type>::type
     >,
     typename average_type<T>::type
   >::type type;
};


} // end namespace alps

#endif // ALPS_TYPE_TRAITS_COVARIANCE_TYPE_H
