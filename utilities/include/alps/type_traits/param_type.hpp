/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_PARAM_TYPE_H
#define ALPS_TYPE_TRAITS_PARAM_TYPE_H

#include <alps/type_traits/is_scalar.hpp>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <typename T> 
struct param_type : public boost::mpl::if_<
      typename is_scalar<typename boost::remove_cv<T>::type>::type
    , T
    , typename boost::add_reference<typename boost::add_const<T>::type>::type
> {};

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_NORM_TYPE_H
