/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_ALEA_TYPE_TAG_H
#define ALPS_ALEA_TYPE_TAG_H

#include <alps/type_traits/type_tag.hpp>
#include <vector>
#include <valarray>

namespace alps {

template <class T>
struct type_tag<std::valarray<T> >
 : public boost::mpl::int_<256 + type_tag<T>::value> {};

template <class T>
struct type_tag<std::vector<T> >
 : public boost::mpl::int_<256 + type_tag<T>::value> {};

} // end namespace alps

#endif // ALPS_ALEA_TYPE_TAG_H
