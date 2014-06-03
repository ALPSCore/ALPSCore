/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef OSIRIS_STD_VALARRAY_H
#define OSIRIS_STD_VALARRAY_H

#include <alps/config.h>

#include <alps/osiris/dump.h>
#include <alps/osiris/std/impl.h>
#include <valarray>

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// deserialize a std::valarray container
template <class T>
inline alps::IDump& operator >> (alps::IDump& dump, std::valarray<T>& x)
{
  x.resize(uint32_t(dump));
  dump.read_array(x.size(),&(x[0]));
  return dump;
}

/// serialize a std::valarray container
template <class T>
inline alps::ODump& operator << (alps::ODump& dump,
                                   const std:: valarray<T>& x)
{
  dump << uint32_t(x.size());
  dump.write_array(x.size(),&(const_cast<std::valarray<T>&>(x)[0]));
  return dump;
}          

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_STD_VALARRAY_H
