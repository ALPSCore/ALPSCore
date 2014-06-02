/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef OSIRIS_BOOST_ARRAY_HPP
#define OSIRIS_BOOST_ARRAY_HPP

#include <alps/config.h>
#include <alps/osiris/dump.h>

#include <boost/array.hpp>

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template<class T, std::size_t N> 
inline alps::IDump& operator>>(alps::IDump& dump, boost::array<T, N>& x)
{
  dump.read_array(N,&(x[0]));
  return dump;
}

template<class T, std::size_t N> 
inline alps::ODump& operator<<(alps::ODump& dump, const boost::array<T,N>& x)
{
  dump.write_array(N,&(x[0]));
  return dump;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_BOOST_ARRAY_HPP
