/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef OSIRIS_BOOST_UBLAS_H
#define OSIRIS_BOOST_UBLAS_H

// #include <palm/config.h>
#include <boost/numeric/ublas/vector.hpp>
#include <alps/osiris/std/impl.h>

/// deserialize a boost::numeric::ublas::vector container

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class T, class STORAGE>
inline alps::IDump& operator >> (alps::IDump& dump,
                                   boost::numeric::ublas::vector<T,STORAGE>& x)
{
  x.resize(uint32_t(dump));
  if (x.size())
    dump.read_array(x.size(),&(x[0]));
  return dump;
}

/// serialize a boost::numeric::ublas::vector container
template <class T, class STORAGE>
inline alps::ODump& operator << (alps::ODump& dump,
                                   const boost::numeric::ublas::vector<T,STORAGE>& x)
{
  dump << uint32_t(x.size());
  if(x.size())
    dump.write_array(x.size(),&(x[0]));
  return dump;
}          

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_BOOST_UBLAS_H
