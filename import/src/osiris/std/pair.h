/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef OSIRIS_STD_PAIR_H
#define OSIRIS_STD_PAIR_H

#include <alps/config.h>
#include <alps/osiris/dump.h>
#include <utility>

//=======================================================================
// pair templates
//-----------------------------------------------------------------------

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class T1, class T2>
inline alps::IDump& operator>>(alps::IDump& dump, std::pair<T1,T2>& x)
{
  return dump >> x.first >> x.second;
}

template <class T1, class T2>
inline alps::ODump& operator<<(alps::ODump& dump, const std::pair<T1,T2>& x)
{
  return dump << x.first << x.second;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_STD_PAIR_H
