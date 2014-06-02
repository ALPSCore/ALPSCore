/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: observableset.h 3696 2010-01-13 15:25:30Z gamperl $ */

#ifndef ALPS_ALEA_OBSERVABLEFACTORY_H
#define ALPS_ALEA_OBSERVABLEFACTORY_H

#include <alps/config.h>
#include <alps/factory.h>
#include <alps/alea/observable.h>

namespace alps {

/** A class to collect the various measurements performed in a simulation
    It is implemented as a map, with std::string as key type */

class ObservableFactory : public factory<uint32_t,Observable>
{
public:
  ObservableFactory();
  template <class T>
  void register_observable() { register_type<T>(T::version); }
};

} // end namespace alps

#endif // ALPS_ALEA_OBSERVABLEFACTORY_H
