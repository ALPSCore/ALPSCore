/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef PARAPACK_MEASUREMENT_H
#define PARAPACK_MEASUREMENT_H

#include <alps/alea.h>

namespace alps {

ALPS_DECL void merge_clone(alps::ObservableSet& total, alps::ObservableSet const& clone,
  bool same_weight);

ALPS_DECL void merge_random_clone(alps::ObservableSet& total, alps::ObservableSet const& clone);

template<typename T>
void add_constant(Observable& obs, T const& val) {
  if (dynamic_cast<SimpleRealObservable*>(&obs) &&
      dynamic_cast<SimpleRealObservable*>(&obs)->count() < 2) obs << val;
}

} // end namespace alps

#endif // PARAPACK_MEASUREMENT_H
