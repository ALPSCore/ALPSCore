/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef PARAPACK_WORKER_H
#define PARAPACK_WORKER_H

#include "measurement.h"
#include "worker_factory.h"
#include "rng_helper.h"
#include <alps/config.h>
#include <alps/lattice.h>
#include <alps/model.h>

namespace alps {
namespace parapack {

//
// workers
//

class dumb_worker : public abstract_worker {
public:
  virtual ~dumb_worker();
  static void print_copyright(std::ostream& out);
  void init_observables(Parameters const& param, ObservableSet& obs);
  void run(ObservableSet& obs);
  void load(IDump& dp);
  void save(ODump& dp) const;
  bool is_thermalized() const;
  double progress() const;
};

class ALPS_DECL mc_worker : public abstract_worker, protected rng_helper {
public:
  mc_worker(Parameters const& params);
  virtual ~mc_worker();
  virtual void load_worker(IDump& dp);
  virtual void save_worker(ODump& dp) const;
};

template<typename G = graph_helper<>::graph_type>
class lattice_mc_worker : public mc_worker, protected graph_helper<G> {
public:
  lattice_mc_worker(Parameters const& params) : mc_worker(params), graph_helper<G>(params) {}
  virtual ~lattice_mc_worker() {}
};

template<typename G = graph_helper<>::graph_type, typename I = short>
class latticemodel_mc_worker : public lattice_mc_worker<G>, protected model_helper<I> {
public:
  latticemodel_mc_worker(Parameters const& params)
    : lattice_mc_worker<G>(params), model_helper<I>(*this, params) {}
  virtual ~latticemodel_mc_worker() {}
};

} // end namespace parapack
} // end namespace alps

#endif // PARAPACK_WORKER_H
