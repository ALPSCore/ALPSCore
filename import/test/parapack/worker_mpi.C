/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2008 by Synge Todo <wistaria@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#include <example/single/ising.h>
#include <example/multiple/ising.h>
#include <parapack/parallel_factory.h>
#include <parapack/util.h>
#include <iostream>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/timer.hpp>

namespace mpi = boost::mpi;

const bool single_worker_registered =
  alps::parapack::worker_factory::instance()
  ->register_worker<single_ising_worker>("ising");
const bool parallel_worker_registered =
  alps::parapack::parallel_worker_factory::instance()
  ->register_worker<parallel_ising_worker>("ising");

int main(int argc, char **argv) {

  mpi::environment env(argc, argv);
  mpi::communicator world;

  alps::Parameters p;
  if (world.rank() == 0) {
    p.parse(std::cin);
    for (int i = 1; i < world.size(); ++i) world.send(i, 0, p);
  } else {
    world.recv(0, 0, p);
  }

  boost::filesystem::path dumpfile = "worker.dump." + alps::id2string(world.rank());

  std::vector<alps::ObservableSet> obs;
  boost::shared_ptr<alps::parapack::abstract_worker> worker;
  if (world.size() > 1)
    worker = alps::parapack::parallel_worker_factory::instance()->make_worker(world, p);
  else
    worker = alps::parapack::worker_factory::instance()->make_worker(p);
  worker->init_observables(p, obs);

  world.barrier();
  boost::timer tm;

  bool dumped = false;
  while (worker->progress() < 1) {
    bool is_thermalized = worker->is_thermalized();
    worker->run(obs);
    if (!is_thermalized && worker->is_thermalized())
      BOOST_FOREACH(alps::ObservableSet& m, obs) m.reset(true);
    if (worker->progress() > 0.5 && !dumped) {
      if (world.rank() == 0)
        std::cerr << "saving to dump files, progress = " << worker->progress() << std::endl;
      alps::OXDRFileDump dump(dumpfile);
      BOOST_FOREACH(alps::ObservableSet& m, obs) dump << m;
      worker->save_worker(dump);
      dumped = true;
    }
  }

  world.barrier();
  double t = tm.elapsed();

  worker.reset();

  if (world.rank() == 0) {
    BOOST_FOREACH(alps::ObservableSet& m, obs) std::cout << m;
    std::cerr << "Elapsed time = " << t << " sec\n";
  }

  // restart from dump
  if (world.size() > 1)
    worker = alps::parapack::parallel_worker_factory::instance()->make_worker(world, p);
  else
    worker = alps::parapack::worker_factory::instance()->make_worker(p);
  {
    alps::IXDRFileDump dump(dumpfile);
    BOOST_FOREACH(alps::ObservableSet& m, obs) dump >> m;
    worker->load_worker(dump);
    if (world.rank() == 0)
      std::cerr << "loading from dump files, progress = " << worker->progress() << std::endl;
  }
  while (worker->progress() < 1) {
    bool is_thermalized = worker->is_thermalized();
    worker->run(obs);
    if (!is_thermalized && worker->is_thermalized())
      BOOST_FOREACH(alps::ObservableSet& m, obs) m.reset(true);
  }
  worker.reset();

  if (world.rank() == 0)
    BOOST_FOREACH(alps::ObservableSet& m, obs) std::cout << m;

  boost::filesystem::remove(dumpfile);
}
