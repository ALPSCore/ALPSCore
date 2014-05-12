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
#include <alps/parapack/clone_mpi.h>
#include <alps/parapack/clone_proxy_mpi.h>
#include <alps/parapack/parallel_factory.h>
#include <alps/parapack/task.h>
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

  if (world.size() < 2) {
    std::cerr << "too little number of processes\n";
    return -1;
  }
  int nw = world.size() - 1;
  if (alps::is_master())
    std::cerr << "number of workers is " << nw << std::endl;

  alps::process_helper_mpi process(world, nw);
  mpi::communicator work = process.comm_work();

  alps::Parameters params;
  if (world.rank() == 0) params.parse(std::cin);

  if (world.rank() == 0) {

    alps::process_group procs;
    procs.group_id = 0;
    for (int p = 1; p < world.size(); ++p) procs.process_list.push_back(alps::Process(p));
    alps::tid_t tid = 0;
    alps::cid_t cid = 0;
    alps::clone_info info;
    std::string base = "clone.dump";

    alps::clone_proxy_mpi proxy(world);

    for (int i = 0; i < 2; ++i) {
      proxy.start(tid, cid, procs, params, boost::filesystem::initial_path(), base, (i == 0));
      bool running = true;
      double breakpoint = (i == 0) ? 0.3 : 1.0;

      while (true) {
        while (world.iprobe(1, alps::mcmp_tag::clone_info)) {
          alps::clone_info_msg_t msg;
          world.recv(1, alps::mcmp_tag::clone_info, msg);
          info = msg.info;
          if (running) {
            std::cerr << "task " << msg.task_id+1 << ", clone " << msg.clone_id+1 << " is "
                      << info.phase() << " (" << info.progress() * 100 << "% done)\n";
            if (info.progress() >= breakpoint) {
              if (info.progress() < 1) {
                std::cerr << "suspending task " << msg.task_id+1
                          << ", clone " << msg.clone_id+1 << std::endl;
                world.send(1, alps::mcmp_tag::clone_suspend);
              } else {
                std::cerr << "halting task " << msg.task_id+1
                          << ", clone " << msg.clone_id+1 << std::endl;
                world.send(1, alps::mcmp_tag::clone_halt);
              }
              running = false;
            }
          }
        }
        if (world.iprobe(1, alps::mcmp_tag::clone_suspend)) {
          alps::clone_info_msg_t msg;
          world.recv(1, alps::mcmp_tag::clone_suspend, msg);
          info = msg.info;
          std::cerr << "task " << msg.task_id+1
                    << ", clone " << msg.clone_id+1 << " is suspended\n";
          alps::oxstream oxs(std::cerr);
          oxs << info;
          break;
        }
        if (world.iprobe(1, alps::mcmp_tag::clone_halt)) {
          alps::clone_halt_msg_t msg;
          world.recv(1, alps::mcmp_tag::clone_halt, msg);
          std::cerr << "task " << msg.task_id+1
                    << ", clone " << msg.clone_id+1 << " is halted\n";
          alps::oxstream oxs(std::cerr);
          oxs << info;
          BOOST_FOREACH(std::string const& file, info.checkpoints())
            boost::filesystem::remove(boost::filesystem::path(file));
          break;
        }
        if (running) {
          sleep(1);
          world.send(1, alps::mcmp_tag::clone_info);
        }
      }
    }

  } else {

    for (int i = 0; i < 2; ++i) {
      alps::clone_create_msg_t msg;
      world.recv(0, alps::mcmp_tag::clone_create, msg);
      alps::abstract_clone* clone = new alps::clone_mpi(world, world, work, msg);
      std::cerr << "worker started on node " << world.rank() << std::endl;
      while (!clone->halted()) clone->run();
      delete clone;
      std::cerr << "worker halted on node " << world.rank() << std::endl;
    }

  }

  process.halt();
  while (!process.check_halted()) {}
}
