/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2010 by Synge Todo <wistaria@comp-phys.org>
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

#include <alps/parapack/filelock.h>
#include <alps/parapack/process.h>
#include <iostream>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/timer.hpp>

namespace mpi = boost::mpi;

int main(int argc, char **argv) {
  mpi::environment env(argc, argv);
  mpi::communicator world;

  if (world.size() < 2) {
    std::cerr << "too little number of processes\n";
    return -1;
  }

  boost::filesystem::path file("filelock_mpi");
  alps::filelock lock(file);

  // serial lock
  if (world.rank() == 0) {
    lock.lock();
    std::cerr << "process #0 lock acquired\n";
    world.barrier();
    sleep(2);
    lock.release();
    std::cerr << "process #0 lock released\n";
  } else if (world.rank() == 1) {
    world.barrier();
    std::cerr << "process #1 lock trying\n";
    lock.lock();
    std::cerr << "process #1 lock acquired\n";
    lock.release();
    std::cerr << "process #1 lock released\n";
  } else {
    world.barrier();
  }

  world.barrier();

  // random lock
  {
    std::cerr << "process #" << world.rank() << " lock trying\n";
    alps::filelock lock2(file, true);
    std::cerr << "process #" << world.rank() << " lock acquired\n";
    sleep(1);
    lock2.release();
  }
  std::cerr << "process #" << world.rank() << " lock released\n";
}
