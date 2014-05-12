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

#include <alps/parapack/process.h>
#include <iostream>

namespace mpi = boost::mpi;

int main(int argc, char** argv) {
  mpi::environment env(argc, argv);
  mpi::communicator world;

  alps::process_helper_mpi process(world, 4);

  if (world.rank() == 0) {
    alps::process_group g1 = process.allocate();
    for (int i = 0; i < g1.process_list.size(); ++i)
      std::cout << g1.process_list[i] << ' ';
    std::cout << std::endl;
    std::cout << process.num_groups() << ' ' << process.num_free() << std::endl;

    alps::process_group g2 = process.allocate();
    for (int i = 0; i < g2.process_list.size(); ++i)
      std::cout << g2.process_list[i] << ' ';
    std::cout << std::endl;
    std::cout << process.num_groups() << ' ' << process.num_free() << std::endl;

    process.release(g1);
    std::cout << process.num_groups() << ' ' << process.num_free() << std::endl;

    g1 = process.allocate();
    for (int i = 0; i < g1.process_list.size(); ++i)
      std::cout << g1.process_list[i] << ' ';
    std::cout << std::endl;
    std::cout << process.num_groups() << ' ' << process.num_free() << std::endl;

    process.release(g2);
    std::cout << process.num_groups() << ' ' << process.num_free() << std::endl;

    process.release(g1);
    std::cout << process.num_groups() << ' ' << process.num_free() << std::endl;
  }

  process.halt();
  while (true) {
    if (process.check_halted()) break;
  }
}
