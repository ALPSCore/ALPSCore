/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2005-2010 by Synge Todo <wistaria@comp-phys.org>
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

int main(int argc, char **argv) {
  mpi::environment env(argc, argv);
  mpi::communicator world;
  if (world.size() >= 4) {
    alps::process_helper_mpi process(world, 4);
    mpi::communicator cg = process.comm_ctrl();
    mpi::communicator cl = process.comm_work();
    mpi::communicator hd = process.comm_head();
    for (int p = 0; p < world.size(); ++p) {
      if (world.rank() == p) {
        std::cout << "rank: " << world.rank();
        if (cg)
          std::cout << ", global rank = " << cg.rank();
        if (cl)
          std::cout << ", local rank = " << cl.rank();
        if (hd)
          std::cout << ", head rank = " << hd.rank();
        std::cout << std::endl;
      }
      std::cout << std::flush;
      world.barrier();
    }
    process.halt();
    while (!process.check_halted()) {}
  }
}
